import Foundation
import MLX

/// The align model in MLX: SPyNet, warp, latent_up with FiLM, vae2 decoder, on the GPU.
///
/// A `.mlxmodel` folder holds `model.safetensors` and `config.json`, written by
/// `python -m whoodeo.mlxport.convert … --app`. The math follows
/// src/whoodeo/mlxport/model.py, which is checked against PyTorch.
/// Layouts are NHWC. Flexible sizes: SPyNet resizes to a multiple of 32
/// itself, the generator pads to a multiple of 4 and the result is cropped.
nonisolated final class MLXAlignModel {
    struct Config: Decodable {
        var step: Int?
        var activation: String
        var in_frames: Int
        var condition_size: Int
        var encoder_blocks: [Int]
        var blocks: Int
        var block_widths: [Int]
        var latent: Int
        var vae_blocks: Int
        var decoder_blocks_run: [Int]
        var align: String?
        /// float16 or float32 for latent_up and the decoder; SPyNet and the warp stay fp32.
        var compute: String?
    }

    static let fileExtension = "mlxmodel"
    private static let dilations = [1, 2, 4]
    private static let factor = 4

    let config: Config
    /// Precision of latent_up and the decoder. float16 as the Core ML export
    /// computes; WHOODEO_MLX_PRECISION=float32 overrides it for checks.
    let dtype: DType
    private let p: [String: MLXArray]
    private let relu: Bool
    private let mean = MLXArray([Float(0.485), 0.456, 0.406])
    private let std = MLXArray([Float(0.229), 0.224, 0.225])

    init(url: URL) throws {
        let data = try Data(contentsOf: url.appendingPathComponent("config.json"))
        config = try JSONDecoder().decode(Config.self, from: data)
        let precision = ProcessInfo.processInfo.environment["WHOODEO_MLX_PRECISION"] ?? config.compute ?? "float16"
        dtype = precision == "float32" ? .float32 : .float16
        let stored = try loadArrays(url: url.appendingPathComponent("model.safetensors"))
        var cast: [String: MLXArray] = [:]
        for (key, value) in stored {
            cast[key] = key.hasPrefix("spynet.") || dtype == .float32 ? value : value.asType(dtype)
        }
        p = cast
        relu = config.activation == "relu"
        guard config.in_frames == 3, p["spynet.basic_module.0.basic_module.0.conv.weight"] != nil else {
            throw CLIError("das MLX-Modell braucht 3 Bilder und SPyNet")
        }
        eval(Array(p.values))
    }

    // MARK: ops, as mlxport/ops.py

    private func act(_ x: MLXArray) -> MLXArray {
        relu ? maximum(x, Float(0)) : x * sigmoid(x)
    }

    private func weight(_ name: String) -> MLXArray {
        guard let value = p[name] else { fatalError("MLX-Modell: \(name) fehlt") }
        return value
    }

    private func conv(_ name: String, _ x: MLXArray, stride: Int = 1, padding: Int = 1, dilation: Int = 1) -> MLXArray {
        var y = conv2d(
            x, weight(name + ".weight"),
            stride: IntOrPair(integerLiteral: stride),
            padding: IntOrPair(integerLiteral: padding),
            dilation: IntOrPair(integerLiteral: dilation)
        )
        if let bias = p[name + ".bias"] { y = y + bias }
        return y
    }

    private static func avgPool2(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        return x.reshaped([s[0], s[1] / 2, 2, s[2] / 2, 2, s[3]]).mean(axes: [2, 4])
    }

    private static func upsampleNearest2(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let wide = broadcast(x.expandedDimensions(axes: [2, 4]), to: [s[0], s[1], 2, s[2], 2, s[3]])
        return wide.reshaped([s[0], 2 * s[1], 2 * s[2], s[3]])
    }

    /// Indices and weights of 1-D bilinear sampling, as torch upsample_bilinear2d.
    private static func sourceIndex(_ sizeIn: Int, _ sizeOut: Int, alignCorners: Bool) -> (MLXArray, MLXArray, MLXArray) {
        var i0 = [Int32](repeating: 0, count: sizeOut)
        var i1 = [Int32](repeating: 0, count: sizeOut)
        var frac = [Float](repeating: 0, count: sizeOut)
        let scale: Float = alignCorners
            ? (sizeOut > 1 ? Float(sizeIn - 1) / Float(sizeOut - 1) : 0)
            : Float(sizeIn) / Float(sizeOut)
        for d in 0..<sizeOut {
            let src: Float = alignCorners ? Float(d) * scale : max((Float(d) + 0.5) * scale - 0.5, 0)
            let low = min(Int(src.rounded(.down)), sizeIn - 1)
            i0[d] = Int32(low)
            i1[d] = Int32(min(low + 1, sizeIn - 1))
            frac[d] = src - Float(low)
        }
        return (MLXArray(i0), MLXArray(i1), MLXArray(frac))
    }

    private static func resizeBilinear(_ x: MLXArray, _ height: Int, _ width: Int, alignCorners: Bool) -> MLXArray {
        let h = x.dim(1), w = x.dim(2)
        if h == height, w == width, !alignCorners { return x }
        let (y0, y1, fy) = sourceIndex(h, height, alignCorners: alignCorners)
        let (x0, x1, fx) = sourceIndex(w, width, alignCorners: alignCorners)
        let fyb = fy.reshaped([1, height, 1, 1])
        let fxb = fx.reshaped([1, 1, width, 1])
        var top = take(x, y0, axis: 1)
        var bottom = take(x, y1, axis: 1)
        top = take(top, x0, axis: 2) * (1 - fxb) + take(top, x1, axis: 2) * fxb
        bottom = take(bottom, x0, axis: 2) * (1 - fxb) + take(bottom, x1, axis: 2) * fxb
        return top * (1 - fyb) + bottom * fyb
    }

    /// common.pad_to: bottom and right to a multiple of `factor`, reflect when it fits.
    private static func padReflect(_ x: MLXArray, _ factor: Int) -> MLXArray {
        let h = x.dim(1), w = x.dim(2)
        let padH = (factor - h % factor) % factor
        let padW = (factor - w % factor) % factor
        if padH == 0 && padW == 0 { return x }
        let reflect = h > padH && w > padW
        let rows = (0..<h).map { Int32($0) } + (0..<padH).map { Int32(reflect ? h - 2 - $0 : h - 1) }
        let cols = (0..<w).map { Int32($0) } + (0..<padW).map { Int32(reflect ? w - 2 - $0 : w - 1) }
        return take(take(x, MLXArray(rows), axis: 1), MLXArray(cols), axis: 2)
    }

    /// grid_sample(bilinear, border, align_corners=True) with the flow in pixels (dx, dy).
    static func flowWarp(_ x: MLXArray, _ flow: MLXArray) -> MLXArray {
        let n = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let flows = split(flow, parts: 2, axis: -1)
        let xs = MLXArray((0..<w).map { Float($0) }).reshaped([1, 1, w])
        let ys = MLXArray((0..<h).map { Float($0) }).reshaped([1, h, 1])
        var gx = xs + flows[0].squeezed(axis: -1)
        var gy = ys + flows[1].squeezed(axis: -1)
        let denomW = Float(max(w - 1, 1)), denomH = Float(max(h - 1, 1))
        gx = 2 * gx / denomW - 1
        gy = 2 * gy / denomH - 1
        let ix = clip((gx + 1) / 2 * Float(w - 1), min: Float(0), max: Float(w - 1))
        let iy = clip((gy + 1) / 2 * Float(h - 1), min: Float(0), max: Float(h - 1))
        let x0f = floor(ix), y0f = floor(iy)
        let fx = (ix - x0f).expandedDimensions(axis: -1)
        let fy = (iy - y0f).expandedDimensions(axis: -1)
        let x0 = x0f.asType(.int32), y0 = y0f.asType(.int32)
        let x1 = minimum(x0 + 1, Int32(w - 1)), y1 = minimum(y0 + 1, Int32(h - 1))
        let flat = x.reshaped([n * h * w, c])
        let base = (MLXArray((0..<n).map { Int32($0 * h * w) })).reshaped([n, 1, 1])
        func gather(_ yi: MLXArray, _ xi: MLXArray) -> MLXArray {
            let index = (base + yi * Int32(w) + xi).reshaped([-1])
            return take(flat, index, axis: 0).reshaped([n, h, w, c])
        }
        return gather(y0, x0) * ((1 - fx) * (1 - fy))
            + gather(y0, x1) * (fx * (1 - fy))
            + gather(y1, x0) * ((1 - fx) * fy)
            + gather(y1, x1) * (fx * fy)
    }

    // MARK: SPyNet

    private func spynetBlock(_ level: Int, _ input: MLXArray) -> MLXArray {
        var x = input
        for index in 0..<5 {
            x = conv("spynet.basic_module.\(level).basic_module.\(index).conv", x, padding: 3)
            if index < 4 { x = maximum(x, Float(0)) }
        }
        return x
    }

    private func computeFlow(_ refIn: MLXArray, _ suppIn: MLXArray) -> MLXArray {
        var ref = [(refIn - mean) / std]
        var supp = [(suppIn - mean) / std]
        for _ in 0..<5 {
            ref.append(Self.avgPool2(ref.last!))
            supp.append(Self.avgPool2(supp.last!))
        }
        ref.reverse()
        supp.reverse()
        let n = refIn.dim(0)
        var flow = MLXArray.zeros([n, refIn.dim(1) / 32, refIn.dim(2) / 32, 2])
        for level in 0..<6 {
            let up = level == 0
                ? flow
                : Self.resizeBilinear(flow, 2 * flow.dim(1), 2 * flow.dim(2), alignCorners: true) * Float(2)
            let input = concatenated([ref[level], Self.flowWarp(supp[level], up), up], axis: -1)
            flow = up + spynetBlock(level, input)
        }
        return flow
    }

    /// Flow (N, H, W, 2) from ref to supp, any size, as spynet.SPyNet.forward.
    func spynet(_ ref: MLXArray, _ supp: MLXArray) -> MLXArray {
        let height = ref.dim(1), width = ref.dim(2)
        let hUp = (height + 31) / 32 * 32, wUp = (width + 31) / 32 * 32
        let refUp = Self.resizeBilinear(ref, hUp, wUp, alignCorners: false)
        let suppUp = Self.resizeBilinear(supp, hUp, wUp, alignCorners: false)
        let flow = Self.resizeBilinear(computeFlow(refUp, suppUp), height, width, alignCorners: false)
        let scale = MLXArray([Float(width) / Float(wUp), Float(height) / Float(hUp)])
        return flow * scale
    }

    /// Previous and next warped onto the middle low and clamped, as align_neighbors.
    func align(_ low: MLXArray) -> MLXArray {
        let parts = split(low, parts: 3, axis: -1)
        let b = low.dim(0)
        let neigh = concatenated([parts[0], parts[2]], axis: 0)
        let ref = concatenated([parts[1], parts[1]], axis: 0)
        let warped = clip(Self.flowWarp(neigh, spynet(ref, neigh)), min: Float(0), max: Float(1))
        return concatenated([warped[0 ..< b], parts[1], warped[b ..< 2 * b]], axis: -1)
    }

    // MARK: latent_up

    private func film(_ condition: MLXArray) -> [(MLXArray, MLXArray)] {
        let hidden = act(matmul(condition, weight("film.inner.weight").T) + weight("film.inner.bias"))
        let out = matmul(hidden, weight("film.outer.weight").T) + weight("film.outer.bias")
        let b = condition.dim(0)
        var pairs: [(MLXArray, MLXArray)] = []
        var start = 0
        for width in config.block_widths {
            let gamma = out[0..., start ..< start + width].reshaped([b, 1, 1, width])
            let beta = out[0..., (start + width) ..< (start + 2 * width)].reshaped([b, 1, 1, width])
            pairs.append((gamma, beta))
            start += 2 * width
        }
        return pairs
    }

    private func block(_ name: String, _ x: MLXArray, dilation: Int) -> MLXArray {
        let h = conv(name + ".conv1", act(x), padding: dilation, dilation: dilation)
        return x + conv(name + ".conv2", act(h))
    }

    private func blocks(
        _ prefix: String, _ count: Int, _ input: MLXArray,
        _ pairs: inout ArraySlice<(MLXArray, MLXArray)>, dilations: [Int] = [1]
    ) -> MLXArray {
        var x = input
        for index in 0..<count {
            x = block("\(prefix).\(index)", x, dilation: dilations[index % dilations.count])
            if let (gamma, beta) = pairs.popFirst() {
                x = x * (1 + gamma) + beta
            }
        }
        return x
    }

    /// Normalized code z of `padded` (B, H, W, 9), sides multiples of 4.
    func code(_ padded: MLXArray, _ condition: MLXArray) -> MLXArray {
        let b = padded.dim(0)
        let frames = config.in_frames
        let all = film(condition)
        let count = config.encoder_blocks.reduce(0, +)
        // The front runs on a frame-major batch of B * frames; repeat a per-batch modulation.
        var front = ArraySlice(all[..<count].map { gamma, beta in
            b > 1 ? (concatenated(Array(repeating: gamma, count: frames), axis: 0),
                     concatenated(Array(repeating: beta, count: frames), axis: 0)) : (gamma, beta)
        })
        var back = ArraySlice(all[count...])
        let stacked = concatenated(split(padded, parts: frames, axis: -1), axis: 0)
        var x = conv("encoder.stem", stacked)
        for (level, n) in config.encoder_blocks.enumerated() {
            x = blocks("encoder.downs.\(level).blocks", n, x, &front)
            x = conv("encoder.downs.\(level).down", x, stride: 2)
        }
        let feats = (0..<frames).map { x[($0 * b) ..< (($0 + 1) * b)] }
        var mixed = conv("mix.inner", concatenated(feats, axis: -1), padding: 0)
        mixed = conv("mix.outer", act(mixed), padding: 0)
        var hidden = feats[frames / 2] + mixed
        hidden = blocks("encoder.mid", config.blocks, hidden, &back, dilations: Self.dilations)
        return conv("encoder.to_stats", act(hidden))
    }

    // MARK: vae2 decoder

    func decode(_ latent: MLXArray) -> MLXArray {
        var none = ArraySlice<(MLXArray, MLXArray)>()
        var x = conv("decoder.from_latent", latent)
        x = blocks("decoder.mid", config.vae_blocks, x, &none, dilations: Self.dilations)
        for (index, n) in config.decoder_blocks_run.enumerated() {
            x = Self.upsampleNearest2(x)
            x = conv("decoder.ups.\(index).mix", x)
            x = blocks("decoder.ups.\(index).blocks", n, x, &none)
        }
        return conv("decoder.to_rgb", act(x))
    }

    // MARK: whole path

    /// `low` (B, H, W, 9) in [0, 1] -> (B, 2H, 2W, 3) in [0, 1].
    func callAsFunction(_ low: MLXArray, condition: MLXArray) -> MLXArray {
        let height = low.dim(1), width = low.dim(2)
        let aligned = align(low)
        let z = code(Self.padReflect(aligned, Self.factor).asType(dtype), condition.asType(dtype))
        let latent = z / weight("vae.latent_scale") + weight("vae.latent_shift")
        let image = clip(decode(latent), min: Float(0), max: Float(1))
        return image[0..., 0 ..< 2 * height, 0 ..< 2 * width].asType(.float32)
    }

    /// Planar 9 × H × W in, planar 3 × 2H × 2W out, as Predictor hands frames around.
    func predict(stack: [Float], width: Int, height: Int, condition: [Float]) -> [Float] {
        let channels = 3 * config.in_frames
        let low = MLXArray(stack, [1, channels, height, width]).transposed(0, 2, 3, 1)
        let cond = MLXArray(condition, [1, config.condition_size])
        let out = self(low, condition: cond).transposed(0, 3, 1, 2)
        return out.asArray(Float.self)
    }
}
