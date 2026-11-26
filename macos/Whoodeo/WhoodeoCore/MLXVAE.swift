import Foundation
import MLX

/// Frozen vae2 encoder (step 044750). The decoder already lives in `MLXAlignModel`.
/// Output is μ in NHWC, 16 channels, at 1/8 of the (padded) master.
nonisolated final class MLXVAEEncoder {
    static let packName = "vae2-044750.safetensors"
    static let factor = 8
    static let latent = 16
    static let midBlocks = 6
    static let encoderBlocks = [1, 2, 2]
    static let dilations = [1, 2, 4]

    private let p: [String: MLXArray]
    private let relu: Bool

    var shift: MLXArray { p["vae.latent_shift"]! }
    var scale: MLXArray { p["vae.latent_scale"]! }

    static func defaultURL() -> URL {
        WhoodeoPaths.trainDirectory.appendingPathComponent(packName)
    }

    init(url: URL = MLXVAEEncoder.defaultURL(), activation: String = "relu") throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError("VAE-Gewichte fehlen: \(url.path)")
        }
        p = try loadArrays(url: url)
        relu = activation == "relu"
        guard p["encoder.stem.weight"] != nil, p["vae.latent_shift"] != nil else {
            throw CLIError("VAE-Paket unvollständig: \(url.lastPathComponent)")
        }
        eval(Array(p.values))
    }

    /// μ of `image` (B, H, W, 3) in [0, 1]. Pads to a multiple of 8.
    func encode(_ image: MLXArray) -> MLXArray {
        let padded = padTo(image, Self.factor)
        var x = conv("encoder.stem", padded)
        for (level, count) in Self.encoderBlocks.enumerated() {
            x = blocks("encoder.downs.\(level).blocks", count, x)
            x = conv("encoder.downs.\(level).down", x, stride: 2)
        }
        x = blocks("encoder.mid", Self.midBlocks, x, dilations: Self.dilations)
        return conv("encoder.to_stats", act(x))
    }

    /// Normalized target z = (μ - shift) * scale, no gradient needed from here.
    func normalizedCode(_ image: MLXArray) -> MLXArray {
        let mu = encode(image)
        let shift = self.shift.reshaped([1, 1, 1, Self.latent])
        let scale = self.scale.reshaped([1, 1, 1, Self.latent])
        return (mu - shift) * scale
    }

    private func act(_ x: MLXArray) -> MLXArray {
        relu ? maximum(x, Float(0)) : x * sigmoid(x)
    }

    private func conv(_ name: String, _ x: MLXArray, stride: Int = 1, padding: Int = 1, dilation: Int = 1) -> MLXArray {
        var y = conv2d(
            x, p[name + ".weight"]!,
            stride: IntOrPair(integerLiteral: stride),
            padding: IntOrPair(integerLiteral: padding),
            dilation: IntOrPair(integerLiteral: dilation)
        )
        if let bias = p[name + ".bias"] { y = y + bias }
        return y
    }

    private func block(_ name: String, _ x: MLXArray, dilation: Int) -> MLXArray {
        let h = conv(name + ".conv1", act(x), padding: dilation, dilation: dilation)
        return x + conv(name + ".conv2", act(h))
    }

    private func blocks(_ prefix: String, _ count: Int, _ input: MLXArray, dilations: [Int] = [1]) -> MLXArray {
        var x = input
        for index in 0..<count {
            x = block("\(prefix).\(index)", x, dilation: dilations[index % dilations.count])
        }
        return x
    }
}

/// Bottom/right pad so both sides divide by `factor`. Reflect when it fits, else repeat the edge.
func padTo(_ x: MLXArray, _ factor: Int) -> MLXArray {
    let h = x.dim(1), w = x.dim(2)
    let padH = (factor - h % factor) % factor
    let padW = (factor - w % factor) % factor
    if padH == 0 && padW == 0 { return x }
    let reflect = h > padH && w > padW
    let rows = (0..<h).map { Int32($0) } + (0..<padH).map { Int32(reflect ? h - 2 - $0 : h - 1) }
    let cols = (0..<w).map { Int32($0) } + (0..<padW).map { Int32(reflect ? w - 2 - $0 : w - 1) }
    return take(take(x, MLXArray(rows), axis: 1), MLXArray(cols), axis: 2)
}
