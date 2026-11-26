import Foundation
import MLX

/// From-scratch init, fine-tune load, FP16 `.mlxmodel` export, and a small AdamW on named arrays.
enum TrainModel {
    static let spynetPrefix = "align.spynet."
    static let frozenPrefixes = ["decoder.", "vae.latent_"]
    static let filmHidden = 64
    static let latent = 16
    static let vaeBlocks = 6
    static let decoderBlocks = [2, 3, 2]
    static let activation = "relu"
    static let conditionSize = 11
    static let inFrames = 3

    static func blockWidths(channels: Int, encoderBlocks: [Int], midBlocks: Int) -> [Int] {
        var width = channels
        var widths: [Int] = []
        for count in encoderBlocks {
            widths.append(contentsOf: repeatElement(width, count: count))
            width *= 2
        }
        widths.append(contentsOf: repeatElement(width, count: midBlocks))
        return widths
    }

    static func makeConfig(_ train: TrainConfig, step: Int, compute: String) -> MLXAlignModel.Config {
        MLXAlignModel.Config(
            step: step,
            activation: activation,
            in_frames: inFrames,
            condition_size: train.conditionSize,
            encoder_blocks: train.encoderBlocks,
            blocks: train.midBlocks,
            block_widths: blockWidths(channels: train.channels, encoderBlocks: train.encoderBlocks, midBlocks: train.midBlocks),
            latent: latent,
            vae_blocks: vaeBlocks,
            decoder_blocks_run: decoderBlocks,
            align: "feature",
            compute: compute
        )
    }

    static func fromScratch(_ train: TrainConfig) throws -> MLXAlignModel {
        seed(train.seed)
        var weights: [String: MLXArray] = [:]
        try loadFrozen(into: &weights)
        let widths = blockWidths(channels: train.channels, encoderBlocks: train.encoderBlocks, midBlocks: train.midBlocks)
        let c = train.channels
        addConv("encoder.stem", out: c, inCh: 3, into: &weights)
        var width = c
        for (level, count) in train.encoderBlocks.enumerated() {
            for index in 0..<count {
                addResidual("encoder.downs.\(level).blocks.\(index)", channels: width, into: &weights)
            }
            let wider = width * 2
            addConv("encoder.downs.\(level).down", out: wider, inCh: width, into: &weights)
            width = wider
        }
        for index in 0..<train.midBlocks {
            addResidual("encoder.mid.\(index)", channels: width, into: &weights)
        }
        addConv("encoder.to_stats", out: latent, inCh: width, into: &weights)
        addConv("align.inner", out: c, inCh: c * 3 + 2, into: &weights)
        addConv("align.outer", out: c, inCh: c, into: &weights)
        addLinear("film.inner", out: filmHidden, inCh: train.conditionSize, into: &weights, zero: false)
        let filmOut = 2 * widths.reduce(0, +)
        addLinear("film.outer", out: filmOut, inCh: filmHidden, into: &weights, zero: true)
        eval(Array(weights.values))
        return MLXAlignModel(config: makeConfig(train, step: 0, compute: "float32"), weights: weights, dtype: train.computeDType)
    }

    static func load(from url: URL, train: TrainConfig) throws -> MLXAlignModel {
        let stored = try loadArrays(url: url.appendingPathComponent("model.safetensors"))
        var weights: [String: MLXArray] = [:]
        for (key, value) in stored { weights[key] = value.asType(.float32) }
        eval(Array(weights.values))
        let data = try Data(contentsOf: url.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(MLXAlignModel.Config.self, from: data)
        if config.block_widths.first != train.channels {
            throw CLIError(
                "Checkpoint hat \(config.block_widths.first ?? 0) Kanäle, der Lauf verlangt \(train.channels)"
            )
        }
        config.compute = "float32"
        return MLXAlignModel(config: config, weights: weights, dtype: train.computeDType)
    }

    static func trainableKeys(_ weights: [String: MLXArray], trainSpynet: Bool = false) -> [String] {
        weights.keys.filter { key in
            if key.hasPrefix(spynetPrefix) { return trainSpynet }
            return !frozenPrefixes.contains(where: { key.hasPrefix($0) })
        }.sorted()
    }

    static func spynetL2Delta(before: [String: MLXArray], after: [String: MLXArray]) -> Float {
        var sum = MLXArray(Float(0))
        for (key, start) in before {
            guard let now = after[key] else { continue }
            let d = now - start
            sum = sum + (d * d).sum()
        }
        eval(sum)
        let value = sum.item(Float.self)
        return max(value, 0).squareRoot()
    }

    static func exportFP16(model: MLXAlignModel, to url: URL, step: Int, train: TrainConfig) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var stored: [String: MLXArray] = [:]
        for (key, value) in model.p {
            stored[key] = key.hasPrefix(spynetPrefix) ? value.asType(.float32) : value.asType(.float16)
        }
        eval(Array(stored.values))
        try MLX.save(arrays: stored, url: url.appendingPathComponent("model.safetensors"))
        var object: [String: Any] = [
            "arch": "latent_up",
            "align": "feature",
            "step": step,
            "activation": activation,
            "in_frames": inFrames,
            "condition": "codec",
            "condition_size": train.conditionSize,
            "encoder_blocks": train.encoderBlocks,
            "blocks": train.midBlocks,
            "block_widths": blockWidths(channels: train.channels, encoderBlocks: train.encoderBlocks, midBlocks: train.midBlocks),
            "latent": latent,
            "vae_activation": activation,
            "vae_blocks": vaeBlocks,
            "decoder_blocks_run": decoderBlocks,
            "compute": "float16",
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url.appendingPathComponent("config.json"))
    }

    static func loadFrozen(into weights: inout [String: MLXArray]) throws {
        let vaeURL = WhoodeoPaths.trainDirectory.appendingPathComponent("vae2-044750.safetensors")
        let spyURL = WhoodeoPaths.trainDirectory.appendingPathComponent("spynet.safetensors")
        guard FileManager.default.fileExists(atPath: vaeURL.path) else {
            throw CLIError("VAE-Gewichte fehlen: \(vaeURL.path)")
        }
        guard FileManager.default.fileExists(atPath: spyURL.path) else {
            throw CLIError("SPyNet-Gewichte fehlen: \(spyURL.path)")
        }
        let vae = try loadArrays(url: vaeURL)
        for (key, value) in vae where key.hasPrefix("decoder.") || key.hasPrefix("vae.latent_") {
            weights[key] = value
        }
        let spy = try loadArrays(url: spyURL)
        for (key, value) in spy { weights[key] = value }
    }

    /// Kaiming-uniform as PyTorch Conv2d/Linear, residual conv2 optionally zero.
    static func addConv(_ name: String, out: Int, inCh: Int, into weights: inout [String: MLXArray], zero: Bool = false) {
        if zero {
            weights[name + ".weight"] = MLXArray.zeros([out, 3, 3, inCh])
            weights[name + ".bias"] = MLXArray.zeros([out])
            return
        }
        let fan = 3 * 3 * inCh
        let bound = 1 / sqrt(Float(fan))
        weights[name + ".weight"] = uniform(low: -bound, high: bound, [out, 3, 3, inCh])
        weights[name + ".bias"] = uniform(low: -bound, high: bound, [out])
    }

    static func addResidual(_ name: String, channels: Int, into weights: inout [String: MLXArray]) {
        addConv(name + ".conv1", out: channels, inCh: channels, into: &weights)
        addConv(name + ".conv2", out: channels, inCh: channels, into: &weights, zero: true)
    }

    static func addLinear(_ name: String, out: Int, inCh: Int, into weights: inout [String: MLXArray], zero: Bool) {
        if zero {
            weights[name + ".weight"] = MLXArray.zeros([out, inCh])
            weights[name + ".bias"] = MLXArray.zeros([out])
            return
        }
        let bound = 1 / sqrt(Float(inCh))
        weights[name + ".weight"] = uniform(low: -bound, high: bound, [out, inCh])
        weights[name + ".bias"] = uniform(low: -bound, high: bound, [out])
    }
}

/// AdamW on named arrays, bias correction on, weight decay 0 like the official trainer.
final class TrainAdamW {
    var learningRate: Float
    /// Per-parameter multiplier on `learningRate` (SPyNet typically 0.1). Missing keys use 1.
    var lrScale: [String: Float] = [:]
    let beta1: Float
    let beta2: Float
    let epsilon: Float
    let weightDecay: Float
    var step: Int = 0
    private var m: [String: MLXArray] = [:]
    private var v: [String: MLXArray] = [:]

    init(learningRate: Float, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, weightDecay: Float = 0) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weightDecay = weightDecay
    }

    func update(keys: [String], params: inout [MLXArray], grads: [MLXArray]) {
        step += 1
        let b1t = 1 - pow(beta1, Float(step))
        let b2t = 1 - pow(beta2, Float(step))
        for index in keys.indices {
            let key = keys[index]
            var p = params[index]
            let g = grads[index]
            let lr = learningRate * (lrScale[key] ?? 1)
            if weightDecay != 0 {
                p = p * (1 - lr * weightDecay)
            }
            let mi = (m[key] ?? zeros(like: g)) * beta1 + g * (1 - beta1)
            let vi = (v[key] ?? zeros(like: g)) * beta2 + g * g * (1 - beta2)
            m[key] = mi
            v[key] = vi
            let mHat = mi / b1t
            let vHat = vi / b2t
            params[index] = p - lr * mHat / (sqrt(vHat) + epsilon)
        }
        eval(params)
        eval(Array(m.values) + Array(v.values))
    }

    func save(to url: URL) throws {
        var arrays: [String: MLXArray] = ["opt.step": MLXArray(Int32(step))]
        for (key, value) in m { arrays["m." + key] = value }
        for (key, value) in v { arrays["v." + key] = value }
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, url: url)
    }

    func load(from url: URL) throws {
        let arrays = try loadArrays(url: url)
        if let s = arrays["opt.step"] {
            eval(s)
            step = Int(s.item(Int32.self))
        }
        m.removeAll()
        v.removeAll()
        for (key, value) in arrays {
            if key.hasPrefix("m.") { m[String(key.dropFirst(2))] = value }
            if key.hasPrefix("v.") { v[String(key.dropFirst(2))] = value }
        }
    }

    var momentCount: Int { m.count }

    var spynetMomentCount: Int {
        m.keys.filter { $0.hasPrefix(TrainModel.spynetPrefix) }.count
    }
}
