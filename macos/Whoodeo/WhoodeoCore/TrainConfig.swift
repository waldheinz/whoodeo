import Foundation
import MLX

/// Hyperparameters of one training run. Stored in `runs.config_json`.
/// Form, CLI flags, help and JSON keys come from `TrainParameterSpec`.
nonisolated struct TrainConfig: Equatable, Sendable {
    var name: String
    var parentCheckpointID: Int? = nil
    var channels: Int = 64
    var encoderBlocks: [Int] = [2, 4]
    var midBlocks: Int = 12
    /// Master crop side in pixels. `nil` is the full frame. Must be 0 or a multiple of 32.
    var cropSize: Int? = 384
    var rungs: [String] = ["480", "720"]
    /// Empty means every preset.
    var presets: [String] = []
    var batch: Int = 2
    var gradAccum: Int = 4
    var learningRate: Float = 1e-4
    var schedule: String = "constant"
    var steps: Int = 1000
    var saveEvery: Int = 500
    var valEvery: Int = 500
    var valCount: Int = 8
    var logEvery: Int = 1
    var dtype: String = "fp32"
    var seed: UInt64 = 42
    /// Prefetch workers. 0 is synchronous (no background queue).
    var workers: Int = 4
    var hidden: Bool = false
    var codecUnknown: Float = 0.2
    var conditionSize: Int = 11
    var latentWeight: Float = 1
    var oklabWeight: Float = 4.5
    var oklabChannels: [Float] = [1, 1, 1]
    var msSSIMWeight: Float = 1
    var msSSIMScales: Int = 5
    var vggLayers: [String: Float] = ["relu2_2": 0.230, "relu3_4": 0.155]
    /// Factor on the VGG layer weights. 0 skips VGG19 entirely (not even loaded).
    var vggWeight: Float = 1
    /// Official SPyNet stays frozen unless this is set. Then it uses `lr * spynetLrScale`.
    var trainSpynet: Bool = false
    var spynetLrScale: Float = 0.1

    var spynetLearningRate: Float { learningRate * spynetLrScale }

    var spynetLabel: String {
        trainSpynet ? String(format: "SPyNet: trainiert (%.2g× LR)", spynetLrScale) : "SPyNet: eingefroren"
    }

    static func spynetLabel(from json: String) -> String {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return "SPyNet: eingefroren" }
        let train: Bool
        if let value = object["train_spynet"] as? Bool { train = value }
        else if let value = object["train_spynet"] as? Int { train = value != 0 }
        else { train = false }
        var scale: Float = 0.1
        if let value = object["spynet_lr_scale"] as? Double { scale = Float(value) }
        else if let value = object["spynet_lr_scale"] as? Int { scale = Float(value) }
        return train ? String(format: "SPyNet: trainiert (%.2g× LR)", scale) : "SPyNet: eingefroren"
    }

    var computeDType: DType {
        switch dtype.lowercased() {
        case "bf16", "bfloat16": .bfloat16
        case "fp16", "float16": .float16
        default: .float32
        }
    }

    var blocksJSON: String {
        let values = encoderBlocks + [midBlocks]
        if let data = try? JSONSerialization.data(withJSONObject: values),
           let text = String(data: data, encoding: .utf8)
        {
            return text
        }
        return "\(values)"
    }

    var configJSON: String {
        var object: [String: Any] = [
            "arch": "latent_up",
            "align": "feature",
            "activation": "relu",
            "in_frames": 3,
            "condition": "codec",
            "condition_size": conditionSize,
            "compute": dtype == "bf16" ? "bfloat16" : "float32",
        ]
        for spec in TrainParameterSpec.all {
            object[spec.key] = spec.json(self)
        }
        if let parentCheckpointID { object["parent_checkpoint_id"] = parentCheckpointID }
        if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8)
        {
            return text
        }
        return "{}"
    }

    /// `configJSON` on top of an older run config; keys this version does not write (e.g. `imported`) stay.
    func configJSON(mergingInto old: String) -> String {
        guard let oldData = old.data(using: .utf8),
              var object = try? JSONSerialization.jsonObject(with: oldData) as? [String: Any],
              let newData = configJSON.data(using: .utf8),
              let fresh = try? JSONSerialization.jsonObject(with: newData) as? [String: Any]
        else { return configJSON }
        object.merge(fresh) { _, new in new }
        if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8)
        {
            return text
        }
        return configJSON
    }

    func validated() throws -> TrainConfig {
        var copy = self
        if copy.channels < 1 { throw CLIError("channels muss positiv sein") }
        if copy.encoderBlocks.contains(where: { $0 < 1 }) || copy.midBlocks < 1 {
            throw CLIError("blocks müssen positiv sein")
        }
        if copy.batch < 1 || copy.gradAccum < 1 {
            throw CLIError("batch und grad-accum müssen mindestens 1 sein")
        }
        if copy.steps < 1 { throw CLIError("steps muss positiv sein") }
        if copy.learningRate <= 0 { throw CLIError("lr muss positiv sein") }
        if let crop = copy.cropSize {
            if crop == 0 {
                copy.cropSize = nil
            } else if crop < 32 || crop % 32 != 0 {
                throw CLIError("crop muss 0 (Vollbild) oder durch 32 teilbar sein, nicht \(crop)")
            }
        }
        switch copy.dtype.lowercased() {
        case "fp32", "float32": copy.dtype = "fp32"
        case "bf16", "bfloat16": copy.dtype = "bf16"
        default: throw CLIError("dtype muss fp32 oder bf16 sein, nicht \(copy.dtype)")
        }
        if copy.schedule != "constant" && copy.schedule != "cosine" {
            throw CLIError("schedule muss constant oder cosine sein")
        }
        if copy.spynetLrScale <= 0 {
            throw CLIError("spynet-lr-scale muss positiv sein")
        }
        if copy.workers < 0 {
            throw CLIError("workers darf nicht negativ sein")
        }
        let weights = [copy.latentWeight, copy.oklabWeight, copy.msSSIMWeight, copy.vggWeight]
        if weights.contains(where: { $0 < 0 || !$0.isFinite }) {
            throw CLIError("Loss-Gewichte dürfen nicht negativ sein")
        }
        if !weights.contains(where: { $0 > 0 }) {
            throw CLIError("mindestens ein Loss-Gewicht muss größer als 0 sein (latent, oklab, ms-ssim oder vgg)")
        }
        return copy
    }

    static func cropArgument(_ text: String) throws -> Int? {
        guard let value = Int(text) else { throw CLIError("ungültiges crop: \(text)") }
        return value == 0 ? nil : value
    }

    static func intList(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "," || $0 == " " }).map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}
