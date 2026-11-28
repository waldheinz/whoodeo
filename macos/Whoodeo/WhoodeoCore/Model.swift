import Foundation

nonisolated struct FrameSize: Hashable {
    var width: Int
    var height: Int
}

/// What the app knows about a loaded `.mlxmodel`.
nonisolated struct PackageInfo {
    var arch: String
    var blocks: Int
    var channels: Int
    var step: Int?
    var inFrames: Int
    /// Length of the codec condition vector, 0 when the model has none.
    var conditionSize: Int

    var takesCondition: Bool { conditionSize > 0 }

    var label: String {
        var parts = ["\(arch) \(blocks)×\(channels)"]
        if let step { parts.append("Schritt \(step)") }
        return parts.joined(separator: " ")
    }

    func shapeError(width: Int, height: Int) -> String? {
        if width >= 4 && width <= 8192 && height >= 4 && height <= 8192 { return nil }
        return "Das Modell erwartet Breite und Höhe 4–8192. Der Film ist \(width)×\(height)."
    }
}

/// Which film and source frames a window stack holds, so the MLX model can reuse the
/// per-frame features of frames it saw in the previous window.
nonisolated struct FrameKeys {
    var film: URL
    var slots: [Int]
    /// Per slot: 0 for the source frame as decoded, otherwise which computed frame stands there
    /// (a time plan's reconstruction), so it never shares features with the plain frame.
    var variants: [Int] = []
}

nonisolated struct Prediction {
    var image: [Float]
    var width: Int
    var height: Int
    var milliseconds: Double
}

/// The MLX 3-frame latent_up (`.mlxmodel`) on the GPU.
nonisolated final class Predictor {
    let info: PackageInfo
    private let mlx: MLXAlignModel
    /// Condition vector fed to a `-cond` model, nil when the model has none.
    private(set) var condition: [Float]?

    init(url: URL) throws {
        guard url.pathExtension == MLXAlignModel.fileExtension else {
            throw CLIError("\(url.lastPathComponent) ist kein .\(MLXAlignModel.fileExtension)")
        }
        let loaded = try MLXAlignModel(url: url)
        let c = loaded.config
        mlx = loaded
        info = PackageInfo(
            arch: "latent_up", blocks: c.blocks, channels: c.block_widths.first ?? 0,
            step: c.step, inFrames: c.in_frames, conditionSize: c.condition_size
        )
        if info.takesCondition {
            setCondition(.unknown)
        }
    }

    func setCondition(_ codec: SourceCodec) {
        guard info.takesCondition else {
            condition = nil
            return
        }
        condition = codec.vector(size: info.conditionSize)
    }

    func predict(stack: [Float], width: Int, height: Int, frames: FrameKeys? = nil) throws -> Prediction {
        if let problem = info.shapeError(width: width, height: height) {
            throw CLIError(problem)
        }
        let count = info.inFrames * 3 * height * width
        guard stack.count == count else {
            throw CLIError("Stapel hat \(stack.count) Werte, erwartet \(count)")
        }
        let started = CFAbsoluteTimeGetCurrent()
        // Padding, alignment, and the crop happen inside the MLX graph.
        let values = condition ?? [Float](repeating: 0, count: info.conditionSize)
        let image = mlx.predict(stack: stack, width: width, height: height, condition: values, frames: frames)
        return Prediction(
            image: image, width: 2 * width, height: 2 * height,
            milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000
        )
    }
}
