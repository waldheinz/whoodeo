import CoreML
import Foundation

nonisolated enum Compute: String, CaseIterable, Identifiable {
    case gpu, ane, auto

    var id: String { rawValue }

    var title: String {
        switch self {
        case .gpu: "GPU"
        case .ane: "Neural Engine"
        case .auto: "Automatisch"
        }
    }

    var units: MLComputeUnits {
        switch self {
        case .gpu: .cpuAndGPU
        case .ane: .cpuAndNeuralEngine
        case .auto: .all
        }
    }

    static func parse(_ text: String) -> Compute? {
        switch text {
        case "gpu": .gpu
        case "ane": .ane
        case "all", "auto": .auto
        default: nil
        }
    }
}

nonisolated struct PackageInfo {
    var arch: String
    var blocks: Int
    var channels: Int
    var stem: Int?
    var vote: Int?
    var resScale: String
    var step: Int?
    var inFrames: Int
    var scale: Int
    var widthMin: Int
    var widthMax: Int
    var heightMin: Int
    var heightMax: Int
    var prefer: Compute
    var inputName: String
    var outputName: String

    var label: String {
        var parts = ["\(arch) \(blocks)×\(channels)"]
        if let stem { parts.append("stem \(stem)") }
        if let vote { parts.append("vote \(vote)") }
        if let step { parts.append("Schritt \(step)") }
        return parts.joined(separator: " ")
    }

    func shapeError(width: Int, height: Int) -> String? {
        if width >= widthMin && width <= widthMax && height >= heightMin && height <= heightMax {
            return nil
        }
        return "Dieses Paket erwartet Breite \(widthMin)–\(widthMax) und Höhe \(heightMin)–\(heightMax). Der Film ist \(width)×\(height)."
    }

    static func read(_ model: MLModel) throws -> PackageInfo {
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard let arch = meta["whoodeo.arch"], !arch.isEmpty else {
            throw CLIError("das Paket hat kein whoodeo.arch")
        }
        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName
        let inputName = inputs["frames"] != nil ? "frames" : (inputs.keys.first ?? "frames")
        let outputName = outputs["image"] != nil ? "image" : (outputs.keys.first ?? "image")
        let prefer = Compute.parse(meta["whoodeo.bench.prefer"] ?? "") ?? .gpu
        return PackageInfo(
            arch: arch,
            blocks: Int(meta["whoodeo.blocks"] ?? "") ?? 0,
            channels: Int(meta["whoodeo.channels"] ?? "") ?? 0,
            stem: Int(meta["whoodeo.stem"] ?? ""),
            vote: Int(meta["whoodeo.vote"] ?? ""),
            resScale: meta["whoodeo.res_scale"] ?? "",
            step: Int(meta["whoodeo.step"] ?? ""),
            inFrames: Int(meta["whoodeo.in_frames"] ?? "") ?? 1,
            scale: Int(meta["whoodeo.scale"] ?? "") ?? 2,
            widthMin: Int(meta["whoodeo.width_min"] ?? "") ?? 1,
            widthMax: Int(meta["whoodeo.width_max"] ?? "") ?? 8192,
            heightMin: Int(meta["whoodeo.height_min"] ?? "") ?? 1,
            heightMax: Int(meta["whoodeo.height_max"] ?? "") ?? 8192,
            prefer: prefer,
            inputName: inputName,
            outputName: outputName
        )
    }
}

nonisolated struct Prediction {
    var image: [Float]
    var width: Int
    var height: Int
    var milliseconds: Double
}

nonisolated final class Predictor {
    let info: PackageInfo
    let model: MLModel
    let unit: Compute
    private var inputArray: MLMultiArray?

    init(url: URL, unit: Compute) throws {
        let config = MLModelConfiguration()
        config.computeUnits = unit.units
        var hints = MLOptimizationHints()
        // One film is one shape. Infrequent lets that shape onto the Neural Engine.
        hints.reshapeFrequency = .infrequent
        hints.specializationStrategy = .fastPrediction
        config.optimizationHints = hints
        // An mlpackage is not a compiled model. macOS does not compile it on load.
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        model = try MLModel(contentsOf: compiled, configuration: config)
        info = try PackageInfo.read(model)
        self.unit = unit
    }

    func predict(stack: [Float], width: Int, height: Int) throws -> Prediction {
        if let problem = info.shapeError(width: width, height: height) {
            throw CLIError(problem)
        }
        let channels = info.inFrames * 3
        let count = channels * height * width
        guard stack.count == count else {
            throw CLIError("Stapel hat \(stack.count) Werte, erwartet \(count)")
        }
        let array = try inputBuffer(channels: channels, height: height, width: width)
        Self.write(stack, into: array)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            info.inputName: MLFeatureValue(multiArray: array),
        ])
        let started = CFAbsoluteTimeGetCurrent()
        let result = try model.prediction(from: features)
        let milliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1000
        guard let output = result.featureValue(for: info.outputName)?.multiArrayValue else {
            throw CLIError("keine Ausgabe \(info.outputName)")
        }
        guard output.dataType == .float32 else {
            throw CLIError("Ausgabe ist nicht float32")
        }
        let (values, outH, outW) = Self.read(output)
        return Prediction(image: values, width: outW, height: outH, milliseconds: milliseconds)
    }

    private func inputBuffer(channels: Int, height: Int, width: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, channels as NSNumber, height as NSNumber, width as NSNumber]
        if let inputArray, inputArray.shape == shape, inputArray.dataType == .float32 {
            return inputArray
        }
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        inputArray = array
        return array
    }

    /// One copy when Core ML allocated a packed NCHW buffer, otherwise one copy per row.
    private static func write(_ values: [Float], into array: MLMultiArray) {
        let count = values.count
        guard count > 0 else { return }
        values.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            let raw = UnsafeMutableRawPointer(mutating: base)
            if packed(array), array.count == count {
                array.dataPointer.copyMemory(from: raw, byteCount: count * MemoryLayout<Float>.stride)
                return
            }
            let shape = array.shape.map(\.intValue)
            let strides = array.strides.map(\.intValue)
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            guard let last = strides.last, last == 1 else {
                var cursor = 0
                func scatter(_ axis: Int, _ offset: Int) {
                    if axis == shape.count {
                        return
                    }
                    if axis == shape.count - 1 {
                        for index in 0..<shape[axis] {
                            pointer[offset + index * strides[axis]] = base[cursor]
                            cursor += 1
                        }
                        return
                    }
                    for index in 0..<shape[axis] {
                        scatter(axis + 1, offset + index * strides[axis])
                    }
                }
                scatter(0, 0)
                return
            }
            var cursor = 0
            func walk(_ axis: Int, _ offset: Int) {
                if axis == shape.count - 1 {
                    pointer.advanced(by: offset).update(from: base.advanced(by: cursor), count: shape[axis])
                    cursor += shape[axis]
                    return
                }
                for index in 0..<shape[axis] {
                    walk(axis + 1, offset + index * strides[axis])
                }
            }
            walk(0, 0)
        }
    }

    private static func read(_ array: MLMultiArray) -> ([Float], Int, Int) {
        let shape = array.shape.map(\.intValue)
        let channels: Int
        let height: Int
        let width: Int
        if shape.count == 4 {
            channels = shape[1]
            height = shape[2]
            width = shape[3]
        } else if shape.count == 3 {
            channels = shape[0]
            height = shape[1]
            width = shape[2]
        } else {
            return ([], 0, 0)
        }
        let count = channels * height * width
        var values = [Float](repeating: 0, count: count)
        guard count > 0 else { return (values, height, width) }
        values.withUnsafeMutableBufferPointer { dest in
            guard let base = dest.baseAddress else { return }
            if packed(array), array.count == count {
                base.update(from: array.dataPointer.bindMemory(to: Float.self, capacity: count), count: count)
                return
            }
            let strides = array.strides.map(\.intValue)
            let offset = shape.count - 3
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            if strides[offset + 2] == 1 {
                for c in 0..<channels {
                    for y in 0..<height {
                        let from = pointer.advanced(by: c * strides[offset] + y * strides[offset + 1])
                        base.advanced(by: (c * height + y) * width).update(from: from, count: width)
                    }
                }
            } else {
                for c in 0..<channels {
                    for y in 0..<height {
                        for x in 0..<width {
                            let index = c * strides[offset] + y * strides[offset + 1] + x * strides[offset + 2]
                            base[(c * height + y) * width + x] = pointer[index]
                        }
                    }
                }
            }
        }
        return (values, height, width)
    }

    private static func packed(_ array: MLMultiArray) -> Bool {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == strides.count, !shape.isEmpty else { return false }
        var step = 1
        for index in shape.indices.reversed() {
            if strides[index] != step { return false }
            step *= max(shape[index], 1)
        }
        return true
    }
}
