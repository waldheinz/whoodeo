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

nonisolated struct FrameSize: Hashable {
    var width: Int
    var height: Int
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
    /// Enumerated input sizes, empty for a flexible range.
    var sizes: [FrameSize] = []
    /// Both sides are padded (reflect, bottom and right) to a multiple of this, 1 for none.
    var pad: Int = 1
    var inputType: MLMultiArrayDataType = .float32

    var label: String {
        var parts = ["\(arch) \(blocks)×\(channels)"]
        if let stem { parts.append("stem \(stem)") }
        if let vote { parts.append("vote \(vote)") }
        if let step { parts.append("Schritt \(step)") }
        return parts.joined(separator: " ")
    }

    /// The size the model sees: each side rounded up to a multiple of `pad`.
    func padded(width: Int, height: Int) -> FrameSize {
        guard pad > 1 else { return FrameSize(width: width, height: height) }
        return FrameSize(
            width: (width + pad - 1) / pad * pad,
            height: (height + pad - 1) / pad * pad
        )
    }

    func shapeError(width: Int, height: Int) -> String? {
        let size = padded(width: width, height: height)
        let note = size.width != width || size.height != height
            ? " (aufgefüllt \(size.width)×\(size.height))" : ""
        if !sizes.isEmpty {
            if sizes.contains(size) { return nil }
            let list = sizes.map { "\($0.width)×\($0.height)" }.joined(separator: ", ")
            return "Dieses Paket kennt nur die Eingangsgrößen \(list). Der Film ist \(width)×\(height)\(note)."
        }
        if size.width >= widthMin && size.width <= widthMax && size.height >= heightMin && size.height <= heightMax {
            return nil
        }
        return "Dieses Paket erwartet Breite \(widthMin)–\(widthMax) und Höhe \(heightMin)–\(heightMax). Der Film ist \(width)×\(height)\(note)."
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
        // "428x240,640x360": width x height.
        let sizes: [FrameSize] = (meta["whoodeo.sizes"] ?? "").split(separator: ",").compactMap { part in
            let sides = part.split(separator: "x")
            guard sides.count == 2, let width = Int(sides[0]), let height = Int(sides[1]) else { return nil }
            return FrameSize(width: width, height: height)
        }
        let inputType = inputs[inputName]?.multiArrayConstraint?.dataType ?? .float32
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
            outputName: outputName,
            sizes: sizes,
            pad: max(Int(meta["whoodeo.pad"] ?? "") ?? 1, 1),
            inputType: inputType == .float16 ? .float16 : .float32
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
        let size = info.padded(width: width, height: height)
        let input = size.width == width && size.height == height
            ? stack
            : Self.pad(stack, channels: channels, width: width, height: height, to: size)
        let array = try inputBuffer(channels: channels, height: size.height, width: size.width)
        if array.dataType == .float16 {
            Self.write16(input, into: array)
        } else {
            Self.write(input, into: array)
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            info.inputName: MLFeatureValue(multiArray: array),
        ])
        let started = CFAbsoluteTimeGetCurrent()
        let result = try model.prediction(from: features)
        let milliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1000
        guard let output = result.featureValue(for: info.outputName)?.multiArrayValue else {
            throw CLIError("keine Ausgabe \(info.outputName)")
        }
        let values: [Float]
        let outH: Int
        let outW: Int
        switch output.dataType {
        case .float32:
            (values, outH, outW) = Self.read(output, as: Float.self)
        case .float16:
            (values, outH, outW) = Self.read(output, as: Float16.self)
        default:
            throw CLIError("Ausgabe ist weder float32 noch float16")
        }
        if outW == size.width * info.scale && outH == size.height * info.scale
            && (size.width != width || size.height != height) {
            let cropW = width * info.scale
            let cropH = height * info.scale
            return Prediction(
                image: Self.crop(values, width: outW, height: outH, to: FrameSize(width: cropW, height: cropH)),
                width: cropW,
                height: cropH,
                milliseconds: milliseconds
            )
        }
        return Prediction(image: values, width: outW, height: outH, milliseconds: milliseconds)
    }

    /// Planar channels padded on the bottom and the right, as whoodeo's pad_to:
    /// reflect when the frame is larger than the pad, replicate otherwise.
    static func pad(_ values: [Float], channels: Int, width: Int, height: Int, to size: FrameSize) -> [Float] {
        let padW = size.width - width
        let padH = size.height - height
        let reflect = height > padH && width > padW
        func source(_ index: Int, _ count: Int) -> Int {
            if index < count { return index }
            return reflect ? 2 * (count - 1) - index : count - 1
        }
        let columns = (0..<size.width).map { source($0, width) }
        var out = [Float](repeating: 0, count: channels * size.width * size.height)
        values.withUnsafeBufferPointer { from in
            out.withUnsafeMutableBufferPointer { to in
                for c in 0..<channels {
                    for y in 0..<size.height {
                        let row = (c * height + source(y, height)) * width
                        let dest = (c * size.height + y) * size.width
                        to.baseAddress!.advanced(by: dest).update(from: from.baseAddress!.advanced(by: row), count: width)
                        for x in width..<size.width {
                            to[dest + x] = from[row + columns[x]]
                        }
                    }
                }
            }
        }
        return out
    }

    /// The top left `size` of planar RGB.
    static func crop(_ values: [Float], width: Int, height: Int, to size: FrameSize) -> [Float] {
        var out = [Float](repeating: 0, count: 3 * size.width * size.height)
        values.withUnsafeBufferPointer { from in
            out.withUnsafeMutableBufferPointer { to in
                for c in 0..<3 {
                    for y in 0..<size.height {
                        to.baseAddress!.advanced(by: (c * size.height + y) * size.width)
                            .update(from: from.baseAddress!.advanced(by: (c * height + y) * width), count: size.width)
                    }
                }
            }
        }
        return out
    }

    private func inputBuffer(channels: Int, height: Int, width: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, channels as NSNumber, height as NSNumber, width as NSNumber]
        if let inputArray, inputArray.shape == shape, inputArray.dataType == info.inputType {
            return inputArray
        }
        let array = try MLMultiArray(shape: shape, dataType: info.inputType)
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

    /// fp16 input, element by element along the array's strides.
    private static func write16(_ values: [Float], into array: MLMultiArray) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == 4, values.count == array.count else { return }
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        values.withUnsafeBufferPointer { source in
            for c in 0..<channels {
                for y in 0..<height {
                    let from = (c * height + y) * width
                    let to = c * strides[1] + y * strides[2]
                    for x in 0..<width {
                        pointer[to + x * strides[3]] = Float16(source[from + x])
                    }
                }
            }
        }
    }

    private static func read<T: BinaryFloatingPoint>(_ array: MLMultiArray, as _: T.Type) -> ([Float], Int, Int) {
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
            if T.self == Float.self, packed(array), array.count == count {
                base.update(from: array.dataPointer.bindMemory(to: Float.self, capacity: count), count: count)
                return
            }
            let strides = array.strides.map(\.intValue)
            let offset = shape.count - 3
            let pointer = array.dataPointer.bindMemory(to: T.self, capacity: array.count)
            if T.self == Float.self, strides[offset + 2] == 1 {
                let floats = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
                for c in 0..<channels {
                    for y in 0..<height {
                        let from = floats.advanced(by: c * strides[offset] + y * strides[offset + 1])
                        base.advanced(by: (c * height + y) * width).update(from: from, count: width)
                    }
                }
            } else {
                for c in 0..<channels {
                    for y in 0..<height {
                        for x in 0..<width {
                            let index = c * strides[offset] + y * strides[offset + 1] + x * strides[offset + 2]
                            base[(c * height + y) * width + x] = Float(pointer[index])
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
