import CoreML
import Foundation

/// Frozen official SPyNet: warp the previous and next low frames onto the middle one.
///
/// The six basic modules run as small, flexible Core ML packages (the official
/// weights are fixed). The resize to a multiple of 32 and the warp stay here:
/// a flexible Core ML graph that includes them does not run.
nonisolated struct SpyNetAligner {
    private let blocks: [MLModel]
    private let inputName: String
    private let outputName: String
    /// Official ImageNet mean and std, as in the SPyNet weights.
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]

    init(url: URL) throws {
        let folder = url.appendingPathComponent("spynet", isDirectory: true)
        var loaded: [MLModel] = []
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        var hints = MLOptimizationHints()
        hints.reshapeFrequency = .frequent
        config.optimizationHints = hints
        for index in 0..<6 {
            let package = folder.appendingPathComponent("block\(index).mlpackage", isDirectory: true)
            guard FileManager.default.fileExists(atPath: package.path) else {
                throw CLIError("dem Paket fehlen die SPyNet-Blöcke (spynet/block\(index).mlpackage)")
            }
            let compiled = package.pathExtension == "mlmodelc" ? package : try MLModel.compileModel(at: package)
            loaded.append(try MLModel(contentsOf: compiled, configuration: config))
        }
        blocks = loaded
        let inputs = loaded[0].modelDescription.inputDescriptionsByName
        let outputs = loaded[0].modelDescription.outputDescriptionsByName
        inputName = inputs["x"] != nil ? "x" : (inputs.keys.first ?? "x")
        outputName = outputs["y"] != nil ? "y" : (outputs.keys.first ?? "y")
    }

    /// `stack` is 9 × height × width, NCHW, previous / middle / next, in 0…1.
    func align(stack: [Float], width: Int, height: Int) throws -> [Float] {
        let plane = height * width
        guard stack.count == 9 * plane else {
            throw CLIError("Ausrichtung erwartet 9 × \(width) × \(height), bekam \(stack.count) Werte")
        }
        let prev = Array(stack[0..<3 * plane])
        let center = Array(stack[3 * plane..<6 * plane])
        let next = Array(stack[6 * plane..<9 * plane])
        let neigh = prev + next
        let ref = center + center
        let flow = try spyNet(ref: ref, supp: neigh, batch: 2, height: height, width: width)
        let warped = warp(image: neigh, flow: flow, batch: 2, channels: 3, height: height, width: width)
        let alignedPrev = warped[0..<3 * plane].map { min(max($0, 0), 1) }
        let alignedNext = warped[3 * plane..<6 * plane].map { min(max($0, 0), 1) }
        return Array(alignedPrev) + center + Array(alignedNext)
    }

    private func spyNet(ref: [Float], supp: [Float], batch: Int, height: Int, width: Int) throws -> [Float] {
        let hUp = ((height + 31) / 32) * 32
        let wUp = ((width + 31) / 32) * 32
        let refUp = resize(ref, batch: batch, channels: 3, inH: height, inW: width, outH: hUp, outW: wUp, alignCorners: false)
        let suppUp = resize(supp, batch: batch, channels: 3, inH: height, inW: width, outH: hUp, outW: wUp, alignCorners: false)
        var flow = try computeFlow(ref: refUp, supp: suppUp, batch: batch, height: hUp, width: wUp)
        flow = resize(flow, batch: batch, channels: 2, inH: hUp, inW: wUp, outH: height, outW: width, alignCorners: false)
        let scaleX = Float(width) / Float(wUp)
        let scaleY = Float(height) / Float(hUp)
        let plane = height * width
        var scaled = flow
        for n in 0..<batch {
            let base = n * 2 * plane
            for i in 0..<plane { scaled[base + i] *= scaleX }
            for i in 0..<plane { scaled[base + plane + i] *= scaleY }
        }
        return scaled
    }

    private func computeFlow(ref: [Float], supp: [Float], batch: Int, height: Int, width: Int) throws -> [Float] {
        var refs = [normalize(ref, batch: batch, height: height, width: width)]
        var supps = [normalize(supp, batch: batch, height: height, width: width)]
        var h = height, w = width
        for _ in 0..<5 {
            refs.append(avgPool2(refs[refs.count - 1], batch: batch, channels: 3, height: h, width: w))
            supps.append(avgPool2(supps[supps.count - 1], batch: batch, channels: 3, height: h, width: w))
            h /= 2
            w /= 2
        }
        refs.reverse()
        supps.reverse()
        var flow = [Float](repeating: 0, count: batch * 2 * (height / 32) * (width / 32))
        var fh = height / 32, fw = width / 32
        for level in 0..<6 {
            let flowUp: [Float]
            if level == 0 {
                flowUp = flow
            } else {
                let up = resize(flow, batch: batch, channels: 2, inH: fh, inW: fw, outH: fh * 2, outW: fw * 2, alignCorners: true)
                flowUp = up.map { $0 * 2 }
                fh *= 2
                fw *= 2
            }
            let warped = warp(image: supps[level], flow: flowUp, batch: batch, channels: 3, height: fh, width: fw)
            let packed = concatChannels([refs[level], warped, flowUp], batch: batch, channels: [3, 3, 2], height: fh, width: fw)
            let delta = try basicModule(packed, level: level, batch: batch, height: fh, width: fw)
            flow = zip(flowUp, delta).map { $0 + $1 }
        }
        return flow
    }

    private func basicModule(_ input: [Float], level: Int, batch: Int, height: Int, width: Int) throws -> [Float] {
        let shape: [NSNumber] = [batch as NSNumber, 8, height as NSNumber, width as NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        try Self.write(input, into: array)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: array),
        ])
        let result = try blocks[level].prediction(from: features)
        guard let output = result.featureValue(for: outputName)?.multiArrayValue else {
            throw CLIError("SPyNet-Block \(level) hat keine Ausgabe \(outputName)")
        }
        return try Self.read(output)
    }

    private static func write(_ values: [Float], into array: MLMultiArray) throws {
        let count = array.count
        guard values.count == count else {
            throw CLIError("SPyNet-Block erwartet \(count) Werte, bekam \(values.count)")
        }
        if array.dataType == .float32, isContiguous(array) {
            values.withUnsafeBufferPointer { src in
                array.dataPointer.copyMemory(from: src.baseAddress!, byteCount: count * 4)
            }
            return
        }
        for i in 0..<count {
            array[i] = NSNumber(value: values[i])
        }
    }

    private static func read(_ array: MLMultiArray) throws -> [Float] {
        let count = array.count
        var values = [Float](repeating: 0, count: count)
        if array.dataType == .float32, isContiguous(array) {
            values.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: array.dataPointer.assumingMemoryBound(to: Float.self), count: count)
            }
            return values
        }
        for i in 0..<count {
            values[i] = array[i].floatValue
        }
        return values
    }

    private static func isContiguous(_ array: MLMultiArray) -> Bool {
        var expected = 1
        for dimension in (0..<array.shape.count).reversed() {
            if array.strides[dimension].intValue != expected { return false }
            expected *= array.shape[dimension].intValue
        }
        return true
    }

    private func normalize(_ rgb: [Float], batch: Int, height: Int, width: Int) -> [Float] {
        let plane = height * width
        var out = rgb
        for n in 0..<batch {
            for c in 0..<3 {
                let base = (n * 3 + c) * plane
                let m = mean[c], s = std[c]
                for i in 0..<plane { out[base + i] = (out[base + i] - m) / s }
            }
        }
        return out
    }

    private func avgPool2(_ input: [Float], batch: Int, channels: Int, height: Int, width: Int) -> [Float] {
        let outH = height / 2, outW = width / 2
        var out = [Float](repeating: 0, count: batch * channels * outH * outW)
        for n in 0..<batch {
            for c in 0..<channels {
                let src = (n * channels + c) * height * width
                let dst = (n * channels + c) * outH * outW
                for y in 0..<outH {
                    for x in 0..<outW {
                        let y0 = y * 2, x0 = x * 2
                        let a = input[src + y0 * width + x0]
                        let b = input[src + y0 * width + x0 + 1]
                        let c0 = input[src + (y0 + 1) * width + x0]
                        let d = input[src + (y0 + 1) * width + x0 + 1]
                        out[dst + y * outW + x] = (a + b + c0 + d) * 0.25
                    }
                }
            }
        }
        return out
    }

    private func resize(_ input: [Float], batch: Int, channels: Int, inH: Int, inW: Int, outH: Int, outW: Int, alignCorners: Bool) -> [Float] {
        if inH == outH && inW == outW { return input }
        var out = [Float](repeating: 0, count: batch * channels * outH * outW)
        for n in 0..<batch {
            for c in 0..<channels {
                let srcOff = (n * channels + c) * inH * inW
                let dstOff = (n * channels + c) * outH * outW
                for y in 0..<outH {
                    let ys: Float
                    if alignCorners {
                        ys = outH == 1 ? 0 : Float(y) * Float(inH - 1) / Float(outH - 1)
                    } else {
                        ys = (Float(y) + 0.5) * Float(inH) / Float(outH) - 0.5
                    }
                    for x in 0..<outW {
                        let xs: Float
                        if alignCorners {
                            xs = outW == 1 ? 0 : Float(x) * Float(inW - 1) / Float(outW - 1)
                        } else {
                            xs = (Float(x) + 0.5) * Float(inW) / Float(outW) - 0.5
                        }
                        out[dstOff + y * outW + x] = sample(input, offset: srcOff, height: inH, width: inW, y: ys, x: xs, border: false)
                    }
                }
            }
        }
        return out
    }

    private func warp(image: [Float], flow: [Float], batch: Int, channels: Int, height: Int, width: Int) -> [Float] {
        let plane = height * width
        var out = [Float](repeating: 0, count: batch * channels * plane)
        for n in 0..<batch {
            let flowOff = n * 2 * plane
            for y in 0..<height {
                for x in 0..<width {
                    let fx = Float(x) + flow[flowOff + y * width + x]
                    let fy = Float(y) + flow[flowOff + plane + y * width + x]
                    for c in 0..<channels {
                        let srcOff = (n * channels + c) * plane
                        let dstOff = (n * channels + c) * plane
                        out[dstOff + y * width + x] = sample(image, offset: srcOff, height: height, width: width, y: fy, x: fx, border: true)
                    }
                }
            }
        }
        return out
    }

    /// `border` is grid_sample padding_mode=border (clamp). Otherwise out-of-range is 0, as in interpolate.
    private func sample(_ src: [Float], offset: Int, height: Int, width: Int, y: Float, x: Float, border: Bool) -> Float {
        let y0 = floorf(y)
        let x0 = floorf(x)
        let wy = y - y0
        let wx = x - x0
        var acc: Float = 0
        for dy in 0...1 {
            for dx in 0...1 {
                var iy = Int(y0) + dy
                var ix = Int(x0) + dx
                let weight = (dy == 0 ? 1 - wy : wy) * (dx == 0 ? 1 - wx : wx)
                if border {
                    iy = min(max(iy, 0), height - 1)
                    ix = min(max(ix, 0), width - 1)
                    acc += weight * src[offset + iy * width + ix]
                } else if iy >= 0 && iy < height && ix >= 0 && ix < width {
                    acc += weight * src[offset + iy * width + ix]
                }
            }
        }
        return acc
    }

    private func concatChannels(_ tensors: [[Float]], batch: Int, channels: [Int], height: Int, width: Int) -> [Float] {
        let totalC = channels.reduce(0, +)
        let plane = height * width
        var out = [Float](repeating: 0, count: batch * totalC * plane)
        for n in 0..<batch {
            var destC = 0
            for (tensor, cCount) in zip(tensors, channels) {
                for c in 0..<cCount {
                    let src = (n * cCount + c) * plane
                    let dst = (n * totalC + destC) * plane
                    for i in 0..<plane { out[dst + i] = tensor[src + i] }
                    destC += 1
                }
            }
        }
        return out
    }
}
