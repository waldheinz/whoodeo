import CoreGraphics
import CoreML
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Encoder and decoder of one vae2, as `whoodeo-coreml --part pair` writes them.
nonisolated struct VAEPair: Identifiable, Hashable {
    var encoder: URL
    var decoder: URL
    var title: String
    var detail: String

    var id: URL { decoder }
}

/// The playground's folder beside Models. The model list does not look in here.
nonisolated enum VAELibrary {
    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("Whoodeo/VAE", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Every decoder with an encoder of the same name beside it, newest first.
    static func pairs() -> [VAEPair] {
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles]
        )) ?? []
        func modified(_ url: URL) -> Date {
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        return urls
            .filter { $0.pathExtension == "mlpackage" && $0.lastPathComponent.hasSuffix("-decoder.mlpackage") }
            .compactMap { decoder -> VAEPair? in
                let stem = decoder.deletingPathExtension().lastPathComponent.dropLast("-decoder".count)
                let encoder = directory.appendingPathComponent("\(stem)-encoder.mlpackage")
                guard FileManager.default.fileExists(atPath: encoder.path) else { return nil }
                let (title, detail) = describe(String(stem))
                return VAEPair(encoder: encoder, decoder: decoder, title: title, detail: detail)
            }
            .sorted { modified($0.decoder) > modified($1.decoder) }
    }

    private static func describe(_ stem: String) -> (String, String) {
        guard let match = stem.firstMatch(of: /^(?:train-(\d{8})-(\d{4})\d{2}-)?vae2-(\d+)x(\d+)-step(\d+)$/) else {
            return (stem, "Core ML")
        }
        var detail = "Schritt \(Int(match.5) ?? 0)"
        if let day = match.1, let time = match.2 {
            let d = Array(day), t = Array(time)
            detail += " · Lauf \(String(d[6...7])).\(String(d[4...5])). \(String(t[0...1])):\(String(t[2...3]))"
        }
        return ("vae2 \(match.3)×\(match.4)", detail)
    }
}

/// A latent in normalized units: (μ - shift) * scale, mean 0 and std 1 per channel. Planar CHW.
nonisolated struct Latent {
    var channels: Int
    var height: Int
    var width: Int
    var values: [Float]

    var plane: Int { height * width }
    var size: FrameSize { FrameSize(width: width, height: height) }
}

nonisolated final class VAECoder {
    let pair: VAEPair
    let channels: Int
    let factor: Int
    let shift: [Float]
    let scale: [Float]
    /// Latent sizes both packages take, in latent cells.
    let latentSizes: [FrameSize]
    let step: Int?
    private let encoder: MLModel
    private let decoder: MLModel
    private let encoderInput: String
    private let encoderOutput: String
    private let decoderInput: String
    private let decoderOutput: String

    init(pair: VAEPair) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        encoder = try MLModel(contentsOf: try MLModel.compileModel(at: pair.encoder), configuration: config)
        decoder = try MLModel(contentsOf: try MLModel.compileModel(at: pair.decoder), configuration: config)
        let meta = decoder.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let encoderMeta = encoder.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard meta["whoodeo.part"] == "decoder", encoderMeta["whoodeo.part"] == "encoder" else {
            throw CLIError("Das ist kein vae2-Paar aus whoodeo-coreml --part pair.")
        }
        func floats(_ key: String) -> [Float] {
            (meta[key] ?? "").split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        }
        shift = floats("whoodeo.latent_shift")
        scale = floats("whoodeo.latent_scale")
        channels = Int(meta["whoodeo.latent"] ?? "") ?? shift.count
        guard channels > 0, shift.count == channels, scale.count == channels else {
            throw CLIError("Dem Decoder fehlen Latent-Shift und -Scale pro Kanal.")
        }
        factor = Int(meta["whoodeo.factor"] ?? "") ?? 8
        latentSizes = (meta["whoodeo.latent_sizes"] ?? "").split(separator: ",").compactMap { part in
            let sides = part.lowercased().split(separator: "x").compactMap { Int($0) }
            return sides.count == 2 ? FrameSize(width: sides[1], height: sides[0]) : nil
        }
        guard !latentSizes.isEmpty else { throw CLIError("Der Decoder nennt keine Latent-Größen.") }
        step = Int(meta["whoodeo.step"] ?? "")
        encoderInput = encoder.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        encoderOutput = encoder.modelDescription.outputDescriptionsByName.keys.first ?? "mu"
        decoderInput = decoder.modelDescription.inputDescriptionsByName.keys.first ?? "latent"
        decoderOutput = decoder.modelDescription.outputDescriptionsByName.keys.first ?? "image"
        self.pair = pair
    }

    /// Planar RGB in 0..1, sides `factor` times one of `latentSizes`.
    func encode(_ rgb: [Float], width: Int, height: Int) throws -> Latent {
        try autoreleasepool {
            let input = try Tensor16.array(rgb, channels: 3, height: height, width: width)
            let result = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                encoderInput: MLFeatureValue(multiArray: input),
            ]))
            guard let output = result.featureValue(for: encoderOutput)?.multiArrayValue else {
                throw CLIError("keine Ausgabe \(encoderOutput)")
            }
            var (values, c, h, w) = Tensor16.read(output)
            guard c == channels else { throw CLIError("Encoder gibt \(c) Kanäle, erwartet \(channels)") }
            let plane = h * w
            values.withUnsafeMutableBufferPointer { v in
                for k in 0..<c {
                    let s = shift[k], g = scale[k]
                    for i in (k * plane)..<((k + 1) * plane) { v[i] = (v[i] - s) * g }
                }
            }
            return Latent(channels: c, height: h, width: w, values: values)
        }
    }

    /// The clamped frame, planar RGB, `factor` times the latent's sides.
    func decode(_ latent: Latent) throws -> (rgb: [Float], width: Int, height: Int) {
        try autoreleasepool {
            var mu = latent.values
            let plane = latent.plane
            mu.withUnsafeMutableBufferPointer { v in
                for k in 0..<latent.channels {
                    let s = shift[k], g = scale[k]
                    for i in (k * plane)..<((k + 1) * plane) { v[i] = v[i] / g + s }
                }
            }
            let input = try Tensor16.array(mu, channels: latent.channels, height: latent.height, width: latent.width)
            let result = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                decoderInput: MLFeatureValue(multiArray: input),
            ]))
            guard let output = result.featureValue(for: decoderOutput)?.multiArrayValue else {
                throw CLIError("keine Ausgabe \(decoderOutput)")
            }
            let (values, _, h, w) = Tensor16.read(output)
            return (values, w, h)
        }
    }
}

/// fp16 MLMultiArray in and out, along the array's strides.
nonisolated enum Tensor16 {
    static func array(_ values: [Float], channels: Int, height: Int, width: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, channels as NSNumber, height as NSNumber, width as NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float16)
        let strides = array.strides.map(\.intValue)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        values.withUnsafeBufferPointer { source in
            for c in 0..<channels {
                for y in 0..<height {
                    let from = (c * height + y) * width
                    let to = c * strides[1] + y * strides[2]
                    if strides[3] == 1 {
                        for x in 0..<width { pointer[to + x] = Float16(source[from + x]) }
                    } else {
                        for x in 0..<width { pointer[to + x * strides[3]] = Float16(source[from + x]) }
                    }
                }
            }
        }
        return array
    }

    static func read(_ array: MLMultiArray) -> ([Float], Int, Int, Int) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count >= 3 else { return ([], 0, 0, 0) }
        let o = shape.count - 3
        let channels = shape[o], height = shape[o + 1], width = shape[o + 2]
        var out = [Float](repeating: 0, count: channels * height * width)
        out.withUnsafeMutableBufferPointer { dest in
            func copy<T: BinaryFloatingPoint>(_: T.Type) {
                let pointer = array.dataPointer.bindMemory(to: T.self, capacity: array.count)
                for c in 0..<channels {
                    for y in 0..<height {
                        let from = c * strides[o] + y * strides[o + 1]
                        let to = (c * height + y) * width
                        for x in 0..<width { dest[to + x] = Float(pointer[from + x * strides[o + 2]]) }
                    }
                }
            }
            switch array.dataType {
            case .float16: copy(Float16.self)
            case .float32: copy(Float.self)
            case .double: copy(Double.self)
            default: break
            }
        }
        return (out, channels, height, width)
    }
}

/// Pictures for the playground: sRGB, planar float in 0..1 on the model side.
nonisolated enum PlaygroundImage {
    static let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    static func load(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }

    static func isImage(_ url: URL) -> Bool {
        guard let type = UTType(filenameExtension: url.pathExtension) else { return false }
        return type.conforms(to: .image)
    }

    /// Scale to cover width x height and cut the middle, 16 bits per channel.
    static func fill(_ image: CGImage, width: Int, height: Int) -> [Float]? {
        let plane = width * height
        var raw = [UInt16](repeating: 0, count: plane * 4)
        let drawn = raw.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 16,
                bytesPerRow: width * 8, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.setFillColor(CGColor(gray: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            let iw = CGFloat(image.width), ih = CGFloat(image.height)
            let s = max(CGFloat(width) / iw, CGFloat(height) / ih)
            let dw = iw * s, dh = ih * s
            context.draw(image, in: CGRect(x: (CGFloat(width) - dw) / 2, y: (CGFloat(height) - dh) / 2, width: dw, height: dh))
            return true
        }
        guard drawn else { return nil }
        var out = [Float](repeating: 0, count: plane * 3)
        let unit = Float(1) / 65535
        raw.withUnsafeBufferPointer { r in
            out.withUnsafeMutableBufferPointer { o in
                for i in 0..<plane {
                    o[i] = Float(r[i * 4]) * unit
                    o[plane + i] = Float(r[i * 4 + 1]) * unit
                    o[2 * plane + i] = Float(r[i * 4 + 2]) * unit
                }
            }
        }
        return out
    }

    /// 16-bit picture of a decoded film frame, so filling it loses nothing.
    static func image16(planar rgb: [Float], width: Int, height: Int) -> CGImage? {
        let plane = width * height
        guard plane > 0, rgb.count >= plane * 3 else { return nil }
        var raw = [UInt16](repeating: 65535, count: plane * 4)
        rgb.withUnsafeBufferPointer { s in
            raw.withUnsafeMutableBufferPointer { d in
                for i in 0..<plane {
                    d[i * 4] = word(s[i])
                    d[i * 4 + 1] = word(s[plane + i])
                    d[i * 4 + 2] = word(s[2 * plane + i])
                }
            }
        }
        let data = raw.withUnsafeBytes { Data($0) }
        return make(data, width: width, height: height, bits: 16,
                    info: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
    }

    static func image(planar rgb: [Float], width: Int, height: Int) -> CGImage? {
        let plane = width * height
        guard plane > 0, rgb.count >= plane * 3 else { return nil }
        var raw = [UInt8](repeating: 255, count: plane * 4)
        rgb.withUnsafeBufferPointer { s in
            raw.withUnsafeMutableBufferPointer { d in
                for i in 0..<plane {
                    d[i * 4] = byte(s[i])
                    d[i * 4 + 1] = byte(s[plane + i])
                    d[i * 4 + 2] = byte(s[2 * plane + i])
                }
            }
        }
        return make(Data(raw), width: width, height: height, bits: 8, info: CGImageAlphaInfo.noneSkipLast.rawValue)
    }

    /// One latent channel as gray, `-range` black and `+range` white.
    static func channel(_ latent: Latent, _ index: Int, range: Float = 3) -> CGImage? {
        let plane = latent.plane
        guard index < latent.channels, plane > 0 else { return nil }
        var raw = [UInt8](repeating: 0, count: plane)
        latent.values.withUnsafeBufferPointer { v in
            for i in 0..<plane { raw[i] = byte((v[index * plane + i] + range) / (2 * range)) }
        }
        guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
        return CGImage(
            width: latent.width, height: latent.height, bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: latent.width, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }

    static func psnr(_ a: [Float], _ b: [Float]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var sum = 0.0
        a.withUnsafeBufferPointer { x in
            b.withUnsafeBufferPointer { y in
                for i in 0..<x.count {
                    let d = Double(min(max(x[i], 0), 1) - min(max(y[i], 0), 1))
                    sum += d * d
                }
            }
        }
        let mse = sum / Double(a.count)
        return mse > 0 ? 10 * log10(1 / mse) : 99
    }

    static func savePNG(_ image: CGImage, to url: URL) throws {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CLIError("PNG lässt sich nicht anlegen")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CLIError("PNG lässt sich nicht schreiben") }
    }

    @inline(__always) static func byte(_ v: Float) -> UInt8 {
        guard v.isFinite else { return 0 }
        return UInt8(min(max(v, 0), 1) * 255 + 0.5)
    }

    @inline(__always) static func word(_ v: Float) -> UInt16 {
        guard v.isFinite else { return 0 }
        return UInt16(min(max(v, 0), 1) * 65535 + 0.5)
    }

    private static func make(_ data: Data, width: Int, height: Int, bits: Int, info: UInt32) -> CGImage? {
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: bits, bitsPerPixel: bits * 4,
            bytesPerRow: width * bits / 2, space: space, bitmapInfo: CGBitmapInfo(rawValue: info),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
        )
    }
}
