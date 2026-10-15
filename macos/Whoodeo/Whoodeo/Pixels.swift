import Accelerate
import CoreImage
import CoreVideo
import Foundation

struct Picture {
    var image: CGImage
    var width: Int
    var height: Int
}

enum Pixels {
    private static let flags = vImage_Flags(kvImageNoFlags)
    private static let scale16 = Float(1) / 65535

    /// CHW float in 0..1, the same scaling as `frame_tensor`.
    static func planar(_ buffer: CVPixelBuffer) throws -> [Float] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let plane = width * height
        var out = [Float](repeating: 0, count: plane * 3)
        try out.withUnsafeMutableBufferPointer { dest in
            guard let base = dest.baseAddress else { throw CLIError("Bildpuffer fehlt") }
            if CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_64RGBALE {
                try from64(buffer, into: base, width: width, height: height)
            } else {
                try fromBGRA(buffer, into: base, width: width, height: height)
            }
        }
        return out
    }

    /// 16-bit picture. Core Image copies the buffer; the view then paints that image.
    static func picture(planar rgb: [Float], width: Int, height: Int) throws -> Picture {
        try picture(try rgba16(rgb, width: width, height: height))
    }

    static func picture(_ buffer: CVPixelBuffer) throws -> Picture {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_64RGBALE else {
            throw CLIError("Bildformat lässt sich nicht zeichnen")
        }
        let space = Preview.space
        CVBufferSetAttachment(buffer, kCVImageBufferCGColorSpaceKey, space, .shouldPropagate)
        CVBufferRemoveAttachment(buffer, kCVImageBufferColorPrimariesKey)
        CVBufferRemoveAttachment(buffer, kCVImageBufferTransferFunctionKey)
        CVBufferRemoveAttachment(buffer, kCVImageBufferYCbCrMatrixKey)
        let source = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: space])
        guard let image = Preview.context.createCGImage(source, from: source.extent, format: .RGBA16, colorSpace: space) else {
            throw CLIError("Bild lässt sich nicht zeichnen")
        }
        return Picture(image: image, width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
    }

    /// Full-range 16-bit RGBA. Callers that encode set their own color attachments.
    static func rgba16(_ rgb: [Float], width: Int, height: Int) throws -> CVPixelBuffer {
        let plane = width * height
        guard width > 0, height > 0, rgb.count >= plane * 3 else {
            throw CLIError("Bild lässt sich nicht zeichnen")
        }
        let buffer = try RGBAPool.take(width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        var planes = [UInt16](repeating: 0, count: plane * 3)
        var alpha = [UInt16](repeating: 65535, count: plane)
        try rgb.withUnsafeBufferPointer { src in
            try planes.withUnsafeMutableBufferPointer { dst in
                guard let sample = src.baseAddress, let out = dst.baseAddress else {
                    throw CLIError("Bildpuffer fehlt")
                }
                for channel in 0..<3 {
                    var from = imageBuffer(
                        UnsafeMutableRawPointer(mutating: sample).advanced(by: plane * channel * MemoryLayout<Float>.stride),
                        width: width,
                        height: height,
                        rowBytes: width * MemoryLayout<Float>.stride
                    )
                    var to = imageBuffer(
                        UnsafeMutableRawPointer(out).advanced(by: plane * channel * MemoryLayout<UInt16>.stride),
                        width: width,
                        height: height,
                        rowBytes: width * MemoryLayout<UInt16>.stride
                    )
                    try demand(vImageConvert_FTo16U(&from, &to, 0, scale16, flags), "16-bit-Bild")
                }
            }
        }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw CLIError("Bildpuffer fehlt") }
        try planes.withUnsafeMutableBufferPointer { src in
            try alpha.withUnsafeMutableBufferPointer { a in
                guard let sample = src.baseAddress, let opaque = a.baseAddress else {
                    throw CLIError("Bildpuffer fehlt")
                }
                var red = imageBuffer(UnsafeMutableRawPointer(sample), width: width, height: height, rowBytes: width * 2)
                var green = imageBuffer(
                    UnsafeMutableRawPointer(sample).advanced(by: plane * 2),
                    width: width,
                    height: height,
                    rowBytes: width * 2
                )
                var blue = imageBuffer(
                    UnsafeMutableRawPointer(sample).advanced(by: plane * 4),
                    width: width,
                    height: height,
                    rowBytes: width * 2
                )
                var cover = imageBuffer(opaque, width: width, height: height, rowBytes: width * 2)
                var dest = imageBuffer(base, width: width, height: height, rowBytes: CVPixelBufferGetBytesPerRow(buffer))
                // Planar16UtoARGB16U writes its four sources in order. RGBA memory wants R, G, B, A.
                try demand(vImageConvert_Planar16UtoARGB16U(&red, &green, &blue, &cover, &dest, flags), "16-bit-Bild")
            }
        }
        return buffer
    }

    /// Nearest-neighbor enlargement, so a block stays a block.
    static func nearest(_ rgb: [Float], width: Int, height: Int, scale: Int) -> [Float] {
        if scale <= 1 { return rgb }
        let dw = width * scale
        let dh = height * scale
        let srcCount = width * height
        guard rgb.count >= srcCount * 3 else { return rgb }
        var out = [Float](repeating: 0, count: dw * dh * 3)
        rgb.withUnsafeBufferPointer { source in
            out.withUnsafeMutableBufferPointer { dest in
                guard let sample = source.baseAddress, let pixel = dest.baseAddress else { return }
                for channel in 0..<3 {
                    let from = sample.advanced(by: srcCount * channel)
                    let to = pixel.advanced(by: dw * dh * channel)
                    for y in 0..<height {
                        let srcRow = from.advanced(by: y * width)
                        for copy in 0..<scale {
                            let dstRow = to.advanced(by: (y * scale + copy) * dw)
                            for x in 0..<width {
                                let value = srcRow[x]
                                let start = x * scale
                                for step in 0..<scale {
                                    dstRow[start + step] = value
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    static func stack(_ frames: [[Float]]) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(frames.reduce(0) { $0 + $1.count })
        for frame in frames {
            out.append(contentsOf: frame)
        }
        return out
    }

    private static func fromBGRA(_ buffer: CVPixelBuffer, into out: UnsafeMutablePointer<Float>, width: Int, height: Int) throws {
        let plane = width * height
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw CLIError("Bildpuffer fehlt") }
        var src = imageBuffer(base, width: width, height: height, rowBytes: CVPixelBufferGetBytesPerRow(buffer))
        var red = imageBuffer(UnsafeMutableRawPointer(out), width: width, height: height, rowBytes: width * 4)
        var green = imageBuffer(UnsafeMutableRawPointer(out.advanced(by: plane)), width: width, height: height, rowBytes: width * 4)
        var blue = imageBuffer(UnsafeMutableRawPointer(out.advanced(by: plane * 2)), width: width, height: height, rowBytes: width * 4)
        var spare = [Float](repeating: 0, count: plane)
        let maxFloat: [Float] = [1, 1, 1, 1]
        let minFloat: [Float] = [0, 0, 0, 0]
        try spare.withUnsafeMutableBufferPointer { alphaBuf in
            guard let alphaBase = alphaBuf.baseAddress else { throw CLIError("Bildpuffer fehlt") }
            var alpha = imageBuffer(alphaBase, width: width, height: height, rowBytes: width * 4)
            try maxFloat.withUnsafeBufferPointer { maxPtr in
                try minFloat.withUnsafeBufferPointer { minPtr in
                    guard let maxBase = maxPtr.baseAddress, let minBase = minPtr.baseAddress else {
                        throw CLIError("Bildpuffer fehlt")
                    }
                    // BGRA bytes are read as A,R,G,B. Send B to blue, G to green, R to red, and drop A.
                    try demand(
                        vImageConvert_ARGB8888toPlanarF(&src, &blue, &green, &red, &alpha, maxBase, minBase, flags),
                        "8-bit-Bild"
                    )
                }
            }
        }
    }

    private static func from64(_ buffer: CVPixelBuffer, into out: UnsafeMutablePointer<Float>, width: Int, height: Int) throws {
        let plane = width * height
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw CLIError("Bildpuffer fehlt") }
        var interleaved = [Float](repeating: 0, count: plane * 4)
        try interleaved.withUnsafeMutableBufferPointer { chunk in
            guard let packedBase = chunk.baseAddress else { throw CLIError("Bildpuffer fehlt") }
            var src = imageBuffer(base, width: width * 4, height: height, rowBytes: CVPixelBufferGetBytesPerRow(buffer))
            var wide = imageBuffer(packedBase, width: width * 4, height: height, rowBytes: width * 16)
            try demand(vImageConvert_16UToF(&src, &wide, 0, scale16, flags), "10-bit-Bild")
            var packed = imageBuffer(packedBase, width: width, height: height, rowBytes: width * 16)
            var red = imageBuffer(UnsafeMutableRawPointer(out), width: width, height: height, rowBytes: width * 4)
            var green = imageBuffer(UnsafeMutableRawPointer(out.advanced(by: plane)), width: width, height: height, rowBytes: width * 4)
            var blue = imageBuffer(UnsafeMutableRawPointer(out.advanced(by: plane * 2)), width: width, height: height, rowBytes: width * 4)
            var spare = [Float](repeating: 0, count: plane)
            try spare.withUnsafeMutableBufferPointer { alphaBuf in
                guard let alphaBase = alphaBuf.baseAddress else { throw CLIError("Bildpuffer fehlt") }
                var alpha = imageBuffer(alphaBase, width: width, height: height, rowBytes: width * 4)
                // RGBA floats: first component is red.
                try demand(vImageConvert_ARGBFFFFtoPlanarF(&packed, &red, &green, &blue, &alpha, flags), "10-bit-Bild")
            }
        }
    }

    private static func imageBuffer<T>(_ data: UnsafeMutablePointer<T>, width: Int, height: Int, rowBytes: Int) -> vImage_Buffer {
        imageBuffer(UnsafeMutableRawPointer(data), width: width, height: height, rowBytes: rowBytes)
    }

    private static func imageBuffer(_ data: UnsafeMutableRawPointer, width: Int, height: Int, rowBytes: Int) -> vImage_Buffer {
        vImage_Buffer(data: data, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: rowBytes)
    }

    private static func demand(_ code: vImage_Error, _ what: String) throws {
        if code != kvImageNoError {
            throw CLIError("\(what) (\(code))")
        }
    }
}

private enum Preview {
    static let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    static let context = CIContext(options: [
        .cacheIntermediates: false,
        .workingColorSpace: space,
    ])
}

private enum RGBAPool {
    static let lock = NSLock()
    static var pool: CVPixelBufferPool?
    static var width = 0
    static var height = 0

    static func take(width: Int, height: Int) throws -> CVPixelBuffer {
        lock.lock()
        defer { lock.unlock() }
        if pool == nil || self.width != width || self.height != height {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBALE,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &created)
            guard status == kCVReturnSuccess, let created else {
                throw CLIError("16-bit-Puffer fehlt (\(status))")
            }
            pool = created
            self.width = width
            self.height = height
        }
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool!, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw CLIError("16-bit-Puffer fehlt (\(status))")
        }
        return buffer
    }
}

enum TensorFile {
    /// int32 n,c,h,w then float32, little-endian, contiguous NCHW.
    static func read(_ url: URL) throws -> (shape: [Int], data: [Float]) {
        let data = try Data(contentsOf: url)
        guard data.count >= 16 else { throw CLIError("\(url.path) is shorter than a header") }
        let shape = data.withUnsafeBytes { raw -> [Int] in
            let ints = raw.bindMemory(to: Int32.self)
            return (0..<4).map { Int(ints[$0]) }
        }
        let count = shape.reduce(1, *)
        let bytes = count * MemoryLayout<Float>.size
        guard shape.allSatisfy({ $0 > 0 }), data.count >= 16 + bytes else {
            throw CLIError("\(url.path) does not match its header")
        }
        var values = [Float](repeating: 0, count: count)
        _ = values.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: 16..<(16 + bytes))
        }
        return (shape, values)
    }

    static func write(_ url: URL, shape: [Int], data: [Float]) throws {
        var header = shape.map { Int32($0) }
        var out = Data(bytes: &header, count: 16)
        data.withUnsafeBytes { out.append(contentsOf: $0) }
        try out.write(to: url)
    }
}

struct CLIError: LocalizedError, CustomStringConvertible {
    var description: String
    var errorDescription: String? { description }
    init(_ description: String) { self.description = description }
}
