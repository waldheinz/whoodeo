import CoreGraphics
import Foundation

nonisolated enum ChannelMode: String, CaseIterable, Identifiable, Sendable {
    case normal, mute, invert

    var id: String { rawValue }

    var title: String {
        switch self {
        case .normal: "Normal"
        case .mute: "Stumm"
        case .invert: "Invertiert"
        }
    }
}

nonisolated struct ChannelSetting: Equatable, Sendable {
    var mode: ChannelMode = .normal
    var gain: Float = 1
    var offset: Float = 0
    /// Take this channel from picture B instead of the blend.
    var fromB = false

    var isPlain: Bool { self == ChannelSetting() }
}

nonisolated enum BlendKind: String, CaseIterable, Identifiable, Sendable {
    case linear, spherical

    var id: String { rawValue }
    var title: String { self == .linear ? "Linear" : "Sphärisch" }
}

/// Everything the result depends on besides the two encoded pictures.
nonisolated struct PlaygroundParams: Equatable, Sendable {
    var blend: Float = 0
    var blendKind: BlendKind = .linear
    var noise: Float = 0
    var noiseSeed: UInt64 = 1
    /// Angle between two noise fields: animating it walks the noise smoothly, same strength.
    var noisePhase: Float = 0
    /// Noise cells in latent cells: 1 is white noise, larger is blotchy.
    var noiseGrain = 1
    var noiseOnlySelected = false
    var selected = 0
    var channels: [ChannelSetting] = Array(repeating: ChannelSetting(), count: 16)
    var solo: Int?
    var swapFirst: Int?
    var swapSecond: Int?
    var gainAll: Float = 1
    var flipH = false
    var flipV = false
    var shiftX = 0
    var shiftY = 0
    /// Average blocks of this many latent cells: 1 is off.
    var coarse = 1
    /// Bits per value over -3..3: 0 is off.
    var bits = 0
    /// Decode at another latent size, nil for the working size.
    var decodeSize: FrameSize?
    var bilinear = true
    var roundTrips = 0
}

/// One picture of the playground, filled to the working size and encoded.
nonisolated struct SlotData {
    var rgb: [Float]
    var width: Int
    var height: Int
    var latent: Latent
    var reconstruction: [Float]
    var original: CGImage?
    var reconstructionImage: CGImage?
    var psnr: Double?
}

nonisolated struct RenderOutput {
    var result: CGImage?
    var side: CGImage?
    var sideTitle: String
    var tiles: [CGImage]
    var psnrA: Double?
    var psnrBlend: Double?
    var width: Int
    var height: Int
    var milliseconds: Double
}

nonisolated enum PlaygroundRender {
    static func encode(_ coder: VAECoder, image: CGImage, size: FrameSize) throws -> SlotData {
        let width = size.width * coder.factor
        let height = size.height * coder.factor
        guard let rgb = PlaygroundImage.fill(image, width: width, height: height) else {
            throw CLIError("Bild lässt sich nicht auf \(width)×\(height) bringen")
        }
        let latent = try coder.encode(rgb, width: width, height: height)
        let back = try coder.decode(latent)
        return SlotData(
            rgb: rgb, width: width, height: height, latent: latent, reconstruction: back.rgb,
            original: PlaygroundImage.image(planar: rgb, width: width, height: height),
            reconstructionImage: PlaygroundImage.image(planar: back.rgb, width: back.width, height: back.height),
            psnr: PlaygroundImage.psnr(rgb, back.rgb)
        )
    }

    static func run(_ coder: VAECoder, a: SlotData, b: SlotData?, p: PlaygroundParams) throws -> RenderOutput {
        let started = CFAbsoluteTimeGetCurrent()
        let b = b.flatMap { $0.latent.size == a.latent.size ? $0 : nil }
        var z = a.latent
        let count = z.channels
        let plane = z.plane
        if let b, p.blend > 0 {
            z = p.blendKind == .linear ? LatentOps.lerp(a.latent, b.latent, p.blend)
                : LatentOps.slerp(a.latent, b.latent, p.blend)
        }
        z.values.withUnsafeMutableBufferPointer { v in
            for c in 0..<min(count, p.channels.count) {
                let setting = p.channels[c]
                if setting.fromB, let b {
                    b.latent.values.withUnsafeBufferPointer { src in
                        for i in 0..<plane { v[c * plane + i] = src[c * plane + i] }
                    }
                }
                let sign: Float = setting.mode == .invert ? -1 : 1
                let gain = setting.mode == .mute ? 0 : setting.gain * sign * p.gainAll
                let offset = setting.mode == .mute ? 0 : setting.offset
                if gain == 1 && offset == 0 { continue }
                for i in (c * plane)..<((c + 1) * plane) { v[i] = v[i] * gain + offset }
            }
            if let solo = p.solo, solo < count {
                for c in 0..<count where c != solo {
                    for i in (c * plane)..<((c + 1) * plane) { v[i] = 0 }
                }
            }
            if let first = p.swapFirst, let second = p.swapSecond, first != second, first < count, second < count {
                for i in 0..<plane { v.swapAt(first * plane + i, second * plane + i) }
            }
        }
        if p.flipH || p.flipV || p.shiftX != 0 || p.shiftY != 0 {
            z = LatentOps.move(z, flipH: p.flipH, flipV: p.flipV, dx: p.shiftX, dy: p.shiftY)
        }
        if p.coarse > 1 { z = LatentOps.coarsen(z, block: p.coarse) }
        if p.noise > 0 {
            let only = p.noiseOnlySelected ? p.selected : nil
            LatentOps.addNoise(&z, sigma: p.noise, seed: p.noiseSeed, phase: p.noisePhase, grain: p.noiseGrain, only: only)
        }
        if p.bits > 0 { LatentOps.quantize(&z, bits: p.bits) }
        let tiles = (0..<count).compactMap { PlaygroundImage.channel(z, $0) }
        if let size = p.decodeSize, size != z.size, coder.latentSizes.contains(size) {
            z = LatentOps.resize(z, to: size, bilinear: p.bilinear)
        }
        var decoded = try coder.decode(z)
        for _ in 0..<p.roundTrips {
            let again = try coder.encode(decoded.rgb, width: decoded.width, height: decoded.height)
            decoded = try coder.decode(again)
        }
        let same = decoded.width == a.width && decoded.height == a.height
        var side: CGImage?
        var sideTitle = ""
        var psnrBlend: Double?
        if let b {
            let t = p.blend
            if t <= 0 {
                side = b.original
                sideTitle = "Original B"
            } else {
                var mix = a.rgb
                mix.withUnsafeMutableBufferPointer { m in
                    b.rgb.withUnsafeBufferPointer { s in
                        for i in 0..<m.count { m[i] = m[i] * (1 - t) + s[i] * t }
                    }
                }
                side = PlaygroundImage.image(planar: mix, width: a.width, height: a.height)
                sideTitle = String(format: "Pixel-Blende %.0f %%", t * 100)
                if same { psnrBlend = PlaygroundImage.psnr(mix, decoded.rgb) }
            }
        } else if same {
            var diff = decoded.rgb
            diff.withUnsafeMutableBufferPointer { d in
                a.rgb.withUnsafeBufferPointer { s in
                    for i in 0..<d.count { d[i] = abs(d[i] - s[i]) * 4 }
                }
            }
            side = PlaygroundImage.image(planar: diff, width: a.width, height: a.height)
            sideTitle = "Differenz zu A ×4"
        } else {
            sideTitle = "Differenz zu A: andere Größe"
        }
        return RenderOutput(
            result: PlaygroundImage.image(planar: decoded.rgb, width: decoded.width, height: decoded.height),
            side: side,
            sideTitle: sideTitle,
            tiles: tiles,
            psnrA: same ? PlaygroundImage.psnr(a.rgb, decoded.rgb) : nil,
            psnrBlend: psnrBlend,
            width: decoded.width,
            height: decoded.height,
            milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000
        )
    }
}

nonisolated enum LatentOps {
    static func lerp(_ a: Latent, _ b: Latent, _ t: Float) -> Latent {
        var out = a
        out.values.withUnsafeMutableBufferPointer { o in
            b.values.withUnsafeBufferPointer { s in
                for i in 0..<o.count { o[i] += (s[i] - o[i]) * t }
            }
        }
        return out
    }

    /// Per latent cell along the great circle between the two channel vectors, so the middle keeps its length.
    static func slerp(_ a: Latent, _ b: Latent, _ t: Float) -> Latent {
        var out = a
        let plane = a.plane, count = a.channels
        out.values.withUnsafeMutableBufferPointer { o in
            a.values.withUnsafeBufferPointer { x in
                b.values.withUnsafeBufferPointer { y in
                    for i in 0..<plane {
                        var dot: Float = 0, na: Float = 0, nb: Float = 0
                        for c in 0..<count {
                            let u = x[c * plane + i], w = y[c * plane + i]
                            dot += u * w; na += u * u; nb += w * w
                        }
                        let denom = (na * nb).squareRoot()
                        let cosine = denom > 0 ? min(max(dot / denom, -1), 1) : 1
                        let omega = acos(cosine)
                        let s = sin(omega)
                        let (wa, wb): (Float, Float) = s < 1e-4 ? (1 - t, t)
                            : (sin((1 - t) * omega) / s, sin(t * omega) / s)
                        for c in 0..<count {
                            o[c * plane + i] = wa * x[c * plane + i] + wb * y[c * plane + i]
                        }
                    }
                }
            }
        }
        return out
    }

    /// Mirror, then shift with wraparound, in latent cells.
    static func move(_ z: Latent, flipH: Bool, flipV: Bool, dx: Int, dy: Int) -> Latent {
        var out = z
        let w = z.width, h = z.height, plane = z.plane
        func wrap(_ v: Int, _ n: Int) -> Int { ((v % n) + n) % n }
        z.values.withUnsafeBufferPointer { s in
            out.values.withUnsafeMutableBufferPointer { o in
                for c in 0..<z.channels {
                    for y in 0..<h {
                        var sy = wrap(y - dy, h)
                        if flipV { sy = h - 1 - sy }
                        for x in 0..<w {
                            var sx = wrap(x - dx, w)
                            if flipH { sx = w - 1 - sx }
                            o[c * plane + y * w + x] = s[c * plane + sy * w + sx]
                        }
                    }
                }
            }
        }
        return out
    }

    /// Average over blocks, then repeat each average over its block.
    static func coarsen(_ z: Latent, block: Int) -> Latent {
        var out = z
        let w = z.width, h = z.height, plane = z.plane
        out.values.withUnsafeMutableBufferPointer { o in
            for c in 0..<z.channels {
                var by = 0
                while by < h {
                    var bx = 0
                    let ey = min(by + block, h)
                    while bx < w {
                        let ex = min(bx + block, w)
                        var sum: Float = 0
                        for y in by..<ey { for x in bx..<ex { sum += o[c * plane + y * w + x] } }
                        let mean = sum / Float((ey - by) * (ex - bx))
                        for y in by..<ey { for x in bx..<ex { o[c * plane + y * w + x] = mean } }
                        bx += block
                    }
                    by += block
                }
            }
        }
        return out
    }

    static func quantize(_ z: inout Latent, bits: Int, range: Float = 3) {
        let levels = Float((1 << bits) - 1)
        let step = 2 * range / max(levels, 1)
        z.values.withUnsafeMutableBufferPointer { v in
            for i in 0..<v.count {
                let clipped = min(max(v[i], -range), range)
                v[i] = levels < 1 ? 0 : ((clipped + range) / step).rounded() * step - range
            }
        }
    }

    /// Resample to another latent size, align_corners false like torch.
    static func resize(_ z: Latent, to size: FrameSize, bilinear: Bool) -> Latent {
        let w = z.width, h = z.height, nw = size.width, nh = size.height
        var out = Latent(channels: z.channels, height: nh, width: nw, values: [Float](repeating: 0, count: z.channels * nw * nh))
        let sx = Float(w) / Float(nw), sy = Float(h) / Float(nh)
        z.values.withUnsafeBufferPointer { s in
            out.values.withUnsafeMutableBufferPointer { o in
                for c in 0..<z.channels {
                    let src = c * h * w, dst = c * nh * nw
                    for y in 0..<nh {
                        let fy = (Float(y) + 0.5) * sy - 0.5
                        for x in 0..<nw {
                            let fx = (Float(x) + 0.5) * sx - 0.5
                            if !bilinear {
                                let ix = min(Int(Float(x) * sx), w - 1), iy = min(Int(Float(y) * sy), h - 1)
                                o[dst + y * nw + x] = s[src + iy * w + ix]
                                continue
                            }
                            let cx = min(max(fx, 0), Float(w - 1)), cy = min(max(fy, 0), Float(h - 1))
                            let x0 = Int(cx), y0 = Int(cy)
                            let x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1)
                            let ax = cx - Float(x0), ay = cy - Float(y0)
                            let top = s[src + y0 * w + x0] * (1 - ax) + s[src + y0 * w + x1] * ax
                            let bottom = s[src + y1 * w + x0] * (1 - ax) + s[src + y1 * w + x1] * ax
                            o[dst + y * nw + x] = top * (1 - ay) + bottom * ay
                        }
                    }
                }
            }
        }
        return out
    }

    /// sigma * (cos(phase) N1 + sin(phase) N2), each field std 1 per channel. Grain above 1 interpolates a coarser field.
    static func addNoise(_ z: inout Latent, sigma: Float, seed: UInt64, phase: Float, grain: Int, only: Int?) {
        let w = z.width, h = z.height, plane = z.plane
        let ca = cos(phase), sa = sin(phase)
        var first = SplitMix(seed: seed &* 2 &+ 1)
        var second = SplitMix(seed: seed &* 2 &+ 2)
        let gw = w / max(grain, 1) + 2, gh = h / max(grain, 1) + 2
        var field = [Float](repeating: 0, count: plane)
        var coarse = [Float](repeating: 0, count: gw * gh)
        for c in 0..<z.channels {
            // Draw both fields for every channel, so one channel's noise does not depend on the others being on.
            for i in 0..<coarse.count { coarse[i] = ca * first.gaussian() + sa * second.gaussian() }
            if grain <= 1 {
                // gw x gh holds more than one plane.
                for i in 0..<plane { field[i] = coarse[i] }
            } else {
                let g = Float(grain)
                var sum: Float = 0, sq: Float = 0
                for y in 0..<h {
                    let fy = Float(y) / g
                    let y0 = Int(fy), ay = fy - Float(y0)
                    for x in 0..<w {
                        let fx = Float(x) / g
                        let x0 = Int(fx), ax = fx - Float(x0)
                        let top = coarse[y0 * gw + x0] * (1 - ax) + coarse[y0 * gw + x0 + 1] * ax
                        let bottom = coarse[(y0 + 1) * gw + x0] * (1 - ax) + coarse[(y0 + 1) * gw + x0 + 1] * ax
                        let v = top * (1 - ay) + bottom * ay
                        field[y * w + x] = v
                        sum += v; sq += v * v
                    }
                }
                let mean = sum / Float(plane)
                let std = max((sq / Float(plane) - mean * mean).squareRoot(), 1e-6)
                for i in 0..<plane { field[i] = (field[i] - mean) / std }
            }
            if let only, only != c { continue }
            z.values.withUnsafeMutableBufferPointer { v in
                for i in 0..<plane { v[c * plane + i] += sigma * field[i] }
            }
        }
    }
}

nonisolated struct SplitMix {
    var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }

    mutating func gaussian() -> Float {
        let u1 = max(uniform(), 1e-7)
        let u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * Float.pi * u2)
    }
}
