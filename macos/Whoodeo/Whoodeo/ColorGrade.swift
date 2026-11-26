import CoreImage
import CoreVideo
import Foundation

/// Light and color after the upscaler. The screen applies it live and an export renders it in,
/// so the frame cache never changes with it. Stored per film; neutral is the plain upscale.
nonisolated struct ColorGrade: Codable, Hashable, Sendable {
    /// Added to every channel, −0.3…0.3.
    var brightness: Double = 0
    /// Spread around middle gray, 0.5…1.5.
    var contrast: Double = 1
    /// Above 1 lifts the midtones, below 1 lowers them. Black and white stay.
    var gamma: Double = 1
    /// 0 is gray, 1 the film as it is.
    var saturation: Double = 1

    static let neutral = ColorGrade()
    static let brightnessRange = -0.3...0.3
    static let contrastRange = 0.5...1.5
    static let gammaRange = 0.5...2.0
    static let saturationRange = 0.0...2.0

    init(brightness: Double = 0, contrast: Double = 1, gamma: Double = 1, saturation: Double = 1) {
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.saturation = saturation
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        brightness = (try? values.decodeIfPresent(Double.self, forKey: .brightness)) ?? 0
        contrast = (try? values.decodeIfPresent(Double.self, forKey: .contrast)) ?? 1
        gamma = (try? values.decodeIfPresent(Double.self, forKey: .gamma)) ?? 1
        saturation = (try? values.decodeIfPresent(Double.self, forKey: .saturation)) ?? 1
    }

    var isNeutral: Bool {
        abs(brightness) < 0.001 && abs(contrast - 1) < 0.001 && abs(gamma - 1) < 0.001 && abs(saturation - 1) < 0.001
    }

    /// Brightness, contrast and saturation, then gamma. The same chain for the screen and the export.
    func filters() -> [CIFilter] {
        guard !isNeutral else { return [] }
        let controls = CIFilter(name: "CIColorControls")!
        controls.setValue(brightness, forKey: kCIInputBrightnessKey)
        controls.setValue(contrast, forKey: kCIInputContrastKey)
        controls.setValue(saturation, forKey: kCIInputSaturationKey)
        let power = CIFilter(name: "CIGammaAdjust")!
        power.setValue(1 / max(gamma, 0.01), forKey: "inputPower")
        return [controls, power]
    }

    func apply(_ image: CIImage) -> CIImage {
        filters().reduce(image) { picture, filter in
            filter.setValue(picture, forKey: kCIInputImageKey)
            return filter.outputImage ?? picture
        }
    }

    // MARK: Auto

    /// Gentle starting values from frames spread over the film: stretch the luma between the
    /// 1st and 99th percentile toward the full range, center it, bring the median toward middle gray,
    /// and nudge a pale or loud film toward ordinary saturation. Every value stays near neutral.
    static func estimate(url: URL, map: FrameMap?, samples wanted: Int = 16) throws -> ColorGrade {
        let film = try Film.open(url, map: map)
        let count = film.facts.count
        let samples = max(1, min(wanted, count))
        var histogram = [Int](repeating: 0, count: 256)
        var saturation = 0.0
        var pixels = 0
        for sample in 0..<samples {
            let index = (2 * sample + 1) * count / (2 * samples)
            let frame = try film.decode(at: index)
            let plane = frame.width * frame.height
            frame.rgb.withUnsafeBufferPointer { rgb in
                var i = 0
                while i < plane {
                    let r = rgb[i], g = rgb[plane + i], b = rgb[2 * plane + i]
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    histogram[min(255, max(0, Int(luma * 255)))] += 1
                    saturation += Double(max(r, g, b) - min(r, g, b))
                    pixels += 1
                    i += 3
                }
            }
        }
        guard pixels > 0 else { return .neutral }
        func percentile(_ p: Double) -> Double {
            let target = Int(Double(pixels) * p)
            var seen = 0
            for (level, amount) in histogram.enumerated() {
                seen += amount
                if seen > target { return Double(level) / 255 }
            }
            return 1
        }
        return propose(
            low: percentile(0.01),
            high: percentile(0.99),
            median: percentile(0.5),
            saturation: saturation / Double(pixels)
        )
    }

    static func propose(low: Double, high: Double, median: Double, saturation: Double) -> ColorGrade {
        let span = max(high - low, 0.2)
        let contrast = clamp(0.92 / span, 0.85...1.3)
        let center = (low + high) / 2
        let brightness = clamp(-(center - 0.5) * contrast, -0.1...0.1)
        let middle = clamp((median - 0.5) * contrast + 0.5 + brightness, 0.05...0.95)
        let gamma = clamp(log(middle) / log(0.46), 0.85...1.2)
        let color = saturation > 0.01 ? clamp(0.2 / saturation, 0.9...1.2) : 1
        func round2(_ value: Double) -> Double { (value * 100).rounded() / 100 }
        return ColorGrade(
            brightness: round2(brightness),
            contrast: round2(contrast),
            gamma: round2(gamma),
            saturation: round2(color)
        )
    }

    private static func clamp(_ value: Double, _ range: ClosedRange<Double>) -> Double {
        min(max(value, range.lowerBound), range.upperBound)
    }
}

/// Renders a grade into 16-bit RGBA frames for an export. The values are graded as stored,
/// gamma-encoded BT.709, without color management in between.
nonisolated final class GradeRenderer {
    let grade: ColorGrade
    private let context = CIContext(options: [
        .workingColorSpace: NSNull(),
        .outputColorSpace: NSNull(),
        .cacheIntermediates: false,
    ])

    init(_ grade: ColorGrade) {
        self.grade = grade
    }

    /// A new buffer with the grade applied. The color tags of the input carry over.
    func render(_ buffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let image = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: NSNull()])
        var output: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_64RGBALE, attrs as CFDictionary, &output)
        guard status == kCVReturnSuccess, let output else {
            throw CLIError("Pixelpuffer \(width)×\(height) fehlt (\(status))")
        }
        context.render(grade.apply(image), to: output, bounds: image.extent, colorSpace: nil)
        CVBufferPropagateAttachments(buffer, output)
        return output
    }
}
