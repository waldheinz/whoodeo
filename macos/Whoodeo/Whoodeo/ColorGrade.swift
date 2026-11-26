import CoreImage
import CoreVideo
import Foundation
import os

/// One point of the tone curve, both axes 0…1 in perceptual (gamma-encoded) values.
nonisolated struct CurvePoint: Codable, Hashable, Sendable {
    var x: Double
    var y: Double
}

/// Light and color after the upscaler: Apple's auto-enhance chain as editable values. The screen
/// applies it live and an export renders it in, so the frame cache never changes with it. Stored per
/// film; neutral is the plain upscale. Auto copies the parameters of `autoAdjustmentFilters` exactly.
nonisolated struct ColorGrade: Codable, Hashable, Sendable {
    /// `CIToneCurve`: five points, ordered in x.
    var curve: [CurvePoint] = ColorGrade.identityCurve
    /// `CIToneCurve` `inputExtrapolate`: the curve continues past its end points, as Auto delivers it.
    var extrapolate = true
    /// `CIHighlightShadowAdjust` `inputShadowAmount`, −1…1. Above 0 lifts the shadows.
    var shadows: Double = 0
    /// `CIHighlightShadowAdjust` `inputHighlightAmount`, 0.3…1. 1 leaves the highlights alone.
    var highlights: Double = 1
    /// `CIHighlightShadowAdjust` `inputRadius`, 0…10. Only Auto sets it; Apple delivers 0.
    var radius: Double = 0
    /// `CIVibrance` `inputAmount`, −1…1.
    var vibrance: Double = 0
    /// `CIFaceBalance` `inputOrigI`: the measured skin tone on the orange–blue axis. Below the default warms.
    var faceI: Double = ColorGrade.neutralI
    /// `CIFaceBalance` `inputOrigQ`: the measured skin tone on the green–magenta axis. Below the default goes magenta.
    var faceQ: Double = ColorGrade.neutralQ
    /// `CIFaceBalance` `inputStrength`, 0…2. Auto delivers 1.
    var faceStrength: Double = 1
    /// `CIFaceBalance` `inputWarmth`, 0…1. 0.5 is neutral; Auto delivers 0.5.
    var faceWarmth: Double = 0.5

    static let neutral = ColorGrade()
    static let identityCurve = [0, 0.25, 0.5, 0.75, 1].map { CurvePoint(x: $0, y: $0) }
    /// `CIFaceBalance` leaves the picture as it is when the measured tone equals these defaults.
    static let neutralI = 0.103905
    static let neutralQ = 0.0176465
    static let shadowRange = -1.0...1.0
    /// The panel shows the highlights as a correction: `highlights − 1`.
    static let highlightRange = -0.7...0.0
    static let vibranceRange = -1.0...1.0
    static let warmthRange = -0.25...0.25
    static let tintRange = -0.25...0.25
    /// Smallest x distance between neighboring curve points.
    static let curveGap = 0.02

    init() {}

    /// Old entries in the film list carry brightness, contrast, gamma and saturation; they load as neutral.
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        func number(_ key: CodingKeys, _ fallback: Double) -> Double {
            ((try? values.decodeIfPresent(Double.self, forKey: key)) ?? nil).flatMap { $0.isFinite ? $0 : nil } ?? fallback
        }
        if let points = (try? values.decodeIfPresent([CurvePoint].self, forKey: .curve)) ?? nil,
           points.count == 5, points.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) {
            curve = points
        }
        extrapolate = ((try? values.decodeIfPresent(Bool.self, forKey: .extrapolate)) ?? nil) ?? true
        shadows = number(.shadows, 0)
        highlights = number(.highlights, 1)
        radius = number(.radius, 0)
        vibrance = number(.vibrance, 0)
        faceI = number(.faceI, Self.neutralI)
        faceQ = number(.faceQ, Self.neutralQ)
        faceStrength = number(.faceStrength, 1)
        faceWarmth = number(.faceWarmth, 0.5)
    }

    private enum CodingKeys: String, CodingKey {
        case curve, extrapolate, shadows, highlights, radius, vibrance, faceI, faceQ, faceStrength, faceWarmth
    }

    // MARK: Panel values

    /// Positive is warmer, negative cooler. Moves `faceI` against Apple's neutral skin tone.
    var warmth: Double {
        get { Self.neutralI - faceI }
        set { faceI = Self.neutralI - newValue }
    }

    /// Positive is magenta, negative green. Moves `faceQ` against Apple's neutral skin tone.
    var tint: Double {
        get { Self.neutralQ - faceQ }
        set { faceQ = Self.neutralQ - newValue }
    }

    /// 0 leaves the highlights alone, −0.7 darkens them the most.
    var highlightCorrection: Double {
        get { highlights - 1 }
        set { highlights = min(max(newValue + 1, 0.3), 1) }
    }

    // MARK: Neutral parts

    var curveIsIdentity: Bool {
        curve.count == 5 && zip(curve, Self.identityCurve).allSatisfy { abs($0.x - $1.x) < 1e-5 && abs($0.y - $1.y) < 1e-5 }
    }

    var faceIsNeutral: Bool {
        (abs(faceI - Self.neutralI) < 1e-6 && abs(faceQ - Self.neutralQ) < 1e-6 && abs(faceWarmth - 0.5) < 1e-6)
            || faceStrength < 1e-6
    }

    var vibranceIsNeutral: Bool { abs(vibrance) < 1e-6 }
    var lightIsNeutral: Bool { abs(shadows) < 1e-6 && abs(highlights - 1) < 1e-6 }

    var isNeutral: Bool {
        curveIsIdentity && faceIsNeutral && vibranceIsNeutral && lightIsNeutral
    }

    // MARK: Filters

    /// Apple's order: face balance, vibrance, tone curve, highlights and shadows. Neutral filters stay out,
    /// so a neutral grade has none. The same chain for the screen and the export; both run it color
    /// managed in Core Image's linear working space, as `autoAdjustmentFilters` expects.
    func filters() -> [CIFilter] {
        var chain: [CIFilter] = []
        if !faceIsNeutral, let face = CIFilter(name: "CIFaceBalance") {
            face.setValue(faceI, forKey: "inputOrigI")
            face.setValue(faceQ, forKey: "inputOrigQ")
            face.setValue(faceStrength, forKey: "inputStrength")
            face.setValue(faceWarmth, forKey: "inputWarmth")
            chain.append(face)
        }
        if !vibranceIsNeutral, let vibrant = CIFilter(name: "CIVibrance") {
            vibrant.setValue(vibrance, forKey: "inputAmount")
            chain.append(vibrant)
        }
        if !curveIsIdentity, let tone = CIFilter(name: "CIToneCurve") {
            for (index, point) in curve.enumerated() {
                tone.setValue(CIVector(x: point.x, y: point.y), forKey: "inputPoint\(index)")
            }
            tone.setValue(extrapolate, forKey: "inputExtrapolate")
            chain.append(tone)
        }
        if !lightIsNeutral, let light = Self.highlightShadowFilter() {
            light.setValue(radius, forKey: "inputRadius")
            light.setValue(shadows, forKey: "inputShadowAmount")
            light.setValue(highlights, forKey: "inputHighlightAmount")
            chain.append(light)
        }
        return chain
    }

    /// `CIHighlightShadowAdjust` as `autoAdjustmentFilters` builds it. Apple's instance runs an older
    /// version of the filter (the `_shadows_noblur` kernel) than `CIFilter(name:)`, which runs version 2
    /// and lifts the shadows visibly differently with the same values. CIFilter keeps that version
    /// through NSSecureCoding, so this is Apple's own Auto instance with neutral values, archived on
    /// macOS 27.2; a copy per use. Falls back to the plain filter if the archive ever stops loading.
    nonisolated(unsafe) private static let highlightShadowTemplate: CIFilter? = {
        let archive = "YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGmCwwXGBkaVSRudWxs1Q0ODxAREhMUExZfEBdDSV9pbnB1dEhpZ2hsaWdodEFtb3VudF5DSV9pbnB1dFJhZGl1c1lDSVZlcnNpb25fEBRDSV9pbnB1dFNoYWRvd0Ftb3VudFYkY2xhc3OAA4ACgASAAoAFIwAAAAAAAAAAIz/wAAAAAAAAEADSGxwdHlokY2xhc3NuYW1lWCRjbGFzc2VzXxAXQ0lIaWdobGlnaHRTaGFkb3dBZGp1c3SjHyAhXxAXQ0lIaWdobGlnaHRTaGFkb3dBZGp1c3RYQ0lGaWx0ZXJYTlNPYmplY3QACAARABoAJAApADIANwBJAEwAUQBTAFoAYABrAIUAlACeALUAvAC+AMAAwgDEAMYAzwDYANoA3wDqAPMBDQERASsBNAAAAAAAAAIBAAAAAAAAACIAAAAAAAAAAAAAAAAAAAE9"
        guard let data = Data(base64Encoded: archive),
              let filter = try? NSKeyedUnarchiver.unarchivedObject(ofClass: CIFilter.self, from: data),
              filter.name == "CIHighlightShadowAdjust" else { return nil }
        return filter
    }()

    private static func highlightShadowFilter() -> CIFilter? {
        (highlightShadowTemplate?.copy() as? CIFilter) ?? CIFilter(name: "CIHighlightShadowAdjust")
    }

    func apply(_ image: CIImage) -> CIImage {
        filters().reduce(image) { picture, filter in
            filter.setValue(picture, forKey: kCIInputImageKey)
            return filter.outputImage ?? picture
        }
    }

    /// The curve with a point moved: x stays between its neighbors with `curveGap` to each, both axes in 0…1.
    func moving(point index: Int, to target: CurvePoint) -> ColorGrade {
        guard curve.indices.contains(index) else { return self }
        var next = self
        let low = index == 0 ? 0 : curve[index - 1].x + Self.curveGap
        let high = index == curve.count - 1 ? 1 : curve[index + 1].x - Self.curveGap
        let x = min(max(target.x, low), max(low, high))
        next.curve[index] = CurvePoint(x: x, y: min(max(target.y, 0), 1))
        return next
    }

    // MARK: Auto

    /// Apple's auto-enhance (as in Photos) for one picture, with face detection and without red-eye.
    /// The values are copied exactly; a filter Apple leaves out stays neutral.
    static func auto(for image: CIImage) -> ColorGrade {
        var grade = ColorGrade.neutral
        func number(_ filter: CIFilter, _ key: String, _ fallback: Double) -> Double {
            (filter.value(forKey: key) as? NSNumber)?.doubleValue ?? fallback
        }
        for filter in image.autoAdjustmentFilters(options: [.redEye: false]) {
            switch filter.name {
            case "CIFaceBalance":
                grade.faceI = number(filter, "inputOrigI", neutralI)
                grade.faceQ = number(filter, "inputOrigQ", neutralQ)
                grade.faceStrength = number(filter, "inputStrength", 1)
                grade.faceWarmth = number(filter, "inputWarmth", 0.5)
            case "CIVibrance":
                grade.vibrance = number(filter, "inputAmount", 0)
            case "CIToneCurve":
                grade.curve = (0..<5).map { index in
                    let vector = filter.value(forKey: "inputPoint\(index)") as? CIVector
                    return vector.map { CurvePoint(x: Double($0.x), y: Double($0.y)) } ?? identityCurve[index]
                }
                grade.extrapolate = number(filter, "inputExtrapolate", 1) != 0
            case "CIHighlightShadowAdjust":
                grade.radius = number(filter, "inputRadius", 0)
                grade.shadows = number(filter, "inputShadowAmount", 0)
                grade.highlights = number(filter, "inputHighlightAmount", 1)
            default:
                Log.model.info("Auto-Farbe ignoriert \(filter.name, privacy: .public)")
            }
        }
        return grade
    }

    /// A frame as Core Image sees it on screen: tagged with its BT.709 profile, so the filters work
    /// on linear light like the layer filters of the screen do.
    static func image(_ buffer: CVPixelBuffer) -> CIImage {
        CIImage(cvPixelBuffer: buffer, options: [.colorSpace: colorSpace(of: buffer)])
    }

    /// The buffer's own profile; BT.709 when it carries none.
    static func colorSpace(of buffer: CVPixelBuffer) -> CGColorSpace {
        if let attached = CVBufferCopyAttachment(buffer, kCVImageBufferCGColorSpaceKey, nil) {
            if CFGetTypeID(attached) == CGColorSpace.typeID { return attached as! CGColorSpace }
        }
        var tags = (CVBufferCopyAttachments(buffer, .shouldPropagate) as? [CFString: Any]) ?? [:]
        if tags[kCVImageBufferColorPrimariesKey] == nil { tags[kCVImageBufferColorPrimariesKey] = kCVImageBufferColorPrimaries_ITU_R_709_2 }
        if tags[kCVImageBufferTransferFunctionKey] == nil { tags[kCVImageBufferTransferFunctionKey] = kCVImageBufferTransferFunction_ITU_R_709_2 }
        if let created = CVImageBufferCreateColorSpaceFromAttachments(tags as CFDictionary)?.takeRetainedValue() {
            return created
        }
        return CGColorSpace(name: CGColorSpace.itur_709)!
    }
}

/// Renders a grade into 16-bit RGBA frames for an export. Like the screen, Core Image works color
/// managed: the frame's BT.709 profile into the linear working space and back into the same profile,
/// so the export matches the preview and the values Auto measured.
nonisolated final class GradeRenderer {
    let grade: ColorGrade
    private let context = CIContext(options: [.cacheIntermediates: false])

    init(_ grade: ColorGrade) {
        self.grade = grade
    }

    /// A new buffer with the grade applied. The color tags of the input carry over.
    func render(_ buffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let space = ColorGrade.colorSpace(of: buffer)
        let image = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: space])
        var output: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_64RGBALE, attrs as CFDictionary, &output)
        guard status == kCVReturnSuccess, let output else {
            throw CLIError("Pixelpuffer \(width)×\(height) fehlt (\(status))")
        }
        context.render(grade.apply(image), to: output, bounds: image.extent, colorSpace: space)
        CVBufferPropagateAttachments(buffer, output)
        return output
    }
}
