import AppKit
import CoreImage
import SwiftUI

/// The tone curve of the grade: a square with a quarter grid, the diagonal for reference, and the
/// curve through its five points as `CIToneCurve` draws it. The points drag with the mouse; x stays
/// ordered with a small gap. A double-click sets the curve back to the diagonal.
struct ToneCurveView: View {
    @Binding var grade: ColorGrade
    @State private var dragging: Int?
    @Environment(\.isEnabled) private var isEnabled

    private static let inset: CGFloat = 6
    private static let grab: CGFloat = 14

    var body: some View {
        let samples = ToneCurveSampler.shared.samples(grade.curve, extrapolate: grade.extrapolate)
        GeometryReader { proxy in
            let plot = CGRect(origin: .zero, size: proxy.size).insetBy(dx: Self.inset, dy: Self.inset)
            Canvas { context, _ in
                draw(in: &context, plot: plot, samples: samples)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .local)
                    .onChanged { drag in changed(drag, plot: plot) }
                    .onEnded { _ in dragging = nil }
            )
        }
        .aspectRatio(1, contentMode: .fit)
        .opacity(isEnabled ? 1 : 0.5)
        .accessibilityElement()
        .accessibilityLabel("Tonwertkurve")
        .accessibilityValue(grade.curve.map { "\(Int(($0.x * 100).rounded())) → \(Int(($0.y * 100).rounded()))" }.joined(separator: ", "))
        .help("Punkte ziehen, um die Tonwerte zu formen. Doppelklick setzt die Kurve zurück.")
    }

    private func draw(in context: inout GraphicsContext, plot: CGRect, samples: [Double]) {
        let frame = Path(roundedRect: plot, cornerRadius: 3)
        context.fill(frame, with: .color(Color.primary.opacity(0.04)))
        var grid = Path()
        for step in 1..<4 {
            let fraction = CGFloat(step) / 4
            grid.move(to: CGPoint(x: plot.minX + fraction * plot.width, y: plot.minY))
            grid.addLine(to: CGPoint(x: plot.minX + fraction * plot.width, y: plot.maxY))
            grid.move(to: CGPoint(x: plot.minX, y: plot.minY + fraction * plot.height))
            grid.addLine(to: CGPoint(x: plot.maxX, y: plot.minY + fraction * plot.height))
        }
        context.stroke(grid, with: .color(Color.primary.opacity(0.08)), lineWidth: 0.5)
        context.stroke(frame, with: .color(Color(nsColor: .separatorColor)), lineWidth: 0.5)

        var diagonal = Path()
        diagonal.move(to: position(CurvePoint(x: 0, y: 0), plot))
        diagonal.addLine(to: position(CurvePoint(x: 1, y: 1), plot))
        context.stroke(diagonal, with: .color(Color.secondary.opacity(0.5)), style: StrokeStyle(lineWidth: 0.75, dash: [3, 3]))

        if samples.count > 1 {
            var curve = Path()
            for (index, value) in samples.enumerated() {
                let point = position(CurvePoint(x: Double(index) / Double(samples.count - 1), y: value), plot)
                if index == 0 { curve.move(to: point) } else { curve.addLine(to: point) }
            }
            var clipped = context
            clipped.clip(to: Path(plot))
            clipped.stroke(curve, with: .color(.accentColor), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
        }

        for (index, point) in grade.curve.enumerated() {
            let center = position(point, plot)
            let dot = Path(ellipseIn: CGRect(x: center.x - 4, y: center.y - 4, width: 8, height: 8))
            context.fill(dot, with: .color(dragging == index ? .accentColor : Color(nsColor: .controlBackgroundColor)))
            context.stroke(dot, with: .color(.accentColor), lineWidth: 1.25)
        }
    }

    private func position(_ point: CurvePoint, _ plot: CGRect) -> CGPoint {
        CGPoint(x: plot.minX + CGFloat(point.x) * plot.width, y: plot.maxY - CGFloat(point.y) * plot.height)
    }

    private func value(_ location: CGPoint, _ plot: CGRect) -> CurvePoint {
        CurvePoint(
            x: Double((location.x - plot.minX) / max(plot.width, 1)),
            y: Double((plot.maxY - location.y) / max(plot.height, 1))
        )
    }

    private func changed(_ drag: DragGesture.Value, plot: CGRect) {
        if dragging == nil {
            // The first event of a press. A double-click resets instead of grabbing a point.
            if drag.translation == .zero, NSApp.currentEvent?.clickCount == 2 {
                grade.curve = ColorGrade.identityCurve
                return
            }
            let start = drag.startLocation
            let nearest = grade.curve.indices.min { a, b in
                distance(position(grade.curve[a], plot), start) < distance(position(grade.curve[b], plot), start)
            }
            guard let nearest, distance(position(grade.curve[nearest], plot), start) <= Self.grab else { return }
            dragging = nearest
        }
        guard let index = dragging else { return }
        let next = grade.moving(point: index, to: value(drag.location, plot))
        if next != grade { grade = next }
    }

    private func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(a.x - b.x, a.y - b.y)
    }
}

/// The curve exactly as `CIToneCurve` maps values: a gray ramp in sRGB through the filter in Core
/// Image's linear working space, the same way the screen and the export run it.
@MainActor
final class ToneCurveSampler {
    static let shared = ToneCurveSampler()
    private let context = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: false])
    private let count = 129
    private let ramp: CIImage
    private let space = CGColorSpace(name: CGColorSpace.sRGB)!
    private let extended = CGColorSpace(name: CGColorSpace.extendedSRGB)!
    private var last: (key: [CurvePoint], extrapolate: Bool, samples: [Double])?

    private init() {
        var pixels = [Float](repeating: 1, count: count * 4)
        for index in 0..<count {
            let value = Float(index) / Float(count - 1)
            pixels[index * 4] = value
            pixels[index * 4 + 1] = value
            pixels[index * 4 + 2] = value
        }
        let data = pixels.withUnsafeBufferPointer { Data(buffer: $0) }
        ramp = CIImage(bitmapData: data, bytesPerRow: count * 16, size: CGSize(width: count, height: 1), format: .RGBAf, colorSpace: space)
    }

    func samples(_ curve: [CurvePoint], extrapolate: Bool) -> [Double] {
        if let last, last.key == curve, last.extrapolate == extrapolate { return last.samples }
        let identity = (0..<count).map { Double($0) / Double(count - 1) }
        guard curve.count == 5, let tone = CIFilter(name: "CIToneCurve") else { return identity }
        tone.setValue(ramp, forKey: kCIInputImageKey)
        for (index, point) in curve.enumerated() {
            tone.setValue(CIVector(x: point.x, y: point.y), forKey: "inputPoint\(index)")
        }
        tone.setValue(extrapolate, forKey: "inputExtrapolate")
        guard let output = tone.outputImage else { return identity }
        var pixels = [Float](repeating: 0, count: count * 4)
        context.render(output, toBitmap: &pixels, rowBytes: count * 16, bounds: CGRect(x: 0, y: 0, width: count, height: 1), format: .RGBAf, colorSpace: extended)
        let samples = (0..<count).map { Double(pixels[$0 * 4]) }
        last = (curve, extrapolate, samples)
        return samples
    }
}
