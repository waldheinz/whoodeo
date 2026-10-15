import SwiftUI

/// Scaling rate against the film's own frame rate. The tick on the dial is realtime.
struct ScaleMeter: View {
    var rate: Double
    var realtime: Double
    /// How many times realtime the right end of the dial stands for. Grows so the needle stays on the scale.
    @State private var span: Double = 2

    var body: some View {
        VStack(spacing: 2) {
            ZStack(alignment: .bottom) {
                dial
                    .frame(height: 128)
                VStack(spacing: -1) {
                    Text(number)
                        .font(.system(size: 34, weight: .medium, design: .rounded))
                        .monospacedDigit()
                        .contentTransition(.numericText(value: rate))
                    Text("fps")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.bottom, 22)
            }
            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        .padding(.horizontal, 8)
        .padding(.top, 2)
        .padding(.bottom, 12)
        .animation(.smooth(duration: 0.45), value: fraction)
        .animation(.smooth(duration: 0.45), value: span)
        .onChange(of: rate, initial: true) { _, _ in refit() }
        .onChange(of: realtime) { _, _ in refit() }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessLabel)
        .help("Neu skalierte Bilder pro Sekunde. Die Marke ist die Echtzeit des Films.")
    }

    private var dial: some View {
        GeometryReader { geo in
            let size = geo.size
            ZStack {
                DialArc(fraction: 1)
                    .stroke(Color.primary.opacity(0.12), style: DialArc.stroke)
                DialArc(fraction: fraction)
                    .stroke(Color.accentColor, style: DialArc.stroke)
                if let mark {
                    DialTick(mark: mark)
                        .stroke(Color.primary, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    Text("1×")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .position(DialArc.point(mark, in: size, extra: 15))
                }
                Text("0")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .position(x: 12, y: size.height - 14)
                Text(endLabel)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
                    .position(x: size.width - 22, y: size.height - 14)
            }
        }
    }

    private var ceiling: Double { max(realtime, 1) * span }

    private var fraction: Double {
        guard rate > 0 else { return 0 }
        return min(rate / ceiling, 1)
    }

    private var mark: Double? {
        guard realtime > 0 else { return nil }
        return min(max(realtime / ceiling, 0), 1)
    }

    private var number: String {
        guard rate >= 0.05 else { return "–" }
        return rate.formatted(.number.precision(.fractionLength(1)))
    }

    private var endLabel: String {
        if span >= 10 { return String(Int(span)) + "×" }
        if abs(span - span.rounded()) < 0.01 { return String(Int(span)) + "×" }
        return span.formatted(.number.precision(.fractionLength(0))) + "×"
    }

    private var caption: String {
        guard realtime > 0 else { return "noch kein Film" }
        guard rate >= 0.05 else { return "Echtzeit \(Self.plain(realtime)) fps" }
        let ratio = rate / realtime
        if abs(ratio - 1) < 0.05 { return "Echtzeit" }
        return "\(Self.plain(ratio))× Echtzeit"
    }

    private var accessLabel: String {
        if rate < 0.05 { return "Skalierung steht. \(caption)" }
        return "Skalierung \(number) Bilder pro Sekunde, \(caption)"
    }

    private func refit() {
        let base = max(realtime, 1)
        var next = span
        if rate > base * next * 0.9 { next = min(next * 2, 16) }
        while next > 2, rate < base * next * 0.28 { next /= 2 }
        if next != span { span = next }
    }

    private static func plain(_ value: Double) -> String {
        if abs(value - value.rounded()) < 0.05 { return String(Int(value.rounded())) }
        return value.formatted(.number.precision(.fractionLength(1)))
    }
}

/// Open dial. Zero is the lower left, full scale the lower right, and the top is halfway.
private struct DialArc: Shape {
    var fraction: Double

    var animatableData: Double {
        get { fraction }
        set { fraction = newValue }
    }

    static let stroke = StrokeStyle(lineWidth: 9, lineCap: .round, lineJoin: .round)

    func path(in rect: CGRect) -> Path {
        let clamped = min(max(fraction, 0), 1)
        guard clamped > 0.004 else { return Path() }
        var path = Path()
        let steps = max(Int((clamped * 48).rounded(.up)), 2)
        for step in 0...steps {
            let t = clamped * Double(step) / Double(steps)
            let point = Self.point(t, in: rect.size)
            if step == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }
        return path
    }

    /// Semicircle standing on its diameter. `t` 0 is the left end, 1 the right end.
    static func point(_ t: Double, in size: CGSize, extra: CGFloat = 0) -> CGPoint {
        let clamped = min(max(t, 0), 1)
        let center = CGPoint(x: size.width / 2, y: size.height - 16)
        let radius = min(size.width / 2 - 36, size.height - 40) + extra
        let degrees = 180 - 180 * clamped
        let angle = degrees * Double.pi / 180
        return CGPoint(x: center.x + radius * cos(angle), y: center.y - radius * sin(angle))
    }
}

private struct DialTick: Shape {
    var mark: Double

    var animatableData: Double {
        get { mark }
        set { mark = newValue }
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: DialArc.point(mark, in: rect.size, extra: -8))
        path.addLine(to: DialArc.point(mark, in: rect.size, extra: 8))
        return path
    }
}
