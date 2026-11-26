import SwiftUI

/// The right-hand panel: what happens to the film on its way to the screen, top to bottom.
/// Source and preprocessing feed the upscaler; light and color work on its output, live.
struct PipelinePanel: View {
    @EnvironmentObject private var session: Session

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                source
                Divider()
                upscaler
                Divider()
                color
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
        }
        .scrollIndicators(.automatic)
        .disabled(session.facts == nil && session.modelLabel.isEmpty)
    }

    private var source: some View {
        VStack(alignment: .leading, spacing: 8) {
            header("Quelle")
            Picker("Doppelte Bilder", selection: Binding(
                get: { session.preprocess.decimate },
                set: { session.setDecimate($0) }
            )) {
                ForEach(DecimateMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .disabled(session.facts == nil || session.exporting)
            .help("Lässt das Bild weg, das ein Transcode in jedem 6er- oder 5er-Takt wiederholt hat. Der Film läuft dann gleichmäßig mit seiner echten Bildrate, der Ton bleibt synchron. Automatisch sucht den Takt im Film.")
            if let progress = session.decimateProgress {
                ProgressView(value: progress) {
                    Text("Sucht doppelte Bilder …")
                }
                .controlSize(.small)
                .font(.caption)
            } else if session.facts != nil {
                Text(sourceSummary)
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var upscaler: some View {
        VStack(alignment: .leading, spacing: 8) {
            header("Upscaler")
            Picker("Quell-Codec", selection: Binding(
                get: { session.codecChoice },
                set: { session.changeCodec($0) }
            )) {
                ForEach(CodecChoice.allCases) { choice in
                    Text(codecTitle(choice)).tag(choice)
                }
            }
            .pickerStyle(.menu)
            .disabled(!session.codecSelectable)
            .help(codecHelp)
            if !session.modelLabel.isEmpty {
                Text(session.modelLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var color: some View {
        VStack(alignment: .leading, spacing: 10) {
            header("Farbe & Licht")
            GradeSlider(title: "Helligkeit", value: $session.grade.brightness, range: ColorGrade.brightnessRange, neutral: 0, signed: true)
            GradeSlider(title: "Kontrast", value: $session.grade.contrast, range: ColorGrade.contrastRange, neutral: 1)
            GradeSlider(title: "Gamma", value: $session.grade.gamma, range: ColorGrade.gammaRange, neutral: 1, logarithmic: true)
            GradeSlider(title: "Sättigung", value: $session.grade.saturation, range: ColorGrade.saturationRange, neutral: 1)
            HStack {
                Button {
                    session.autoGrade()
                } label: {
                    if session.estimatingGrade {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Auto")
                    }
                }
                .disabled(session.facts == nil || session.estimatingGrade)
                .help("Schätzt ruhige Startwerte aus 16 Bildern des Films. Danach lässt sich alles weiter verstellen.")
                Spacer()
                Button("Zurücksetzen") {
                    session.resetGrade()
                }
                .disabled(session.grade.isNeutral)
            }
            .controlSize(.small)
            Text("Wirkt nach dem Upscale auf Vorschau und Export. Der Cache bleibt, wie er ist.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Section header in the type of the sidebar's section headers. The order is the flow, top to bottom.
    private func header(_ title: String) -> some View {
        Text(title)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    private var sourceSummary: String {
        guard let facts = session.facts else { return "–" }
        let rate = { (fps: Double) in fps.formatted(.number.precision(.fractionLength(0...3))) }
        let frames = facts.count.formatted(.number)
        var text = facts.cycle == nil
            ? "\(rate(facts.fps)) fps · \(frames) Bilder"
            : "\(rate(facts.sourceFps)) → \(rate(facts.fps)) fps · \(frames) Bilder"
        if !session.decimateNote.isEmpty { text += " · \(session.decimateNote)" }
        return text
    }

    /// "Automatisch (h264)" names what automatic picks for the open film.
    private func codecTitle(_ choice: CodecChoice) -> String {
        guard choice == .auto else { return choice.title }
        return "Automatisch (\(session.automaticCodec.title))"
    }

    private var codecHelp: String {
        guard session.selectedModel?.hasCodecs == true else {
            return "Dieses Modell hat keine Codec-Bedingung."
        }
        let film = session.filmCodec.map { "Film: \($0.title)" } ?? "kein Film"
        let active = session.activeCodec.map { "aktiv: \($0.title)" } ?? "kein Codec"
        return "Codec, auf den der Upscaler eingestellt ist. \(film) · \(active)"
    }
}

/// One grade control: title, value, and a slider. A double-click on the value sets it back.
private struct GradeSlider: View {
    var title: String
    @Binding var value: Double
    var range: ClosedRange<Double>
    var neutral: Double
    var signed = false
    /// The track runs in powers of two, so 1 sits in the middle of 0.5…2.
    var logarithmic = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(label)
                    .monospacedDigit()
                    .foregroundStyle(abs(value - neutral) < 0.001 ? .secondary : .primary)
                    .onTapGesture(count: 2) { value = neutral }
                    .help("Doppelklick setzt den Wert zurück.")
            }
            Group {
                if logarithmic {
                    Slider(value: Binding(get: { log2(value) }, set: { value = pow(2, $0) }),
                           in: log2(range.lowerBound)...log2(range.upperBound))
                } else {
                    Slider(value: $value, in: range)
                }
            }
            .controlSize(.small)
            .labelsHidden()
        }
    }

    private var label: String {
        let text = value.formatted(.number.precision(.fractionLength(2)))
        return signed && value > 0.0005 ? "+" + text : text
    }
}
