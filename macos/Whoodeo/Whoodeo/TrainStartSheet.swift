import SwiftUI

/// What the start sheet opens with: new run, fine-tune from a checkpoint, or resume.
struct TrainStartRequest: Identifiable {
    let id = UUID()
    var mode: TrainLauncher.Mode
    var base: TrainConfig
    var title: String
    var subtitle: String
}

/// Form built from `TrainParameterSpec`. Architecture is read-only for fine-tune and resume.
struct TrainStartSheet: View {
    let request: TrainStartRequest
    /// Run names that already exist; a new run may not reuse one.
    let existingNames: Set<String>
    let trainingRunning: Bool
    let onLaunch: (_ runName: String, _ arguments: [String]) throws -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var values: [String: String] = [:]
    @State private var lastWeights: [String: String] = [:]
    @State private var name = ""
    @State private var placeholder = TrainConfig.automaticName()
    @State private var launchError: String?

    private var isResume: Bool {
        if case .resume = request.mode { return true }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                Text(request.title)
                    .font(.title2.weight(.semibold))
                Text(request.subtitle)
                    .foregroundStyle(.secondary)
            }
            .padding([.horizontal, .top], 20)
            .padding(.bottom, 8)

            Form {
                Section {
                    if isResume {
                        LabeledContent("Lauf", value: request.base.name)
                    } else {
                        TextField("Lauf-Name", text: $name, prompt: Text(placeholder))
                    }
                } footer: {
                    if !isResume {
                        Text("Leer lassen für einen automatischen Namen. Das Log steht unter Models/<Name>/train.log.")
                    }
                }
                ForEach(TrainParameterSection.allCases, id: \.self) { section in
                    Section {
                        ForEach(TrainParameterSpec.specs(in: section)) { spec in
                            row(spec)
                        }
                    } header: {
                        sectionHeader(section)
                    } footer: {
                        if section == .loss { lossFooter }
                    }
                }
            }
            .formStyle(.grouped)

            HStack(spacing: 12) {
                if let message = problem ?? launchError {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .font(.callout)
                        .lineLimit(3)
                } else if trainingRunning {
                    Label("Es läuft schon ein Training. Zwei Läufe teilen sich die GPU.", systemImage: "info.circle")
                        .foregroundStyle(.secondary)
                        .font(.callout)
                }
                Spacer()
                Button("Abbrechen", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Training starten") { start() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(problem != nil)
            }
            .padding(20)
        }
        .frame(width: 600, height: 720)
        .onAppear {
            values = request.base.parameterValues
            lastWeights = TrainConfig(name: "").parameterValues
        }
    }

    // MARK: Rows

    private func locked(_ spec: TrainParameterSpec) -> Bool {
        spec.scope == .newOnly && request.mode.locksArchitecture
    }

    @ViewBuilder
    private func sectionHeader(_ section: TrainParameterSection) -> some View {
        if section == .architecture && request.mode.locksArchitecture {
            Label(
                "\(section.title) – \(isResume ? "aus dem Lauf" : "vom Eltern-Checkpoint")",
                systemImage: "lock.fill"
            )
        } else {
            Text(section.title)
        }
    }

    @ViewBuilder
    private func row(_ spec: TrainParameterSpec) -> some View {
        if locked(spec) {
            LabeledContent {
                Text(values[spec.key] ?? "")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            } label: {
                label(spec)
            }
        } else if spec.section == .loss {
            lossRow(spec)
        } else {
            switch spec.kind {
            case .bool:
                Toggle(isOn: boolBinding(spec)) { label(spec) }
            case .choice(let options):
                Picker(selection: text(spec)) {
                    ForEach(options, id: \.self) { Text($0).tag($0) }
                } label: {
                    label(spec)
                }
            case .int, .float, .list:
                LabeledContent {
                    field(spec)
                } label: {
                    label(spec)
                }
            }
        }
    }

    private func label(_ spec: TrainParameterSpec) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(spec.label)
            Text(spec.help)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .help("--\(spec.flag)")
    }

    private func field(_ spec: TrainParameterSpec, width: CGFloat = 140) -> some View {
        TextField("", text: text(spec), prompt: Text(spec.defaultText))
            .labelsHidden()
            .multilineTextAlignment(.trailing)
            .monospacedDigit()
            .frame(width: width)
    }

    /// Switch plus weight: off writes 0 (the term is not computed), on brings back the last weight.
    private func lossRow(_ spec: TrainParameterSpec) -> some View {
        let on = Binding<Bool>(
            get: { weight(spec.key) > 0 },
            set: { enabled in
                if enabled {
                    let previous = lastWeights[spec.key] ?? spec.defaultText
                    values[spec.key] = (Float(previous) ?? 0) > 0 ? previous : "1"
                } else {
                    if weight(spec.key) > 0 { lastWeights[spec.key] = values[spec.key] }
                    values[spec.key] = "0"
                }
            }
        )
        return LabeledContent {
            HStack(spacing: 10) {
                field(spec, width: 70)
                    .disabled(!on.wrappedValue)
                Toggle("", isOn: on)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
            }
        } label: {
            label(spec)
                .opacity(on.wrappedValue ? 1 : 0.55)
        }
    }

    @ViewBuilder
    private var lossFooter: some View {
        let pixel = ["w_oklab", "w_ms_ssim", "w_vgg"].contains { weight($0) > 0 }
        VStack(alignment: .leading, spacing: 2) {
            Text("Gewicht 0 = Term aus, er wird gar nicht berechnet.")
            if !pixel && weight("w_latent") > 0 {
                Text("Alle Bild-Losses sind aus: Der VAE-Decode im Training wird übersprungen.")
            } else if weight("w_vgg") == 0 && pixel {
                Text("VGG ist aus: VGG19 wird nicht geladen.")
            }
        }
    }

    // MARK: Values

    private func weight(_ key: String) -> Float {
        Float((values[key] ?? "").replacingOccurrences(of: ",", with: ".")) ?? 0
    }

    private func text(_ spec: TrainParameterSpec) -> Binding<String> {
        Binding(
            get: { values[spec.key] ?? "" },
            set: { values[spec.key] = $0 }
        )
    }

    private func boolBinding(_ spec: TrainParameterSpec) -> Binding<Bool> {
        Binding(
            get: { values[spec.key] == "true" },
            set: { values[spec.key] = $0 ? "true" : "false" }
        )
    }

    private var runName: String {
        if isResume { return request.base.name }
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? placeholder : trimmed
    }

    /// The config the CLI will get, validated by the same code as `whoodeo-train`.
    private var built: Result<TrainConfig, Error> {
        Result {
            var config = request.base
            config.name = runName
            try config.apply(values, scope: request.mode.locksArchitecture ? .always : nil)
            return try config.validated()
        }
    }

    private var problem: String? {
        if !isResume {
            let candidate = runName
            if candidate.contains("/") || candidate.hasPrefix(".") || candidate.contains(" ") {
                return "Der Lauf-Name darf keine Leerzeichen, „/“ oder einen führenden Punkt enthalten."
            }
            if existingNames.contains(candidate) {
                return "Ein Lauf „\(candidate)“ existiert schon."
            }
        }
        switch built {
        case .success: return nil
        case .failure(let error):
            // Same message as the CLI, with the form label instead of the flag.
            var message = error.localizedDescription
            for spec in TrainParameterSpec.all {
                message = message.replacingOccurrences(of: "--\(spec.flag):", with: "\(spec.label):")
            }
            return message
        }
    }

    private func start() {
        guard case .success(let config) = built else { return }
        do {
            try onLaunch(config.name, TrainLauncher.arguments(mode: request.mode, config: config))
            dismiss()
        } catch {
            launchError = error.localizedDescription
        }
    }
}
