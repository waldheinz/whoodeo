import AppKit
import SwiftUI

/// Whoodeo › Einstellungen… (⌘,). One grouped form for the frame cache, as the macOS settings of Apple's own apps.
struct SettingsView: View {
    @EnvironmentObject private var session: Session
    @ObservedObject private var usage = CacheUsage.shared
    @AppStorage(CacheSettings.codecKey) private var codec = CacheCodec.standard
    @AppStorage(CacheSettings.directoryKey) private var directory = ""
    @AppStorage(CacheSettings.limitEnabledKey) private var limitEnabled = false
    @AppStorage(CacheSettings.limitGBKey) private var limitGB = CacheSettings.defaultLimitGB
    /// The folder waiting for "Verschieben" or "Nur wechseln". `.some(nil)` is the default folder.
    @State private var pending: URL??
    @State private var clearTarget: CacheBudget.Entry?

    var body: some View {
        Form {
            formatSection
            locationSection
            limitSection
            usageSection
        }
        .formStyle(.grouped)
        .frame(width: 560)
        .frame(minHeight: 560)
        .onAppear { usage.refresh() }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                usage.noteGrowth()
            }
        }
        .onChange(of: limitEnabled) { _, _ in limitChanged() }
        .onChange(of: limitGB) { _, _ in limitChanged() }
        .alert("Cache verschieben?", isPresented: askMove, presenting: pending) { target in
            Button("Verschieben") { usage.relocate(to: target, move: true) }
            Button("Nur wechseln") { usage.relocate(to: target, move: false) }
            Button("Abbrechen", role: .cancel) {}
        } message: { target in
            Text("Der Cache belegt \(Self.size(usage.total ?? 0)). „Verschieben“ kopiert ihn nach „\((target ?? CacheSettings.defaultRoot).path)“ und löscht die alten Dateien erst, wenn alles angekommen ist. „Nur wechseln“ lässt die alten Dateien liegen und fängt im neuen Ordner leer an.")
        }
        .confirmationDialog(
            "Cache von „\(clearTarget.map(name(of:)) ?? "")“ löschen?",
            isPresented: askClear,
            presenting: clearTarget
        ) { entry in
            Button("\(Self.size(entry.bytes)) löschen", role: .destructive) { usage.clear(filmID: entry.id) }
            Button("Abbrechen", role: .cancel) {}
        } message: { _ in
            Text("Die hochskalierten Bilder dieses Films werden bei Bedarf neu gerechnet. Der Film selbst bleibt in der Liste.")
        }
    }

    // MARK: Sections

    private var formatSection: some View {
        Section {
            Picker("ProRes-Variante", selection: $codec) {
                ForEach(CacheCodec.allCases) { flavor in
                    Text("\(flavor.title) (\(flavor.sizeNote))").tag(flavor)
                }
            }
        } header: {
            Text("Format")
        } footer: {
            Text("Gilt für neu angelegte Caches. Bestehende Caches behalten ihr Format und bleiben lesbar. Der Export in ProRes 4444 oder ProRes 422 HQ kopiert die Bilder nur dann unverändert, wenn der Cache dasselbe Format hat; sonst wird neu kodiert.")
                .foregroundStyle(.secondary)
        }
    }

    private var locationSection: some View {
        Section("Speicherort") {
            LabeledContent("Ordner") {
                Text(currentPath)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .multilineTextAlignment(.trailing)
            }
            if !usage.available {
                Label("Der Ordner ist nicht erreichbar. Ist das Laufwerk angeschlossen? Bis dahin wird nichts gespeichert und nichts gelöscht.", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
            if let progress = usage.moveProgress {
                ProgressView(value: progress) {
                    Text("Cache wird verschoben …")
                } currentValueLabel: {
                    Text(progress.formatted(.percent.precision(.fractionLength(0))))
                }
            }
            if let error = usage.moveError {
                Label(error, systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.red)
            }
            HStack {
                Button("Im Finder zeigen") {
                    NSWorkspace.shared.activateFileViewerSelecting([CacheSettings.root])
                }
                .disabled(!usage.available)
                Spacer()
                Button("Standard") { choose(nil) }
                    .disabled(directory.isEmpty)
                    .help(CacheSettings.defaultRoot.path)
                Button("Ändern …") { pickFolder() }
            }
            .disabled(usage.isMoving)
        }
    }

    private var limitSection: some View {
        Section {
            Toggle("Größe begrenzen", isOn: $limitEnabled)
            if limitEnabled {
                LabeledContent("Höchstens") {
                    HStack(spacing: 6) {
                        TextField("GB", value: $limitGB, format: .number)
                            .labelsHidden()
                            .multilineTextAlignment(.trailing)
                            .frame(width: 70)
                        Stepper("GB", value: $limitGB, in: 5...10_000, step: 5)
                            .labelsHidden()
                        Text("GB")
                    }
                }
            }
        } header: {
            Text("Speicherlimit")
        } footer: {
            Text("Bei Erreichen pausiert das Hintergrundfüllen; bestehende Caches werden nicht gelöscht. Bilder, die du gezielt ansiehst, abspielst oder exportierst, werden weiter gespeichert.")
                .foregroundStyle(.secondary)
        }
    }

    private var usageSection: some View {
        Section("Belegung") {
            LabeledContent("Gesamt") {
                HStack(spacing: 6) {
                    if usage.scanning { ProgressView().controlSize(.small) }
                    Text(totalText).monospacedDigit()
                }
            }
            if let fraction = limitFraction {
                ProgressView(value: fraction)
                    .tint(fraction >= 1 ? .orange : .accentColor)
            }
            if usage.entries.isEmpty {
                Text(usage.available ? "Noch kein Film im Cache." : "–")
                    .foregroundStyle(.secondary)
            }
            ForEach(usage.entries) { entry in
                LabeledContent {
                    HStack(spacing: 10) {
                        Text(Self.size(entry.bytes)).monospacedDigit()
                        Button("Löschen …") { clearTarget = entry }
                            .controlSize(.small)
                            .disabled(usage.isMoving)
                    }
                } label: {
                    Text(name(of: entry))
                    if let detail = detail(of: entry) { Text(detail) }
                }
            }
        }
    }

    // MARK: Actions

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Auswählen"
        panel.message = "Ordner für den Cache wählen. In einem Ordner mit anderen Dateien entsteht ein Unterordner „Whoodeo Cache“."
        panel.directoryURL = CacheSettings.root.deletingLastPathComponent()
        guard panel.runModal() == .OK, let url = panel.url else { return }
        choose(CacheSettings.cacheFolder(for: url))
    }

    /// Nil is the default folder. An empty or unreachable cache switches at once; otherwise ask.
    private func choose(_ target: URL?) {
        let destination = target ?? CacheSettings.defaultRoot
        guard destination.standardizedFileURL.path != CacheSettings.root.standardizedFileURL.path else { return }
        if usage.available, !usage.entries.isEmpty {
            pending = .some(target)
        } else {
            usage.relocate(to: target, move: false)
        }
    }

    private func limitChanged() {
        if limitGB < 1 { limitGB = CacheSettings.defaultLimitGB }
        NotificationCenter.default.post(name: CacheSettings.changed, object: nil)
    }

    // MARK: Text

    private var askMove: Binding<Bool> {
        Binding(get: { pending != nil }, set: { if !$0 { pending = nil } })
    }

    private var askClear: Binding<Bool> {
        Binding(get: { clearTarget != nil }, set: { if !$0 { clearTarget = nil } })
    }

    private var currentPath: String {
        directory.isEmpty ? CacheSettings.defaultRoot.path : directory
    }

    private var limitFraction: Double? {
        guard limitEnabled, let total = usage.total, limitGB > 0 else { return nil }
        return min(Double(total) / (Double(limitGB) * 1e9), 1)
    }

    private var totalText: String {
        guard let total = usage.total else { return usage.available ? "wird gezählt …" : "nicht erreichbar" }
        if limitEnabled { return "\(Self.size(total)) von \(limitGB) GB" }
        return "\(Self.size(total)) (unbegrenzt)"
    }

    private func name(of entry: CacheBudget.Entry) -> String {
        if let film = session.films.first(where: { $0.id == entry.id }) {
            return URL(fileURLWithPath: film.path).deletingPathExtension().lastPathComponent
        }
        return "Unbekannter Film"
    }

    private func detail(of entry: CacheBudget.Entry) -> String? {
        guard let film = session.films.first(where: { $0.id == entry.id }) else { return String(entry.id.prefix(12)) + "…" }
        return film.subtitle
    }

    private static func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
