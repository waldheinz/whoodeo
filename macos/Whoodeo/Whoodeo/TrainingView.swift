import Charts
import Combine
import SwiftUI

/// Stammbaum der Läufe und Live-Trainingskurven, im Modelle-Modus des Hauptfensters.
struct TrainingView: View {
    @EnvironmentObject private var session: Session
    @StateObject private var browser = TrainingBrowser()
    @AppStorage("trainingMetric") private var metric = "total"
    @AppStorage("trainingEMA") private var smooth = true
    @AppStorage("trainingLogScale") private var logScale = false
    @State private var deleteTarget: CheckpointRecord?
    @State private var stopTarget: RunRecord?
    @State private var deleteRunTarget: RunRecord?
    @State private var startRequest: TrainStartRequest?

    var body: some View {
        NavigationSplitView {
            tree
                .navigationSplitViewColumnWidth(min: 280, ideal: 360, max: 520)
        } detail: {
            detail
        }
        .navigationSplitViewStyle(.balanced)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    startRequest = newRunRequest()
                } label: {
                    Label("Neuer Lauf", systemImage: "plus")
                }
                .help("Neues Training von Grund auf starten")
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    browser.refresh(immediately: true)
                } label: {
                    Label("Aktualisieren", systemImage: "arrow.clockwise")
                }
                .help("Jetzt aus der Datenbank lesen")
            }
        }
        .sheet(item: $startRequest) { request in
            TrainStartSheet(
                request: request,
                existingNames: browser.runNames,
                trainingRunning: browser.anyRunning,
                onLaunch: launch
            )
        }
        .onAppear { browser.start() }
        .onDisappear { browser.stop() }
        .onChange(of: browser.selection) { _, _ in
            browser.onSelectionChange()
        }
        .alert(
            "Checkpoint \(deleteTarget.map { "Schritt \($0.step)" } ?? "") löschen?",
            isPresented: Binding(
                get: { deleteTarget != nil },
                set: { if !$0 { deleteTarget = nil } }
            )
        ) {
            Button("Löschen", role: .destructive) {
                if let target = deleteTarget {
                    browser.deleteCheckpoint(target)
                }
                deleteTarget = nil
            }
            Button("Abbrechen", role: .cancel) { deleteTarget = nil }
        } message: {
            Text("Der Checkpoint, seine Dateien unter Models und die zugeordneten Caches werden gelöscht. Das lässt sich nicht rückgängig machen.")
        }
        .alert(
            "Lauf nach dem aktuellen Schritt pausieren und einen Checkpoint speichern?",
            isPresented: Binding(
                get: { stopTarget != nil },
                set: { if !$0 { stopTarget = nil } }
            )
        ) {
            Button("Lauf stoppen") {
                if let run = stopTarget {
                    browser.requestStop(run)
                }
                stopTarget = nil
            }
            Button("Abbrechen", role: .cancel) { stopTarget = nil }
        }
        .alert(
            deleteRunTitle,
            isPresented: Binding(
                get: { deleteRunTarget != nil },
                set: { if !$0 { deleteRunTarget = nil } }
            )
        ) {
            if let run = deleteRunTarget, browser.childRuns(of: run.id).isEmpty {
                Button("Lauf löschen", role: .destructive) {
                    browser.deleteRun(run)
                    deleteRunTarget = nil
                }
            }
            Button("Abbrechen", role: .cancel) { deleteRunTarget = nil }
        } message: {
            Text(deleteRunMessage)
        }
    }

    private var deleteRunTitle: String {
        guard let run = deleteRunTarget else { return "Lauf löschen?" }
        if !browser.childRuns(of: run.id).isEmpty {
            return "Lauf „\(run.name)“ kann nicht gelöscht werden"
        }
        return "Lauf „\(run.name)“ löschen?"
    }

    private var deleteRunMessage: String {
        guard let run = deleteRunTarget else { return "" }
        let children = browser.childRuns(of: run.id)
        if !children.isEmpty {
            let names = children.map(\.name).joined(separator: ", ")
            return "Dieser Lauf hat Kind-Läufe (\(names)). Bitte zuerst die Kind-Läufe löschen."
        }
        var text = "Der Lauf, alle seine Checkpoints, Optimizer-Dateien und zugehörigen Caches werden unwiderruflich gelöscht."
        if run.status == "imported" {
            text += " Der Lauf ist importiert."
        }
        return text
    }

    private var treeSelection: Binding<TrainingSelection?> {
        Binding(
            get: { browser.selection },
            set: { next in
                browser.selection = next
                if NSEvent.modifierFlags.contains(.option),
                   case .checkpoint(let id) = next,
                   let checkpoint = browser.checkpoint(id)
                {
                    applyCheckpoint(checkpoint, option: true)
                }
            }
        )
    }

    // MARK: Start training

    private func newRunRequest() -> TrainStartRequest {
        TrainStartRequest(
            mode: .new,
            base: TrainConfig(name: ""),
            title: "Neuer Lauf",
            subtitle: "Training von Grund auf. Die Architektur ist frei wählbar."
        )
    }

    private func fineTuneRequest(_ checkpoint: CheckpointRecord) -> TrainStartRequest? {
        guard let run = browser.run(id: checkpoint.runID) else { return nil }
        return TrainStartRequest(
            mode: .fineTune(checkpointID: checkpoint.id),
            base: TrainConfig.fineTuneBase(parent: run, checkpointID: checkpoint.id, name: ""),
            title: "Fine-Tune ab Checkpoint #\(checkpoint.id)",
            subtitle: "\(run.name), Schritt \(checkpoint.step). Die Architektur kommt vom Checkpoint, der Rest ist frei."
        )
    }

    private func resumeRequest(_ run: RunRecord) -> TrainStartRequest {
        let step = browser.lastCheckpointStep(of: run.id).map { " ab Schritt \($0)" } ?? ""
        return TrainStartRequest(
            mode: .resume(runName: run.name),
            base: TrainConfig(run: run),
            title: "Lauf fortsetzen",
            subtitle: "\(run.name)\(step). Die Architektur bleibt, alles andere lässt sich ändern."
        )
    }

    private func launch(_ name: String, _ arguments: [String]) throws {
        let browser = browser
        try TrainLauncher.launch(arguments: arguments, runName: name) { code, _ in
            if code != 0 {
                let tail = TrainLauncher.logTail(runName: name)
                browser.launchNote = "whoodeo-train (\(name)) endete mit Code \(code)" + (tail.isEmpty ? "" : ": \(tail)")
            }
            browser.refresh(immediately: true)
        }
        browser.launchNote = nil
        browser.pendingSelectName = name
        for delay in [1.5, 5.0, 12.0] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { browser.refresh(immediately: true) }
        }
    }

    private func logButton(_ run: RunRecord) -> some View {
        let url = TrainLauncher.logURL(runName: run.name)
        return Group {
            if FileManager.default.fileExists(atPath: url.path) {
                Button("Log öffnen") { NSWorkspace.shared.open(url) }
                    .help("Öffnet Models/\(run.name)/train.log")
            }
        }
    }

    private func applyCheckpoint(_ checkpoint: CheckpointRecord, option: Bool) {
        guard let url = browser.packageURL(checkpoint) else { return }
        if option {
            if !session.isActiveModel(url) {
                session.toggleComparison(url)
            }
        } else if !session.isActiveModel(url) {
            session.selection = url
        }
    }

    private func setReference(_ checkpoint: CheckpointRecord) {
        guard let url = browser.packageURL(checkpoint) else { return }
        if session.isReferenceModel(url) {
            session.setComparison(nil)
        } else if !session.isActiveModel(url) {
            session.setComparison(url)
        }
    }

    // MARK: Tree

    private var tree: some View {
        List(selection: treeSelection) {
            if browser.tree.isEmpty && !browser.loading {
                Text("Keine Läufe in der Datenbank.")
                    .foregroundStyle(.secondary)
            }
            ForEach(browser.flatRows) { row in
                treeRow(row)
                    .tag(row.id)
                    .listRowInsets(EdgeInsets(top: 2, leading: 8 + CGFloat(row.depth) * 14, bottom: 2, trailing: 8))
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Modelle")
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if let note = browser.launchNote {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            if let error = browser.error {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder
    private func treeRow(_ row: TrainingFlatRow) -> some View {
        switch row.id {
        case .run(let id):
            if let node = browser.node(id) {
                HStack(alignment: .top, spacing: 6) {
                    Button {
                        browser.toggleExpanded(id)
                    } label: {
                        Image(systemName: browser.expanded.contains(id) ? "chevron.down" : "chevron.right")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 12, height: 16)
                    }
                    .buttonStyle(.plain)
                    runRow(node)
                }
            }
        case .checkpoint(let id):
            if let checkpoint = browser.checkpoint(id) {
                checkpointRow(checkpoint)
            }
        }
    }

    private func runRow(_ node: RunTreeNode) -> some View {
        let run = node.run
        let loss = browser.headline(for: run.id)
        return VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(run.name)
                    .font(.body.weight(.medium))
                    .lineLimit(1)
                StatusBadge(status: run.status)
            }
            HStack(spacing: 8) {
                Text("\(run.channels) ch")
                Text(run.cropSize.map { "Crop \($0)" } ?? "Vollbild")
                if let step = loss.step {
                    Text("Schritt \(step)")
                }
                if let value = loss.loss {
                    Text(Self.lossText(value))
                        .monospacedDigit()
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .padding(.vertical, 2)
    }

    private func checkpointRow(_ checkpoint: CheckpointRecord) -> some View {
        let url = browser.packageURL(checkpoint)
        let active = url.map(session.isActiveModel) ?? false
        let reference = url.map(session.isReferenceModel) ?? false
        return VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Image(systemName: "cube")
                    .foregroundStyle(active ? Color.orange : Color.secondary)
                    .frame(width: 16)
                Text("Schritt \(checkpoint.step)")
                    .font(.body.weight(.medium))
                Text(checkpoint.exportDtype)
                    .foregroundStyle(.secondary)
                if active {
                    RoleBadge("Aktiv", color: .orange)
                }
                if reference {
                    RoleBadge("Referenz", color: .blue)
                }
                Spacer(minLength: 8)
            }
            HStack(spacing: 8) {
                if let loss = checkpoint.valLoss {
                    Text("Val \(Self.lossText(loss))")
                }
                if let psnr = checkpoint.valPSNR {
                    Text(String(format: "PSNR %.2f", psnr))
                }
                if let ssim = checkpoint.valSSIM {
                    Text(String(format: "SSIM %.3f", ssim))
                }
                if let bytes = browser.cacheBytes[checkpoint.id], bytes > 0 {
                    Text("Cache \(Self.bytesText(bytes))")
                } else if browser.cacheCount(for: checkpoint.id) > 0 {
                    Text("\(browser.cacheCount(for: checkpoint.id)) Caches")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .padding(.vertical, 2)
        .contextMenu {
            if url != nil, !active {
                Button("Aktivieren") { applyCheckpoint(checkpoint, option: false) }
            }
            if url != nil {
                Button(reference ? "Referenz entfernen" : "Als Referenz") {
                    setReference(checkpoint)
                }
            }
            Divider()
            Button("Löschen…", role: .destructive) {
                deleteTarget = checkpoint
            }
        }
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        if let run = browser.selectedRun {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header(run)
                    lossChart(run)
                    extras(run)
                    ValPreviewPane(
                        runName: run.name,
                        preferredStep: browser.selectedCheckpoint?.step,
                        steps: browser.valSteps[run.name] ?? []
                    )
                }
                .padding(20)
                .frame(maxWidth: 920, alignment: .leading)
                // Width limit stays inside; the scroll view itself fills the column so its scroller sits at the window edge.
                .frame(maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationTitle(run.name)
        } else {
            ContentUnavailableView(
                "Einen Lauf wählen",
                systemImage: "chart.xyaxis.line",
                description: Text("Links steht der Stammbaum der Modelle. Ein Lauf zeigt hier seine Kurven.")
            )
        }
    }

    private func header(_ run: RunRecord) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(run.name)
                    .font(.title2.weight(.semibold))
                Text("#\(run.id)")
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
                    .help("Lauf-ID für whoodeo-train")
                StatusBadge(status: run.status)
                Spacer()
                logButton(run)
                if run.status == "running" {
                    Button(browser.checkpointPending ? "Angefordert…" : "Checkpoint jetzt") {
                        browser.requestCheckpoint(run)
                    }
                    .disabled(browser.checkpointPending)
                    .help("Schreibt nach dem nächsten Schritt einen sichtbaren Checkpoint, ohne das Training zu stoppen")
                    Button(browser.stopPending ? "Stopp angefordert…" : "Lauf stoppen") {
                        stopTarget = run
                    }
                    .disabled(browser.stopPending)
                    .help("Pausiert nach dem aktuellen Schritt und schreibt einen Checkpoint")
                } else {
                    if run.status != "imported", browser.lastCheckpointStep(of: run.id) != nil {
                        Button("Fortsetzen…") { startRequest = resumeRequest(run) }
                            .help("Lauf ab dem letzten Checkpoint weitertrainieren")
                    }
                    Button("Lauf löschen", role: .destructive) {
                        deleteRunTarget = run
                    }
                    .help("Löscht den Lauf, seine Checkpoints, Optimizer-Dateien und zugehörigen Caches")
                }
            }
            FlowFacts(items: headerFacts(run))
            if let notes = run.notes, !notes.isEmpty {
                Text(notes)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if let checkpoint = browser.selectedCheckpoint {
                Divider()
                checkpointHeader(checkpoint)
            }
        }
    }

    private func checkpointHeader(_ checkpoint: CheckpointRecord) -> some View {
        let url = browser.packageURL(checkpoint)
        let active = url.map(session.isActiveModel) ?? false
        let reference = url.map(session.isReferenceModel) ?? false
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Schritt \(checkpoint.step)")
                    .font(.headline)
                Text("#\(checkpoint.id)")
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
                    .help("Checkpoint-ID für whoodeo-train")
                if active { RoleBadge("Aktiv", color: .orange) }
                if reference { RoleBadge("Referenz", color: .blue) }
                Text(checkpoint.exportDtype)
                    .foregroundStyle(.secondary)
                Spacer()
                if url != nil {
                    if !active {
                        Button("Aktivieren") { applyCheckpoint(checkpoint, option: false) }
                    }
                    Button(reference ? "Referenz entfernen" : "Als Referenz") {
                        setReference(checkpoint)
                    }
                }
                Button("Fine-Tune…") { startRequest = fineTuneRequest(checkpoint) }
                    .help("Neuen Lauf ab diesem Checkpoint starten")
                Button("Checkpoint löschen…", role: .destructive) {
                    deleteTarget = checkpoint
                }
            }
            FlowFacts(items: [
                "Angelegt \(Self.dateText(checkpoint.createdAt))",
                checkpoint.relPath,
            ])
        }
    }

    private func headerFacts(_ run: RunRecord) -> [String] {
        var items: [String] = []
        items.append("\(run.channels) Kanäle")
        if let blocks = Self.blocksText(run.blocksJSON) {
            items.append("Blöcke \(blocks)")
        }
        items.append(run.cropSize.map { "Crop \($0)" } ?? "Vollbild")
        items.append(run.dtype.uppercased())
        if let parent = run.parentCheckpointID {
            if let step = browser.parentStep(of: run), let name = browser.parentName(of: run) {
                items.append("Fine-Tune von \(name), Schritt \(step)")
            } else {
                items.append("Fine-Tune von Checkpoint \(parent)")
            }
        } else {
            items.append("from scratch")
        }
        let sources = browser.sourceCounts
        if sources.train + sources.val > 0 {
            items.append("Titel \(sources.train) train / \(sources.val) val")
        }
        items.append("Start \(Self.dateText(run.startedAt))")
        if let finished = run.finishedAt {
            items.append("Ende \(Self.dateText(finished))")
        }
        if let config = Self.configBits(run.configJSON) {
            items.append(contentsOf: config)
        }
        return items
    }

    private func lossChart(_ run: RunRecord) -> some View {
        let series = browser.chartSeries(metric: metric, smooth: smooth)
        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Text("Verlust")
                    .font(.headline)
                Picker("Metrik", selection: $metric) {
                    ForEach(browser.lossMetricNames, id: \.self) { name in
                        Text(Self.metricTitle(name)).tag(name)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 200)
                .disabled(browser.lossMetricNames.isEmpty)
                Toggle("Glättung", isOn: $smooth)
                    .help("EMA über die Train-Kurve, nützlich bei verrauschten Step-Losses")
                    .disabled(series.train.isEmpty)
                Toggle("Log", isOn: $logScale)
                    .help("Logarithmische Senkrechte")
                    .disabled(series.train.isEmpty && series.val.isEmpty)
                Spacer()
            }
            if series.train.isEmpty && series.val.isEmpty {
                emptyChart
            } else {
                Chart {
                    ForEach(series.train) { point in
                        LineMark(
                            x: .value("Schritt", point.step),
                            y: .value("Loss", point.value),
                            series: .value("Kurve", series.trainLabel)
                        )
                        .foregroundStyle(by: .value("Kurve", series.trainLabel))
                        .interpolationMethod(.monotone)
                    }
                    ForEach(series.val) { point in
                        LineMark(
                            x: .value("Schritt", point.step),
                            y: .value("Loss", point.value),
                            series: .value("Kurve", "Val")
                        )
                        .foregroundStyle(by: .value("Kurve", "Val"))
                        .interpolationMethod(.monotone)
                        PointMark(
                            x: .value("Schritt", point.step),
                            y: .value("Loss", point.value)
                        )
                        .foregroundStyle(by: .value("Kurve", "Val"))
                        .symbolSize(24)
                    }
                }
                .chartForegroundStyleScale([
                    series.trainLabel: Color.accentColor,
                    "Val": Color.orange,
                ])
                .chartXAxisLabel("Schritt")
                .chartYScale(domain: .automatic, type: logScale ? .log : .linear)
                .frame(height: 280)
                .padding(.top, 4)
            }
        }
    }

    private var emptyChart: some View {
        HStack(spacing: 12) {
            Image(systemName: "chart.xyaxis.line")
                .font(.title2)
                .foregroundStyle(.tertiary)
            VStack(alignment: .leading, spacing: 2) {
                Text("Keine Trainingskurve gespeichert")
                    .font(.callout.weight(.medium))
                Text("Importierte Modelle haben keine Step-Metriken. Ein neuer Lauf mit whoodeo-train schreibt sie hier live hinein.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func extras(_ run: RunRecord) -> some View {
        let tiles = browser.statTiles()
        let extra = browser.chartSeries(metric: browser.secondaryMetric, smooth: false)
        return Group {
            if !tiles.isEmpty || !extra.train.isEmpty || !extra.val.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Validierung & Tempo")
                            .font(.headline)
                        if !browser.secondaryNames.isEmpty {
                            Picker("Kennzahl", selection: $browser.secondaryMetric) {
                                ForEach(browser.secondaryNames, id: \.self) { name in
                                    Text(Self.metricTitle(name)).tag(name)
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(maxWidth: 180)
                        }
                        Spacer()
                    }
                    if !tiles.isEmpty {
                        FlowFacts(items: tiles)
                    }
                    if !extra.train.isEmpty || !extra.val.isEmpty {
                        Chart {
                            ForEach(extra.val.isEmpty ? extra.train : extra.val) { point in
                                LineMark(
                                    x: .value("Schritt", point.step),
                                    y: .value("Wert", point.value)
                                )
                                .foregroundStyle(Color.secondary)
                                .interpolationMethod(.monotone)
                                PointMark(
                                    x: .value("Schritt", point.step),
                                    y: .value("Wert", point.value)
                                )
                                .foregroundStyle(Color.secondary)
                                .symbolSize(20)
                            }
                        }
                        .chartXAxisLabel("Schritt")
                        .frame(height: 140)
                    }
                }
            }
        }
    }

    // MARK: Formatting

    static func lossText(_ value: Double) -> String {
        if value >= 10 { return String(format: "%.2f", value) }
        if value >= 1 { return String(format: "%.3f", value) }
        return String(format: "%.4f", value)
    }

    static func bytesText(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        return formatter.string(fromByteCount: bytes)
    }

    static func dateText(_ raw: String) -> String {
        let berlin = TimeZone(identifier: "Europe/Berlin") ?? .current
        let out = DateFormatter()
        out.locale = Locale(identifier: "de_DE")
        out.timeZone = berlin
        out.dateFormat = "d.M.yyyy, HH:mm"
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = iso.date(from: raw) { return out.string(from: date) }
        iso.formatOptions = [.withInternetDateTime]
        if let date = iso.date(from: raw) { return out.string(from: date) }
        let local = DateFormatter()
        local.locale = Locale(identifier: "en_US_POSIX")
        local.timeZone = berlin
        for format in ["yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd HH:mm:ss"] {
            local.dateFormat = format
            if let date = local.date(from: raw) { return out.string(from: date) }
        }
        return raw
    }

    static func metricTitle(_ name: String) -> String {
        switch name {
        case "total": "Gesamt"
        case "latent_l1": "Latent L1"
        case "oklab": "Oklab"
        case "ms_ssim": "MS-SSIM"
        case "vgg": "VGG"
        case "psnr": "PSNR"
        case "ssim": "SSIM"
        case "lr": "Lernrate"
        case "seconds": "Sekunden / Schritt"
        case "samples_per_sec": "Samples / s"
        default: name.replacingOccurrences(of: "_", with: " ")
        }
    }

    static func blocksText(_ json: String) -> String? {
        guard let data = json.data(using: .utf8),
              let values = try? JSONSerialization.jsonObject(with: data) as? [Int],
              !values.isEmpty
        else { return nil }
        return values.map(String.init).joined(separator: "/")
    }

    static func configBits(_ json: String) -> [String]? {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        var bits: [String] = []
        if let batch = object["batch"] as? Int, let accum = object["grad_accum"] as? Int {
            bits.append("Batch \(batch) × \(accum)")
        }
        if let lr = object["lr"] as? Double {
            bits.append(String(format: "lr %.1e", lr))
        } else if let lr = object["lr"] as? Int {
            bits.append("lr \(lr)")
        }
        if let seed = object["seed"] as? Int {
            bits.append("Seed \(seed)")
        }
        bits.append(TrainConfig.spynetLabel(from: json))
        if let workers = object["workers"] as? Int {
            bits.append(workers == 0 ? "Prefetch aus" : "Worker \(workers)")
        }
        return bits.isEmpty ? nil : bits
    }
}

// MARK: - Status and facts

private struct RoleBadge: View {
    var text: String
    var color: Color

    init(_ text: String, color: Color) {
        self.text = text
        self.color = color
    }

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}

private struct StatusBadge: View {
    let status: String

    var body: some View {
        HStack(spacing: 4) {
            if status == "running" {
                ProgressView()
                    .controlSize(.mini)
            }
            Text(label)
        }
        .font(.caption.weight(.semibold))
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(color.opacity(0.18), in: Capsule())
        .foregroundStyle(color)
        .accessibilityLabel(label)
    }

    private var label: String {
        switch status {
        case "running": "läuft"
        case "paused": "pausiert"
        case "finished": "fertig"
        case "imported": "importiert"
        case "failed": "fehlgeschlagen"
        default: status
        }
    }

    private var color: Color {
        switch status {
        case "running": .green
        case "paused": .orange
        case "finished": .blue
        case "failed": .red
        default: .secondary
        }
    }
}

private struct FlowFacts: View {
    let items: [String]

    var body: some View {
        Text(items.joined(separator: "  ·  "))
            .font(.callout)
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
    }
}

// MARK: - Browser

enum TrainingSelection: Hashable {
    case run(Int)
    case checkpoint(Int)
}

struct TrainingFlatRow: Identifiable, Hashable {
    var id: TrainingSelection
    var depth: Int
}

struct ChartPoint: Identifiable, Hashable {
    var step: Int
    var value: Double
    var id: Int { step }
}

struct ChartSeries {
    var train: [ChartPoint]
    var val: [ChartPoint]
    var trainLabel: String
}

@MainActor
final class TrainingBrowser: ObservableObject {
    @Published var tree: [RunTreeNode] = []
    @Published var selection: TrainingSelection?
    @Published var metrics: [MetricPoint] = []
    @Published var cacheBytes: [Int: Int64] = [:]
    @Published var cacheCounts: [Int: Int] = [:]
    @Published var sourceCounts: (train: Int, val: Int) = (0, 0)
    @Published var headlines: [Int: (step: Int, loss: Double)] = [:]
    @Published var secondaryMetric = "psnr"
    @Published var error: String?
    @Published var valSteps: [String: [Int]] = [:]
    @Published var loading = false
    @Published var expanded: Set<Int> = []
    @Published var checkpointPending = false
    @Published var stopPending = false
    /// Exit note of a training the app started, shown under the tree.
    @Published var launchNote: String?
    /// Selected once the run shows up after a start from the app.
    var pendingSelectName: String?
    private var pendingRunID: Int?
    private var pendingAfterStep = 0

    private var timer: Timer?
    private var watches: [DispatchSourceFileSystemObject] = []
    private var lastSignature = ""
    private var cacheSignature = ""
    private var selectedRunID: Int?

    var selectedRun: RunRecord? {
        guard let id = runID(of: selection) else { return nil }
        return findRun(id, in: tree)
    }

    var selectedCheckpoint: CheckpointRecord? {
        guard case .checkpoint(let id) = selection else { return nil }
        return findCheckpoint(id, in: tree)
    }

    var lossMetricNames: [String] {
        let reserved: Set<String> = ["psnr", "ssim", "lr", "seconds", "samples_per_sec"]
        var names = Set(metrics.map(\.name).filter { !reserved.contains($0) && !$0.hasPrefix("_") })
        if names.contains("total") || !names.isEmpty {
            // Keep a stable order: total first, then known terms, then the rest.
            let preferred = ["total", "latent_l1", "oklab", "ms_ssim", "vgg"]
            let ordered = preferred.filter { names.contains($0) } + names.subtracting(preferred).sorted()
            return ordered
        }
        return []
    }

    var secondaryNames: [String] {
        let wanted = ["psnr", "ssim", "lr", "seconds", "samples_per_sec"]
        return wanted.filter { name in metrics.contains { $0.name == name } }
    }

    func start() {
        if expanded.isEmpty {
            // First open: expand every run so checkpoints are visible.
        }
        refresh(immediately: true)
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh(immediately: false) }
        }
        if let timer { RunLoop.main.add(timer, forMode: .common) }
        startWatch()
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        for watch in watches { watch.cancel() }
        watches.removeAll()
    }

    func refresh(immediately: Bool) {
        if loading && !immediately { return }
        loading = true
        let current = selectedRunID
        let keepExpanded = expanded
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                guard let db = WhoodeoDB.shared else {
                    throw WhoodeoDBError("Datenbank nicht geöffnet")
                }
                let tree = try db.runTree()
                var headlines: [Int: (step: Int, loss: Double)] = [:]
                for id in tree.flatMap(\.allRunIDs) {
                    let train = try db.lastMetrics(runID: id, split: "train")
                    if let total = train["total"] {
                        headlines[id] = (Int(train["_step"] ?? 0), total)
                    }
                }
                var metrics: [MetricPoint] = []
                var sources: (Int, Int) = (0, 0)
                if let current, tree.containsRun(current) {
                    metrics = try db.metrics(runID: current)
                    var found = try db.runSourceCounts(runID: current)
                    if found.train + found.val == 0 {
                        let counts = try db.counts()
                        found = (counts.train, counts.val)
                    }
                    sources = found
                }
                let cacheSig = try db.counts().cacheFiles.description
                    + tree.flatMap(\.allCheckpoints).map { "\($0.id):\($0.step)" }.joined()
                let sizes: [Int: Int64]
                let recount: [Int: Int]
                if cacheSig != self?.cacheSignature {
                    sizes = try db.cacheBytesByCheckpoint()
                    recount = try db.cacheCountsByCheckpoint()
                } else {
                    sizes = self?.cacheBytes ?? [:]
                    recount = self?.cacheCounts ?? [:]
                }
                var stopPending = false
                if let current, tree.containsRun(current) {
                    let status = Self.status(of: current, in: tree)
                    if status == "running" {
                        stopPending = try db.hasStopRequest(runID: current)
                    }
                }
                var valSteps: [String: [Int]] = [:]
                for run in tree.flatMap(\.allRuns) {
                    valSteps[run.name] = ValPreview.steps(forRun: run.name)
                }
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.error = nil
                    self.tree = tree
                    self.headlines = headlines
                    let allIDs = Set(tree.flatMap(\.allRunIDs))
                    if keepExpanded.isEmpty {
                        self.expanded = allIDs
                    } else {
                        self.expanded.formUnion(allIDs.subtracting(keepExpanded))
                    }
                    if let name = self.pendingSelectName,
                       let run = Self.flattenRuns(tree).first(where: { $0.name == name })
                    {
                        self.selection = .run(run.id)
                        self.expanded.insert(run.id)
                        self.pendingSelectName = nil
                    }
                    if self.selection == nil, let first = tree.first {
                        self.selection = .run(first.run.id)
                    } else if let sel = self.selection, !tree.contains(sel) {
                        self.selection = tree.first.map { .run($0.run.id) }
                    }
                    let selected = self.runID(of: self.selection)
                    self.lastSignature = Self.signature(tree)
                    self.loading = false
                    if cacheSig != self.cacheSignature {
                        self.cacheBytes = sizes
                        self.cacheCounts = recount
                        self.cacheSignature = cacheSig
                    }
                    if selected != current {
                        self.reloadDetails(runID: selected)
                    } else {
                        self.metrics = metrics
                        self.sourceCounts = sources
                    }
                    self.clearCheckpointPendingIfReady()
                    self.stopPending = stopPending
                    self.valSteps = valSteps
                }
            } catch {
                DispatchQueue.main.async {
                    self?.error = error.localizedDescription
                    self?.loading = false
                }
            }
        }
    }

    func reloadDetails(runID: Int?) {
        selectedRunID = runID
        guard let runID else {
            metrics = []
            sourceCounts = (0, 0)
            return
        }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                guard let db = WhoodeoDB.shared else { return }
                let metrics = try db.metrics(runID: runID)
                var sources = try db.runSourceCounts(runID: runID)
                if sources.train + sources.val == 0 {
                    let counts = try db.counts()
                    sources = (counts.train, counts.val)
                }
                DispatchQueue.main.async {
                    guard self?.selectedRunID == runID else { return }
                    self?.metrics = metrics
                    self?.sourceCounts = sources
                    if let self, !self.secondaryNames.contains(self.secondaryMetric) {
                        self.secondaryMetric = self.secondaryNames.first ?? "psnr"
                    }
                }
            } catch {
                DispatchQueue.main.async { self?.error = error.localizedDescription }
            }
        }
    }

    var flatRows: [TrainingFlatRow] {
        flatten(tree, depth: 0)
    }

    func toggleExpanded(_ runID: Int) {
        if expanded.contains(runID) { expanded.remove(runID) } else { expanded.insert(runID) }
    }

    func node(_ id: Int) -> RunTreeNode? { findNode(id, in: tree) }

    func run(id: Int) -> RunRecord? { findRun(id, in: tree) }

    var runNames: Set<String> { Set(Self.flattenRuns(tree).map(\.name)) }

    var anyRunning: Bool { Self.flattenRuns(tree).contains { $0.status == "running" } }

    func lastCheckpointStep(of runID: Int) -> Int? {
        node(runID)?.checkpoints.map(\.step).max()
    }

    func checkpoint(_ id: Int) -> CheckpointRecord? { findCheckpoint(id, in: tree) }

    func expandedBinding(_ runID: Int) -> Binding<Bool> {
        Binding(
            get: { self.expanded.contains(runID) },
            set: { open in
                if open { self.expanded.insert(runID) } else { self.expanded.remove(runID) }
            }
        )
    }

    private func flatten(_ nodes: [RunTreeNode], depth: Int) -> [TrainingFlatRow] {
        var rows: [TrainingFlatRow] = []
        for node in nodes {
            rows.append(TrainingFlatRow(id: .run(node.run.id), depth: depth))
            if expanded.contains(node.run.id) {
                for checkpoint in node.checkpoints {
                    rows.append(TrainingFlatRow(id: .checkpoint(checkpoint.id), depth: depth + 1))
                }
                rows.append(contentsOf: flatten(node.children, depth: depth + 1))
            }
        }
        return rows
    }

    func requestCheckpoint(_ run: RunRecord) {
        do {
            let db = try WhoodeoDB.openDefault()
            try db.requestCheckpoint(runID: run.id)
            pendingRunID = run.id
            pendingAfterStep = findNode(run.id, in: tree)?.checkpoints.map(\.step).max() ?? 0
            checkpointPending = true
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func clearCheckpointPendingIfReady() {
        guard checkpointPending, let id = pendingRunID else { return }
        let step = findNode(id, in: tree)?.checkpoints.map(\.step).max() ?? 0
        if step > pendingAfterStep {
            checkpointPending = false
            pendingRunID = nil
        }
    }

    func requestStop(_ run: RunRecord) {
        do {
            let db = try WhoodeoDB.openDefault()
            try db.requestStop(runID: run.id)
            stopPending = true
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }

    func deleteRun(_ run: RunRecord) {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                guard let db = WhoodeoDB.shared else { throw WhoodeoDBError("Datenbank nicht geöffnet") }
                try db.deleteRun(id: run.id, removeFiles: true)
                DispatchQueue.main.async {
                    if self?.runID(of: self?.selection) == run.id {
                        self?.selection = nil
                    }
                    self?.refresh(immediately: true)
                }
            } catch {
                DispatchQueue.main.async { self?.error = error.localizedDescription }
            }
        }
    }

    func childRuns(of runID: Int) -> [RunRecord] {
        guard let node = findNode(runID, in: tree) else { return [] }
        return Self.flattenRuns(node.children)
    }

    func deleteCheckpoint(_ checkpoint: CheckpointRecord) {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                guard let db = WhoodeoDB.shared else { throw WhoodeoDBError("Datenbank nicht geöffnet") }
                try db.deleteCheckpoint(id: checkpoint.id, removeFiles: true)
                DispatchQueue.main.async {
                    if self?.selection == .checkpoint(checkpoint.id) {
                        self?.selection = .run(checkpoint.runID)
                    }
                    self?.refresh(immediately: true)
                }
            } catch {
                DispatchQueue.main.async { self?.error = error.localizedDescription }
            }
        }
    }

    func headline(for runID: Int) -> (step: Int?, loss: Double?) {
        if selectedRunID == runID {
            let train = metrics.filter { $0.split == "train" && $0.name == "total" }.sorted { $0.step < $1.step }
            if let last = train.last {
                return (last.step, last.value)
            }
        }
        if let held = headlines[runID] {
            return (held.step, held.loss)
        }
        if let run = findNode(runID, in: tree) {
            let step = run.checkpoints.map(\.step).max()
            let loss = run.checkpoints.last(where: { $0.valLoss != nil })?.valLoss
            return (step, loss)
        }
        return (nil, nil)
    }

    func chartSeries(metric: String, smooth: Bool) -> ChartSeries {
        var train = metrics.filter { $0.split == "train" && $0.name == metric }
            .map { ChartPoint(step: $0.step, value: $0.value) }
            .sorted { $0.step < $1.step }
        let val = metrics.filter { $0.split == "val" && $0.name == metric }
            .map { ChartPoint(step: $0.step, value: $0.value) }
            .sorted { $0.step < $1.step }
        if smooth, train.count >= 3 {
            train = Self.ema(train, alpha: 0.2)
        }
        return ChartSeries(train: train, val: val, trainLabel: smooth ? "Train (EMA)" : "Train")
    }

    func statTiles() -> [String] {
        var tiles: [String] = []
        if let last = metrics.filter({ $0.split == "val" && $0.name == "psnr" }).max(by: { $0.step < $1.step }) {
            tiles.append(String(format: "Val-PSNR %.2f", last.value))
        }
        if let last = metrics.filter({ $0.split == "val" && $0.name == "ssim" }).max(by: { $0.step < $1.step }) {
            tiles.append(String(format: "Val-SSIM %.3f", last.value))
        }
        if let last = metrics.filter({ $0.split == "train" && $0.name == "lr" }).max(by: { $0.step < $1.step }) {
            tiles.append(String(format: "lr %.2e", last.value))
        }
        if let last = metrics.filter({ $0.split == "train" && $0.name == "seconds" }).max(by: { $0.step < $1.step }), last.value > 0 {
            tiles.append(String(format: "%.1f s / Schritt", last.value))
            tiles.append(String(format: "%.2f Schritte / s", 1 / last.value))
        }
        if let last = metrics.filter({ $0.split == "train" && $0.name == "samples_per_sec" }).max(by: { $0.step < $1.step }) {
            tiles.append(String(format: "%.1f Samples / s", last.value))
        }
        return tiles
    }

    func cacheCount(for id: Int) -> Int { cacheCounts[id] ?? 0 }

    func packageURL(_ checkpoint: CheckpointRecord) -> URL? {
        let url = WhoodeoPaths.modelsDirectory.appendingPathComponent(checkpoint.relPath)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func parentStep(of run: RunRecord) -> Int? {
        guard let parent = run.parentCheckpointID else { return nil }
        return findCheckpoint(parent, in: tree)?.step
    }

    func parentName(of run: RunRecord) -> String? {
        guard let parent = run.parentCheckpointID else { return nil }
        return findCheckpoint(parent, in: tree)?.runName
    }

    // Called when selection changes
    func onSelectionChange() {
        let id = runID(of: selection)
        if id != selectedRunID {
            reloadDetails(runID: id)
        }
    }

    private func startWatch() {
        let db = WhoodeoPaths.databaseURL
        let paths = [
            WhoodeoPaths.supportDirectory,
            db,
            URL(fileURLWithPath: db.path + "-wal"),
            URL(fileURLWithPath: db.path + "-shm"),
        ]
        for url in paths {
            let fd = Darwin.open(url.path, O_EVTONLY)
            guard fd >= 0 else { continue }
            let source = DispatchSource.makeFileSystemObjectSource(
                fileDescriptor: fd,
                eventMask: [.write, .extend, .rename, .delete, .attrib],
                queue: .main
            )
            source.setEventHandler { [weak self] in
                self?.refresh(immediately: false)
            }
            source.setCancelHandler { Darwin.close(fd) }
            source.resume()
            watches.append(source)
        }
    }

    private func runID(of selection: TrainingSelection?) -> Int? {
        switch selection {
        case .run(let id): id
        case .checkpoint(let id): findCheckpoint(id, in: tree)?.runID
        case nil: nil
        }
    }

    private func findRun(_ id: Int, in nodes: [RunTreeNode]) -> RunRecord? {
        findNode(id, in: nodes)?.run
    }

    private static func status(of runID: Int, in nodes: [RunTreeNode]) -> String? {
        for node in nodes {
            if node.run.id == runID { return node.run.status }
            if let found = status(of: runID, in: node.children) { return found }
        }
        return nil
    }

    private static func flattenRuns(_ nodes: [RunTreeNode]) -> [RunRecord] {
        nodes.flatMap { [$0.run] + flattenRuns($0.children) }
    }

    private func findNode(_ id: Int, in nodes: [RunTreeNode]) -> RunTreeNode? {
        for node in nodes {
            if node.run.id == id { return node }
            if let found = findNode(id, in: node.children) { return found }
        }
        return nil
    }

    private func findCheckpoint(_ id: Int, in nodes: [RunTreeNode]) -> CheckpointRecord? {
        for node in nodes {
            if let found = node.checkpoints.first(where: { $0.id == id }) { return found }
            if let found = findCheckpoint(id, in: node.children) { return found }
        }
        return nil
    }

    private func metricsOf(runID: Int, split: String, name: String) -> [MetricPoint] {
        guard selectedRunID == runID else { return [] }
        return metrics.filter { $0.split == split && $0.name == name }.sorted { $0.step < $1.step }
    }

    private static func ema(_ points: [ChartPoint], alpha: Double) -> [ChartPoint] {
        var value = points[0].value
        return points.map { point in
            value = alpha * point.value + (1 - alpha) * value
            return ChartPoint(step: point.step, value: value)
        }
    }

    private static func signature(_ tree: [RunTreeNode]) -> String {
        tree.map { node in
            let checks = node.checkpoints.map { "\($0.id):\($0.step):\($0.visibleInApp):\($0.valLoss ?? -1)" }.joined(separator: ",")
            return "\(node.run.id):\(node.run.status):\(node.run.finishedAt ?? ""):[\(checks)]{\(signature(node.children))}"
        }.joined(separator: "|")
    }

    private static func cacheCounts(in tree: [RunTreeNode]) -> [Int: Int] {
        var found: [Int: Int] = [:]
        func walk(_ nodes: [RunTreeNode]) {
            for node in nodes {
                // cacheCount is a run total; split later if needed. We fill per-run equally? No:
                // we'll just leave per-checkpoint counts to be computed separately.
                walk(node.children)
            }
        }
        walk(tree)
        return found
    }
}

private extension Array where Element == RunTreeNode {
    var allRunIDs: [Int] { flatMap(\.allRunIDs) }

    func containsRun(_ id: Int) -> Bool {
        contains { $0.run.id == id || $0.children.containsRun(id) }
    }

    func contains(_ selection: TrainingSelection) -> Bool {
        switch selection {
        case .run(let id): containsRun(id)
        case .checkpoint(let id): contains(where: { $0.checkpoints.contains { $0.id == id } || $0.children.contains(.checkpoint(id)) })
        }
    }
}

private extension RunTreeNode {
    var allRunIDs: [Int] { [run.id] + children.flatMap(\.allRunIDs) }
    var allCheckpoints: [CheckpointRecord] { checkpoints + children.flatMap(\.allCheckpoints) }
    var allRuns: [RunRecord] { [run] + children.flatMap(\.allRuns) }
}

// Reload details when the selection changes.
extension TrainingView {
    // Hook is applied via onChange in body; kept here so the browser stays simple.
}

