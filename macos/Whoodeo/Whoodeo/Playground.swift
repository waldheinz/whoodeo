import AppKit
import Combine
import os
import SwiftUI
import UniformTypeIdentifiers

/// Where a playground picture comes from: one frame of a film, or a picture file.
nonisolated struct SlotSource: Equatable, Sendable {
    var url: URL
    var isFilm: Bool
    var index = 0
    var count = 0

    var name: String { url.lastPathComponent }
}

/// Films the playground reads frames from, and the coder, touched only on its queue.
nonisolated final class PlaygroundWorker: @unchecked Sendable {
    let queue = DispatchQueue(label: "meins.whoodeo.playground", qos: .userInitiated)
    private var films: [URL: Film] = [:]

    /// On `queue`. The frame as a 16-bit picture, and the film's frame count.
    func picture(_ source: SlotSource) throws -> (CGImage, Int) {
        if !source.isFilm {
            guard let image = PlaygroundImage.load(source.url) else {
                throw CLIError("\(source.name) lässt sich nicht als Bild lesen")
            }
            return (image, 0)
        }
        let film: Film
        if let held = films[source.url] {
            film = held
        } else {
            film = try Self.wait { try await Film.load(source.url) }
            // A and B: two open films are enough.
            if films.count >= 2 { films.removeAll() }
            films[source.url] = film
        }
        let count = film.facts.count
        let index = min(max(source.index, 0), max(count - 1, 0))
        let frame = try Self.wait { try await film.frame(at: index) }
        guard let image = PlaygroundImage.image16(planar: frame.rgb, width: frame.width, height: frame.height) else {
            throw CLIError("Bild \(index + 1) lässt sich nicht lesen")
        }
        return (image, count)
    }

    private static func wait<T>(_ body: @escaping () async throws -> T) throws -> T {
        var result: Result<T, Error>?
        let gate = DispatchSemaphore(value: 0)
        Task.detached {
            do { result = .success(try await body()) } catch { result = .failure(error) }
            gate.signal()
        }
        gate.wait()
        return try result!.get()
    }
}

final class PlaygroundModel: ObservableObject {
    @Published private(set) var pairs: [VAEPair] = []
    @Published var pairID: URL? { didSet { if pairID != oldValue { loadCoder() } } }
    @Published var workSize: FrameSize? { didSet { if workSize != oldValue { prepareAll() } } }
    @Published private(set) var latentSizes: [FrameSize] = []
    @Published private(set) var channelCount = 16
    @Published var params = PlaygroundParams() { didSet { if params != oldValue { scheduleRender() } } }
    @Published private(set) var sourceA: SlotSource?
    @Published private(set) var sourceB: SlotSource?
    @Published private(set) var originalA: CGImage?
    @Published private(set) var reconstructionA: CGImage?
    @Published private(set) var reconstructionPSNR: Double?
    @Published private(set) var output: RenderOutput?
    @Published private(set) var status = ""
    @Published private(set) var problem: String?
    @Published private(set) var working = false
    @Published var animateNoise = false
    @Published var animateBlend = false

    private(set) var coder: VAECoder?
    private var dataA: SlotData?
    private var dataB: SlotData?
    private let worker = PlaygroundWorker()
    private var rendering = false
    private var dirty = false
    private var generation = 0
    private var blendPhase: Float = 0
    private var started = false

    var pair: VAEPair? { pairs.first { $0.id == pairID } }
    var hasB: Bool { dataB != nil }

    func start(films: [OpenFilm]) {
        guard !started else { return }
        started = true
        refreshPairs()
        // A first look: two moments of the first film.
        if let film = films.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            let url = URL(fileURLWithPath: film.path)
            let count = max(film.frames, 1)
            sourceA = SlotSource(url: url, isFilm: true, index: count / 3, count: film.frames)
            sourceB = SlotSource(url: url, isFilm: true, index: count * 2 / 3, count: film.frames)
        }
        if pairID == nil { pairID = pairs.first?.id }
    }

    func refreshPairs() {
        pairs = VAELibrary.pairs()
        if let pairID, !pairs.contains(where: { $0.id == pairID }) { self.pairID = pairs.first?.id }
    }

    func openFolder() {
        NSWorkspace.shared.open(VAELibrary.directory)
    }

    private func loadCoder() {
        generation += 1
        coder = nil
        dataA = nil
        dataB = nil
        output = nil
        originalA = nil
        reconstructionA = nil
        guard let pair else {
            status = ""
            return
        }
        let token = generation
        working = true
        status = "Lade \(pair.title) …"
        worker.queue.async {
            let loaded = Result { try VAECoder(pair: pair) }
            DispatchQueue.main.async {
                guard token == self.generation else { return }
                self.working = false
                switch loaded {
                case .failure(let error):
                    Log.playground.error("VAE: \(error.localizedDescription, privacy: .public)")
                    self.problem = error.localizedDescription
                    self.status = ""
                case .success(let coder):
                    Log.playground.info("VAE \(pair.decoder.lastPathComponent, privacy: .public)")
                    self.problem = nil
                    self.coder = coder
                    self.latentSizes = coder.latentSizes
                    self.channelCount = coder.channels
                    if self.params.channels.count != coder.channels {
                        self.params.channels = Array(repeating: ChannelSetting(), count: coder.channels)
                    }
                    let stepText = coder.step.map { " · Schritt \($0)" } ?? ""
                    self.status = "\(pair.title)\(stepText)"
                    let preferred = FrameSize(width: 160, height: 90)
                    let size = coder.latentSizes.contains(preferred) ? preferred : coder.latentSizes[coder.latentSizes.count / 2]
                    if self.workSize == size { self.prepareAll() } else { self.workSize = size }
                }
            }
        }
    }

    // MARK: Pictures

    func choose(_ slot: Int, film: OpenFilm) {
        let source = SlotSource(url: URL(fileURLWithPath: film.path), isFilm: true, index: max(film.frames, 1) / 2, count: film.frames)
        set(slot, source)
    }

    func pickFile(_ slot: Int) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image, .movie, .audiovisualContent] + FilmDrop.contentTypes
        panel.prompt = "Wählen"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        set(slot, SlotSource(url: url, isFilm: !PlaygroundImage.isImage(url)))
    }

    func clearB() {
        sourceB = nil
        dataB = nil
        scheduleRender()
    }

    func setFrame(_ slot: Int, _ index: Int) {
        guard var source = slot == 0 ? sourceA : sourceB, source.isFilm else { return }
        let clamped = max(0, source.count > 0 ? min(index, source.count - 1) : index)
        guard clamped != source.index else { return }
        source.index = clamped
        set(slot, source)
    }

    private func set(_ slot: Int, _ source: SlotSource) {
        if slot == 0 { sourceA = source } else { sourceB = source }
        prepare(slot)
    }

    private func prepareAll() {
        prepare(0)
        prepare(1)
    }

    private func prepare(_ slot: Int) {
        guard let coder, let size = workSize, let source = slot == 0 ? sourceA : sourceB else { return }
        let token = generation
        working = true
        worker.queue.async { [worker] in
            let made = Result { () -> (SlotData, Int) in
                let (image, count) = try worker.picture(source)
                return (try PlaygroundRender.encode(coder, image: image, size: size), count)
            }
            DispatchQueue.main.async {
                self.working = false
                guard token == self.generation, size == self.workSize,
                      source == (slot == 0 ? self.sourceA : self.sourceB) else { return }
                switch made {
                case .failure(let error):
                    Log.playground.error("picture \(slot): \(error.localizedDescription, privacy: .public)")
                    self.problem = error.localizedDescription
                case .success(let (data, count)):
                    Log.playground.info("picture \(slot) \(data.width)x\(data.height) roundtrip \(data.psnr ?? 0, format: .fixed(precision: 2)) dB")
                    self.problem = nil
                    if source.isFilm, count != source.count {
                        var updated = source
                        updated.count = count
                        if slot == 0 { self.sourceA = updated } else { self.sourceB = updated }
                    }
                    if slot == 0 {
                        self.dataA = data
                        self.originalA = data.original
                        self.reconstructionA = data.reconstructionImage
                        self.reconstructionPSNR = data.psnr
                    } else {
                        self.dataB = data
                    }
                    self.scheduleRender()
                }
            }
        }
    }

    // MARK: Result

    private func scheduleRender() {
        guard let coder, let a = dataA else { return }
        if rendering {
            dirty = true
            return
        }
        rendering = true
        let b = dataB
        let p = params
        worker.queue.async {
            let made = Result { try PlaygroundRender.run(coder, a: a, b: b, p: p) }
            DispatchQueue.main.async {
                self.rendering = false
                switch made {
                case .failure(let error):
                    Log.playground.error("render: \(error.localizedDescription, privacy: .public)")
                    self.problem = error.localizedDescription
                case .success(let out):
                    Log.playground.debug("render \(out.width)x\(out.height) \(out.milliseconds, format: .fixed(precision: 0)) ms")
                    self.output = out
                }
                if self.dirty {
                    self.dirty = false
                    self.scheduleRender()
                }
            }
        }
    }

    func tick() {
        guard !rendering else { return }
        if animateNoise {
            if params.noise == 0 { params.noise = 0.5 }
            params.noisePhase += 0.12
        }
        if animateBlend, hasB {
            blendPhase += 0.04
            params.blend = (1 - cos(blendPhase)) / 2
        }
    }

    func reroll() {
        params.noiseSeed = UInt64.random(in: 1...UInt64(UInt32.max))
    }

    func resetChannels() {
        params.channels = Array(repeating: ChannelSetting(), count: channelCount)
        params.solo = nil
        params.swapFirst = nil
        params.swapSecond = nil
        params.gainAll = 1
    }

    func resetAll() {
        animateNoise = false
        animateBlend = false
        var fresh = PlaygroundParams()
        fresh.channels = Array(repeating: ChannelSetting(), count: channelCount)
        fresh.selected = min(params.selected, channelCount - 1)
        params = fresh
    }

    func save() {
        guard let image = output?.result else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "vae-playground.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try PlaygroundImage.savePNG(image, to: url) } catch { problem = error.localizedDescription }
    }

    func channelBinding(_ index: Int) -> Binding<ChannelSetting> {
        Binding(
            get: { index < self.params.channels.count ? self.params.channels[index] : ChannelSetting() },
            set: { value in
                guard index < self.params.channels.count else { return }
                self.params.channels[index] = value
            }
        )
    }
}

struct PlaygroundMenuButton: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("VAE-Playground") { openWindow(id: PlaygroundView.windowID) }
            .keyboardShortcut("l", modifiers: [.command, .shift])
    }
}

struct PlaygroundView: View {
    static let windowID = "vae-playground"

    @EnvironmentObject private var session: Session
    @StateObject private var model = PlaygroundModel()
    private let timer = Timer.publish(every: 1.0 / 12, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 8) {
                if model.pairs.isEmpty && !model.working {
                    missing
                } else {
                    panes
                    channelStrip
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            controls
                .frame(width: 340)
        }
        .frame(minWidth: 1100, minHeight: 720)
        .onAppear { model.start(films: session.films) }
        .onReceive(timer) { _ in model.tick() }
    }

    private var missing: some View {
        VStack(spacing: 12) {
            Image(systemName: "square.stack.3d.down.right").font(.system(size: 40)).foregroundStyle(.secondary)
            Text("Kein VAE-Paar gefunden").font(.title2)
            Text("Der Playground braucht Encoder und Decoder eines vae2 im Ordner VAE. Auf dem Mac:")
                .foregroundStyle(.secondary)
            Text(".venv/bin/whoodeo-coreml <vae2-Lauf oder Checkpoint> --part pair")
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
            HStack {
                Button("Ordner öffnen") { model.openFolder() }
                Button("Erneut suchen") { model.refreshPairs() }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Panes

    private var panes: some View {
        let out = model.output
        return Grid(horizontalSpacing: 8, verticalSpacing: 8) {
            GridRow {
                pane("Original A", model.originalA, caption: model.sourceA.map { caption($0) })
                pane("Rekonstruktion A", model.reconstructionA, caption: psnrText(model.reconstructionPSNR, "zu A"))
            }
            GridRow {
                pane(out?.sideTitle ?? "", out?.side, caption: nil)
                pane(
                    "Ergebnis",
                    out?.result,
                    caption: resultCaption(out),
                    accent: true
                )
            }
        }
    }

    private func resultCaption(_ out: RenderOutput?) -> String? {
        guard let out else { return nil }
        var parts: [String] = []
        if let a = psnrText(out.psnrA, "zu A") { parts.append(a) }
        if let b = psnrText(out.psnrBlend, "zur Pixel-Blende") { parts.append(b) }
        parts.append("\(out.width)×\(out.height)")
        parts.append(String(format: "%.0f ms", out.milliseconds))
        return parts.joined(separator: " · ")
    }

    private func psnrText(_ value: Double?, _ against: String) -> String? {
        value.map { String(format: "%.2f dB %@", $0, against) }
    }

    private func caption(_ source: SlotSource) -> String {
        source.isFilm ? "\(source.name) · Bild \(source.index + 1)" : source.name
    }

    private func pane(_ title: String, _ image: CGImage?, caption: String?, accent: Bool = false) -> some View {
        VStack(spacing: 4) {
            HStack {
                Text(title).font(.headline)
                Spacer()
                if let caption { Text(caption).font(.caption).foregroundStyle(.secondary).monospacedDigit() }
            }
            ZStack {
                RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.85))
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else if model.working {
                    ProgressView().controlSize(.small)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(accent ? Color.accentColor : Color.clear, lineWidth: 2)
            )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Channels

    private var channelStrip: some View {
        let tiles = model.output?.tiles ?? []
        let columns = Array(repeating: GridItem(.flexible(), spacing: 4), count: max(model.channelCount / 2, 1))
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Latent-Kanäle").font(.headline)
                Text("schwarz −3σ, weiß +3σ · Klick wählt, Rechtsklick für mehr")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            LazyVGrid(columns: columns, spacing: 4) {
                ForEach(0..<model.channelCount, id: \.self) { index in
                    tile(index, index < tiles.count ? tiles[index] : nil)
                }
            }
        }
        .frame(height: 190)
    }

    private func tile(_ index: Int, _ image: CGImage?) -> some View {
        let setting = index < model.params.channels.count ? model.params.channels[index] : ChannelSetting()
        let selected = model.params.selected == index
        return ZStack(alignment: .topLeading) {
            Rectangle().fill(Color.black)
            if let image {
                Image(decorative: image, scale: 1)
                    .interpolation(.none)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            HStack(spacing: 2) {
                Text("\(index)")
                if model.params.solo == index { badge("S", .yellow) }
                if setting.mode == .mute { badge("M", .red) }
                if setting.mode == .invert { badge("I", .orange) }
                if setting.fromB { badge("B", .teal) }
                if setting.gain != 1 || setting.offset != 0 { badge("±", .purple) }
                if index == model.params.swapFirst || index == model.params.swapSecond { badge("⇄", .green) }
            }
            .font(.caption2.bold())
            .padding(3)
            .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 3))
            .foregroundStyle(.white)
            .padding(2)
        }
        .aspectRatio(16.0 / 9.0, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(RoundedRectangle(cornerRadius: 4).stroke(selected ? Color.accentColor : Color.gray.opacity(0.4), lineWidth: selected ? 3 : 1))
        .contentShape(Rectangle())
        .onTapGesture { model.params.selected = index }
        .contextMenu {
            Button(setting.mode == .mute ? "Stumm aus" : "Stumm") {
                model.params.channels[index].mode = setting.mode == .mute ? .normal : .mute
            }
            Button(model.params.solo == index ? "Solo aus" : "Solo") {
                model.params.solo = model.params.solo == index ? nil : index
            }
            Button(setting.mode == .invert ? "Invertieren aus" : "Invertieren") {
                model.params.channels[index].mode = setting.mode == .invert ? .normal : .invert
            }
            Button(setting.fromB ? "Wieder aus der Blende" : "Aus Bild B nehmen") {
                model.params.channels[index].fromB.toggle()
            }
            .disabled(!model.hasB)
            Divider()
            Button("Kanal zurücksetzen") {
                model.params.channels[index] = ChannelSetting()
                if model.params.solo == index { model.params.solo = nil }
            }
        }
    }

    private func badge(_ text: String, _ color: Color) -> some View {
        Text(text).foregroundStyle(color)
    }

    // MARK: Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                vaeSection
                slotSection(0)
                slotSection(1)
                blendSection
                noiseSection
                channelSection
                spaceSection
                decodeSection
                HStack {
                    Button("Alles zurücksetzen") { model.resetAll() }
                    Spacer()
                    Button("Ergebnis sichern …") { model.save() }
                        .disabled(model.output?.result == nil)
                }
                if let problem = model.problem {
                    Text(problem).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            .padding(12)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            content()
        }
    }

    private var vaeSection: some View {
        section("VAE") {
            Picker("Paar", selection: $model.pairID) {
                ForEach(model.pairs) { pair in
                    Text("\(pair.title) · \(pair.detail)").tag(Optional(pair.id))
                }
            }
            .labelsHidden()
            HStack {
                Spacer()
                Button { model.refreshPairs() } label: { Image(systemName: "arrow.clockwise") }
                    .help("VAE-Ordner neu lesen")
                Button { model.openFolder() } label: { Image(systemName: "folder") }
                    .help("VAE-Ordner im Finder zeigen")
            }
            Picker("Arbeitsgröße", selection: $model.workSize) {
                ForEach(model.latentSizes, id: \.self) { size in
                    Text("\(size.width * 8)×\(size.height * 8) (Latent \(size.width)×\(size.height))").tag(Optional(size))
                }
            }
            HStack(spacing: 6) {
                if model.working { ProgressView().controlSize(.small) }
                Text(model.status).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func slotSection(_ slot: Int) -> some View {
        let source = slot == 0 ? model.sourceA : model.sourceB
        return section(slot == 0 ? "Bild A" : "Bild B") {
            HStack {
                Menu(source?.name ?? "Keins") {
                    ForEach(session.films) { film in
                        Button(film.name) { model.choose(slot, film: film) }
                    }
                    if !session.films.isEmpty { Divider() }
                    Button("Film oder Bild wählen …") { model.pickFile(slot) }
                    if slot == 1 && source != nil {
                        Divider()
                        Button("Kein Bild B") { model.clearB() }
                    }
                }
            }
            if let source, source.isFilm, source.count > 1 {
                FrameSlider(index: source.index, count: source.count) { model.setFrame(slot, $0) }
            }
        }
    }

    private var blendSection: some View {
        section("Blende A → B") {
            HStack {
                Slider(value: $model.params.blend, in: 0...1)
                Text(String(format: "%3.0f %%", model.params.blend * 100)).monospacedDigit().frame(width: 44, alignment: .trailing)
            }
            .disabled(!model.hasB)
            HStack {
                Picker("Art", selection: $model.params.blendKind) {
                    ForEach(BlendKind.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Toggle("Pendeln", isOn: $model.animateBlend)
            }
            .disabled(!model.hasB)
            Text("Links unten steht die Pixel-Blende derselben Mischung zum Vergleich.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var noiseSection: some View {
        section("Rauschen") {
            HStack {
                Text("σ")
                Slider(value: $model.params.noise, in: 0...2)
                Text(String(format: "%.2f", model.params.noise)).monospacedDigit().frame(width: 36, alignment: .trailing)
            }
            Picker("Körnung", selection: $model.params.noiseGrain) {
                Text("fein").tag(1)
                Text("2").tag(2)
                Text("4").tag(4)
                Text("8").tag(8)
                Text("16").tag(16)
            }
            .pickerStyle(.segmented)
            HStack {
                Toggle("Nur Kanal \(model.params.selected)", isOn: $model.params.noiseOnlySelected)
                Spacer()
                Toggle("Wabern", isOn: $model.animateNoise)
                Button("Würfeln") { model.reroll() }
            }
        }
    }

    private var channelSection: some View {
        let index = model.params.selected
        let channel = model.channelBinding(index)
        return section("Kanal \(index)") {
            Picker("Modus", selection: channel.mode) {
                ForEach(ChannelMode.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            HStack {
                Text("Faktor").frame(width: 52, alignment: .leading)
                Slider(value: channel.gain, in: 0...4)
                Text(String(format: "%.2f", channel.wrappedValue.gain)).monospacedDigit().frame(width: 36, alignment: .trailing)
            }
            HStack {
                Text("Versatz").frame(width: 52, alignment: .leading)
                Slider(value: channel.offset, in: -3...3)
                Text(String(format: "%+.2f", channel.wrappedValue.offset)).monospacedDigit().frame(width: 36, alignment: .trailing)
            }
            HStack {
                Toggle("Solo", isOn: Binding(
                    get: { model.params.solo == index },
                    set: { model.params.solo = $0 ? index : nil }
                ))
                Toggle("Aus Bild B", isOn: channel.fromB)
                    .disabled(!model.hasB)
            }
            HStack {
                Text("Tauschen")
                Picker("Tauschen", selection: $model.params.swapFirst) {
                    Text("–").tag(Int?.none)
                    ForEach(0..<model.channelCount, id: \.self) { Text("\($0)").tag(Optional($0)) }
                }
                .labelsHidden()
                Text("⇄")
                Picker("mit", selection: $model.params.swapSecond) {
                    Text("–").tag(Int?.none)
                    ForEach(0..<model.channelCount, id: \.self) { Text("\($0)").tag(Optional($0)) }
                }
                .labelsHidden()
            }
            HStack {
                Text("Alle ×").frame(width: 52, alignment: .leading)
                Slider(value: $model.params.gainAll, in: 0...3)
                Text(String(format: "%.2f", model.params.gainAll)).monospacedDigit().frame(width: 36, alignment: .trailing)
            }
            Button("Alle Kanäle zurücksetzen") { model.resetChannels() }
        }
    }

    private var spaceSection: some View {
        section("Raum und Kompression") {
            HStack {
                Toggle("Spiegeln ↔", isOn: $model.params.flipH)
                Toggle("↕", isOn: $model.params.flipV)
            }
            Stepper("Verschieben x: \(model.params.shiftX)", value: $model.params.shiftX, in: -80...80)
            Stepper("Verschieben y: \(model.params.shiftY)", value: $model.params.shiftY, in: -45...45)
            Picker("Vergröbern", selection: $model.params.coarse) {
                Text("aus").tag(1)
                Text("2×2").tag(2)
                Text("4×4").tag(4)
                Text("8×8").tag(8)
            }
            .pickerStyle(.segmented)
            Picker("Bits pro Wert", selection: $model.params.bits) {
                Text("aus").tag(0)
                Text("6").tag(6)
                Text("4").tag(4)
                Text("3").tag(3)
                Text("2").tag(2)
                Text("1").tag(1)
            }
            .pickerStyle(.segmented)
        }
    }

    private var decodeSection: some View {
        section("Dekodieren") {
            Picker("Latent ziehen auf", selection: $model.params.decodeSize) {
                Text("Arbeitsgröße").tag(FrameSize?.none)
                ForEach(model.latentSizes.filter { $0 != model.workSize }, id: \.self) { size in
                    Text("\(size.width)×\(size.height) → \(size.width * 8)×\(size.height * 8)").tag(Optional(size))
                }
            }
            Picker("Interpolation", selection: $model.params.bilinear) {
                Text("Bilinear").tag(true)
                Text("Nearest").tag(false)
            }
            .pickerStyle(.segmented)
            .disabled(model.params.decodeSize == nil)
            Stepper("Round-Trips: \(model.params.roundTrips)", value: $model.params.roundTrips, in: 0...30)
            Text("Round-Trips kodieren das Ergebnis erneut und dekodieren es wieder, so sieht man die Drift.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// Frame slider that loads on release, with steps of one.
private struct FrameSlider: View {
    var index: Int
    var count: Int
    var commit: (Int) -> Void
    @State private var value: Double = 0
    @State private var editing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Slider(value: $value, in: 0...Double(max(count - 1, 1))) { active in
                editing = active
                if !active { commit(Int(value.rounded())) }
            }
            HStack {
                Button { commit(index - 1) } label: { Image(systemName: "chevron.left") }
                    .buttonStyle(.borderless)
                Text("Bild \(Int(value.rounded()) + 1) von \(count)").font(.caption).monospacedDigit()
                Button { commit(index + 1) } label: { Image(systemName: "chevron.right") }
                    .buttonStyle(.borderless)
            }
        }
        .onAppear { value = Double(index) }
        .onChange(of: index) { _, new in if !editing { value = Double(new) } }
    }
}
