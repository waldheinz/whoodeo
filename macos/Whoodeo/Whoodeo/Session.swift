import AppKit
import Combine
import CoreGraphics
import Darwin
import Foundation
import os
import UniformTypeIdentifiers

enum DisplayZoom: Hashable {
    case fit, full, double, percent(Int)
}

/// Logarithmic zoom. 100% sits in the middle, between 12.5% and 800%.
enum ZoomScale {
    static let minimum: CGFloat = 0.125
    static let maximum: CGFloat = 8

    static func position(for magnification: CGFloat) -> Double {
        let clamped = min(max(magnification, minimum), maximum)
        let span = log2(maximum / minimum)
        return Double(log2(clamped / minimum) / span)
    }

    static func magnification(for position: Double) -> CGFloat {
        let span = log2(maximum / minimum)
        let value = log2(minimum) + CGFloat(position) * span
        return min(max(pow(2, value), minimum), maximum)
    }

    /// Slider position of 100%. The system tick uses the same track as the thumb.
    static let hundred = position(for: 1)
}

@MainActor
final class Session: ObservableObject {
    nonisolated let objectWillChange = ObservableObjectPublisher()

    @Published var zoomToFit = true
    @Published var magnification: CGFloat = 1
    @Published private(set) var zoomPlacement = 0
    @Published var unit: Compute = .gpu
    @Published var codec: Codec = .hevc
    @Published var index = 0
    @Published var frameCount = 0
    @Published var inPoint = 0
    @Published var outPoint = 0
    @Published var playing = false
    @Published var busy = false
    @Published var exporting = false
    @Published var recon: CGImage?
    @Published var source: CGImage?
    @Published var pixelWidth = 0
    @Published var pixelHeight = 0
    @Published var milliseconds: Double?
    @Published var modelLabel = ""
    @Published var status = "Modell in der Seitenleiste wählen"
    @Published var failure: String?
    @Published var library: [LibraryModel] = []
    @Published var selection: URL?
    @Published var confirmDelete = false
    @Published var deleteName = ""
    @Published var loadingModel = false
    @Published private(set) var coverage = Coverage(count: 0)
    /// Newly scaled frames per second over the last two seconds. Zero when nothing is computing.
    @Published private(set) var scaleRate: Double = 0
    /// Idle fill of missing frames. Playback and export still store what they compute.
    @Published var fillCache: Bool {
        didSet {
            guard fillCache != oldValue else { return }
            UserDefaults.standard.set(fillCache, forKey: Self.fillCacheKey)
            if fillCache {
                resetFillHolds()
                Log.fill.info("fill an")
                scheduleFill()
            } else {
                Log.fill.info("fill aus")
            }
        }
    }

    private let engine = Engine()
    private var modelURL: URL?
    private var deleteTarget: URL?
    private var librarySource: DispatchSourceFileSystemObject?
    private var facts: FilmFacts?
    private var store: FrameStore?
    private var cache: [Int: Shown] = [:]
    private var cacheOrder: [Int] = []
    private var playGeneration = 0
    private var epoch = 0
    private var scrubbing = false
    private var filling: Int?
    /// Holes that failed once. One bad frame is skipped. Two failures in a row stop
    /// the walk until the next seek or until the switch is turned on again.
    private var fillSkipped: Set<Int> = []
    private var fillStrikes = 0
    private var fillHalted = false
    private var fillError: String?
    private var fillReportedDone = false
    private var fillAnnounced = false
    private var scaleInstants: [ContinuousClock.Instant] = []
    private var scaleSmooth: Double = 0
    private var scaleWatch: Task<Void, Never>?
    private var playFetching = false
    /// ProRes decode for the blue span. Independent of `playFetching`, which is a model frame.
    private var cacheDecoding = false
    private var playStalled = false
    private var playTask: Task<Void, Never>?
    /// Film time. Frame `playEpochFrame` was on screen at `playEpoch`.
    private var playEpoch: ContinuousClock.Instant?
    private var playEpochFrame = 0
    private var scrubTask: Task<Void, Never>?
    private var keys: Any?
    private var honorPrefer = true
    private var recentDrop: (url: URL, at: Date)?
    private var sliderHoldingHundred = false
    private static let fillCacheKey = "fillCache"
    private static let lastModelKey = "lastModel"
    private var didRestoreModel = false

    init() {
        let stored = UserDefaults.standard.object(forKey: Self.fillCacheKey) as? Bool
        fillCache = stored ?? true
        Session.current = self
        startLibraryWatch()
        if !AppDelegate.pendingFilms.isEmpty {
            let pending = AppDelegate.pendingFilms
            AppDelegate.pendingFilms = []
            _ = openFilms(pending)
        }
    }

    static weak var current: Session?

    /// Bild number currently being filled. Zero when the background walker is idle.
    var fillingBild: Int {
        guard let filling else { return 0 }
        return filling + 1
    }

    var cachedCount: Int { coverage.filled }

    /// The open film's frame rate. Zero when no film is open. The gauge calls this realtime.
    var realtimeRate: Double { facts?.fps ?? 0 }

    /// Count frames the model just produced. Cache hits are not scaling.
    func noteScaled(_ count: Int) {
        guard count > 0 else { return }
        let now = ContinuousClock.now
        for _ in 0..<count {
            scaleInstants.append(now)
        }
        publishScale(now)
        guard scaleWatch == nil else { return }
        scaleWatch = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                guard let self else { return }
                self.publishScale(.now)
                if self.scaleInstants.isEmpty { break }
            }
            self?.scaleWatch = nil
        }
    }

    private func publishScale(_ now: ContinuousClock.Instant) {
        scaleInstants.removeAll { now - $0 > .seconds(2) }
        let instant: Double
        if let first = scaleInstants.first, scaleInstants.count >= 2 {
            let span = Self.seconds(now - first)
            instant = span >= 0.25 ? Double(scaleInstants.count - 1) / span : scaleSmooth
        } else {
            instant = 0
        }
        if instant == 0 {
            scaleSmooth *= 0.5
        } else if scaleSmooth == 0 {
            scaleSmooth = instant
        } else {
            scaleSmooth += (instant - scaleSmooth) * 0.2
        }
        if scaleSmooth < 0.25 {
            scaleSmooth = 0
            if scaleRate != 0 { scaleRate = 0 }
            return
        }
        let shown = (scaleSmooth * 10).rounded() / 10
        // The tenth stays put until the smoothed rate has moved by half a frame per second.
        if scaleRate == 0 || abs(scaleSmooth - scaleRate) >= 0.5 {
            if shown != scaleRate { scaleRate = shown }
        }
    }

    private func resetScale() {
        scaleInstants.removeAll()
        scaleSmooth = 0
        scaleWatch?.cancel()
        scaleWatch = nil
        if scaleRate != 0 { scaleRate = 0 }
    }

    var backgroundFillError: String { fillError ?? "" }

    var scriptModelName: String {
        guard let selection, let model = library.first(where: { $0.url == selection }) else { return "" }
        return model.detail
    }

    /// POSIX path of the file the user opened. Empty when no film is open.
    var scriptFilmPath: String { facts?.url.path ?? "" }

    var scriptModelNames: [String] { library.map(\.detail) }

    /// Match the sidebar line, for example "Schritt 21250 · flexibel", or a package name.
    /// Returns a message when nothing unique matches.
    func chooseModel(named query: String) -> String? {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Der Modellname ist leer." }
        func exact(_ model: LibraryModel) -> Bool {
            model.detail.caseInsensitiveCompare(trimmed) == .orderedSame
                || model.url.lastPathComponent.caseInsensitiveCompare(trimmed) == .orderedSame
        }
        var hits = library.filter(exact)
        if hits.isEmpty {
            hits = library.filter {
                $0.detail.localizedCaseInsensitiveContains(trimmed)
                    || $0.url.lastPathComponent.localizedCaseInsensitiveContains(trimmed)
            }
        }
        guard hits.count == 1, let model = hits.first else {
            let names = (hits.isEmpty ? library : hits).map(\.detail).joined(separator: ", ")
            if hits.isEmpty { return "Kein Modell „\(trimmed)“. Vorhanden: \(names)" }
            return "Mehrere Modelle passen auf „\(trimmed)“: \(names)"
        }
        selection = model.url
        openSelection()
        return nil
    }

    /// Open a film from a path. Returns a message when the file cannot be opened.
    func openFilm(at path: String) -> String? {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return "Die Datei fehlt." }
        guard openFilms([url]) else { return "Das ist kein Film." }
        return nil
    }

    func play() {
        guard !playing else { return }
        togglePlay()
    }

    func pause() {
        guard playing else { return }
        finishPlay()
    }

    /// `hevc`, `prores`, or `prores4444`. Returns a message when the name is unknown.
    func setCodec(named name: String) -> String? {
        guard let parsed = Codec(rawValue: name) else {
            return "Codec ist hevc, prores oder prores4444."
        }
        codec = parsed
        return nil
    }

    /// `bild` is the number shown in the window, starting at 1.
    func setInBild(_ bild: Int) {
        guard frameCount > 0 else { return }
        inPoint = min(max(bild, 1), frameCount) - 1
        if inPoint > outPoint { outPoint = inPoint }
    }

    /// `bild` is the number shown in the window, starting at 1.
    func setOutBild(_ bild: Int) {
        guard frameCount > 0 else { return }
        outPoint = min(max(bild, 1), frameCount) - 1
        if outPoint < inPoint { inPoint = outPoint }
    }

    /// Start an export of the in/out range. Returns a message when it cannot start.
    /// The write continues on the engine queue. A file already at `path` is replaced.
    func exportFilm(to path: String) -> String? {
        guard modelURL != nil, facts != nil, frameCount > 0 else { return "Modell oder Film fehlt." }
        guard !exporting else { return "Ein Export läuft schon." }
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        var directory: ObjCBool = false
        let folder = url.deletingLastPathComponent().path
        guard FileManager.default.fileExists(atPath: folder, isDirectory: &directory), directory.boolValue else {
            return "Der Ordner fehlt."
        }
        startExport(url)
        return nil
    }

    /// `bild` is the number shown in the window, starting at 1.
    func seekToBild(_ bild: Int) {
        guard frameCount > 0 else { return }
        scrubTask?.cancel()
        scrubbing = false
        stopPlay()
        go(min(max(bild, 1), frameCount) - 1)
    }

    var canExport: Bool { modelURL != nil && facts != nil && !exporting && frameCount > 0 }

    var displayZoom: DisplayZoom {
        if zoomToFit { return .fit }
        let percent = Int((magnification * 100).rounded())
        if percent == 100 { return .full }
        if percent == 200 { return .double }
        return .percent(percent)
    }

    func chooseZoom(_ zoom: DisplayZoom) {
        sliderHoldingHundred = false
        switch zoom {
        case .fit:
            zoomToFit = true
        case .full:
            zoomToFit = false
            magnification = 1
        case .double:
            zoomToFit = false
            magnification = 2
        case .percent:
            return
        }
        zoomPlacement += 1
    }

    /// Slider position is logarithmic. Near 100% the thumb stays there until it is pulled clear.
    func setSliderZoom(_ position: Double) {
        let proposed = ZoomScale.magnification(for: position)
        let distance = abs(log2(proposed))
        let next: CGFloat
        if distance < 0.55 || (sliderHoldingHundred && distance < 1.05) {
            next = 1
            sliderHoldingHundred = true
        } else {
            next = proposed
            sliderHoldingHundred = false
        }
        if !zoomToFit, abs(magnification - next) < 0.0001 { return }
        zoomToFit = false
        magnification = next
    }

    func noteZoom(fits: Bool, magnification: CGFloat) {
        if zoomToFit == fits, abs(self.magnification - magnification) < 0.0001 { return }
        sliderHoldingHundred = false
        zoomToFit = fits
        self.magnification = magnification
    }

    var placeholder: String {
        if failure != nil { return failure! }
        if modelURL == nil && library.isEmpty {
            return "whoodeo-coreml legt neue Modelle in diese Sammlung. Vorhandene Pakete kannst du importieren."
        }
        if modelURL == nil { return "Wähle links ein Modell. Die Vorschau startet, sobald ein Film offen ist." }
        if facts == nil { return "Öffne einen Film oder leg ihn hier ab." }
        if busy { return "rechnet…" }
        return ""
    }

    func installKeys() {
        startLibraryWatch()
        if keys != nil { return }
        keys = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            if event.modifierFlags.contains(.command) { return event }
            switch event.keyCode {
            case 49:
                self.togglePlay()
                return nil
            case 123:
                self.step(-1)
                return nil
            case 124:
                self.step(1)
                return nil
            default:
                break
            }
            switch event.charactersIgnoringModifiers {
            case "i":
                self.markIn()
                return nil
            case "o":
                self.markOut()
                return nil
            default:
                return event
            }
        }
    }

    func importModel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.treatsFilePackagesAsDirectories = false
        panel.allowedContentTypes = [UTType(filenameExtension: "mlpackage") ?? .package]
        panel.prompt = "Importieren"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let dest = ModelLibrary.uniqueDestination(for: url)
        if dest.standardizedFileURL != url.standardizedFileURL {
            do {
                try FileManager.default.copyItem(at: url, to: dest)
            } catch {
                failure = error.localizedDescription
                status = failure!
                return
            }
        }
        reloadLibrary()
        selection = dest
        openSelection()
    }

    func openSelection() {
        guard let selection else { return }
        UserDefaults.standard.set(selection.lastPathComponent, forKey: Self.lastModelKey)
        if selection == modelURL { return }
        loadModel(selection)
    }

    func askDelete(_ url: URL? = nil) {
        guard let url = url ?? selection else { return }
        deleteTarget = url
        if let model = library.first(where: { $0.url == url }) {
            deleteName = "\(model.title), \(model.detail)"
        } else {
            deleteName = url.deletingPathExtension().lastPathComponent
        }
        confirmDelete = true
    }

    func deleteSelected() {
        guard let target = deleteTarget else { return }
        let removingCurrent = target == modelURL
        let removingSelection = target == selection
        NSWorkspace.shared.recycle([target]) { [weak self] _, error in
            let message = error?.localizedDescription
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let message {
                    self.failure = message
                    self.status = self.failure!
                    return
                }
                if removingCurrent { self.clearModel() }
                if removingSelection {
                    self.selection = nil
                    if UserDefaults.standard.string(forKey: Self.lastModelKey) == target.lastPathComponent {
                        UserDefaults.standard.removeObject(forKey: Self.lastModelKey)
                    }
                }
                self.deleteTarget = nil
                self.reloadLibrary()
            }
        }
    }

    func openFilms(_ urls: [URL]) -> Bool {
        guard let url = urls.first(where: FilmDrop.isFilm) else { return false }
        let key = url.standardizedFileURL
        if let recentDrop, recentDrop.url == key, Date().timeIntervalSince(recentDrop.at) < 0.5 {
            return true
        }
        recentDrop = (key, Date())
        loadFilm(url)
        return true
    }

    func pickFilm() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.allowedContentTypes = FilmDrop.contentTypes
        panel.prompt = "Öffnen"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        loadFilm(url)
    }

    func pickExport() {
        guard canExport else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.quickTimeMovie]
        panel.nameFieldStringValue = "recon.mov"
        panel.canCreateDirectories = true
        let accessory = NSHostingView(rootView: ExportAccessory(session: self))
        accessory.frame.size = accessory.fittingSize
        panel.accessoryView = accessory
        guard panel.runModal() == .OK, let url = panel.url else { return }
        startExport(url)
    }

    func scrub(_ next: Int) {
        scrubTask?.cancel()
        scrubbing = false
        stopPlay()
        go(next)
    }

    /// Move the playhead. A cached frame is shown at once. Inference waits until the thumb rests.
    func preview(_ next: Int) {
        guard frameCount > 0 else { return }
        stopPlay()
        scrubbing = true
        filling = nil
        index = min(max(next, 0), frameCount - 1)
        if let shown = cache[index], shown.recon != nil {
            apply(shown)
        } else {
            let token = epoch
            let wanted = index
            engine.request(wanted, store: store, need: .cached) { [weak self] result in
                Task { @MainActor in
                    guard let self, token == self.epoch, self.scrubbing, self.index == wanted else { return }
                    guard case .success(let shown) = result, shown.recon != nil else { return }
                    self.remember(shown)
                    self.apply(shown)
                }
            }
        }
        scrubTask?.cancel()
        scrubTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            guard !Task.isCancelled else { return }
            self.endPreview()
        }
    }

    func endPreview() {
        scrubTask?.cancel()
        scrubTask = nil
        guard scrubbing else { return }
        scrubbing = false
        showCurrent()
    }

    func step(_ delta: Int) {
        scrubTask?.cancel()
        scrubbing = false
        stopPlay()
        go(index + delta)
    }

    func togglePlay() {
        guard frameCount > 0 else { return }
        if playing {
            finishPlay()
            return
        }
        scrubbing = false
        scrubTask?.cancel()
        playing = true
        playGeneration += 1
        playStalled = false
        playEpoch = .now
        playEpochFrame = index
        source = nil
        Log.play.info("Wiedergabe ab Bild \(self.index + 1, privacy: .public)")
        playClock(playGeneration)
    }

    /// The original for the frame on screen, built when the mouse button goes down.
    func originalNow() -> CGImage? {
        guard pixelWidth > 0, pixelHeight > 0 else { return nil }
        var image: CGImage?
        let gate = DispatchSemaphore(value: 0)
        engine.original(index: index, width: pixelWidth, height: pixelHeight) {
            image = $0
            gate.signal()
        }
        gate.wait()
        return image
    }

    func markIn() {
        inPoint = index
        if inPoint > outPoint { outPoint = inPoint }
    }

    func markOut() {
        outPoint = index
        if outPoint < inPoint { inPoint = outPoint }
    }

    func changeUnit(_ next: Compute) {
        honorPrefer = false
        unit = next
        guard let modelURL else { return }
        loadingModel = true
        failure = nil
        status = "lädt \(next.title)"
        releaseArchive()
        engine.setUnit(next, url: modelURL) { [weak self] result in
            Task { @MainActor in
                self?.applyModel(result)
            }
        }
    }

    private func loadModel(_ url: URL) {
        honorPrefer = true
        loadingModel = true
        modelURL = url
        failure = nil
        status = "lädt \(url.lastPathComponent)"
        releaseArchive()
        engine.openModel(url, unit: unit) { [weak self] result in
            Task { @MainActor in
                self?.applyModel(result)
            }
        }
    }

    private func applyModel(_ result: Result<PackageInfo, Error>) {
        switch result {
        case .success(let info):
            modelLabel = info.label
            if honorPrefer && info.prefer != unit, let modelURL {
                honorPrefer = false
                unit = info.prefer
                loadingModel = true
                status = "lädt \(info.prefer.title)"
                engine.setUnit(info.prefer, url: modelURL) { [weak self] result in
                    Task { @MainActor in self?.applyModel(result) }
                }
                return
            }
            loadingModel = false
            failure = nil
            if facts == nil {
                status = "Film öffnen"
            }
            showCurrent()
        case .failure(let error):
            loadingModel = false
            failure = error.localizedDescription
            status = failure!
        }
    }

    private func loadFilm(_ url: URL) {
        releaseArchive()
        busy = true
        failure = nil
        frameCount = 0
        coverage = Coverage(count: 0)
        status = "öffnet \(url.lastPathComponent)"
        engine.openFilm(url) { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let facts):
                    self.facts = facts
                    self.frameCount = facts.count
                    self.index = 0
                    self.inPoint = 0
                    self.outPoint = max(facts.count - 1, 0)
                    self.busy = false
                    self.failure = nil
                    self.status = "\(facts.width)×\(facts.height) · \(facts.count) Bilder"
                    self.showCurrent()
                case .failure(let error):
                    self.busy = false
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
            }
        }
    }

    private func startLibraryWatch() {
        reloadLibrary()
        if librarySource != nil { return }
        let fd = Darwin.open(ModelLibrary.directory.path, O_EVTONLY)
        guard fd >= 0 else { return }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: [.write, .rename, .delete],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            self?.reloadLibrary()
        }
        source.setCancelHandler { Darwin.close(fd) }
        source.resume()
        librarySource = source
    }

    private func reloadLibrary() {
        library = ModelLibrary.models()
        if let selection, !library.contains(where: { $0.url == selection }) {
            self.selection = nil
            if selection == modelURL { clearModel() }
        }
        guard !didRestoreModel else { return }
        didRestoreModel = true
        guard selection == nil,
              let name = UserDefaults.standard.string(forKey: Self.lastModelKey),
              let match = library.first(where: { $0.url.lastPathComponent == name })
        else { return }
        selection = match.url
        openSelection()
    }

    private func clearModel() {
        loadingModel = false
        modelURL = nil
        modelLabel = ""
        pixelWidth = 0
        pixelHeight = 0
        releaseArchive()
        failure = nil
        status = facts == nil ? "Modell in der Seitenleiste wählen" : status
    }

    /// Drop the pictures on screen. The archive on disk stays with its model.
    private func releaseArchive() {
        epoch += 1
        stopPlay()
        scrubTask?.cancel()
        scrubbing = false
        filling = nil
        resetFillHolds()
        store = nil
        cache.removeAll()
        cacheOrder.removeAll()
        recon = nil
        source = nil
        milliseconds = nil
        coverage = Coverage(count: frameCount)
        resetScale()
    }

    private func adoptStore() {
        guard store == nil, let facts, let modelURL, facts.count > 0 else { return }
        do {
            let opened = try FrameStore.open(film: facts.url, model: modelURL, unit: unit, count: facts.count)
            store = opened
            coverage = opened.coverage()
        } catch {
            failure = error.localizedDescription
            status = failure!
        }
    }

    private func showCurrent() {
        guard modelURL != nil, facts != nil else { return }
        resetFillHolds()
        adoptStore()
        if let shown = cache[index], shown.recon != nil {
            apply(shown)
            scheduleFill()
            return
        }
        busy = true
        if recon == nil { status = "rechnet…" }
        let wanted = index
        let token = epoch
        filling = nil
        engine.request(wanted, store: store, need: .display) { [weak self] result in
            Task { @MainActor in
                guard let self, token == self.epoch else { return }
                self.deliver(result)
            }
        }
    }

    private func go(_ next: Int) {
        guard frameCount > 0 else { return }
        index = min(max(next, 0), frameCount - 1)
        showCurrent()
    }

    private func stopPlay() {
        playing = false
        playGeneration += 1
        playTask?.cancel()
        playTask = nil
        playEpoch = nil
        playFetching = false
        cacheDecoding = false
        playStalled = false
    }

    private func finishPlay() {
        stopPlay()
        Log.play.info("Pause bei Bild \(self.index + 1, privacy: .public)")
        if let shown = cache[index], shown.recon != nil {
            apply(shown)
        }
        attachSource()
        scheduleFill()
    }

    /// Decoded pictures kept ahead of the playhead. One model frame fits inside this cushion.
    private static let playAheadTarget = 12

    /// Show cached frames on film time. A late picture jumps to the frame that is due,
    /// instead of stretching every following interval by the lateness.
    private func playClock(_ generation: Int) {
        guard playing, generation == playGeneration else { return }
        schedulePlayAhead()
        let fps = max(facts?.fps ?? 25, 1)
        let next = index + 1
        guard next < frameCount else {
            finishPlay()
            return
        }
        if playEpoch == nil {
            playEpoch = .now
            playEpochFrame = index
        }
        let due = playEpoch!.advanced(by: .seconds(Double(next - playEpochFrame) / fps))
        playTask = Task { @MainActor [weak self] in
            if due > ContinuousClock.now {
                try? await Task.sleep(until: due, clock: .continuous)
            }
            guard let self, !Task.isCancelled, self.playing, generation == self.playGeneration else { return }
            self.showDueFrames(generation, fps: fps)
        }
    }

    private func showDueFrames(_ generation: Int, fps: Double) {
        guard playing, generation == playGeneration, let epoch = playEpoch else { return }
        let elapsed = Self.seconds(ContinuousClock.now - epoch)
        let dueIndex = playEpochFrame + Int((elapsed * fps).rounded(.down))
        let target = min(frameCount - 1, max(index + 1, dueIndex))
        var show = index
        var frame = index + 1
        while frame <= target {
            guard cache[frame]?.recon != nil else { break }
            show = frame
            frame += 1
        }
        guard show != index, let shown = cache[show] else {
            let missing = index + 1
            if !playStalled {
                Log.play.info("Wiedergabe wartet auf Bild \(missing, privacy: .public)")
            }
            playStalled = true
            if missing < frameCount, !coverage.contains(missing) { status = "rechnet…" }
            schedulePlayAhead()
            return
        }
        if playStalled {
            Log.play.info("Wiedergabe weiter bei Bild \(show + 1, privacy: .public)")
        }
        playStalled = false
        index = show
        applyPlay(shown)
        playClock(generation)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    /// Cached frames already decoded after the playhead.
    private func playBufferReady() -> Int {
        var ready = 0
        var frame = index + 1
        while ready < Self.playAheadTarget, frame < frameCount, cache[frame]?.recon != nil {
            ready += 1
            frame += 1
        }
        return ready
    }

    /// Next cached frame in the cushion that is not decoded yet. The immediate next hole is included.
    private func nextPlayDecode() -> Int? {
        for step in 1...Self.playAheadTarget {
            let candidate = index + step
            if candidate >= frameCount { return nil }
            if cache[candidate]?.recon != nil { continue }
            if step > 1, !coverage.contains(candidate) { return nil }
            return candidate
        }
        return nil
    }

    private func schedulePlayAhead() {
        guard playing else { return }
        if playBufferReady() < Self.playAheadTarget, let target = nextPlayDecode() {
            if coverage.contains(target) {
                if !cacheDecoding { beginCacheDecode(target) }
            } else if !playFetching, filling == nil {
                // The frame under the playhead is still gray. One model, not a second fill of the same hole.
                beginPlayFetch(target)
                return
            }
        }
        // One model frame takes longer than eight film frames, so a thinner cushion would stutter.
        // Cache decode does not take the model queue, so fill keeps running beside it.
        guard playBufferReady() >= 8, !playFetching, filling == nil else { return }
        scheduleFill(allowWhilePlaying: true)
    }

    /// Cached frames ahead of the playhead. The model queue is left to fill.
    private func beginCacheDecode(_ target: Int) {
        guard let store else { return }
        cacheDecoding = true
        let generation = playGeneration
        let token = epoch
        engine.decodeCached(from: target, store: store, limit: Self.playAheadTarget) { [weak self] result in
            Task { @MainActor in
                guard let self, generation == self.playGeneration, token == self.epoch else { return }
                self.cacheDecoding = false
                guard self.playing else { return }
                switch result {
                case .success(let frames):
                    guard let first = frames.first, first.recon != nil else { return }
                    for frame in frames {
                        var copy = frame
                        copy.ahead = []
                        self.remember(copy)
                    }
                    // The clock stops when the next picture is not ready yet, and nothing else
                    // starts it. A late decode has to pick the timeline back up.
                    if self.playStalled, let next = frames.first(where: { $0.index == self.index + 1 }) {
                        self.playStalled = false
                        self.index = next.index
                        self.playEpoch = .now
                        self.playEpochFrame = next.index
                        self.applyPlay(next)
                        self.playClock(generation)
                        return
                    }
                    self.schedulePlayAhead()
                case .failure(let error):
                    Log.play.error("Wiedergabe liest Bild \(target + 1, privacy: .public): \(error.localizedDescription, privacy: .public)")
                    guard target == self.index + 1 else { return }
                    self.stopPlay()
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
            }
        }
    }

    private func beginPlayFetch(_ target: Int) {
        playFetching = true
        let generation = playGeneration
        let token = epoch
        engine.request(target, store: store, need: .play, run: Self.playAheadTarget) { [weak self] result in
            Task { @MainActor in
                guard let self, generation == self.playGeneration else { return }
                self.playFetching = false
                guard token == self.epoch, self.playing else { return }
                switch result {
                case .success(let shown):
                    var frames = [shown]
                    frames.append(contentsOf: shown.ahead)
                    var fresh = 0
                    for frame in frames {
                        if frame.stored { self.coverage.insert(frame.index) }
                        if !frame.fromCache { fresh += 1 }
                    }
                    self.noteScaled(fresh)
                    guard shown.recon != nil else {
                        self.schedulePlayAhead()
                        return
                    }
                    for frame in frames {
                        var copy = frame
                        copy.ahead = []
                        self.remember(copy)
                    }
                    if self.playStalled, shown.index == self.index + 1 {
                        self.playStalled = false
                        self.index = shown.index
                        self.playEpoch = .now
                        self.playEpochFrame = shown.index
                        self.applyPlay(shown)
                        self.playClock(generation)
                        return
                    }
                    self.schedulePlayAhead()
                case .failure(let error):
                    guard target == self.index + 1 else {
                        self.schedulePlayAhead()
                        return
                    }
                    self.stopPlay()
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
            }
        }
    }

    private func applyPlay(_ shown: Shown) {
        recon = shown.recon?.image
        if source != nil { source = nil }
        if pixelWidth != shown.width { pixelWidth = shown.width }
        if pixelHeight != shown.height { pixelHeight = shown.height }
        failure = nil
        let label = shown.fromCache ? "Wiedergabe · Cache" : "Wiedergabe · \(unit.title)"
        if status != label { status = label }
    }

    private func attachSource() {
        let wanted = index
        let token = epoch
        let width = pixelWidth
        let height = pixelHeight
        guard width > 0, height > 0 else { return }
        engine.original(index: wanted, width: width, height: height) { [weak self] image in
            Task { @MainActor in
                guard let self, token == self.epoch, self.index == wanted, !self.playing else { return }
                self.source = image
            }
        }
    }

    private func deliver(_ result: Result<Shown, Error>) {
        busy = false
        switch result {
        case .success(let shown):
            if shown.stored { coverage.insert(shown.index) }
            if !shown.fromCache { noteScaled(1) }
            guard shown.recon != nil else {
                scheduleFill()
                return
            }
            remember(shown)
            if shown.index == index {
                apply(shown)
                if !playing { scheduleFill() }
            }
        case .failure(let error):
            playing = false
            failure = error.localizedDescription
            status = failure!
        }
    }

    private func apply(_ shown: Shown) {
        recon = shown.recon?.image
        source = shown.source?.image
        pixelWidth = shown.width
        pixelHeight = shown.height
        milliseconds = shown.milliseconds
        failure = nil
        if shown.fromCache {
            status = "Bild \(shown.index + 1) von \(frameCount) · Cache · \(unit.title)"
        } else {
            let timing = String(format: "%.0f ms", shown.milliseconds)
            status = "Bild \(shown.index + 1) von \(frameCount) · \(timing) · \(unit.title)"
        }
    }

    private func resetFillHolds() {
        fillSkipped.removeAll()
        fillStrikes = 0
        fillHalted = false
        fillError = nil
        fillReportedDone = false
        fillAnnounced = false
    }

    /// The next hole to the right of the slider, then holes in front of it.
    /// A frame that already failed is left out until the next seek.
    private func nextFillIndex() -> Int? {
        guard var hole = coverage.nextMissing(after: index) else { return nil }
        var steps = 0
        while fillSkipped.contains(hole) {
            steps += 1
            if steps > fillSkipped.count { return nil }
            guard let later = coverage.nextMissing(after: hole), later != hole else { return nil }
            hole = later
        }
        return hole
    }

    private func scheduleFill(allowWhilePlaying: Bool = false) {
        guard fillCache else { return }
        guard !fillHalted else { return }
        if filling != nil || playFetching || exporting || scrubbing || busy { return }
        if playing && !allowWhilePlaying { return }
        guard modelURL != nil, let next = nextFillIndex() else {
            if !fillReportedDone {
                fillReportedDone = true
                if fillSkipped.isEmpty {
                    Log.fill.info("fill fertig")
                } else {
                    Log.fill.info("fill überspringt \(self.fillSkipped.count, privacy: .public) Bilder, sonst nichts offen")
                }
            }
            return
        }
        let token = epoch
        filling = next
        fillReportedDone = false
        if fillAnnounced {
            Log.fill.debug("fill Bild \(next + 1, privacy: .public)")
        } else {
            fillAnnounced = true
            Log.fill.info("fill startet Bild \(next + 1, privacy: .public)")
        }
        engine.request(next, store: store, need: .fill) { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                if self.filling == next { self.filling = nil }
                guard token == self.epoch else { return }
                switch result {
                case .success(let shown):
                    self.fillStrikes = 0
                    self.fillError = nil
                    if shown.stored { self.coverage.insert(shown.index) }
                    if !shown.fromCache { self.noteScaled(1) }
                    Log.fill.debug("fill speichert Bild \(shown.index + 1, privacy: .public)")
                    if self.playing {
                        self.schedulePlayAhead()
                        return
                    }
                    guard !self.scrubbing, !self.exporting, !self.busy else { return }
                    self.scheduleFill()
                case .failure(let error):
                    if error is CancellationError {
                        Log.fill.info("fill Bild \(next + 1, privacy: .public) verdrängt")
                        if self.playing { self.schedulePlayAhead() } else { self.scheduleFill() }
                        return
                    }
                    let text = error.localizedDescription
                    self.fillError = text
                    self.fillSkipped.insert(next)
                    self.fillStrikes += 1
                    Log.fill.error("fill Bild \(next + 1, privacy: .public): \(text, privacy: .public)")
                    if self.fillStrikes >= 2 {
                        self.fillHalted = true
                        self.fillAnnounced = false
                        Log.fill.error("fill steht nach zwei Fehlern hintereinander")
                        return
                    }
                    if self.playing { self.schedulePlayAhead() } else { self.scheduleFill() }
                }
            }
        }
    }

    private func remember(_ shown: Shown) {
        guard shown.recon != nil else { return }
        if cache[shown.index] == nil {
            cacheOrder.append(shown.index)
        }
        cache[shown.index] = shown
        while cacheOrder.count > 16 {
            var farthest = 0
            var distance = -1
            for (offset, cached) in cacheOrder.enumerated() where cached != index {
                // Keep the cushion in front of the playhead. Frames behind go first.
                let gap = cached < index ? abs(cached - index) + frameCount : cached - index
                if gap > distance {
                    distance = gap
                    farthest = offset
                }
            }
            let dropped = cacheOrder.remove(at: farthest)
            cache.removeValue(forKey: dropped)
        }
    }

    private func startExport(_ url: URL) {
        stopPlay()
        scrubTask?.cancel()
        scrubbing = false
        filling = nil
        exporting = true
        failure = nil
        adoptStore()
        guard let store else {
            exporting = false
            failure = "Cache fehlt"
            status = failure!
            return
        }
        let range = inPoint..<(outPoint + 1)
        let token = epoch
        status = "exportiert 0 / \(range.count)"
        engine.export(url: url, codec: codec, range: range, store: store, progress: { [weak self] done, total in
            Task { @MainActor in
                self?.status = "exportiert \(done) / \(total)"
            }
        }, stored: { [weak self] index in
            Task { @MainActor in
                guard let self, token == self.epoch else { return }
                self.coverage.insert(index)
                self.noteScaled(1)
            }
        }, done: { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                self.exporting = false
                switch result {
                case .success:
                    self.failure = nil
                    self.status = "geschrieben \(url.lastPathComponent)"
                case .failure(let error):
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
                self.scheduleFill()
            }
        })
    }

    func cancelExport() {
        engine.cancelExportNow()
    }
}

import SwiftUI

struct ExportAccessory: View {
    @ObservedObject var session: Session

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Codec", selection: $session.codec) {
                ForEach(Codec.allCases) { codec in
                    Text(codec.title).tag(codec)
                }
            }
            .pickerStyle(.radioGroup)
            Text("Bilder \(session.inPoint + 1)–\(session.outPoint + 1)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(width: 240)
    }
}
