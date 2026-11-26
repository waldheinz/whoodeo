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
    @Published var zoomToFit = true
    @Published var magnification: CGFloat = 1
    @Published private(set) var zoomPlacement = 0
    @Published var codec: Codec = .hevc
    /// Codec condition of a latent_up with FiLM. Automatic follows the open film.
    @Published private(set) var codecChoice: CodecChoice = .auto
    /// The codec of the open film, from its picture stream. Nil without a film.
    @Published private(set) var filmCodec: SourceCodec?
    /// The codec the loaded package was exported for. Nil for a package without FiLM.
    @Published private(set) var activeCodec: SourceCodec?
    @Published var index = 0
    @Published var frameCount = 0
    @Published var inPoint = 0
    @Published var outPoint = 0
    @Published var playing = false
    @Published var busy = false
    @Published var exporting = false
    /// A frame of this film and model is on screen.
    @Published private(set) var hasPicture = false
    @Published var pixelWidth = 0
    @Published var pixelHeight = 0
    @Published var milliseconds: Double?
    @Published var modelLabel = ""
    @Published var status = "Modell in der Seitenleiste wählen"
    @Published var failure: String?
    @Published var library: [LibraryModel] = []
    @Published var selection: URL?
    /// Sidebar line whose upscale a click into a still shows instead of the original. Never the main model.
    @Published private(set) var comparison: URL?
    /// What the mouse button or a script asks for. The upscale when nothing is pressed.
    @Published private(set) var wantedSource = ScreenSource.upscale
    /// What the picture shows right now. A comparison frame that is still computing leaves the upscale.
    @Published private(set) var screenSource = ScreenSource.upscale
    /// Frame the comparison model is computing, nil when it is idle.
    @Published private(set) var comparisonComputing: Int?
    /// The comparison model package is loading.
    @Published private(set) var comparisonLoading = false
    @Published private(set) var comparisonError: String?
    @Published var confirmDelete = false
    @Published var deleteName = ""
    @Published var confirmRemove = false
    @Published var removeName = ""
    @Published private(set) var films: [OpenFilm] = []
    @Published var filmSelection: String?
    @Published var loadingModel = false
    @Published private(set) var coverage = Coverage(count: 0)
    /// Newly scaled frames per second. Zero when nothing is computing.
    /// Slow film stays visible: the pace follows the gap between frames, not a two-second window.
    @Published private(set) var scaleRate: Double = 0
    /// Preprocessing of the open film, as chosen in the pipeline panel.
    @Published private(set) var preprocess = PreprocessConfig.none
    /// The scan for repeated frames: 0…1 while it runs, nil otherwise.
    @Published private(set) var decimateProgress: Double?
    /// What automatic found, or why the scan stopped. Empty when there is nothing to say.
    @Published private(set) var decimateNote = ""
    /// Light and color after the upscaler, per film. The screen shows it at once; an export renders it in.
    @Published var grade = ColorGrade.neutral {
        didSet {
            guard grade != oldValue else { return }
            stage?.screenIfCreated()?.setGrade(grade)
            saveGradeSoon()
        }
    }
    /// The auto button is sampling the film.
    @Published private(set) var estimatingGrade = false
    /// Why the background fill waits for the cache: limit reached or folder not reachable. Nil otherwise.
    @Published private(set) var cacheNote: String?
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
    /// Length of the loaded package's condition input, 0 without one.
    private var conditionSize = 0
    private var deleteTarget: URL?
    private var librarySource: DispatchSourceFileSystemObject?
    /// The open film after preprocessing: `count` and `fps` as shown, `sourceCount` and `sourceFps` of the file.
    @Published private(set) var facts: FilmFacts?
    /// Which source frames the open film keeps. Nil when it keeps all of them.
    private var frameMap: FrameMap?
    private var decimateToken = 0
    private var decimateCancel: OSAllocatedUnfairLock<Bool>?
    private var gradeSave: Task<Void, Never>?
    /// The app is quitting; nothing new goes to the engines.
    private var quitting = false
    /// The cache folder is being moved or switched; no archive opens until it is done.
    private var cacheSuspended = false
    private var cacheObserver: NSObjectProtocol?
    private var store: FrameStore?
    /// The open film's originals for the screen. Nil without a film.
    private var originals: Originals?
    /// The comparison model's archive for the open film and codec. Opened on first use.
    private var comparisonStore: FrameStore?
    private var comparisonOpened = false
    /// Runs the comparison model, one frame on request. Created on the first missing frame.
    private var comparisonEngine: Engine?
    private var comparisonPackage: URL?
    /// Counts comparison resets. A result from before the last reset is dropped.
    private var comparisonEpoch = 0
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
    private var scalePace = ScalePace()
    private var scaleOrigin: ContinuousClock.Instant?
    private var scaleWatch: Task<Void, Never>?
    private var playFetching = false
    private var scrubTask: Task<Void, Never>?
    private var keys: Any?
    private var recentDrop: (url: URL, at: Date)?
    private var sliderHoldingHundred = false
    private var filmList = FilmList.empty
    /// Digest of the film on screen. Nil when the viewer has no film.
    private var shownFilmID: String?
    /// The picture view. Its screen shows every frame.
    private weak var stage: PictureStage?
    private var removeTarget: String?
    private var loadingFilmID: String?
    private static let fillCacheKey = "fillCache"
    private static let lastModelKey = "lastModel"
    private var didRestoreModel = false
    /// Counts model loads. A load that finishes after a newer one started is dropped.
    private var modelLoad = 0

    init() {
        let stored = UserDefaults.standard.object(forKey: Self.fillCacheKey) as? Bool
        fillCache = stored ?? true
        Session.current = self
        cacheObserver = NotificationCenter.default.addObserver(forName: CacheSettings.changed, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.cacheSettingsChanged() }
        }
        CacheUsage.shared.refresh()
        if !CacheSettings.isAvailable {
            cacheNote = "Cache-Ordner nicht erreichbar"
            Log.fill.error("Cache-Ordner nicht erreichbar: \(CacheSettings.root.path, privacy: .public)")
        }
        reloadFilms()
        DispatchQueue.global(qos: .utility).async {
            FrameStore.pruneCaches { id in
                FilmLibrary.load().films.contains { $0.id == id }
            }
        }
        startLibraryWatch()
        if !AppDelegate.pendingFilms.isEmpty {
            let pending = AppDelegate.pendingFilms
            AppDelegate.pendingFilms = []
            _ = openFilms(pending)
        } else if let film = filmList.films.first(where: { $0.id == filmList.current }) {
            filmSelection = film.id
            if FileManager.default.fileExists(atPath: film.path) {
                loadFilm(URL(fileURLWithPath: film.path), id: film.id)
            }
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

    /// Frames the running job still has to scale. Zero when fill and export are idle.
    var scaleRemaining: Int {
        guard scaleRate > 0 else { return 0 }
        if exporting {
            return coverage.missing(from: inPoint, until: outPoint + 1)
        }
        guard fillCache, !fillHalted else { return 0 }
        return max(0, frameCount - coverage.filled - fillSkipped.count)
    }

    /// Count frames the model just produced. Cache hits are not scaling.
    func noteScaled(_ count: Int) {
        guard count > 0 else { return }
        let now = ContinuousClock.now
        if scaleOrigin == nil { scaleOrigin = now }
        scalePace.add(count, at: Self.seconds(now - scaleOrigin!))
        if scaleRate != scalePace.shown { scaleRate = scalePace.shown }
        guard scaleWatch == nil else { return }
        scaleWatch = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                guard let self else { return }
                self.publishScale(.now)
                if self.scalePace.isIdle { break }
            }
            self?.scaleWatch = nil
        }
    }

    private func publishScale(_ now: ContinuousClock.Instant) {
        guard let origin = scaleOrigin else { return }
        scalePace.revise(at: Self.seconds(now - origin))
        if scaleRate != scalePace.shown { scaleRate = scalePace.shown }
    }

    private func resetScale() {
        scalePace.reset()
        scaleOrigin = nil
        scaleWatch?.cancel()
        scaleWatch = nil
        if scaleRate != 0 { scaleRate = 0 }
    }

    var backgroundFillError: String { fillError ?? cacheNote ?? "" }

    var scriptModelName: String {
        guard let selection, let model = library.first(where: { $0.url == selection }) else { return "" }
        return model.detail
    }

    /// ID of the film on screen. Empty when none is shown.
    var scriptCurrentFilmID: String { shownFilmID ?? "" }

    /// Name of the film on screen without its extension, for the window title. Nil when none is shown.
    var filmTitle: String? {
        guard let shownFilmID, let film = films.first(where: { $0.id == shownFilmID }) else { return nil }
        return URL(fileURLWithPath: film.path).deletingPathExtension().lastPathComponent
    }

    func scriptFilmRecords() -> [OpenFilm] { films }

    var scriptModelNames: [String] { library.map(\.detail) }

    /// Match the sidebar line, for example "Schritt 21250 · flexibel", or a package name.
    /// Returns a message when nothing unique matches.
    func chooseModel(named query: String) -> String? {
        switch matchModel(query) {
        case .failure(let problem):
            return problem.message
        case .success(let model):
            selection = model.url
            openSelection()
            return nil
        }
    }

    /// Sidebar line of the comparison model, empty when there is none.
    var scriptComparisonName: String {
        guard let comparison else { return "" }
        return library.first { $0.url == comparison }?.detail ?? ""
    }

    /// Short name of the comparison model for the note over the picture, for example "latent_up 12×128 · Schritt 6500".
    var comparisonLabel: String {
        guard let comparison, let model = library.first(where: { $0.url == comparison }) else { return "" }
        let step = model.detail.components(separatedBy: " · ").first ?? model.detail
        return "\(model.title) · \(step)"
    }

    /// Like `chooseModel`. An empty name clears the comparison.
    func chooseComparison(named query: String) -> String? {
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            setComparison(nil)
            return nil
        }
        switch matchModel(query) {
        case .failure(let problem):
            return problem.message
        case .success(let model):
            guard model.url != selection else { return "Das Hauptmodell kann nicht zugleich Vergleich sein." }
            setComparison(model.url)
            return nil
        }
    }

    private struct ModelMismatch: Error {
        let message: String
    }

    private func matchModel(_ query: String) -> Result<LibraryModel, ModelMismatch> {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .failure(ModelMismatch(message: "Der Modellname ist leer.")) }
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
            if hits.isEmpty { return .failure(ModelMismatch(message: "Kein Modell „\(trimmed)“. Vorhanden: \(names)")) }
            return .failure(ModelMismatch(message: "Mehrere Modelle passen auf „\(trimmed)“: \(names)"))
        }
        return .success(model)
    }

    /// Add a film and show it. The same path keeps its id. Nil means it was not added; `failure` says why.
    @discardableResult
    func addFilm(at path: String) -> String? {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        let id: String
        do {
            id = try FilmLibrary.mutate { list in
                try FilmLibrary.insert(url, into: &list)
            }
        } catch {
            let text = error.localizedDescription
            failure = text
            status = text
            return nil
        }
        reloadFilms()
        if let problem = showFilm(id: id) {
            failure = problem
            status = problem
            return nil
        }
        return id
    }

    /// Show a film that is already in the list. Does not add or remove.
    func showFilm(id: String) -> String? {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Die Film-ID ist leer." }
        var known = false
        do {
            try FilmLibrary.mutate { list in
                guard list.films.contains(where: { $0.id == trimmed }) else { return }
                known = true
                list.current = trimmed
            }
        } catch {
            let text = error.localizedDescription
            failure = text
            status = text
            return text
        }
        guard known else { return "Kein Film mit dieser ID." }
        reloadFilms()
        filmSelection = trimmed
        guard let film = films.first(where: { $0.id == trimmed }) else { return "Kein Film mit dieser ID." }
        if shownFilmID == trimmed, facts != nil || busy { return nil }
        loadFilm(URL(fileURLWithPath: film.path), id: trimmed)
        return nil
    }

    /// Drop one film and delete its cache. The movie file stays. A message means it is still in the list.
    func removeFilm(id: String) -> String? {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard FilmLibrary.load().films.contains(where: { $0.id == trimmed }) else {
            return "Kein Film mit dieser ID."
        }
        if exporting && shownFilmID == trimmed { return "Ein Export läuft schon." }
        if shownFilmID == trimmed {
            closeShownFilm()
            let gate = DispatchSemaphore(value: 0)
            engine.settle { gate.signal() }
            gate.wait()
        }
        do {
            try FrameStore.removeCache(id: trimmed)
            try FilmLibrary.mutate { list in
                list.films.removeAll { $0.id == trimmed }
                if list.current == trimmed { list.current = "" }
            }
        } catch {
            reloadFilms()
            let text = error.localizedDescription
            failure = text
            status = text
            return text
        }
        reloadFilms()
        if filmSelection == trimmed { filmSelection = nil }
        return nil
    }

    func askRemoveFilm(_ id: String? = nil) {
        guard let id = id ?? filmSelection else { return }
        guard let film = films.first(where: { $0.id == id }) else { return }
        if exporting && shownFilmID == id {
            failure = "Ein Export läuft schon."
            status = failure!
            return
        }
        removeTarget = id
        removeName = film.name
        confirmRemove = true
    }

    func removeSelectedFilm() {
        guard let id = removeTarget else { return }
        removeTarget = nil
        if let problem = removeFilm(id: id) {
            failure = problem
            status = problem
        }
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
            return "python -m whoodeo.mlxport.convert <Checkpoint> --app legt neue Modelle in diese Sammlung."
        }
        if modelURL == nil { return "Wähle links ein Modell. Die Vorschau startet, sobald ein Film offen ist." }
        if facts == nil { return "Öffne einen Film oder leg ihn hier ab." }
        if busy { return "rechnet…" }
        return ""
    }

    /// The picture view appears with the first frame, so the still goes up once it is here.
    func attachStage(_ stage: PictureStage) {
        guard self.stage !== stage else { return }
        self.stage = stage
        stage.screen().setGrade(grade)
        if !playing { showStill() }
    }

    func installKeys() {
        startLibraryWatch()
        if keys != nil { return }
        keys = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            if event.modifierFlags.contains(.command) { return event }
            // The playground has its own sliders; arrows and space stay there.
            if event.window?.identifier?.rawValue.contains(PlaygroundView.windowID) == true { return event }
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

    func openSelection() {
        if comparison != nil, comparison == selection { setComparison(nil) }
        guard let selection, let (url, codec) = resolvedPackage() else { return }
        UserDefaults.standard.set(selection.lastPathComponent, forKey: Self.lastModelKey)
        if url == modelURL { return }
        if exporting {
            failure = "Während des Exports bleibt das Modell."
            status = failure!
            return
        }
        loadModel(url, codec: codec)
    }

    /// The sidebar line that is selected.
    var selectedModel: LibraryModel? {
        guard let selection else { return nil }
        return library.first { $0.url == selection }
    }

    /// The codec menu works for a latent_up with FiLM, and not while an export runs.
    var codecSelectable: Bool { selectedModel?.hasCodecs == true && !exporting }

    /// The codec automatic would pick now: that of the open film, unknown without one.
    var automaticCodec: SourceCodec { filmCodec ?? .unknown }

    /// The codec actually fed: automatic MPEG-2 uses unknown until that slot has been trained.
    private func wantedCodec() -> SourceCodec {
        let wanted = codecChoice.resolve(film: filmCodec)
        if codecChoice == .auto, wanted == .mpeg2 {
            return .unknown
        }
        return wanted
    }

    /// The selected package, and the codec to feed a `-cond` package. The codec is nil without FiLM.
    private func resolvedPackage() -> (url: URL, codec: SourceCodec?)? {
        guard let selection else { return nil }
        guard let model = selectedModel else { return (selection, nil) }
        guard model.conditionInput else { return (model.url, nil) }
        let wanted = wantedCodec()
        if codecChoice == .auto, filmCodec == .mpeg2, wanted == .unknown {
            Log.model.info("MPEG-2 noch ohne Training, nehme unknown")
        }
        return (model.url, wanted)
    }

    /// Suffix of the frame cache: the condition of a `-cond` package, then the preprocessing.
    /// Nil for a package without FiLM on an unprocessed film, so older caches keep their names.
    private func conditionTag() -> String? {
        let condition = conditionSize > 0 ? (activeCodec ?? wantedCodec()).cacheTag : nil
        return Self.archiveTag(condition, frameMap)
    }

    private static func archiveTag(_ condition: String?, _ map: FrameMap?) -> String? {
        let parts = [condition, map?.cacheTag].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: "-")
    }

    /// Menu: automatic or a fixed codec. Swaps the package or the vector; the frame cache follows.
    func changeCodec(_ next: CodecChoice) {
        guard !exporting else {
            failure = "Während des Exports bleibt der Codec."
            status = failure!
            return
        }
        codecChoice = next
        Log.model.info("Codec-Wahl \(next.rawValue, privacy: .public)")
        if !syncCodecPackage() {
            showCurrent()
        }
    }

    /// `auto`, `h264`, `mpeg4`, `mpeg2`, or `unknown`. Returns a message when the name is unknown.
    func setCodecChoice(named name: String) -> String? {
        guard let parsed = CodecChoice.parse(name) else {
            return "Codec ist auto, h264, mpeg4, mpeg2 oder unknown."
        }
        changeCodec(parsed)
        return nil
    }

    /// Load the package of the codec in force if it is not the one loaded, or switch the
    /// condition vector of a `-cond` package. True when a load started.
    @discardableResult
    private func syncCodecPackage() -> Bool {
        guard !exporting, let (url, codec) = resolvedPackage() else { return false }
        if url == modelURL {
            if conditionSize > 0, let codec, codec != activeCodec {
                Log.model.info("Codec \(codec.rawValue, privacy: .public) am selben Paket")
                activeCodec = codec
                engine.setCondition(codec)
                reopenConditionArchives()
            }
            return false
        }
        Log.model.info("Codec \(codec?.rawValue ?? "-", privacy: .public): \(url.lastPathComponent, privacy: .public)")
        loadModel(url, codec: codec)
        return true
    }

    /// Drop and reopen the frame caches after the condition of a loaded `-cond` package changes.
    /// The model stays; fill and a comparison cache start over, the comparison engine is kept.
    private func reopenConditionArchives() {
        epoch += 1
        stopPlay(clear: true)
        scrubTask?.cancel()
        scrubbing = false
        filling = nil
        resetFillHolds()
        store?.flush()
        store = nil
        comparisonStore?.flush()
        comparisonStore = nil
        comparisonOpened = false
        comparisonComputing = nil
        hasPicture = false
        milliseconds = nil
        coverage = Coverage(count: frameCount)
        resetScale()
        if let codec = activeCodec {
            comparisonEngine?.setCondition(codec)
        }
    }

    /// End of the line under the picture: the codec for a package with FiLM, empty otherwise.
    private var codecNote: String {
        activeCodec.map { " · \($0.title)" } ?? ""
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
        let removingCurrent = modelURL == target
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
        guard let url = urls.first(where: FilmDrop.isFilm) else {
            Log.drop.info("openFilms: kein Film unter \(urls.count) URL(s)")
            return false
        }
        let key = url.standardizedFileURL
        if let recentDrop, recentDrop.url == key, Date().timeIntervalSince(recentDrop.at) < 0.5 {
            Log.drop.info("openFilms: doppelter Drop ignoriert")
            return true
        }
        recentDrop = (key, Date())
        let opened = addFilm(at: key.path) != nil
        if opened {
            Log.drop.info("openFilms: geöffnet \(key.path, privacy: .public)")
        } else {
            Log.drop.error("openFilms: \(self.failure ?? "Fehler", privacy: .public) bei \(key.path, privacy: .public)")
        }
        return opened
    }

    func pickFilm() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.allowedContentTypes = FilmDrop.contentTypes
        panel.prompt = "Öffnen"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        _ = addFilm(at: url.path)
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
        adoptStore()
        if store?.contains(index) == true {
            display()
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
        // The comparison is for stills. Playback starts on the upscale.
        if wantedSource == .comparison { showSource(.upscale) }
        playing = true
        playGeneration += 1
        Log.play.info("Wiedergabe ab Bild \(self.index + 1, privacy: .public)")
        beginPlayback()
    }

    /// While the mouse button is down. Without a comparison model the original, also during playback.
    /// With one, the comparison of the still; a click during playback pauses first and stays paused.
    func showAlternate(_ show: Bool) {
        showSource(show ? (comparison == nil ? .original : .comparison) : .upscale)
    }

    func showSource(_ next: ScreenSource) {
        wantedSource = next == .comparison && comparison == nil ? .upscale : next
        if wantedSource == .comparison, playing {
            finishPlay()
            Log.play.info("Vergleich hält die Wiedergabe an")
        }
        refreshSource()
    }

    /// Point the screen at the wanted source for the frame it shows.
    private func refreshSource() {
        guard hasPicture, let stage, let reel = reel() else { return }
        let source = screenSource(for: reel)
        stage.screen().setSource(source, reel: reel)
        screenSource = source
    }

    /// The comparison shows only in a still and only once its frame is stored. Until then the upscale
    /// stays and the comparison model computes the frame.
    private func screenSource(for reel: Reel) -> ScreenSource {
        switch wantedSource {
        case .upscale, .original:
            return wantedSource
        case .comparison:
            guard comparison != nil, !playing else { return .upscale }
            if reel.comparison?.contains(index) == true { return .comparison }
            requestComparison(index)
            return .upscale
        }
    }

    /// The option-click and the context menu of a sidebar line.
    func toggleComparison(_ url: URL) {
        setComparison(comparison == url ? nil : url)
    }

    /// The main model cannot be its own comparison. A comparison on screen gives way to the upscale.
    func setComparison(_ url: URL?) {
        let next = url == selection ? nil : url
        guard next != comparison else { return }
        if wantedSource == .comparison { showSource(.upscale) }
        comparison = next
        dropComparison()
        Log.model.info("Vergleich \(next?.lastPathComponent ?? "-", privacy: .public)")
    }

    /// Forget the comparison archive and its engine. Frames it stored stay on disk.
    private func dropComparison() {
        comparisonEpoch += 1
        if let engine = comparisonEngine {
            let store = comparisonStore
            engine.settle { store?.flush() }
        }
        comparisonEngine = nil
        comparisonStore = nil
        comparisonPackage = nil
        comparisonOpened = false
        comparisonComputing = nil
        comparisonLoading = false
        comparisonError = nil
    }

    /// The package of the comparison line for the codec in force, the same choice the main model follows.
    private func adoptComparison() {
        guard !comparisonOpened, let comparison, let facts, let shownFilmID, facts.count > 0,
              let model = library.first(where: { $0.url == comparison }) else { return }
        comparisonOpened = true
        let package = model.url
        guard package != modelURL else { return }
        do {
            let tag = Self.archiveTag(model.conditionInput ? wantedCodec().cacheTag : nil, frameMap)
            comparisonStore = try FrameStore.open(filmID: shownFilmID, model: package, count: facts.count, condition: tag)
            comparisonPackage = package
        } catch {
            comparisonError = error.localizedDescription
            Log.model.error("Vergleichs-Cache: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Compute one frame with the comparison model and store it in its archive. Its own engine loads
    /// the film and the package on first use, so the main model and its fill keep going.
    private func requestComparison(_ frame: Int) {
        guard comparisonComputing != frame, let facts, let shownFilmID,
              let package = comparisonPackage, let store = comparisonStore else { return }
        let token = comparisonEpoch
        let engine: Engine
        if let comparisonEngine {
            engine = comparisonEngine
        } else {
            engine = Engine()
            comparisonEngine = engine
            comparisonLoading = true
            Log.model.info("Vergleich lädt \(package.lastPathComponent, privacy: .public)")
            engine.openFilm(facts.url, filmID: shownFilmID, map: frameMap) { _ in }
            let codec = activeCodec ?? wantedCodec()
            engine.openModel(package, condition: codec) { [weak self] result in
                Task { @MainActor in
                    guard let self, token == self.comparisonEpoch else { return }
                    self.comparisonLoading = false
                    if case .failure(let error) = result {
                        self.comparisonError = error.localizedDescription
                        Log.model.error("Vergleichsmodell: \(error.localizedDescription, privacy: .public)")
                    }
                }
            }
        }
        comparisonComputing = frame
        comparisonError = nil
        engine.request(frame, store: store) { [weak self] result in
            Task { @MainActor in
                guard let self, token == self.comparisonEpoch else { return }
                if self.comparisonComputing == frame { self.comparisonComputing = nil }
                switch result {
                case .success(let shown):
                    if !shown.fromCache {
                        let timing = String(format: "%.0f ms", shown.milliseconds)
                        Log.model.info("Vergleich rechnet Bild \(frame + 1, privacy: .public) · \(timing, privacy: .public)")
                    }
                    if self.wantedSource == .comparison { self.refreshSource() }
                case .failure(let error):
                    if error is CancellationError { return }
                    self.comparisonError = error.localizedDescription
                    Log.model.error("Vergleich Bild \(frame + 1, privacy: .public): \(error.localizedDescription, privacy: .public)")
                }
            }
        }
    }

    /// The comparison archive holds the frame under the playhead. For scripts.
    var comparisonCached: Bool {
        adoptComparison()
        return comparisonStore?.contains(index) == true
    }

    func markIn() {
        inPoint = index
        if inPoint > outPoint { outPoint = inPoint }
    }

    func markOut() {
        outPoint = index
        if outPoint < inPoint { inPoint = outPoint }
    }

    private func loadModel(_ url: URL, codec: SourceCodec? = nil) {
        loadingModel = true
        modelURL = url
        activeCodec = codec
        failure = nil
        status = "lädt \(url.lastPathComponent)"
        releaseArchive()
        modelLoad += 1
        let token = modelLoad
        engine.openModel(url, condition: codec) { [weak self] result in
            Task { @MainActor in
                guard let self, token == self.modelLoad else { return }
                self.applyModel(result)
            }
        }
    }

    private func applyModel(_ result: Result<PackageInfo, Error>) {
        switch result {
        case .success(let info):
            modelLabel = info.label
            conditionSize = info.takesCondition ? info.conditionSize : 0
            if info.takesCondition {
                let codec = activeCodec ?? wantedCodec()
                activeCodec = codec
                engine.setCondition(codec)
                Log.model.info("Modell \(self.modelURL?.lastPathComponent ?? "", privacy: .public) · Codec \(codec.rawValue, privacy: .public) · Wahl \(self.codecChoice.rawValue, privacy: .public) · Film \(self.filmCodec?.rawValue ?? "-", privacy: .public)")
            } else {
                activeCodec = nil
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

    private func loadFilm(_ url: URL, id: String) {
        if shownFilmID == id, busy { return }
        cancelDecimateScan()
        facts = nil
        frameCount = 0
        shownFilmID = id
        loadingFilmID = id
        busy = true
        failure = nil
        originals = nil
        let entry = films.first { $0.id == id }
        preprocess = entry?.preprocess ?? .none
        grade = entry?.grade ?? .neutral
        // A scan from before resolves at once. The film checks that it was made for this file.
        let scan = preprocess.decimate == .off ? nil : Decimate.cached(filmID: id)
        let resolved = scan.map { Decimate.resolve(preprocess.decimate, diffs: $0) }
        frameMap = resolved?.map
        decimateNote = resolved?.note ?? ""
        releaseArchive()
        status = "öffnet \(url.lastPathComponent)"
        engine.openFilm(url, filmID: id, map: frameMap) { [weak self] result in
            Task { @MainActor in
                guard let self, self.loadingFilmID == id else { return }
                switch result {
                case .success(let facts):
                    self.loadingFilmID = nil
                    if facts.cycle == nil { self.frameMap = nil }
                    if self.preprocess.decimate != .off, scan?.count != facts.sourceCount {
                        self.startDecimateScan(url: url, id: id)
                    }
                    self.facts = facts
                    self.originals = Originals(url: facts.url, map: self.frameMap)
                    self.frameCount = facts.count
                    self.index = 0
                    self.inPoint = 0
                    self.outPoint = max(facts.count - 1, 0)
                    self.busy = false
                    self.failure = nil
                    self.status = "\(facts.width)×\(facts.height) · \(facts.count) Bilder"
                    self.filmCodec = facts.codec
                    Log.model.info("Film-Codec \(facts.codecName, privacy: .public) → \(facts.codec.rawValue, privacy: .public)")
                    self.rememberShownFacts(facts, id: id)
                    // Automatic follows the film: another codec loads another package, and that shows the frame.
                    if !self.syncCodecPackage() {
                        self.showCurrent()
                    }
                case .failure(let error):
                    self.loadingFilmID = nil
                    self.busy = false
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
            }
        }
    }

    private func reloadFilms() {
        filmList = FilmLibrary.load()
        films = filmList.films
    }

    private func rememberShownFacts(_ facts: FilmFacts, id: String) {
        do {
            try FilmLibrary.mutate { list in
                FilmLibrary.remember(id, width: facts.width, height: facts.height, frames: facts.count, in: &list)
            }
        } catch {
            Log.fill.error("Filmliste nicht geschrieben: \(error.localizedDescription, privacy: .public)")
        }
        reloadFilms()
    }

    /// Drop the picture. The list entry and the cache stay.
    private func closeShownFilm() {
        cancelDecimateScan()
        frameMap = nil
        preprocess = .none
        decimateNote = ""
        shownFilmID = nil
        filmCodec = nil
        loadingFilmID = nil
        facts = nil
        originals = nil
        frameCount = 0
        index = 0
        inPoint = 0
        outPoint = 0
        busy = false
        failure = nil
        releaseArchive()
        status = modelURL == nil ? "Modell in der Seitenleiste wählen" : "Film öffnen"
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
        let urls = library.map(\.url)
        DispatchQueue.global(qos: .utility).async {
            FrameStore.pruneModelCaches(keep: Set(urls.map(FrameStore.modelKey(of:))))
        }
        if let selection, !library.contains(where: { $0.url == selection }) {
            self.selection = nil
        }
        if let modelURL, !urls.contains(modelURL) { clearModel() }
        if let comparison, !library.contains(where: { $0.url == comparison }) { setComparison(nil) }
        if selection != nil, modelURL != nil, !loadingModel {
            syncCodecPackage()
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
        modelLoad += 1
        loadingModel = false
        modelURL = nil
        conditionSize = 0
        activeCodec = nil
        modelLabel = ""
        pixelWidth = 0
        pixelHeight = 0
        releaseArchive()
        failure = nil
        status = facts == nil ? "Modell in der Seitenleiste wählen" : status
    }

    /// Quit: stop playback, fill and scans, then wait off the main thread until both engines
    /// have finished their current frame. `done` runs on the main actor.
    func prepareToQuit(done: @escaping @MainActor () -> Void) {
        guard !quitting else { return }
        quitting = true
        epoch += 1
        stopPlay()
        scrubTask?.cancel()
        cancelDecimateScan()
        let engines = [engine] + (comparisonEngine.map { [$0] } ?? [])
        Log.fill.info("Beenden: warte auf das Modell")
        DispatchQueue.global(qos: .userInitiated).async {
            let started = CFAbsoluteTimeGetCurrent()
            let idle = engines.map { $0.shutdown(timeout: 20) }.allSatisfy { $0 }
            let waited = String(format: "%.1f s", CFAbsoluteTimeGetCurrent() - started)
            Log.fill.info("Beenden: Modell \(idle ? "frei" : "noch beschäftigt", privacy: .public) nach \(waited, privacy: .public)")
            DispatchQueue.main.async {
                MainActor.assumeIsolated { done() }
            }
        }
    }

    /// Publish the open index before the process exits. Dropping the archive publishes as well.
    func flushStore() {
        store?.flush()
        comparisonStore?.flush()
    }

    /// Drop the pictures on screen. The archive on disk stays with its model.
    private func releaseArchive() {
        epoch += 1
        stopPlay(clear: true)
        scrubTask?.cancel()
        scrubbing = false
        filling = nil
        resetFillHolds()
        store = nil
        dropComparison()
        hasPicture = false
        milliseconds = nil
        coverage = Coverage(count: frameCount)
        resetScale()
    }

    private func adoptStore() {
        guard store == nil, !cacheSuspended, let facts, let modelURL, let shownFilmID, facts.count > 0 else { return }
        do {
            let opened = try FrameStore.open(filmID: shownFilmID, model: modelURL, count: facts.count, condition: conditionTag())
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
        if store?.contains(index) == true {
            display()
            scheduleFill()
            return
        }
        busy = true
        if !hasPicture { status = "rechnet…" }
        let wanted = index
        let token = epoch
        filling = nil
        engine.request(wanted, store: store) { [weak self] result in
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

    /// Stop playback. The picture stays until the next still replaces it, unless `clear` drops it.
    private func stopPlay(clear: Bool = false) {
        playing = false
        playGeneration += 1
        playFetching = false
        if clear {
            stage?.screenIfCreated()?.stop()
        } else {
            stage?.screenIfCreated()?.yield()
        }
    }

    /// Pause on the frame the screen shows, then treat it like any other still.
    private func finishPlay() {
        if let frame = stage?.screenIfCreated()?.pause() {
            index = frame
        }
        stopPlay()
        Log.play.info("Pause bei Bild \(self.index + 1, privacy: .public)")
        showCurrent()
    }

    /// Playback hands ProRes samples, or the originals or the comparison while the mouse is down, to the system video display.
    private func beginPlayback() {
        adoptStore()
        guard pixelWidth > 0, let stage, let reel = reel() else {
            playing = false
            return
        }
        if coverage.contains(index) {
            if status != "Wiedergabe · Cache" { status = "Wiedergabe · Cache" }
        } else if status != "rechnet…" {
            status = "rechnet…"
        }
        stage.screen().play(from: index, reel: reel) { [weak self] event in
            self?.handlePlayback(event)
        }
        scheduleFill(allowWhilePlaying: true)
    }

    private func handlePlayback(_ event: FrameScreen.Event) {
        switch event {
        case .show(let frame):
            guard playing, frame >= 0, frame < frameCount else { return }
            if frame != index { index = frame }
            if status != "Wiedergabe · Cache" { status = "Wiedergabe · Cache" }
        case .needs(let frame):
            guard playing else { return }
            if frame <= index + 1, !coverage.contains(frame), status != "rechnet…" {
                status = "rechnet…"
            }
            ensureFrame(frame)
        case .ended:
            guard playing else { return }
            finishPlay()
        case .failed(let message):
            guard playing else { return }
            stopPlay()
            failure = message
            status = message
            Log.play.error("Wiedergabe: \(message, privacy: .public)")
        }
    }

    /// The frame under or just ahead of the playhead is missing. Store it, and let the video layer show the ProRes sample.
    private func ensureFrame(_ frame: Int) {
        guard playing, frame >= 0, frame < frameCount else { return }
        if coverage.contains(frame) || playFetching || filling == frame { return }
        playFetching = true
        let generation = playGeneration
        let token = epoch
        engine.request(frame, store: store) { [weak self] result in
            Task { @MainActor in
                guard let self, generation == self.playGeneration else { return }
                self.playFetching = false
                guard token == self.epoch else { return }
                switch result {
                case .success(let shown):
                    if shown.stored { self.coverage.insert(shown.index) }
                    if !shown.fromCache { self.noteScaled(1) }
                    guard self.playing else { return }
                    self.scheduleFill(allowWhilePlaying: true)
                case .failure(let error):
                    if error is CancellationError {
                        if self.playing { self.continuePlaybackWork() }
                        return
                    }
                    guard frame <= self.index + 1 else { return }
                    self.stopPlay()
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
            }
        }
    }

    private func continuePlaybackWork() {
        scheduleFill(allowWhilePlaying: playing)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    private func deliver(_ result: Result<Shown, Error>) {
        busy = false
        switch result {
        case .success(let shown):
            if shown.stored { coverage.insert(shown.index) }
            if !shown.fromCache { noteScaled(1) }
            if shown.index == index, shown.stored, !playing {
                display(milliseconds: shown.fromCache ? nil : shown.milliseconds)
            }
            if !playing { scheduleFill() }
        case .failure(let error):
            playing = false
            failure = error.localizedDescription
            status = failure!
        }
    }

    /// Show the stored frame under the playhead as a still.
    private func display(milliseconds computed: Double? = nil) {
        guard let size = store?.dimensions() else { return }
        pixelWidth = size.width
        pixelHeight = size.height
        hasPicture = true
        milliseconds = computed
        failure = nil
        if let computed {
            let timing = String(format: "%.0f ms", computed)
            status = "Bild \(index + 1) von \(frameCount) · \(timing)\(codecNote)"
        } else {
            status = "Bild \(index + 1) von \(frameCount) · Cache\(codecNote)"
        }
        showStill()
    }

    private func showStill() {
        guard hasPicture, let stage, let reel = reel() else { return }
        let source = screenSource(for: reel)
        stage.screen().show(index, reel: reel, source: source)
        screenSource = source
    }

    /// What the screen may show for the open film and model.
    private func reel() -> Reel? {
        guard let store, let facts else { return nil }
        adoptComparison()
        return Reel(store: store, originals: originals, comparison: comparisonStore, fps: max(facts.fps, 1), count: frameCount)
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
        guard fillCache, !quitting, !cacheSuspended else { return }
        guard !fillHalted else { return }
        // At the limit the walk waits. Nothing is deleted; frames looked at on purpose are still stored.
        if CacheBudget.isFull {
            if cacheNote != Self.limitNote {
                cacheNote = Self.limitNote
                Log.fill.info("fill pausiert: Cache-Limit erreicht")
            }
            return
        }
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
        engine.request(next, store: store) { [weak self] result in
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
                        self.continuePlaybackWork()
                        return
                    }
                    guard !self.scrubbing, !self.exporting, !self.busy else { return }
                    self.scheduleFill()
                case .failure(let error):
                    if error is CancellationError {
                        Log.fill.info("fill Bild \(next + 1, privacy: .public) verdrängt")
                        self.continuePlaybackWork()
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
                    self.continuePlaybackWork()
                }
            }
        }
    }

    private static let limitNote = "Cache-Limit erreicht – pausiert"

    /// Location, limit, or contents changed in the settings window. A fill that waited for room goes on.
    private func cacheSettingsChanged() {
        let note: String? = !CacheSettings.isAvailable
            ? "Cache-Ordner nicht erreichbar"
            : (CacheBudget.isFull && fillCache ? Self.limitNote : nil)
        let resumed = cacheNote == Self.limitNote && note == nil
        if note == Self.limitNote, cacheNote != note { Log.fill.info("fill pausiert: Cache-Limit erreicht") }
        if cacheNote != note { cacheNote = note }
        if resumed {
            Log.fill.info("fill geht weiter: Platz im Cache")
            resetFillHolds()
            if !playing { scheduleFill() }
        }
    }

    /// Why the cache folder cannot change right now. Nil when it can.
    var cacheBusyReason: String? {
        if exporting { return "Während eines Exports bleibt der Cache-Ordner, wie er ist." }
        if decimateProgress != nil { return "Bitte warten, bis der Scan nach doppelten Bildern fertig ist." }
        return nil
    }

    /// Close every archive and wait until both engines let go of them. The film and model stay loaded.
    func suspendCache(done: @escaping @MainActor () -> Void) {
        cacheSuspended = true
        store?.flush()
        comparisonStore?.flush()
        let engines = [engine] + (comparisonEngine.map { [$0] } ?? [])
        releaseArchive()
        status = "Cache wird umgestellt…"
        let group = DispatchGroup()
        for engine in engines {
            group.enter()
            engine.settle { group.leave() }
        }
        group.notify(queue: .main) {
            MainActor.assumeIsolated { done() }
        }
    }

    /// Open the archive of the frame on screen again, from the folder now configured.
    func resumeCache() {
        cacheSuspended = false
        cacheSettingsChanged()
        if shownFilmID != nil, modelURL != nil, facts != nil {
            showCurrent()
        } else if status == "Cache wird umgestellt…" {
            status = ""
        }
    }

    /// Delete one film's frame caches; the film stays in the list. A message means nothing was deleted.
    func clearCache(filmID: String) -> String? {
        let shown = shownFilmID == filmID
        if shown, exporting { return "Ein Export läuft gerade." }
        if shown {
            store?.flush()
            let engines = [engine] + (comparisonEngine.map { [$0] } ?? [])
            cancelDecimateScan()
            releaseArchive()
            for engine in engines {
                let gate = DispatchSemaphore(value: 0)
                engine.settle { gate.signal() }
                gate.wait()
            }
        }
        defer { if shown { showCurrent() } }
        do {
            try FrameStore.removeCache(id: filmID)
            Log.fill.info("Cache eines Films geleert")
            return nil
        } catch {
            return error.localizedDescription
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
        engine.export(url: url, codec: codec, range: range, grade: grade, store: store, progress: { [weak self] done, total in
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

    // MARK: Pipeline

    /// Pipeline panel and scripts: which repeated frames to drop. Scans the film once, then
    /// switches the film the app sees. The frame cache of the other choice stays on disk.
    func setDecimate(_ mode: DecimateMode) {
        guard mode != preprocess.decimate else { return }
        guard !exporting else {
            failure = "Während des Exports bleibt die Vorverarbeitung."
            status = failure!
            return
        }
        preprocess.decimate = mode
        Log.model.info("Decimate \(mode.rawValue, privacy: .public)")
        if let id = shownFilmID {
            let config = preprocess
            do {
                try FilmLibrary.mutate { FilmLibrary.remember(id, preprocess: config, in: &$0) }
            } catch {
                Log.fill.error("Filmliste nicht geschrieben: \(error.localizedDescription, privacy: .public)")
            }
            reloadFilms()
        }
        cancelDecimateScan()
        decimateNote = ""
        guard let facts, let id = shownFilmID else { return }
        if mode == .off {
            switchMap(nil)
            return
        }
        guard let scan = Decimate.cached(filmID: id, count: facts.sourceCount) else {
            startDecimateScan(url: facts.url, id: id)
            return
        }
        let resolved = Decimate.resolve(mode, diffs: scan)
        decimateNote = resolved.note
        switchMap(resolved.map)
    }

    /// `off`, `auto`, `cycle6` or `cycle5`. Returns a message when the name is unknown.
    func setDecimate(named name: String) -> String? {
        guard let mode = DecimateMode.parse(name) else {
            return "Decimate ist off, auto, cycle6 oder cycle5."
        }
        setDecimate(mode)
        return nil
    }

    /// One line for the panel and for scripts, for example "30 → 25 fps · 44847 Bilder".
    var pipelineSummary: String {
        guard let facts else { return "" }
        if let decimateProgress {
            return "sucht doppelte Bilder … \(Int((decimateProgress * 100).rounded())) %"
        }
        let rate = { (fps: Double) in fps.formatted(.number.precision(.fractionLength(0...3))) }
        var text = facts.cycle == nil
            ? "\(rate(facts.fps)) fps · \(facts.count) Bilder"
            : "\(rate(facts.sourceFps)) → \(rate(facts.fps)) fps · \(facts.count) Bilder"
        if !decimateNote.isEmpty { text += " · \(decimateNote)" }
        return text
    }

    private func cancelDecimateScan() {
        decimateToken += 1
        decimateCancel?.withLock { $0 = true }
        decimateCancel = nil
        decimateProgress = nil
    }

    /// Measure the frame differences on a background thread. The film stays as it is until the scan is done.
    private func startDecimateScan(url: URL, id: String) {
        cancelDecimateScan()
        let token = decimateToken
        let cancel = OSAllocatedUnfairLock(initialState: false)
        decimateCancel = cancel
        decimateProgress = 0
        decimateNote = ""
        Log.model.info("Decimate-Scan startet \(url.lastPathComponent, privacy: .public)")
        let report: @Sendable (Double) -> Void = { fraction in
            Task { @MainActor in self.noteDecimateProgress(fraction, token: token) }
        }
        Task.detached(priority: .utility) { [weak self] in
            let started = CFAbsoluteTimeGetCurrent()
            let result = Result {
                try Decimate.scan(url: url, progress: report, cancelled: { cancel.withLock { $0 } })
            }
            if case .success(let diffs) = result {
                try? Decimate.store(diffs, filmID: id)
                let seconds = String(format: "%.1f s", CFAbsoluteTimeGetCurrent() - started)
                Log.model.info("Decimate-Scan \(diffs.count, privacy: .public) Bilder in \(seconds, privacy: .public)")
            }
            await self?.finishDecimateScan(result, token: token, id: id)
        }
    }

    private func noteDecimateProgress(_ fraction: Double, token: Int) {
        guard token == decimateToken, decimateProgress != nil else { return }
        decimateProgress = fraction
    }

    private func finishDecimateScan(_ result: Result<[Float], Error>, token: Int, id: String) {
        guard token == decimateToken, shownFilmID == id else { return }
        decimateCancel = nil
        decimateProgress = nil
        switch result {
        case .success(let diffs):
            let resolved = Decimate.resolve(preprocess.decimate, diffs: diffs)
            decimateNote = resolved.note
            switchMap(resolved.map)
        case .failure(let error):
            if error is CancellationError { return }
            decimateNote = "Suche fehlgeschlagen: \(error.localizedDescription)"
            Log.model.error("Decimate-Scan: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Show the film through another frame map. The playhead stays on the same source frame,
    /// the in and out points span the film again, and the caches of the new map take over.
    private func switchMap(_ map: FrameMap?) {
        guard map != frameMap, facts != nil, let id = shownFilmID else { return }
        let sourceFrame = frameMap.map { $0.raw[min(max(index, 0), $0.count - 1)] } ?? index
        releaseArchive()
        frameMap = map
        busy = true
        engine.applyMap(map) { [weak self] result in
            Task { @MainActor in
                guard let self, self.shownFilmID == id else { return }
                self.busy = false
                switch result {
                case .success(let facts):
                    if facts.cycle == nil { self.frameMap = nil }
                    self.facts = facts
                    self.originals = Originals(url: facts.url, map: self.frameMap)
                    self.frameCount = facts.count
                    self.index = min(self.frameMap?.index(forSource: sourceFrame) ?? sourceFrame, max(facts.count - 1, 0))
                    self.inPoint = 0
                    self.outPoint = max(facts.count - 1, 0)
                    self.coverage = Coverage(count: facts.count)
                    self.status = "\(facts.width)×\(facts.height) · \(facts.count) Bilder"
                    self.rememberShownFacts(facts, id: id)
                    Log.model.info("Film \(facts.count, privacy: .public) Bilder · \(facts.fps, privacy: .public) fps")
                    self.showCurrent()
                case .failure(let error):
                    self.failure = error.localizedDescription
                    self.status = self.failure!
                }
            }
        }
    }

    /// Fill the sliders with values sampled from the film. They stay editable.
    func autoGrade() {
        guard let facts, !estimatingGrade else { return }
        estimatingGrade = true
        let url = facts.url
        let map = frameMap
        let id = shownFilmID
        Task.detached(priority: .userInitiated) { [weak self] in
            let result = Result { try ColorGrade.estimate(url: url, map: map) }
            await self?.finishAutoGrade(result, id: id)
        }
    }

    private func finishAutoGrade(_ result: Result<ColorGrade, Error>, id: String?) {
        estimatingGrade = false
        guard shownFilmID == id else { return }
        switch result {
        case .success(let proposed):
            grade = proposed
            Log.model.info("Auto-Farbe h \(proposed.brightness) k \(proposed.contrast) g \(proposed.gamma) s \(proposed.saturation)")
        case .failure(let error):
            failure = error.localizedDescription
            status = failure!
        }
    }

    func resetGrade() {
        grade = .neutral
    }

    /// Brightness, contrast, gamma and saturation for scripts, "0 1 1 1" when neutral.
    var scriptGrade: String {
        [grade.brightness, grade.contrast, grade.gamma, grade.saturation]
            .map { String(format: "%g", $0) }
            .joined(separator: " ")
    }

    /// Four numbers as in `scriptGrade`, `neutral`, or `auto`. Returns a message when the text does not parse.
    func setGrade(text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespaces).lowercased()
        switch trimmed {
        case "neutral", "":
            resetGrade()
            return nil
        case "auto":
            autoGrade()
            return nil
        default:
            let values = trimmed.split(separator: " ").compactMap { Double($0) }
            guard values.count == 4 else {
                return "Farbe ist „Helligkeit Kontrast Gamma Sättigung“, neutral oder auto."
            }
            func clamp(_ value: Double, _ range: ClosedRange<Double>) -> Double {
                min(max(value, range.lowerBound), range.upperBound)
            }
            grade = ColorGrade(
                brightness: clamp(values[0], ColorGrade.brightnessRange),
                contrast: clamp(values[1], ColorGrade.contrastRange),
                gamma: clamp(values[2], ColorGrade.gammaRange),
                saturation: clamp(values[3], ColorGrade.saturationRange)
            )
            return nil
        }
    }

    /// Slider drags change the grade many times a second; the film list is written once they rest.
    private func saveGradeSoon() {
        gradeSave?.cancel()
        guard let id = shownFilmID else { return }
        let value = grade
        gradeSave = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled, let self else { return }
            do {
                try FilmLibrary.mutate { FilmLibrary.remember(id, grade: value, in: &$0) }
            } catch {
                Log.fill.error("Filmliste nicht geschrieben: \(error.localizedDescription, privacy: .public)")
            }
            self.reloadFilms()
        }
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
