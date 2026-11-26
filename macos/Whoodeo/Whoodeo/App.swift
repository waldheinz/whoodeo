import CoreMedia
import CoreVideo
import Foundation
import MLX
import SwiftUI

@main
struct Main {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        if let command = args.first, command.hasPrefix("--") {
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try CLI.run(command, Array(args.dropFirst()))
                    exit(0)
                } catch {
                    let ns = error as NSError
                    var text = ns.localizedDescription
                    if let underlying = ns.userInfo[NSUnderlyingErrorKey] as? NSError {
                        text += "\n\(underlying.localizedDescription)"
                    }
                    fputs("\(text)\n", stderr)
                    exit(1)
                }
            }
            dispatchMain()
        }
        WhoodeoApp.main()
    }
}

struct WhoodeoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var session = Session()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(session)
        }
        .defaultSize(width: 1280, height: 800)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Film öffnen…") { session.pickFilm() }
                    .keyboardShortcut("o")
                Divider()
                Button("Exportieren…") { session.pickExport() }
                    .keyboardShortcut("e")
                    .disabled(!session.canExport)
            }
            CommandGroup(after: .pasteboard) {
                Button("Modell in den Papierkorb legen") { session.askDelete() }
                    .keyboardShortcut(.delete, modifiers: .command)
                    .disabled(session.selection == nil)
            }
            CommandGroup(after: .sidebar) {
                Button("Einpassen") { session.chooseZoom(.fit) }
                    .disabled(session.frameCount == 0)
                Button("100 %") { session.chooseZoom(.full) }
                    .disabled(session.frameCount == 0)
                Divider()
                Toggle("Cache im Hintergrund füllen", isOn: $session.fillCache)
                Divider()
                PlaygroundMenuButton()
            }
        }
        Settings {
            SettingsView()
                .environmentObject(session)
        }
        Window("VAE-Playground", id: PlaygroundView.windowID) {
            PlaygroundView()
                .environmentObject(session)
        }
        .defaultSize(width: 1500, height: 950)
    }
}

nonisolated enum CLI {
    static let usage = """
    Whoodeo --check model.mlxmodel in.f32 out.f32
    Whoodeo --decode video index out.f32
    Whoodeo --export model.mlxmodel video out.mov [--frames N] [--codec hevc|prores|prores4444]
    Whoodeo --pace model.mlxmodel video [--frames N]
    Whoodeo --vae-roundtrip picture-or-video
    Whoodeo --upscale model.mlxmodel video index [out.f32] [--source-codec auto|h264|mpeg4|mpeg2|unknown]
    Whoodeo --decimate video [auto|cycle6|cycle5]
    Whoodeo --move-cache folder|default

    --upscale scales one frame through the frame cache, as the window does: a latent_up with
    FiLM uses the condition of the given codec (auto: the film's codec; MPEG-2 automatic
    falls back to unknown). The frame comes from that cache when it is there. --export takes
    --source-codec too. --decimate scans a film for repeated frames (or reuses the app's scan),
    prints the cycle it finds and the frames left, and keeps the scan for the app.
    --move-cache moves the frame cache as Einstellungen › Speicherort › Verschieben does
    (copy, check every file, then delete the old files) and makes the app use the folder;
    quit the app first. A folder with other files gets a "Whoodeo Cache" folder inside.

    The f32 file is little-endian int32 n,c,h,w followed by float32 NCHW.
    """

    static func run(_ command: String, _ args: [String]) throws {
        switch command {
        case "--check": try check(args)
        case "--decode": try decode(args)
        case "--export": try export(args)
        case "--pace": try pace(args)
        case "--vae-roundtrip": try vaeRoundtrip(args)
        case "--upscale": try upscale(args)
        case "--decimate": try decimate(args)
        case "--move-cache": try moveCache(args)
        default: throw CLIError(usage)
        }
    }

    private static func moveCache(_ args: [String]) throws {
        guard args.count == 1 else { throw CLIError(usage) }
        let target: URL? = args[0] == "default"
            ? nil
            : CacheSettings.cacheFolder(for: URL(fileURLWithPath: (args[0] as NSString).expandingTildeInPath, isDirectory: true))
        let source = CacheSettings.root
        let destination = target ?? CacheSettings.defaultRoot
        guard source.standardizedFileURL.path != destination.standardizedFileURL.path else {
            fputs("Der Cache liegt schon in \(destination.path)\n", stdout)
            return
        }
        if CacheSettings.isAvailable {
            var last = -1
            try CacheMover.move(from: source, to: destination) { fraction in
                let percent = Int(fraction * 100)
                if percent / 10 != last / 10 {
                    last = percent
                    fputs("\(percent) %\n", stdout)
                }
            }
        }
        if target != nil {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            CacheSettings.writeMarker(in: destination)
        }
        CacheSettings.setRoot(target)
        fputs("Cache-Ordner: \(CacheSettings.root.path)\n", stdout)
    }

    private static func check(_ args: [String]) throws {
        let (paths, _, _) = try flags(args)
        guard paths.count == 3 else { throw CLIError(usage) }
        let predictor = try Predictor(url: URL(fileURLWithPath: paths[0]))
        let (shape, data) = try TensorFile.read(URL(fileURLWithPath: paths[1]))
        guard shape.count == 4, shape[0] == 1 else { throw CLIError("expected one NCHW tensor") }
        let predicted = try predictor.predict(stack: data, width: shape[3], height: shape[2])
        try TensorFile.write(
            URL(fileURLWithPath: paths[2]),
            shape: [1, 3, predicted.height, predicted.width],
            data: predicted.image
        )
        fputs(String(format: "%.3f ms\n", predicted.milliseconds), stdout)
    }

    private static func decode(_ args: [String]) throws {
        guard args.count == 3, let index = Int(args[1]), index >= 0 else { throw CLIError(usage) }
        let (url, _) = try filmID(for: args[0])
        let film = try wait { try await Film.load(url) }
        let frame = try wait { try await film.frame(at: index) }
        try TensorFile.write(
            URL(fileURLWithPath: args[2]),
            shape: [1, 3, frame.height, frame.width],
            data: frame.rgb
        )
        fputs("\(frame.width)x\(frame.height)\n", stdout)
    }

    /// Warm two frames, then print decode, model and draw times for the result view.
    private static func pace(_ args: [String]) throws {
        let (paths, frames, _) = try flags(args)
        guard paths.count == 2 else { throw CLIError(usage) }
        let count = frames > 0 ? frames : 8
        let predictor = try Predictor(url: URL(fileURLWithPath: paths[0]))
        let (url, _) = try filmID(for: paths[1])
        let film = try wait { try await Film.load(url) }
        let last = min(count + 2, film.facts.count)
        for index in 0..<last {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let window = try wait { try await film.window(at: index, frames: predictor.info.inFrames) }
            let decode = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
            let modelStart = CFAbsoluteTimeGetCurrent()
            let predicted = try predictor.predict(
                stack: window.stack,
                width: window.current.width,
                height: window.current.height,
                frames: window.frames
            )
            let model = (CFAbsoluteTimeGetCurrent() - modelStart) * 1000
            let drawStart = CFAbsoluteTimeGetCurrent()
            _ = try Pixels.picture(planar: predicted.image, width: predicted.width, height: predicted.height)
            let draw = (CFAbsoluteTimeGetCurrent() - drawStart) * 1000
            let gib = Double(1 << 30)
            let active = Double(Memory.activeMemory) / gib
            let cache = Double(Memory.cacheMemory) / gib
            let peak = Double(Memory.peakMemory) / gib
            Memory.peakMemory = 0
            if index < 2 { continue }
            let total = decode + model + draw
            fputs(String(
                format: "frame %d  decode %.1f  model %.1f  draw %.1f  total %.1f  (%.2f fps)  MLX active %.2f  cache %.2f  peak %.2f GiB\n",
                index, decode, model, draw, total, 1000 / total, active, cache, peak
            ), stdout)
        }
    }

    /// The newest playground pair on one picture (a video's middle frame), filled to each latent size.
    private static func vaeRoundtrip(_ args: [String]) throws {
        let (paths, _, _) = try flags(args)
        guard paths.count == 1 else { throw CLIError(usage) }
        guard let pair = VAELibrary.pairs().first else { throw CLIError("Kein VAE-Paar in \(VAELibrary.directory.path)") }
        let coder = try VAECoder(pair: pair)
        let url = URL(fileURLWithPath: (paths[0] as NSString).expandingTildeInPath)
        let image: CGImage
        if PlaygroundImage.isImage(url) {
            guard let loaded = PlaygroundImage.load(url) else { throw CLIError("Bild lässt sich nicht lesen") }
            image = loaded
        } else {
            let film = try wait { try await Film.load(url) }
            let frame = try wait { try await film.frame(at: film.facts.count / 2) }
            guard let made = PlaygroundImage.image16(planar: frame.rgb, width: frame.width, height: frame.height) else {
                throw CLIError("Bild lässt sich nicht lesen")
            }
            image = made
        }
        fputs("\(pair.title) · \(pair.detail)\n", stdout)
        for size in coder.latentSizes {
            let data = try PlaygroundRender.encode(coder, image: image, size: size)
            let z = data.latent.values
            let mean = z.reduce(0, +) / Float(z.count)
            let std = (z.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(z.count)).squareRoot()
            let started = CFAbsoluteTimeGetCurrent()
            _ = try coder.decode(data.latent)
            let decodeMs = (CFAbsoluteTimeGetCurrent() - started) * 1000
            fputs(String(
                format: "%dx%d  roundtrip %.2f dB  z mean %.2f std %.2f  decode %.1f ms\n",
                data.width, data.height, data.psnr ?? 0, mean, std, decodeMs
            ), stdout)
        }
    }

    /// Scan or reuse the frame differences, then report what the pipeline panel would show.
    private static func decimate(_ args: [String]) throws {
        guard args.count == 1 || args.count == 2 else { throw CLIError(usage) }
        let mode = args.count == 2 ? DecimateMode.parse(args[1]) : .auto
        guard let mode, mode != .off else { throw CLIError(usage) }
        let (url, id) = try filmID(for: args[0])
        let film = try Film.open(url)
        let started = CFAbsoluteTimeGetCurrent()
        let diffs: [Float]
        if let cached = Decimate.cached(filmID: id, count: film.source.count) {
            diffs = cached
            fputs("Scan aus dem Cache\n", stdout)
        } else {
            diffs = try Decimate.scan(url: url, progress: { _ in }, cancelled: { false })
            try Decimate.store(diffs, filmID: id)
            let seconds = CFAbsoluteTimeGetCurrent() - started
            fputs(String(format: "Scan %d Bilder in %.1f s (%.0f Bilder/s)\n", diffs.count, seconds, Double(diffs.count) / seconds), stdout)
        }
        let resolved = Decimate.resolve(mode, diffs: diffs)
        guard let map = resolved.map else {
            fputs("\(mode.rawValue): \(resolved.note.isEmpty ? "nichts weggelassen" : resolved.note)\n", stdout)
            return
        }
        film.apply(map)
        let dropped = Set(0..<map.rawCount).subtracting(map.raw).sorted()
        var phases: [(from: Int, phase: Int)] = []
        for frame in dropped where phases.last?.phase != frame % map.cycle {
            phases.append((frame, frame % map.cycle))
        }
        let shown = phases.prefix(12).map { "ab \($0.from) Phase \($0.phase)" }.joined(separator: ", ")
        fputs("\(mode.rawValue) \(resolved.note): Zyklus \(map.cycle), \(film.source.count) → \(film.facts.count) Bilder, \(film.source.fps) → \(film.facts.fps) fps\n", stdout)
        fputs("Dauer \(film.time(for: film.facts.count).seconds) s (Quelle \(CMTimeAdd(film.time(for: 0), CMTimeMultiply(film.sourceDuration, multiplier: Int32(film.source.count))).seconds) s)\n", stdout)
        fputs("Phasenwechsel \(phases.count): \(shown)\n", stdout)
    }

    /// `--source-codec X` taken out of the arguments. Nil when it is not there.
    private static func sourceCodec(_ args: [String]) throws -> ([String], CodecChoice?) {
        guard let at = args.firstIndex(of: "--source-codec") else { return (args, nil) }
        guard at + 1 < args.count, let choice = CodecChoice.parse(args[at + 1]) else { throw CLIError(usage) }
        var rest = args
        rest.removeSubrange(at...(at + 1))
        return (rest, choice)
    }

    /// The codec the window would feed: automatic MPEG-2 uses unknown until that slot is trained.
    private static func wantedCodec(_ choice: CodecChoice, film: SourceCodec) -> SourceCodec {
        let wanted = choice.resolve(film: film)
        if choice == .auto, wanted == .mpeg2 { return .unknown }
        return wanted
    }

    /// Cache suffix of a `-cond` package, nil without FiLM.
    private static func conditionTag(package: URL, codec: SourceCodec?) -> String? {
        package.deletingPathExtension().lastPathComponent.hasSuffix("-cond") ? codec?.cacheTag : nil
    }

    /// The named package, and the codec a `-cond` package would be fed.
    private static func resolvePackage(_ path: String, choice: CodecChoice?, film: FilmFacts) -> (URL, SourceCodec?) {
        let given = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let model = LibraryModel.read(given)
        guard model.conditionInput else { return (given, nil) }
        return (given, wantedCodec(choice ?? .auto, film: film.codec))
    }

    /// One frame through the cache of the package the window would pick. Prints package, codec, and cache hit.
    private static func upscale(_ args: [String]) throws {
        let (rest, choice) = try sourceCodec(args)
        let (paths, _, _) = try flags(rest)
        guard paths.count == 3 || paths.count == 4, let index = Int(paths[2]), index >= 0 else { throw CLIError(usage) }
        let (video, id) = try filmID(for: paths[1])
        let film = try wait { try await Film.load(video) }
        guard index < film.facts.count else { throw CLIError("der Film hat \(film.facts.count) Bilder") }
        let (package, codec) = resolvePackage(paths[0], choice: choice ?? .auto, film: film.facts)
        let tag = conditionTag(package: package, codec: codec)
        let store = try FrameStore.open(filmID: id, model: package, count: film.facts.count, condition: tag)
        defer { store.flush() }
        fputs("Film-Codec \(film.facts.codecName) → \(film.facts.codec.rawValue)\n", stdout)
        fputs("Paket \(package.lastPathComponent) · Codec \(codec?.rawValue ?? "-")\n", stdout)
        fputs("Cache \(id.prefix(12))…/\(FrameStore.folderName(model: package, condition: tag).prefix(12))…\n", stdout)
        var image: [Float]
        var width: Int
        var height: Int
        if store.contains(index) {
            let buffer = try store.image(at: index)
            width = CVPixelBufferGetWidth(buffer)
            height = CVPixelBufferGetHeight(buffer)
            image = try Pixels.planar(buffer)
            fputs("Bild \(index + 1): aus dem Cache\n", stdout)
        } else {
            let predictor = try Predictor(url: package)
            if predictor.info.takesCondition {
                predictor.setCondition(codec ?? .unknown)
            }
            let window = try wait { try await film.window(at: index, frames: predictor.info.inFrames) }
            let predicted = try predictor.predict(stack: window.stack, width: window.current.width, height: window.current.height, frames: window.frames)
            _ = try store.put(rgb: predicted.image, width: predicted.width, height: predicted.height, index: index)
            image = predicted.image
            width = predicted.width
            height = predicted.height
            fputs(String(format: "Bild %d: gerechnet, %.1f ms\n", index + 1, predicted.milliseconds), stdout)
        }
        if paths.count == 4 {
            try TensorFile.write(URL(fileURLWithPath: paths[3]), shape: [1, 3, height, width], data: image)
            fputs("\(width)x\(height) → \(paths[3])\n", stdout)
        }
    }

    private static func export(_ args: [String]) throws {
        let (rest, choice) = try sourceCodec(args)
        let (paths, frames, codec) = try flags(rest)
        guard paths.count == 3 else { throw CLIError(usage) }
        let video = URL(fileURLWithPath: paths[1]).standardizedFileURL
        let probe = try wait { try await Film.load(video) }
        let (modelURL, sourceCodec) = resolvePackage(paths[0], choice: choice, film: probe.facts)
        if let sourceCodec {
            fputs("Paket \(modelURL.lastPathComponent) · Codec \(sourceCodec.rawValue)\n", stdout)
        }
        let predictor = try Predictor(url: modelURL)
        if predictor.info.takesCondition {
            predictor.setCondition(sourceCodec ?? .unknown)
        }
        let id = try FilmLibrary.mutate { list in
            let id = try FilmLibrary.insert(video, into: &list)
            if list.current.isEmpty { list.current = id }
            return id
        }
        let film = try wait { try await Film.load(video) }
        try FilmLibrary.mutate { list in
            FilmLibrary.remember(id, width: film.facts.width, height: film.facts.height, frames: film.facts.count, in: &list)
        }
        let store = try FrameStore.open(filmID: id, model: modelURL, count: film.facts.count, condition: conditionTag(package: modelURL, codec: sourceCodec))
        let count = frames > 0 ? min(frames, film.facts.count) : film.facts.count
        try Export.write(
            film: film,
            predictor: predictor,
            store: store,
            url: URL(fileURLWithPath: paths[2]),
            codec: codec,
            range: 0..<count,
            progress: { done, total in fputs("\r\(done)/\(total)", stdout) },
            stored: { _ in },
            cancelled: { false }
        )
        fputs("\nwrote \(count) \(paths[2])\n", stdout)
    }

    /// Listed films keep the id they already have. Anything else uses the digest it would get when added.
    /// A cache directory whose name is not a listed film id is pruned the next time the app starts.
    private static func filmID(for path: String) throws -> (URL, String) {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
        if let existing = FilmLibrary.load().films.first(where: { $0.path == url.path }) {
            return (url, existing.id)
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError("Die Datei fehlt.")
        }
        return (url, FrameStore.filmDigest(of: url))
    }

    private static func flags(_ args: [String]) throws -> ([String], Int, Codec) {
        var paths: [String] = []
        var frames = 0
        var codec = Codec.hevc
        var index = 0
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--frames":
                index += 1
                guard index < args.count, let parsed = Int(args[index]), parsed > 0 else { throw CLIError(usage) }
                frames = parsed
            case "--codec":
                index += 1
                guard index < args.count, let parsed = Codec(rawValue: args[index]) else { throw CLIError(usage) }
                codec = parsed
            default:
                paths.append(arg)
            }
            index += 1
        }
        return (paths, frames, codec)
    }

    private static func wait<T>(_ body: @escaping () async throws -> T) throws -> T {
        var result: Result<T, Error>?
        let gate = DispatchSemaphore(value: 0)
        Task {
            do { result = .success(try await body()) }
            catch { result = .failure(error) }
            gate.signal()
        }
        gate.wait()
        return try result!.get()
    }
}
