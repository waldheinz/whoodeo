import Foundation
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
                Button("Modell importieren…") { session.importModel() }
                    .keyboardShortcut("o", modifiers: [.command, .shift])
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
            }
        }
    }
}

nonisolated enum CLI {
    static let usage = """
    Whoodeo --check model.mlpackage in.f32 out.f32 [--unit gpu|ane|all]
    Whoodeo --decode video index out.f32
    Whoodeo --export model.mlpackage video out.mov [--frames N] [--unit gpu|ane|all] [--codec hevc|prores|prores4444]
    Whoodeo --pace model.mlpackage video [--frames N] [--unit gpu|ane|all]

    The f32 file is little-endian int32 n,c,h,w followed by float32 NCHW.
    """

    static func run(_ command: String, _ args: [String]) throws {
        switch command {
        case "--check": try check(args)
        case "--decode": try decode(args)
        case "--export": try export(args)
        case "--pace": try pace(args)
        default: throw CLIError(usage)
        }
    }

    private static func check(_ args: [String]) throws {
        let (paths, unit, _, _) = try flags(args)
        guard paths.count == 3 else { throw CLIError(usage) }
        let predictor = try Predictor(url: URL(fileURLWithPath: paths[0]), unit: unit)
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
        let film = try wait { try await Film.load(URL(fileURLWithPath: args[0])) }
        let frame = try film.frame(at: index)
        try TensorFile.write(
            URL(fileURLWithPath: args[2]),
            shape: [1, 3, frame.height, frame.width],
            data: frame.rgb
        )
        fputs("\(frame.width)x\(frame.height)\n", stdout)
    }

    /// Warm two frames, then print decode, model and draw times for the result view.
    private static func pace(_ args: [String]) throws {
        let (paths, unit, frames, _) = try flags(args)
        guard paths.count == 2 else { throw CLIError(usage) }
        let count = frames > 0 ? frames : 8
        let predictor = try Predictor(url: URL(fileURLWithPath: paths[0]), unit: unit)
        let film = try wait { try await Film.load(URL(fileURLWithPath: paths[1])) }
        let last = min(count + 2, film.facts.count)
        for index in 0..<last {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let window = try film.window(at: index, frames: predictor.info.inFrames)
            let decode = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
            let modelStart = CFAbsoluteTimeGetCurrent()
            let predicted = try predictor.predict(
                stack: window.stack,
                width: window.current.width,
                height: window.current.height
            )
            let model = (CFAbsoluteTimeGetCurrent() - modelStart) * 1000
            let drawStart = CFAbsoluteTimeGetCurrent()
            _ = try Pixels.picture(planar: predicted.image, width: predicted.width, height: predicted.height)
            let draw = (CFAbsoluteTimeGetCurrent() - drawStart) * 1000
            if index < 2 { continue }
            let total = decode + model + draw
            fputs(String(
                format: "frame %d  decode %.1f  model %.1f  draw %.1f  total %.1f  (%.2f fps)\n",
                index, decode, model, draw, total, 1000 / total
            ), stdout)
        }
    }

    private static func export(_ args: [String]) throws {
        let (paths, unit, frames, codec) = try flags(args)
        guard paths.count == 3 else { throw CLIError(usage) }
        let modelURL = URL(fileURLWithPath: paths[0])
        let predictor = try Predictor(url: modelURL, unit: unit)
        let film = try wait { try await Film.load(URL(fileURLWithPath: paths[1])) }
        let store = try FrameStore.open(film: film.facts.url, model: modelURL, unit: unit, count: film.facts.count)
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

    private static func flags(_ args: [String]) throws -> ([String], Compute, Int, Codec) {
        var paths: [String] = []
        var unit = Compute.gpu
        var frames = 0
        var codec = Codec.hevc
        var index = 0
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--unit":
                index += 1
                guard index < args.count, let parsed = Compute.parse(args[index]) else { throw CLIError(usage) }
                unit = parsed
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
        return (paths, unit, frames, codec)
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
