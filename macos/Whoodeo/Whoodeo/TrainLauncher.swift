import Darwin
import Foundation

/// Starts `whoodeo-train` as its own process session. Quitting the app does not stop the training;
/// stdout and stderr go to `Models/<run>/train.log`.
enum TrainLauncher {
    enum Mode: Equatable {
        case new
        case fineTune(checkpointID: Int)
        case resume(runName: String)

        var locksArchitecture: Bool { self != .new }
    }

    /// Next to the app in the build folder, or inside the bundle if it is ever embedded there.
    static func binaryURL() -> URL? {
        let bundle = Bundle.main.bundleURL
        let candidates: [URL] = [
            bundle.deletingLastPathComponent().appendingPathComponent("whoodeo-train"),
            bundle.appendingPathComponent("Contents/MacOS/whoodeo-train"),
            bundle.appendingPathComponent("Contents/Helpers/whoodeo-train"),
            bundle.appendingPathComponent("Contents/Resources/whoodeo-train"),
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    static func logURL(runName: String) -> URL {
        WhoodeoPaths.modelsDirectory
            .appendingPathComponent(runName, isDirectory: true)
            .appendingPathComponent("train.log")
    }

    /// Command line from `TrainParameterSpec`. Fine-tune and resume leave out the architecture.
    static func arguments(mode: Mode, config: TrainConfig) -> [String] {
        var args: [String]
        switch mode {
        case .new: args = ["start", "--name", config.name]
        case .fineTune(let id): args = ["start", "--name", config.name, "--from", String(id)]
        case .resume(let name): args = ["resume", name]
        }
        for spec in TrainParameterSpec.all {
            if spec.scope == .newOnly && mode.locksArchitecture { continue }
            let text = spec.read(config)
            if case .bool = spec.kind {
                args.append(text == "true" ? "--\(spec.flag)" : "--no-\(spec.flag)")
            } else {
                args += ["--\(spec.flag)", text]
            }
        }
        return args
    }

    /// Spawns the trainer with `setsid`, stdin from /dev/null and the log appended.
    /// `finished` runs on the main queue when the process ends while the app still runs.
    @discardableResult
    static func launch(
        arguments: [String], runName: String,
        finished: @escaping @MainActor (_ code: Int32, _ seconds: TimeInterval) -> Void
    ) throws -> pid_t {
        guard let binary = binaryURL() else {
            throw CLIError("whoodeo-train nicht gefunden (erwartet neben Whoodeo.app)")
        }
        let log = logURL(runName: runName)
        try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fd = open(log.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { throw CLIError("train.log lässt sich nicht öffnen: \(log.path)") }
        defer { close(fd) }

        let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withInternetDateTime])
        let quoted = arguments.map { $0.isEmpty || $0.contains(" ") ? "'\($0)'" : $0 }.joined(separator: " ")
        let header = "\n=== \(stamp)  whoodeo-train \(quoted)\n"
        _ = header.withCString { write(fd, $0, strlen($0)) }

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, fd, 1)
        posix_spawn_file_actions_adddup2(&actions, fd, 2)

        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        // Own session (no SIGHUP from the app), no inherited descriptors, default signal handling.
        let flags = POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK
        posix_spawnattr_setflags(&attr, Int16(flags))
        var defaults: sigset_t = 0
        for signal in [SIGINT, SIGTERM, SIGHUP, SIGPIPE, SIGUSR1, SIGQUIT] {
            defaults |= sigset_t(1) << sigset_t(signal - 1)
        }
        posix_spawnattr_setsigdefault(&attr, &defaults)
        var mask: sigset_t = 0
        posix_spawnattr_setsigmask(&attr, &mask)

        let argv: [UnsafeMutablePointer<CChar>?] = ([binary.path] + arguments).map { strdup($0) } + [nil]
        defer { for pointer in argv { free(pointer) } }
        var pid: pid_t = 0
        let started = Date()
        let result = posix_spawn(&pid, binary.path, &actions, &attr, argv, environ)
        guard result == 0 else {
            throw CLIError("whoodeo-train ließ sich nicht starten: \(String(cString: strerror(result)))")
        }
        let child = pid
        DispatchQueue.global(qos: .utility).async {
            var status: Int32 = 0
            while waitpid(child, &status, 0) < 0 && errno == EINTR {}
            let signal = status & 0x7f
            let code = signal == 0 ? (status >> 8) & 0xff : 128 + signal
            let seconds = Date().timeIntervalSince(started)
            DispatchQueue.main.async { finished(code, seconds) }
        }
        return pid
    }

    /// Last non-empty lines of the log, for a short error note.
    static func logTail(runName: String, lines: Int = 2) -> String {
        guard let text = try? String(contentsOf: logURL(runName: runName), encoding: .utf8) else { return "" }
        return text.split(separator: "\n").suffix(lines).joined(separator: " · ")
    }
}
