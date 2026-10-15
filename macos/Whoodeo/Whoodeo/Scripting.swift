import AppKit

/// AppleScript access to the open session. The picture stays with the slider:
/// seek, play and pause are the same actions as the controls in the window.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func application(_ application: NSApplication, delegateHandlesKey key: String) -> Bool {
        Self.keys.contains(key)
    }

    @objc var scriptFrame: Int {
        get {
            MainActor.assumeIsolated {
                guard let session = Session.current, session.frameCount > 0 else { return 0 }
                return session.index + 1
            }
        }
        set { MainActor.assumeIsolated { Session.current?.seekToBild(newValue) } }
    }

    @objc var scriptFrameCount: Int {
        MainActor.assumeIsolated { Session.current?.frameCount ?? 0 }
    }

    @objc var scriptPlaying: Bool {
        get { MainActor.assumeIsolated { Session.current?.playing ?? false } }
        set {
            MainActor.assumeIsolated {
                guard let session = Session.current else { return }
                if newValue { session.play() } else { session.pause() }
            }
        }
    }

    @objc var scriptFillEnabled: Bool {
        get { MainActor.assumeIsolated { Session.current?.fillCache ?? false } }
        set { MainActor.assumeIsolated { Session.current?.fillCache = newValue } }
    }

    @objc var scriptStatus: String {
        MainActor.assumeIsolated { Session.current?.status ?? "" }
    }

    @objc var scriptFillError: String {
        MainActor.assumeIsolated { Session.current?.backgroundFillError ?? "" }
    }

    @objc var scriptCachedFrames: Int {
        MainActor.assumeIsolated { Session.current?.cachedCount ?? 0 }
    }

    @objc var scriptFilling: Int {
        MainActor.assumeIsolated { Session.current?.fillingBild ?? 0 }
    }

    @objc var scriptModel: String {
        get { MainActor.assumeIsolated { Session.current?.scriptModelName ?? "" } }
        set {
            MainActor.assumeIsolated {
                guard let problem = Session.current?.chooseModel(named: newValue) else { return }
                NSScriptCommand.current()?.scriptErrorNumber = errAENoSuchObject
                NSScriptCommand.current()?.scriptErrorString = problem
            }
        }
    }

    @objc var scriptModels: [String] {
        MainActor.assumeIsolated { Session.current?.scriptModelNames ?? [] }
    }

    @objc var scriptCodec: String {
        get { MainActor.assumeIsolated { Session.current?.codec.rawValue ?? "hevc" } }
        set {
            MainActor.assumeIsolated {
                guard let problem = Session.current?.setCodec(named: newValue) else { return }
                NSScriptCommand.current()?.scriptErrorNumber = errAECoercionFail
                NSScriptCommand.current()?.scriptErrorString = problem
            }
        }
    }

    @objc var scriptInPoint: Int {
        get {
            MainActor.assumeIsolated {
                guard let session = Session.current, session.frameCount > 0 else { return 0 }
                return session.inPoint + 1
            }
        }
        set { MainActor.assumeIsolated { Session.current?.setInBild(newValue) } }
    }

    @objc var scriptOutPoint: Int {
        get {
            MainActor.assumeIsolated {
                guard let session = Session.current, session.frameCount > 0 else { return 0 }
                return session.outPoint + 1
            }
        }
        set { MainActor.assumeIsolated { Session.current?.setOutBild(newValue) } }
    }

    @objc var scriptExporting: Bool {
        MainActor.assumeIsolated { Session.current?.exporting ?? false }
    }

    @objc var scriptFilmPath: String {
        MainActor.assumeIsolated { Session.current?.scriptFilmPath ?? "" }
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            if let session = Session.current {
                _ = session.openFilms(urls)
            } else {
                Self.pendingFilms.append(contentsOf: urls)
            }
        }
    }

    static var pendingFilms: [URL] = []

    private static let keys: Set<String> = [
        "scriptFrame", "scriptFrameCount", "scriptPlaying", "scriptFillEnabled",
        "scriptStatus", "scriptFillError", "scriptCachedFrames", "scriptFilling",
        "scriptModel", "scriptModels",
        "scriptCodec", "scriptInPoint", "scriptOutPoint", "scriptExporting",
        "scriptFilmPath",
    ]
}

@objc(WhoodeoPlayCommand)
final class WhoodeoPlayCommand: NSScriptCommand {
    override func performDefaultImplementation() -> Any? {
        MainActor.assumeIsolated { Session.current?.play() }
        return nil
    }
}

@objc(WhoodeoPauseCommand)
final class WhoodeoPauseCommand: NSScriptCommand {
    override func performDefaultImplementation() -> Any? {
        MainActor.assumeIsolated { Session.current?.pause() }
        return nil
    }
}

@objc(WhoodeoSeekCommand)
final class WhoodeoSeekCommand: NSScriptCommand {
    override func performDefaultImplementation() -> Any? {
        let bild: Int?
        if let value = directParameter as? Int {
            bild = value
        } else if let number = directParameter as? NSNumber {
            bild = number.intValue
        } else {
            bild = nil
        }
        guard let bild else {
            scriptErrorNumber = errAECoercionFail
            return nil
        }
        MainActor.assumeIsolated { Session.current?.seekToBild(bild) }
        return nil
    }
}

@objc(WhoodeoOpenFilmCommand)
final class WhoodeoOpenFilmCommand: NSScriptCommand {
    override func performDefaultImplementation() -> Any? {
        let path: String?
        if let text = directParameter as? String {
            path = text
        } else if let url = directParameter as? URL {
            path = url.path
        } else if let url = directParameter as? NSURL {
            path = url.path
        } else {
            path = nil
        }
        guard let path else {
            scriptErrorNumber = errAECoercionFail
            return nil
        }
        var problem: String?
        MainActor.assumeIsolated {
            problem = Session.current?.openFilm(at: path)
        }
        if let problem {
            scriptErrorNumber = errAENoSuchObject
            scriptErrorString = problem
        }
        return nil
    }
}

@objc(WhoodeoExportCommand)
final class WhoodeoExportCommand: NSScriptCommand {
    override func performDefaultImplementation() -> Any? {
        let path: String?
        if let text = directParameter as? String {
            path = text
        } else if let url = directParameter as? URL {
            path = url.path
        } else if let url = directParameter as? NSURL {
            path = url.path
        } else {
            path = nil
        }
        guard let path else {
            scriptErrorNumber = errAECoercionFail
            return nil
        }
        var problem: String?
        MainActor.assumeIsolated {
            problem = Session.current?.exportFilm(to: path)
        }
        if let problem {
            scriptErrorNumber = errAEEventFailed
            scriptErrorString = problem
        }
        return nil
    }
}
