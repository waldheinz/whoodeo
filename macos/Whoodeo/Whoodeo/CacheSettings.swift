import Combine
import CoreMedia
import CoreVideo
import Foundation
import os

/// ProRes flavor of new frame caches. Every archive records its own flavor in `meta.json`,
/// so changing the choice leaves existing caches readable and appendable as they are.
nonisolated enum CacheCodec: String, CaseIterable, Identifiable, Codable, Sendable {
    case proRes422Proxy, proRes422LT, proRes422, proRes422HQ, proRes4444

    var id: String { rawValue }

    /// Archives from before the setting have no flavor in `meta.json`; they are ProRes 4444.
    static let legacy = CacheCodec.proRes4444
    static let standard = CacheCodec.proRes422LT

    var title: String {
        switch self {
        case .proRes422Proxy: "ProRes 422 Proxy"
        case .proRes422LT: "ProRes 422 LT"
        case .proRes422: "ProRes 422"
        case .proRes422HQ: "ProRes 422 HQ"
        case .proRes4444: "ProRes 4444"
        }
    }

    /// Rough size against ProRes 4444, from Apple's ProRes target data rates.
    var sizeNote: String {
        switch self {
        case .proRes422Proxy: "ca. 15 % von 4444"
        case .proRes422LT: "ca. 35 % von 4444"
        case .proRes422: "ca. 50 % von 4444"
        case .proRes422HQ: "ca. 75 % von 4444"
        case .proRes4444: "größte Dateien"
        }
    }

    var codecType: CMVideoCodecType {
        switch self {
        case .proRes422Proxy: kCMVideoCodecType_AppleProRes422Proxy
        case .proRes422LT: kCMVideoCodecType_AppleProRes422LT
        case .proRes422: kCMVideoCodecType_AppleProRes422
        case .proRes422HQ: kCMVideoCodecType_AppleProRes422HQ
        case .proRes4444: kCMVideoCodecType_AppleProRes4444
        }
    }

    /// What the encoder is fed. 4:2:2 flavors drop the chroma anyway, so they get 10-bit 4:2:2.
    var pixelFormat: OSType {
        self == .proRes4444 ? kCVPixelFormatType_4444AYpCbCr16 : kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
    }
}

/// Where the frame caches live, which flavor new ones get, and how large they may grow.
/// Plain `UserDefaults`, read from any thread. The settings window writes the same keys.
nonisolated enum CacheSettings {
    static let codecKey = "cacheCodec"
    static let directoryKey = "cacheDirectory"
    static let limitEnabledKey = "cacheLimitEnabled"
    static let limitGBKey = "cacheLimitGB"
    static let defaultLimitGB = 100
    /// Posted on the main thread after the location, the limit, or the cache contents changed.
    static let changed = Notification.Name("WhoodeoCacheSettingsChanged")

    /// Marks a folder as a Whoodeo cache, so a chosen folder can be told apart from one with other files.
    static let markerName = ".whoodeo-cache"

    static var defaultRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Whoodeo/Cache", isDirectory: true)
    }

    static var codec: CacheCodec {
        get { UserDefaults.standard.string(forKey: codecKey).flatMap(CacheCodec.init(rawValue:)) ?? .standard }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: codecKey) }
    }

    /// The configured folder. The default folder when none is set.
    static var root: URL {
        guard let path = UserDefaults.standard.string(forKey: directoryKey), !path.isEmpty else { return defaultRoot }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static var isDefaultRoot: Bool {
        root.standardizedFileURL.path == defaultRoot.standardizedFileURL.path
    }

    /// Nil goes back to the default folder.
    static func setRoot(_ url: URL?) {
        if let url, url.standardizedFileURL.path != defaultRoot.standardizedFileURL.path {
            UserDefaults.standard.set(url.standardizedFileURL.path, forKey: directoryKey)
        } else {
            UserDefaults.standard.removeObject(forKey: directoryKey)
        }
    }

    /// The default folder is created on demand. A custom folder must exist already: when its drive
    /// is not connected, creating it would put an empty folder under /Volumes on the system disk.
    static var isAvailable: Bool {
        if isDefaultRoot { return true }
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: root.path, isDirectory: &directory) && directory.boolValue
    }

    /// The root, created when it is the default. Throws when a custom folder is not reachable.
    static func availableRoot() throws -> URL {
        let url = root
        guard isAvailable else {
            throw CLIError("Cache-Ordner „\(url.path)“ ist nicht erreichbar. Ist das Laufwerk angeschlossen?")
        }
        if isDefaultRoot {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    /// Bytes, or nil when the limit is off.
    static var limitBytes: Int64? {
        guard UserDefaults.standard.bool(forKey: limitEnabledKey) else { return nil }
        let stored = UserDefaults.standard.integer(forKey: limitGBKey)
        let gb = stored > 0 ? stored : defaultLimitGB
        return Int64(gb) * 1_000_000_000
    }

    /// The folder to use for a choice in the open panel. An empty folder or an existing cache is used
    /// as it is; a folder with other files gets a "Whoodeo Cache" folder inside, so pruning never
    /// touches anything that is not a cache.
    static func cacheFolder(for chosen: URL) -> URL {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: chosen.path))?.filter { !$0.hasPrefix(".") } ?? []
        let marked = FileManager.default.fileExists(atPath: chosen.appendingPathComponent(markerName).path)
        if marked || files.isEmpty || chosen.standardizedFileURL.path == defaultRoot.standardizedFileURL.path {
            return chosen
        }
        return chosen.appendingPathComponent("Whoodeo Cache", isDirectory: true)
    }

    static func writeMarker(in root: URL) {
        let marker = root.appendingPathComponent(markerName)
        if !FileManager.default.fileExists(atPath: marker.path) {
            FileManager.default.createFile(atPath: marker.path, contents: Data("Whoodeo frame cache\n".utf8))
        }
    }

    /// Film folders are named by the 64-digit hex digest of the film.
    static func isFilmFolderName(_ name: String) -> Bool {
        let base = name.hasSuffix(".deleting") ? String(name.dropLast(".deleting".count)) : name
        return base.count == 64 && base.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }
}

/// Bytes of the frame caches on disk. A walk sets the total; every stored frame adds its bytes,
/// so the limit holds without walking the folder on each frame.
nonisolated enum CacheBudget {
    private static let total = OSAllocatedUnfairLock<Int64?>(initialState: nil)

    static var bytes: Int64? { total.withLock { $0 } }

    static func set(_ bytes: Int64) { total.withLock { $0 = bytes } }

    static func add(_ bytes: Int) { total.withLock { if $0 != nil { $0! += Int64(bytes) } } }

    /// True when a limit is set and reached. Before the first walk the size is unknown and nothing is full.
    static var isFull: Bool {
        guard let limit = CacheSettings.limitBytes, let bytes else { return false }
        return bytes >= limit
    }

    struct Entry: Identifiable, Hashable, Sendable {
        var id: String
        var bytes: Int64
    }

    /// Size of every film folder under the root, largest first. Hidden files and claims being deleted are left out.
    static func scan(_ root: URL) -> [Entry] {
        let manager = FileManager.default
        guard let children = try? manager.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
        ) else { return [] }
        var entries: [Entry] = []
        for child in children where !child.lastPathComponent.hasSuffix(".deleting") {
            guard (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
            entries.append(Entry(id: child.lastPathComponent, bytes: size(of: child)))
        }
        return entries.sorted { $0.bytes > $1.bytes }
    }

    static func size(of url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileSizeKey]
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys) else { return 0 }
        var sum: Int64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: Set(keys)), values.isRegularFile == true else { continue }
            sum += Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
        }
        return sum
    }
}

/// The settings window's view of the cache: total, per film, and a running move.
@MainActor
final class CacheUsage: ObservableObject {
    static let shared = CacheUsage()

    @Published private(set) var entries: [CacheBudget.Entry] = []
    @Published private(set) var total: Int64?
    @Published private(set) var scanning = false
    @Published private(set) var available = CacheSettings.isAvailable
    /// 0...1 while a move runs.
    @Published private(set) var moveProgress: Double?
    @Published var moveError: String?

    private var pending = false

    private init() {}

    /// Walk the folder in the background. A call during a walk runs one more walk afterwards.
    func refresh() {
        if scanning {
            pending = true
            return
        }
        scanning = true
        available = CacheSettings.isAvailable
        let root = CacheSettings.root
        let reachable = available
        DispatchQueue.global(qos: .utility).async {
            let found = reachable ? CacheBudget.scan(root) : []
            let sum = found.reduce(Int64(0)) { $0 + $1.bytes }
            if reachable { CacheBudget.set(sum) }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    self.entries = found
                    self.total = reachable ? sum : nil
                    self.scanning = false
                    if self.pending {
                        self.pending = false
                        self.refresh()
                    }
                    NotificationCenter.default.post(name: CacheSettings.changed, object: nil)
                }
            }
        }
    }

    /// The latest count of stored frames, without a walk. The window calls this while it is open.
    func noteGrowth() {
        guard let bytes = CacheBudget.bytes, bytes != total else { return }
        total = bytes
    }

    var isMoving: Bool { moveProgress != nil }

    /// Switch to `target`. With `move`, the open archives close first, every file is copied and checked
    /// by size, and only then the old files are deleted. A failed copy leaves the old cache in place.
    func relocate(to target: URL?, move: Bool) {
        guard !isMoving else { return }
        let source = CacheSettings.root
        let destination = target ?? CacheSettings.defaultRoot
        guard source.standardizedFileURL.path != destination.standardizedFileURL.path else { return }
        moveError = nil
        if let reason = Session.current?.cacheBusyReason {
            moveError = reason
            return
        }
        guard move, CacheSettings.isAvailable else {
            Session.current?.suspendCache {
                Self.adopt(target, destination: destination)
                Session.current?.resumeCache()
                self.refresh()
            }
            return
        }
        moveProgress = 0
        Session.current?.suspendCache {
            DispatchQueue.global(qos: .userInitiated).async {
                let result = Result {
                    try CacheMover.move(from: source, to: destination) { fraction in
                        DispatchQueue.main.async {
                            MainActor.assumeIsolated { self.moveProgress = fraction }
                        }
                    }
                }
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        self.moveProgress = nil
                        switch result {
                        case .success:
                            Self.adopt(target, destination: destination)
                        case .failure(let error):
                            self.moveError = error.localizedDescription
                            Log.fill.error("Cache verschieben: \(error.localizedDescription, privacy: .public)")
                        }
                        Session.current?.resumeCache()
                        self.refresh()
                    }
                }
            }
        }
    }

    private static func adopt(_ target: URL?, destination: URL) {
        if target != nil {
            try? FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            CacheSettings.writeMarker(in: destination)
        }
        CacheSettings.setRoot(target)
        Log.fill.info("Cache-Ordner: \(destination.path, privacy: .public)")
    }

    /// Remove one film's caches. The film stays in the list; its frames are computed again when needed.
    func clear(filmID: String) {
        if let problem = Session.current?.clearCache(filmID: filmID) {
            moveError = problem
        }
        refresh()
    }
}

/// Copies a cache folder across volumes, checks every file, then removes the source.
nonisolated enum CacheMover {
    static func move(from source: URL, to destination: URL, progress: @escaping (Double) -> Void) throws {
        let manager = FileManager.default
        let names = try manager.contentsOfDirectory(atPath: source.path).filter { name in
            !name.hasSuffix(".deleting") && name != ".DS_Store" && name != CacheSettings.markerName
        }
        try manager.createDirectory(at: destination, withIntermediateDirectories: true)
        for name in names where manager.fileExists(atPath: destination.appendingPathComponent(name).path) {
            throw CLIError("Im Ziel liegt schon „\(name)“. Nichts wurde verschoben.")
        }
        var files: [(from: URL, to: URL, size: Int64)] = []
        for name in names {
            let top = source.appendingPathComponent(name)
            var directory: ObjCBool = false
            manager.fileExists(atPath: top.path, isDirectory: &directory)
            if !directory.boolValue {
                files.append((top, destination.appendingPathComponent(name), fileSize(top)))
                continue
            }
            guard let walker = manager.enumerator(at: top, includingPropertiesForKeys: [.isRegularFileKey]) else { continue }
            for case let file as URL in walker {
                guard (try? file.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
                let relative = String(file.standardizedFileURL.path.dropFirst(source.standardizedFileURL.path.count + 1))
                files.append((file, destination.appendingPathComponent(relative), fileSize(file)))
            }
        }
        let needed = files.reduce(Int64(0)) { $0 + $1.size }
        // exFAT drives report 0 for "important usage", so the larger of the two counts.
        let values = try? destination.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        let free = max(values?.volumeAvailableCapacityForImportantUsage ?? 0, Int64(values?.volumeAvailableCapacity ?? 0))
        if free < needed + 1_000_000_000 {
            throw CLIError("Im Ziel ist nicht genug Platz (\(ByteCountFormatter.string(fromByteCount: needed, countStyle: .file)) nötig).")
        }
        var copied: [URL] = []
        do {
            var done: Int64 = 0
            for file in files {
                try manager.createDirectory(at: file.to.deletingLastPathComponent(), withIntermediateDirectories: true)
                copied.append(file.to)
                try copy(file.from, to: file.to) { bytes in
                    done += Int64(bytes)
                    progress(needed > 0 ? Double(done) / Double(needed) : 1)
                }
                guard fileSize(file.to) == file.size else {
                    throw CLIError("„\(file.from.lastPathComponent)“ kam nicht vollständig an.")
                }
            }
        } catch {
            // Only what this move wrote; the top-level names were checked to be new.
            for name in names { try? manager.removeItem(at: destination.appendingPathComponent(name)) }
            throw error
        }
        for name in names {
            try? manager.removeItem(at: source.appendingPathComponent(name))
        }
        progress(1)
        Log.fill.info("Cache verschoben: \(files.count, privacy: .public) Dateien, \(needed, privacy: .public) Bytes")
    }

    private static func fileSize(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? -1)
    }

    /// Chunked copy, so progress moves within a large `frames` file. The copy is synchronized before it counts.
    private static func copy(_ from: URL, to: URL, step: (Int) -> Void) throws {
        let input = try FileHandle(forReadingFrom: from)
        defer { try? input.close() }
        guard FileManager.default.createFile(atPath: to.path, contents: nil) else {
            throw CLIError("„\(to.path)“ lässt sich nicht anlegen.")
        }
        let output = try FileHandle(forWritingTo: to)
        defer { try? output.close() }
        while true {
            let chunk = try autoreleasepool { try input.read(upToCount: 16 << 20) }
            guard let chunk, !chunk.isEmpty else { break }
            try output.write(contentsOf: chunk)
            step(chunk.count)
        }
        try output.synchronize()
    }
}
