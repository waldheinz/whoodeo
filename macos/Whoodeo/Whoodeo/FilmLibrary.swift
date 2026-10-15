import Foundation
import os

/// One film the user has opened and not removed. `id` is the digest from the moment it was added.
nonisolated struct OpenFilm: Codable, Identifiable, Hashable, Sendable {
    var id: String
    var path: String
    var width: Int
    var height: Int
    var frames: Int

    var name: String { URL(fileURLWithPath: path).lastPathComponent }

    var subtitle: String {
        if !FileManager.default.fileExists(atPath: path) { return "Datei fehlt" }
        guard frames > 0 else { return "Film" }
        let pictures = frames == 1 ? "1 Bild" : "\(frames) Bilder"
        return "\(width)×\(height) · \(pictures)"
    }
}

nonisolated struct FilmList: Codable, Sendable {
    var current: String
    var films: [OpenFilm]

    static let empty = FilmList(current: "", films: [])
}

/// The opened-film list in Application Support. The cache directories use the same ids.
/// Do not take the cache gate while this lock is held. Prune holds the gate and then reads the list.
nonisolated enum FilmLibrary {
    private static let lock = NSLock()

    static var file: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Whoodeo/films.json")
    }

    static func load() -> FilmList {
        lock.lock()
        defer { lock.unlock() }
        return read()
    }

    static func mutate<T>(_ body: (inout FilmList) throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        var list = read()
        let value = try body(&list)
        try write(list)
        return value
    }

    /// The same path keeps the id it got the first time. A new path gets a new digest.
    static func insert(_ url: URL, into list: inout FilmList) throws -> String {
        let file = url.standardizedFileURL
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw CLIError("Die Datei fehlt.")
        }
        guard FilmDrop.isFilm(file) else { throw CLIError("Das ist kein Film.") }
        if let existing = list.films.first(where: { $0.path == file.path }) {
            return existing.id
        }
        let id = FrameStore.filmDigest(of: file)
        if list.films.contains(where: { $0.id == id }) {
            throw CLIError("Diese ID gehört schon zu einem anderen Film.")
        }
        list.films.append(OpenFilm(id: id, path: file.path, width: 0, height: 0, frames: 0))
        return id
    }

    static func remember(_ id: String, width: Int, height: Int, frames: Int, in list: inout FilmList) {
        guard let index = list.films.firstIndex(where: { $0.id == id }) else { return }
        list.films[index].width = width
        list.films[index].height = height
        list.films[index].frames = frames
    }

    private static func read() -> FilmList {
        let url = file
        guard FileManager.default.fileExists(atPath: url.path) else { return .empty }
        do {
            return try JSONDecoder().decode(FilmList.self, from: Data(contentsOf: url))
        } catch {
            Log.fill.error("films.json nicht lesbar: \(error.localizedDescription, privacy: .public)")
            return .empty
        }
    }

    private static func write(_ list: FilmList) throws {
        let url = file
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(list).write(to: url, options: .atomic)
    }
}
