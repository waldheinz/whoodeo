import AppKit
import os
import UniformTypeIdentifiers

/// Movie files the window accepts when they are dropped onto it.
nonisolated enum FilmDrop {
    static var contentTypes: [UTType] {
        var types: [UTType] = [.movie, .mpeg4Movie, .quickTimeMovie, .video]
        if let mkv = UTType(filenameExtension: "mkv"), !types.contains(where: { $0 == mkv }) {
            types.append(mkv)
        }
        return types
    }

    static var pasteboardTypes: [NSPasteboard.PasteboardType] {
        // Finder puts public.file-url on a file drag, not public.mpeg-4. A view
        // registered only for movie types never hears about the drop.
        var types = contentTypes.map { NSPasteboard.PasteboardType($0.identifier) }
        if !types.contains(.fileURL) { types.append(.fileURL) }
        return types
    }

    /// Types for SwiftUI `onDrop`. SwiftUI hands over only the types listed here, so
    /// without public.file-url a Finder drop arrives as a bare movie and the URL is lost.
    static var dropTypes: [UTType] {
        [.fileURL] + contentTypes
    }

    static func isFilm(_ url: URL) -> Bool {
        guard !url.hasDirectoryPath,
              let type = UTType(filenameExtension: url.pathExtension)
        else { return false }
        return contentTypes.contains { type.conforms(to: $0) }
    }

    @MainActor
    static func firstFilm(in info: NSDraggingInfo) -> URL? {
        let pasteboard = info.draggingPasteboard
        let types = (pasteboard.types ?? []).map(\.rawValue).joined(separator: ", ")
        Log.drop.debug("Pasteboard-Typen: \(types, privacy: .public)")
        let options: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingFileURLsOnly: true]
        if let urls = (pasteboard.readObjects(forClasses: [NSURL.self], options: options) as? [URL])?
            .compactMap({ resolved($0) }),
           let film = urls.first(where: isFilm) {
            return film
        }
        Log.drop.info("Kein Film auf dem Pasteboard, Typen: \(types, privacy: .public)")
        for type in [NSPasteboard.PasteboardType.fileURL] + pasteboardTypes {
            guard let text = pasteboard.string(forType: type),
                  let url = resolved(URL(string: text)),
                  isFilm(url) else { continue }
            return url
        }
        return nil
    }

    static func read(_ providers: [NSItemProvider], done: @escaping ([URL]) -> Void) {
        Log.drop.info("SwiftUI-Drop mit \(providers.count) Provider(n)")
        guard !providers.isEmpty else {
            DispatchQueue.main.async { done([]) }
            return
        }
        let found = OrderedURLs()
        let group = DispatchGroup()
        for (index, provider) in providers.enumerated() {
            group.enter()
            let identifiers = provider.registeredTypeIdentifiers
            Log.drop.info("Provider \(index): \(identifiers.joined(separator: ", "), privacy: .public)")
            let finish: (URL?) -> Void = { url in
                let film = url.map(isFilm) ?? false
                Log.drop.info("Provider \(index): URL \(url?.path ?? "keine", privacy: .public), Film: \(film)")
                if let url, film {
                    found.store(index, url)
                }
                group.leave()
            }
            loadURL(provider, finish)
        }
        group.notify(queue: .main) { done(found.ordered()) }
    }

    /// Reads the file URL itself. Loading the data or a file representation of a
    /// movie type copies the whole film first (593 MB from SMB) and hands over a
    /// temporary copy, so that path only runs in place.
    private static func loadURL(_ provider: NSItemProvider, _ finish: @escaping (URL?) -> Void) {
        let identifiers = provider.registeredTypeIdentifiers
        if identifiers.contains(UTType.fileURL.identifier) {
            provider.loadDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier) { data, error in
                if let error {
                    Log.drop.error("file-url nicht lesbar: \(error.localizedDescription, privacy: .public)")
                }
                finish(url(from: data))
            }
            return
        }
        guard let movie = identifiers.first(where: { id in
            UTType(id).map { type in contentTypes.contains { type.conforms(to: $0) } } ?? false
        }) else {
            Log.drop.info("Provider ohne file-url und ohne Filmtyp")
            finish(nil)
            return
        }
        provider.loadInPlaceFileRepresentation(forTypeIdentifier: movie) { url, inPlace, error in
            if let error {
                Log.drop.error("\(movie, privacy: .public) nicht lesbar: \(error.localizedDescription, privacy: .public)")
            }
            // A copy disappears when this handler returns. Only the original is any use.
            guard inPlace else {
                Log.drop.error("Nur eine Kopie bekommen: \(url?.path ?? "keine", privacy: .public)")
                finish(nil)
                return
            }
            finish(url)
        }
    }

    private static func url(from item: Any?) -> URL? {
        resolved(rawURL(from: item))
    }

    /// Finder may hand over a file reference URL (file:///.file/id=…) without a file name.
    private static func resolved(_ url: URL?) -> URL? {
        guard let url else { return nil }
        if url.isFileURL, (url as NSURL).isFileReferenceURL() {
            return (url as NSURL).filePathURL ?? url
        }
        return url
    }

    private static func rawURL(from item: Any?) -> URL? {
        if let url = item as? URL { return url }
        if let text = item as? String { return URL(string: text) }
        if let text = item as? NSString { return URL(string: text as String) }
        if let data = item as? Data {
            if let url = URL(dataRepresentation: data, relativeTo: nil) { return url }
            if let text = String(data: data, encoding: .utf8) { return URL(string: text) }
        }
        return nil
    }
}

private nonisolated final class OrderedURLs: @unchecked Sendable {
    private var items: [Int: URL] = [:]
    private let lock = NSLock()

    func store(_ index: Int, _ url: URL) {
        lock.lock()
        items[index] = url
        lock.unlock()
    }

    func ordered() -> [URL] {
        lock.lock()
        let urls = items.sorted { $0.key < $1.key }.map(\.value)
        lock.unlock()
        return urls
    }
}
