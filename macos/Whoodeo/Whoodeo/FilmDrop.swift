import AppKit
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
        contentTypes.map { NSPasteboard.PasteboardType($0.identifier) }
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
        let options: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingFileURLsOnly: true]
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: options) as? [URL],
           let film = urls.first(where: isFilm) {
            return film
        }
        for type in [NSPasteboard.PasteboardType.fileURL] + pasteboardTypes {
            guard let text = pasteboard.string(forType: type),
                  let url = URL(string: text),
                  isFilm(url) else { continue }
            return url
        }
        return nil
    }

    static func read(_ providers: [NSItemProvider], done: @escaping ([URL]) -> Void) {
        guard !providers.isEmpty else {
            DispatchQueue.main.async { done([]) }
            return
        }
        let found = OrderedURLs()
        let group = DispatchGroup()
        for (index, provider) in providers.enumerated() {
            group.enter()
            let finish: (URL?) -> Void = { url in
                if let url, isFilm(url) {
                    found.store(index, url)
                }
                group.leave()
            }
            if provider.canLoadObject(ofClass: URL.self) {
                let held = HeldProvider(provider)
                _ = provider.loadObject(ofClass: URL.self) { object, _ in
                    if let object {
                        finish(object)
                    } else {
                        loadFileURL(held.provider, finish)
                    }
                }
            } else {
                loadFileURL(provider, finish)
            }
        }
        group.notify(queue: .main) { done(found.ordered()) }
    }

    private static func loadFileURL(_ provider: NSItemProvider, _ finish: @escaping (URL?) -> Void) {
        let identifiers = provider.registeredTypeIdentifiers
        let identifier = identifiers.contains(UTType.fileURL.identifier)
            ? UTType.fileURL.identifier
            : (identifiers.first ?? UTType.fileURL.identifier)
        provider.loadItem(forTypeIdentifier: identifier, options: nil) { item, _ in
            finish(url(from: item))
        }
    }

    private static func url(from item: Any?) -> URL? {
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

/// `NSItemProvider` is not sendable. The load callback only uses it after the call has returned.
private nonisolated final class HeldProvider: @unchecked Sendable {
    let provider: NSItemProvider

    init(_ provider: NSItemProvider) {
        self.provider = provider
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
