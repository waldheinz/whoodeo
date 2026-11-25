import Foundation
import SwiftUI

nonisolated struct LibraryModel: Identifiable, Hashable {
    var url: URL
    var title: String
    var detail: String
    var symbol: String
    var tint: Color
    /// A FiLM model: the codec condition vector is an input (`…-cond`).
    var conditionInput: Bool = false

    var id: URL { url }

    var hasCodecs: Bool { conditionInput }

    static func read(_ url: URL) -> LibraryModel {
        let stem = url.deletingPathExtension().lastPathComponent
        guard let match = stem.firstMatch(
            of: /^(?:train-(\d{8})-(\d{4})\d{2}-)?latent_up-(\d+)x(\d+)-step(\d+)-(\d+)f-hd-align(?:-cond)?$/
        ) else {
            return LibraryModel(
                url: url, title: stem, detail: "MLX",
                symbol: "cube", tint: .secondary
            )
        }
        let blocks = String(match.3)
        let channels = String(match.4)
        let step = Int(match.5) ?? 0
        // "-3f" behind the step: a model on 3 frames, previous, current, next.
        let frames = Int(match.6) ?? 3
        var detail = "Schritt \(step)"
        if let day = match.1, let time = match.2 {
            // Run started at, e.g. 22.10. 01:35
            let d = Array(day), t = Array(time)
            detail += " · Lauf \(String(d[6...7])).\(String(d[4...5])). \(String(t[0...1])):\(String(t[2...3]))"
        }
        var model = LibraryModel(
            url: url,
            title: "latent_up \(blocks)×\(channels) · \(frames) Bilder",
            detail: detail,
            symbol: "arrow.up.left.and.arrow.down.right",
            tint: .orange
        )
        if stem.hasSuffix("-cond") {
            model.conditionInput = true
            model.detail += " · Codec-Bedingung"
        }
        return model
    }
}

nonisolated enum ModelLibrary {
    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("Whoodeo/Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func models() -> [LibraryModel] {
        models(in: directory)
    }

    /// The sidebar line of this package, read from its folder.
    static func family(of package: URL) -> LibraryModel {
        let wanted = package.standardizedFileURL.resolvingSymlinksInPath()
        let folder = wanted.deletingLastPathComponent()
        return models(in: folder).first {
            $0.url.standardizedFileURL.resolvingSymlinksInPath() == wanted
        } ?? LibraryModel.read(package)
    }

    static func models(in folder: URL) -> [LibraryModel] {
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls
            .filter { $0.pathExtension == MLXAlignModel.fileExtension }
            .sorted { modified($0) > modified($1) }
            .map(LibraryModel.read)
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
