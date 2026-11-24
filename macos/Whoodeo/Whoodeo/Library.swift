import Foundation
import SwiftUI

nonisolated struct LibraryModel: Identifiable, Hashable {
    var url: URL
    var title: String
    var detail: String
    var symbol: String
    var tint: Color
    /// A FiLM package: one file with the condition vector as an input (`…-hd-cond`).
    var conditionInput: Bool = false

    var id: URL { url }

    var hasCodecs: Bool { conditionInput }

    static func read(_ url: URL) -> LibraryModel {
        let stem = url.deletingPathExtension().lastPathComponent
        guard let match = stem.firstMatch(
            of: /^(?:train-(\d{8})-(\d{4})\d{2}-)?(vote|edsr|latent_up)-(\d+)x(\d+)-step(\d+)(?:-(\d+)f)?-([a-z]+)(?:-align)?(?:-cond)?$/
        ) else {
            return LibraryModel(
                url: url, title: stem, detail: "Core ML",
                symbol: "cube", tint: .secondary
            )
        }
        let arch = String(match.3)
        let blocks = String(match.4)
        let channels = String(match.5)
        let step = Int(match.6) ?? 0
        // "-3f" behind the step: a model on 3 frames, previous, current, next.
        let frames = match.7.flatMap { Int($0) } ?? 1
        // The size variant (hd, or range from before the app was GPU only) is not shown.
        var detail = "Schritt \(step)"
        if let day = match.1, let time = match.2 {
            // Run started at, e.g. 22.10. 01:35
            let d = Array(day), t = Array(time)
            detail += " · Lauf \(String(d[6...7])).\(String(d[4...5])). \(String(t[0...1])):\(String(t[2...3]))"
        }
        var model = LibraryModel(
            url: url,
            title: "\(arch) \(blocks)×\(channels)" + (frames > 1 ? " · \(frames) Bilder" : ""),
            detail: detail,
            symbol: "arrow.up.left.and.arrow.down.right",
            tint: arch == "latent_up" ? .purple : .blue
        )
        if stem.contains("-align") {
            model.detail += " · Ausrichtung"
        }
        if stem.hasSuffix("-cond") {
            model.conditionInput = true
            model.detail += " · Codec-Bedingung"
        }
        if url.pathExtension == MLXAlignModel.fileExtension {
            model.title += " · MLX"
            model.tint = .orange
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
            .filter { $0.pathExtension == "mlpackage" || $0.pathExtension == MLXAlignModel.fileExtension }
            .sorted { modified($0) > modified($1) }
            .map(LibraryModel.read)
    }

    static func uniqueDestination(for source: URL) -> URL {
        let folder = directory
        var dest = folder.appendingPathComponent(source.lastPathComponent)
        if dest.standardizedFileURL == source.standardizedFileURL { return dest }
        let stem = source.deletingPathExtension().lastPathComponent
        var number = 2
        while FileManager.default.fileExists(atPath: dest.path) {
            dest = folder.appendingPathComponent("\(stem)-\(number).\(source.pathExtension)")
            number += 1
        }
        return dest
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
