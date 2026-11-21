import Foundation
import SwiftUI

nonisolated struct LibraryModel: Identifiable, Hashable {
    var url: URL
    var title: String
    var detail: String
    var symbol: String
    var tint: Color
    /// A latent_up with FiLM: one package per codec, the same network otherwise.
    /// `url` is then the package of the unknown codec, or the first one there is. Empty without FiLM.
    var variants: [SourceCodec: URL] = [:]

    var id: URL { url }

    var hasCodecs: Bool { !variants.isEmpty }

    /// Every package behind this line of the sidebar.
    var urls: [URL] { variants.isEmpty ? [url] : SourceCodec.allCases.compactMap { variants[$0] } }

    /// The package for `codec`. A missing codec falls back to unknown, then to `url`.
    /// Without FiLM it is always `url`, and the codec is nil.
    func package(for codec: SourceCodec) -> (url: URL, codec: SourceCodec?) {
        guard !variants.isEmpty else { return (url, nil) }
        if let hit = variants[codec] { return (hit, codec) }
        if let unknown = variants[.unknown] { return (unknown, .unknown) }
        let first = SourceCodec.allCases.first { variants[$0] != nil }!
        return (variants[first]!, first)
    }

    /// "…-hd-h264" gives ("…-hd", h264). Names without a codec at the end give nil.
    static func splitCodec(_ stem: String) -> (base: String, codec: SourceCodec)? {
        guard let dash = stem.lastIndex(of: "-"),
              let codec = SourceCodec(rawValue: String(stem[stem.index(after: dash)...]))
        else { return nil }
        return (String(stem[..<dash]), codec)
    }

    static func read(_ url: URL) -> LibraryModel {
        let stem = url.deletingPathExtension().lastPathComponent
        guard let match = stem.firstMatch(
            of: /^(?:train-(\d{8})-(\d{4})\d{2}-)?(vote|edsr|latent_up)-(\d+)x(\d+)-step(\d+)(?:-(\d+)f)?-([a-z]+)(?:-(?:h264|mpeg4|mpeg2|unknown))?$/
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
        return LibraryModel(
            url: url,
            title: "\(arch) \(blocks)×\(channels)" + (frames > 1 ? " · \(frames) Bilder" : ""),
            detail: detail,
            symbol: "arrow.up.left.and.arrow.down.right",
            tint: arch == "latent_up" ? .purple : .blue
        )
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

    /// The sidebar line a package belongs to, with the codec packages beside it in its folder.
    static func family(of package: URL) -> LibraryModel {
        let wanted = package.standardizedFileURL.resolvingSymlinksInPath()
        let folder = wanted.deletingLastPathComponent()
        return models(in: folder).first {
            $0.urls.contains { $0.standardizedFileURL.resolvingSymlinksInPath() == wanted }
        } ?? LibraryModel.read(package)
    }

    static func models(in folder: URL) -> [LibraryModel] {
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )) ?? []
        let packages = urls
            .filter { $0.pathExtension == "mlpackage" }
            .sorted { modified($0) > modified($1) }
        // The codec packages of one export become one line, at the place of the newest.
        var order: [String] = []
        var groups: [String: [SourceCodec: URL]] = [:]
        var plain: [String: URL] = [:]
        for url in packages {
            let stem = url.deletingPathExtension().lastPathComponent
            if let (base, codec) = LibraryModel.splitCodec(stem) {
                let key = "codec:" + base
                if groups[key] == nil { order.append(key) }
                groups[key, default: [:]][codec] = url
            } else {
                order.append(stem)
                plain[stem] = url
            }
        }
        return order.map { key in
            if let url = plain[key] { return LibraryModel.read(url) }
            let variants = groups[key]!
            let main = variants[.unknown] ?? SourceCodec.allCases.compactMap { variants[$0] }.first!
            var model = LibraryModel.read(main)
            model.variants = variants
            let names = SourceCodec.allCases.filter { variants[$0] != nil }.map(\.title)
            model.detail += " · Codecs " + names.joined(separator: ", ")
            return model
        }
    }

    static func uniqueDestination(for source: URL) -> URL {
        let folder = directory
        var dest = folder.appendingPathComponent(source.lastPathComponent)
        if dest.standardizedFileURL == source.standardizedFileURL { return dest }
        let stem = source.deletingPathExtension().lastPathComponent
        var number = 2
        while FileManager.default.fileExists(atPath: dest.path) {
            dest = folder.appendingPathComponent("\(stem)-\(number).mlpackage")
            number += 1
        }
        return dest
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
