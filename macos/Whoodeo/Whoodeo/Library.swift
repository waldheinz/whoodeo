import Foundation
import SwiftUI

nonisolated struct LibraryModel: Identifiable, Hashable {
    var url: URL
    var title: String
    var detail: String
    var symbol: String
    var tint: Color

    var id: URL { url }

    static func read(_ url: URL) -> LibraryModel {
        let stem = url.deletingPathExtension().lastPathComponent
        guard let match = stem.firstMatch(of: /(vote|edsr)-(\d+)x(\d+)-step(\d+)-flex$/) else {
            return LibraryModel(
                url: url, title: stem, detail: "Core ML",
                symbol: "cube", tint: .secondary
            )
        }
        let arch = String(match.1)
        let blocks = String(match.2)
        let channels = String(match.3)
        let step = Int(match.4) ?? 0
        return LibraryModel(
            url: url,
            title: "\(arch) \(blocks)×\(channels)",
            detail: "Schritt \(step) · flexibel",
            symbol: "arrow.up.left.and.arrow.down.right",
            tint: .blue
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
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls
            .filter { $0.pathExtension == "mlpackage" }
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
            dest = folder.appendingPathComponent("\(stem)-\(number).mlpackage")
            number += 1
        }
        return dest
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
