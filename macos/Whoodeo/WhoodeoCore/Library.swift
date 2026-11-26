import Foundation
import os

nonisolated enum LibraryTint: String, Hashable, Sendable {
    case orange
    case secondary
}

nonisolated struct LibraryModel: Identifiable, Hashable {
    var url: URL
    var title: String
    var detail: String
    var symbol: String
    var tint: LibraryTint
    /// A FiLM model: the codec condition vector is an input (`…-cond`).
    var conditionInput: Bool = false
    var checkpointID: Int? = nil
    var runName: String? = nil
    var step: Int? = nil
    var parentStep: Int? = nil

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
            tint: .orange,
            runName: WhoodeoDB.runName(forPackage: url),
            step: step
        )
        if stem.hasSuffix("-cond") {
            model.conditionInput = true
            model.detail += " · Codec-Bedingung"
        }
        return model
    }

    static func from(checkpoint: CheckpointRecord, modelsRoot: URL) -> LibraryModel {
        let url = modelsRoot.appendingPathComponent(checkpoint.relPath)
        let config = Self.config(checkpoint.configJSON)
        let blocks = config.blocks ?? Self.lastBlock(checkpoint.blocksJSON) ?? 0
        let frames = config.inFrames ?? 3
        var detail = "Schritt \(checkpoint.step)"
        if let runDate = Self.runDate(checkpoint.runName) {
            detail += " · Lauf \(runDate)"
        } else if !checkpoint.runName.isEmpty {
            detail += " · \(checkpoint.runName)"
        }
        if let parent = checkpoint.parentStep {
            let origin = checkpoint.parentRunName.map { " von \($0), Schritt \(parent)" } ?? " von Schritt \(parent)"
            detail += " ·\(origin)"
        }
        if (config.conditionSize ?? 0) > 0 {
            detail += " · Codec-Bedingung"
        }
        return LibraryModel(
            url: url,
            title: "latent_up \(blocks)×\(checkpoint.channels) · \(frames) Bilder",
            detail: detail,
            symbol: "arrow.up.left.and.arrow.down.right",
            tint: .orange,
            conditionInput: (config.conditionSize ?? 0) > 0,
            checkpointID: checkpoint.id,
            runName: checkpoint.runName,
            step: checkpoint.step,
            parentStep: checkpoint.parentStep
        )
    }

    private struct ConfigBits {
        var blocks: Int?
        var inFrames: Int?
        var conditionSize: Int?
    }

    private static func config(_ json: String) -> ConfigBits {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return ConfigBits() }
        return ConfigBits(
            blocks: object["blocks"] as? Int,
            inFrames: object["in_frames"] as? Int,
            conditionSize: object["condition_size"] as? Int
        )
    }

    private static func lastBlock(_ json: String) -> Int? {
        guard let data = json.data(using: .utf8),
              let values = try? JSONSerialization.jsonObject(with: data) as? [Int]
        else { return nil }
        return values.last
    }

    private static func runDate(_ name: String) -> String? {
        guard let match = name.firstMatch(of: /^train-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/) else { return nil }
        return "\(match.3).\(match.2). \(match.4):\(match.5)"
    }
}

nonisolated enum ModelLibrary {
    static var directory: URL { WhoodeoPaths.modelsDirectory }

    static func models() -> [LibraryModel] {
        models(in: directory)
    }

    /// The sidebar line of this package, read from its folder or the database.
    static func family(of package: URL) -> LibraryModel {
        let wanted = package.standardizedFileURL.resolvingSymlinksInPath()
        let folder = wanted.deletingLastPathComponent()
        return models(in: folder).first {
            $0.url.standardizedFileURL.resolvingSymlinksInPath() == wanted
        } ?? LibraryModel.read(package)
    }

    static func models(in folder: URL) -> [LibraryModel] {
        if folder.standardizedFileURL.resolvingSymlinksInPath()
            == directory.standardizedFileURL.resolvingSymlinksInPath(),
           let db = WhoodeoDB.shared
        {
            do {
                return try modelsFromDatabase(db, folder: folder)
            } catch {
                Log.model.error("Modellliste aus der Datenbank: \(error.localizedDescription, privacy: .public)")
            }
        }
        return modelsFromFolder(folder)
    }

    private static func modelsFromDatabase(_ db: WhoodeoDB, folder: URL) throws -> [LibraryModel] {
        _ = try db.adoptOrphanModels()
        let records = try db.allCheckpoints()
        let known = Set(
            (try db.allCheckpoints()).map {
                folder.appendingPathComponent($0.relPath).standardizedFileURL.resolvingSymlinksInPath()
            }
        )
        var seen = Set<URL>()
        var models: [LibraryModel] = []
        for record in records {
            let url = folder.appendingPathComponent(record.relPath)
            let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
            guard FileManager.default.fileExists(atPath: resolved.path) else { continue }
            models.append(LibraryModel.from(checkpoint: record, modelsRoot: folder))
            seen.insert(resolved)
        }
        // Packages that appeared between bootstrap and this read, or that failed to import.
        // Every checkpoint in the database is listed. visible_in_app is no longer a filter.
        for fallback in modelsFromFolder(folder) {
            let resolved = fallback.url.standardizedFileURL.resolvingSymlinksInPath()
            if seen.contains(resolved) || known.contains(resolved) { continue }
            models.append(fallback)
        }
        return models.sorted { modified($0.url) > modified($1.url) }
    }

    /// The sidebar line of this package, read from its folder.
    static func modelsFromFolder(_ folder: URL) -> [LibraryModel] {
        return WhoodeoDB.collectPackages(in: folder)
            .sorted { modified($0) > modified($1) }
            .map(LibraryModel.read)
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
