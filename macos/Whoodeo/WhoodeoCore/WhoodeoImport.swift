import Foundation
import GRDB

nonisolated struct ImportReport: Sendable {
    var sourcesInserted = 0
    var mastersInserted = 0
    var variantsInserted = 0
    var runsInserted = 0
    var checkpointsInserted = 0
    var cacheFilesInserted = 0
    var missingFiles: [String] = []
    var notes: [String] = []
}

/// Imports that fill the database from files: the dataset catalog (once), `.mlxmodel` packages that
/// appear in `Models/`, and archive folders under the cache root. Each reads first and writes only
/// when there is something new, so a running trainer is not held up. Files are never moved or renamed.
nonisolated extension WhoodeoDB {
    /// Imports sources, masters, and low variants from the dataset's `catalog.json`. Existing rows are kept.
    func importCatalog(root: URL) throws -> ImportReport {
        let catalogURL = root.appendingPathComponent("catalog.json")
        let data = try Data(contentsOf: catalogURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let titles = json["titles"] as? [String: Any]
        else {
            throw WhoodeoDBError("catalog.json hat kein titles-Objekt: \(catalogURL.path)")
        }
        let now = Self.timestamp()
        let exists = { (rel: String) in FileManager.default.fileExists(atPath: root.appendingPathComponent(rel).path) }
        var report = ImportReport()
        try pool.write { db in
            try Self.setSetting(db, root.path, forKey: "dataset_root")
            try Self.importDegrade(db, from: json)
            for (name, raw) in titles.sorted(by: { $0.key < $1.key }) {
                guard !name.hasPrefix("._"), let entry = raw as? [String: Any] else { continue }
                let frames = Self.intValue(entry["frames"]) ?? 0
                guard frames > 0 else {
                    report.notes.append("übersprungen \(name): keine frames")
                    continue
                }
                let source = try Self.insertOrFind(db, into: "sources", [
                    "name": name, "origin_path": entry["origin"] as? String, "frames": frames,
                    "split": Self.titleSplit(name), "added_at": now,
                ], uniqueBy: ["name"])
                if source.inserted { report.sourcesInserted += 1 }
                let masters = entry["masters"] as? [String: Any] ?? [:]
                for (rung, masterRaw) in masters.sorted(by: { $0.key < $1.key }) {
                    guard let master = masterRaw as? [String: Any] else { continue }
                    let rel = "orig/\(name).\(rung).mkv"
                    if rel.split(separator: "/").contains(where: { $0.hasPrefix("._") }) { continue }
                    if !exists(rel) { report.missingFiles.append(rel) }
                    let masterRow = try Self.insertOrFind(db, into: "source_masters", [
                        "source_id": source.id, "rung": rung, "rel_path": rel,
                        "width": Self.intValue(master["width"]) ?? 0, "height": Self.intValue(master["height"]) ?? 0,
                    ], uniqueBy: ["source_id", "rung"])
                    if masterRow.inserted { report.mastersInserted += 1 }
                    var seen = Set<String>()
                    for item in master["low"] as? [Any] ?? [] {
                        guard let preset = item as? String, !preset.isEmpty, !preset.hasPrefix("._"),
                              !preset.contains("/"), preset != ".", preset != "..", seen.insert(preset).inserted
                        else { continue }
                        let variantRel = "low/\(preset)/\(name).\(rung).mkv"
                        if !exists(variantRel) { report.missingFiles.append(variantRel) }
                        try Self.ensurePreset(db, name: preset)
                        let variant = try Self.insertOrFind(db, into: "source_variants", [
                            "master_id": masterRow.id, "preset": preset,
                            "rel_path": variantRel,
                        ], uniqueBy: ["master_id", "preset"])
                        if variant.inserted { report.variantsInserted += 1 }
                    }
                }
            }
        }
        report.notes.append("Katalog importiert: \(root.path)")
        return report
    }

    /// Imports `.mlxmodel` packages in `Models/` that are not yet in the database, grouped into runs by folder
    /// or name. Packages whose `config.json` cannot be read are skipped.
    func adoptOrphanModels() throws -> ImportReport {
        var report = ImportReport()
        let packages = Self.collectPackages(in: WhoodeoPaths.modelsDirectory)
        let orphans = try pool.read { db in try packages.filter { try Self.checkpointID(db, forModel: $0) == nil } }
        var metas: [URL: PackageMeta] = [:]
        for url in orphans {
            do {
                metas[url] = try Self.packageMeta(url)
            } catch {
                report.notes.append("übersprungen \(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
        guard !metas.isEmpty else { return report }
        let groups = Dictionary(grouping: packages.sorted { $0.lastPathComponent < $1.lastPathComponent }, by: Self.runName(forPackage:))
        try pool.write { db in
            for (runName, group) in groups.sorted(by: { $0.key < $1.key }) {
                var runID: Int?
                for url in group {
                    if let id = try Self.checkpointID(db, forModel: url) {
                        runID = try Int.fetchOne(db, sql: "SELECT run_id FROM checkpoints WHERE id = ?", arguments: [id])
                        continue
                    }
                    guard let meta = metas[url] else { continue }
                    let target: Int
                    if let runID {
                        target = runID
                    } else if let found = try Int.fetchOne(db, sql: "SELECT id FROM runs WHERE name = ?", arguments: [runName]) {
                        target = found
                    } else {
                        target = try Self.insertRun(
                            db, name: runName, parentCheckpointID: nil, channels: meta.channels,
                            blocksJSON: meta.blocksJSON, dtype: meta.dtype, cropSize: nil, configJSON: meta.configJSON,
                            status: "imported", startedAt: Self.startedAt(fromRunName: runName) ?? Self.timestamp(),
                            notes: "Importiertes Modell aus der App-Bibliothek"
                        )
                        report.runsInserted += 1
                    }
                    runID = target
                    let checkpoint = try Self.insertOrFind(db, into: "checkpoints", [
                        "run_id": target, "step": meta.step, "rel_path": Self.relativeModelPath(of: url),
                        "export_dtype": meta.dtype, "created_at": Self.fileTimestamp(url) ?? Self.timestamp(),
                    ], uniqueBy: ["run_id", "step"])
                    if checkpoint.inserted { report.checkpointsInserted += 1 }
                }
            }
        }
        return report
    }

    /// Walks the cache root and records archive folders that belong to a known checkpoint.
    /// Does not rename, move, or delete any cache file.
    func associateExistingCaches(at root: URL) throws -> ImportReport {
        var report = ImportReport()
        guard FileManager.default.fileExists(atPath: root.path) else {
            report.notes.append("Cache-Ordner nicht erreichbar: \(root.path)")
            return report
        }
        let (storedRoot, packages, known) = try pool.read { db in
            (
                try Self.setting(db, "cache_root"),
                try Row.fetchAll(db, sql: "SELECT id, rel_path FROM checkpoints").map { ($0[0] as Int, $0[1] as String) },
                Set(try String.fetchAll(db, sql: "SELECT rel_path FROM cache_files"))
            )
        }
        let models = WhoodeoPaths.modelsDirectory
        var keyToCheckpoint: [String: Int] = [:]
        for (id, rel) in packages {
            keyToCheckpoint[ModelCacheKey.modelKey(of: models.appendingPathComponent(rel))] = id
        }
        var found: [(checkpointID: Int, filmID: String, modelKey: String, relPath: String, condition: String?)] = []
        for film in keyToCheckpoint.isEmpty ? [] : Self.subfolders(of: root) {
            let filmName = film.lastPathComponent
            guard ModelCacheKey.isFilmFolderName(filmName) else { continue }
            for archive in Self.subfolders(of: film) {
                let name = archive.lastPathComponent
                let rel = "\(filmName)/\(name)"
                guard !known.contains(rel),
                      let parsed = ModelCacheKey.parseArchiveName(name),
                      let checkpointID = keyToCheckpoint[parsed.modelKey]
                else { continue }
                found.append((checkpointID, filmName, parsed.modelKey, rel, parsed.condition))
            }
        }
        if storedRoot != root.path || !found.isEmpty {
            try pool.write { db in
                try Self.setSetting(db, root.path, forKey: "cache_root")
                for item in found {
                    if try Self.registerCache(
                        db, checkpointID: item.checkpointID, filmID: item.filmID, modelKey: item.modelKey,
                        relPath: item.relPath, condition: item.condition
                    ) {
                        report.cacheFilesInserted += 1
                    }
                }
            }
        }
        report.notes.append("Cache zugeordnet unter \(root.path)")
        return report
    }

    /// Inserts a row unless its unique key exists; returns the row id either way.
    private static func insertOrFind(
        _ db: Database, into table: String, _ values: [String: (any DatabaseValueConvertible)?], uniqueBy keys: [String]
    ) throws -> (id: Int, inserted: Bool) {
        let columns = values.keys.sorted()
        try db.execute(
            sql: "INSERT OR IGNORE INTO \(table)(\(columns.joined(separator: ", "))) VALUES(\(columns.map { ":" + $0 }.joined(separator: ", ")))",
            arguments: StatementArguments(values)
        )
        let inserted = db.changesCount > 0
        let filter = keys.map { "\($0) = :\($0)" }.joined(separator: " AND ")
        guard let id = try Int.fetchOne(
            db, sql: "SELECT id FROM \(table) WHERE \(filter)", arguments: StatementArguments(values.filter { keys.contains($0.key) })
        ) else {
            throw WhoodeoDBError("\(table): Zeile fehlt nach dem Einfügen")
        }
        return (id, inserted)
    }

    // MARK: - Package and catalog helpers

    /// Stable 10% validation split from the title name (FNV-1a), independent of catalog order.
    static func titleSplit(_ name: String) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in name.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 1_099_511_628_211
        }
        return hash % 10 == 0 ? "val" : "train"
    }

    /// FiLM codec label (`h264` / `mpeg4` / `mpeg2`), from any hyphen-separated token of the preset name.
    static func codec(forPreset preset: String) -> String {
        let parts = preset.lowercased().split(separator: "-", omittingEmptySubsequences: true).map(String.init)
        for part in parts {
            switch part {
            case "x264", "h264", "avc": return "h264"
            case "mpeg4": return "mpeg4"
            case "mpeg2": return "mpeg2"
            default: continue
            }
        }
        return parts.first ?? preset
    }

    /// Writes `degrade` recipes and `rungs` from a catalog.json document.
    static func importDegrade(_ db: Database, from json: [String: Any]) throws {
        if let degrade = json["degrade"] as? [String: Any] {
            let scale = Self.intValue(degrade["scale"]) ?? 2
            let flags = (degrade["scale_flags"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "bicubic"
            let pix = (degrade["pix_fmt"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "yuv420p"
            // Optional `frame_ratios: {name: [num, den]}`; otherwise the built-in ratio (1/1 if unknown).
            let ratios = degrade["frame_ratios"] as? [String: Any] ?? [:]
            if let variants = degrade["variants"] as? [String: Any] {
                for (name, raw) in variants.sorted(by: { $0.key < $1.key }) {
                    guard let args = raw as? String, !args.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                    var ratio = WhoodeoDB.builtInFrameRatio(name)
                    if let pair = ratios[name] as? [Any], pair.count == 2,
                       let num = Self.intValue(pair[0]), let den = Self.intValue(pair[1]), num > 0, den > 0
                    {
                        ratio = (num, den)
                    }
                    try WhoodeoDB.upsertPreset(
                        db, name: name, codec: codec(forPreset: name), ffmpegArgs: args,
                        scale: scale, scaleFlags: flags, pixFmt: pix,
                        frameRatioNum: ratio.num, frameRatioDen: ratio.den
                    )
                }
            }
        }
        if let rungs = json["rungs"], JSONSerialization.isValidJSONObject(rungs),
           let data = try? JSONSerialization.data(withJSONObject: rungs),
           let text = String(data: data, encoding: .utf8)
        {
            try WhoodeoDB.setSetting(db, text, forKey: "rungs")
        }
    }

    /// Inserts a placeholder preset when a low file names one that is not in `degrade_presets` yet.
    static func ensurePreset(_ db: Database, name: String) throws {
        if try String.fetchOne(db, sql: "SELECT name FROM degrade_presets WHERE name = ?", arguments: [name]) != nil {
            return
        }
        let ratio = WhoodeoDB.builtInFrameRatio(name)
        try WhoodeoDB.upsertPreset(
            db, name: name, codec: codec(forPreset: name), ffmpegArgs: "",
            frameRatioNum: ratio.num, frameRatioDen: ratio.den
        )
    }

    static func collectPackages(in folder: URL) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        var found: [URL] = []
        for url in items {
            if url.lastPathComponent.hasPrefix("._") { continue }
            if url.pathExtension == MLXAlignModel.fileExtension {
                found.append(url)
                continue
            }
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                found.append(contentsOf: collectPackages(in: url))
            }
        }
        return found
    }

    /// Visible subfolders, without `._*` and `*.deleting`.
    private static func subfolders(of folder: URL) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return items.filter { url in
            let name = url.lastPathComponent
            return !name.hasPrefix("._") && !name.hasSuffix(".deleting")
                && (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
        }
    }

    static func runName(forPackage url: URL) -> String {
        let models = WhoodeoPaths.modelsDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let parent = url.deletingLastPathComponent().standardizedFileURL.resolvingSymlinksInPath()
        let root = models.path.hasSuffix("/") ? models.path : models.path + "/"
        if parent.path != models.path && parent.path.hasPrefix(root) {
            return parent.lastPathComponent
        }
        let stem = url.deletingPathExtension().lastPathComponent
        if let match = stem.firstMatch(of: /^(train-\d{8}-\d{6})-/) {
            return String(match.1)
        }
        return stem
    }

    struct PackageMeta {
        var step: Int
        var channels: Int
        var blocksJSON: String
        var dtype: String
        var configJSON: String
    }

    static func packageMeta(_ url: URL) throws -> PackageMeta {
        let data = try Data(contentsOf: url.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MLXAlignModel.Config.self, from: data)
        let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let blocks = config.encoder_blocks + [config.blocks]
        let dtype: String
        switch (config.compute ?? (raw["compute"] as? String))?.lowercased() {
        case "bfloat16", "bf16": dtype = "bf16"
        case "float32", "fp32": dtype = "fp32"
        default: dtype = "fp16"
        }
        var stored: [String: Any] = [
            "arch": raw["arch"] ?? "latent_up",
            "in_frames": config.in_frames,
            "condition": raw["condition"] ?? "codec",
            "condition_size": config.condition_size,
            "encoder_blocks": config.encoder_blocks,
            "blocks": config.blocks,
            "align": raw["align"] ?? "feature",
            "compute": config.compute ?? "float16",
            "imported": true,
        ]
        if let checkpoint = raw["checkpoint"] { stored["checkpoint"] = checkpoint }
        return PackageMeta(
            step: config.step ?? 0,
            channels: config.block_widths.first ?? 0,
            blocksJSON: jsonText(blocks) ?? "\(blocks)",
            dtype: dtype,
            configJSON: jsonText(stored, options: [.sortedKeys]) ?? "{}"
        )
    }

    private static func jsonText(_ object: Any, options: JSONSerialization.WritingOptions = []) -> String? {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: options) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func startedAt(fromRunName name: String) -> String? {
        guard let match = name.firstMatch(of: /^train-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/) else { return nil }
        return "\(match.1)-\(match.2)-\(match.3)T\(match.4):\(match.5):\(match.6)"
    }

    static func fileTimestamp(_ url: URL) -> String? {
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
        guard let date = values?.contentModificationDate else { return nil }
        return timestamp(date)
    }

    private static func intValue(_ raw: Any?) -> Int? {
        if let value = raw as? Int { return value }
        if let value = raw as? Double { return Int(value) }
        if let value = raw as? String { return Int(value) }
        return nil
    }

    // MARK: - File helpers

    static func folderSize(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            let values = try? file.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile == true else { continue }
            total += Int64(values?.fileSize ?? 0)
        }
        return total
    }

    /// Removes `url` only if it lies inside `root`.
    static func removeIfSafe(_ url: URL, under root: URL) throws {
        let file = url.standardizedFileURL.resolvingSymlinksInPath()
        let folder = root.standardizedFileURL.resolvingSymlinksInPath()
        let prefix = folder.path.hasSuffix("/") ? folder.path : folder.path + "/"
        guard file.path == folder.path || file.path.hasPrefix(prefix) else {
            throw WhoodeoDBError("Löschen außerhalb des erlaubten Ordners: \(file.path)")
        }
        if FileManager.default.fileExists(atPath: file.path) {
            try FileManager.default.removeItem(at: file)
        }
    }
}
