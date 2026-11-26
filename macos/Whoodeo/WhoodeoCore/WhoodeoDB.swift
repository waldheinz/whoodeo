import Foundation
import os
import SQLite3

/// Application Support folder, the SQLite database, and the model packages.
nonisolated enum WhoodeoPaths {
    static let appDefaultsSuite = "meins.whoodeo.Whoodeo"
    static let cacheDirectoryKey = "cacheDirectory"
    static let defaultDatasetRootPath = "/Volumes/Black Elem/whoodeo/train"

    static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("Whoodeo", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var databaseURL: URL {
        supportDirectory.appendingPathComponent("whoodeo.sqlite")
    }

    static var modelsDirectory: URL {
        let url = supportDirectory.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var defaultCacheRoot: URL {
        supportDirectory.appendingPathComponent("Cache", isDirectory: true)
    }

    /// Frozen VAE, VGG19, and SPyNet assets for the Swift trainer.
    static var trainDirectory: URL {
        let url = supportDirectory.appendingPathComponent("Train", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// The app's cache folder from its defaults, otherwise the default under Application Support.
    static func resolvedCacheRoot() -> URL {
        if let path = UserDefaults.standard.string(forKey: cacheDirectoryKey), !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        if let path = UserDefaults(suiteName: appDefaultsSuite)?.string(forKey: cacheDirectoryKey), !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return defaultCacheRoot
    }
}

nonisolated struct WhoodeoDBError: Error, LocalizedError, CustomStringConvertible {
    var description: String
    var errorDescription: String? { description }
    init(_ description: String) { self.description = description }
}

nonisolated struct SourceRecord: Equatable, Identifiable, Sendable {
    var id: Int
    var name: String
    var originPath: String?
    var frames: Int
    var split: String
    var enabled: Bool
    var notes: String?
    var addedAt: String
}

nonisolated struct RunRecord: Equatable, Identifiable, Sendable {
    var id: Int
    var name: String
    var parentCheckpointID: Int?
    var channels: Int
    var blocksJSON: String
    var dtype: String
    var cropSize: Int?
    var configJSON: String
    var status: String
    var startedAt: String
    var finishedAt: String?
    var notes: String?
}

nonisolated struct CheckpointRecord: Equatable, Identifiable, Sendable {
    var id: Int
    var runID: Int
    var step: Int
    var relPath: String
    var exportDtype: String
    var optimizerPath: String?
    var valLoss: Double?
    var valPSNR: Double?
    var valSSIM: Double?
    var visibleInApp: Bool
    var createdAt: String
    var runName: String
    var channels: Int
    var blocksJSON: String
    var parentCheckpointID: Int?
    var parentStep: Int?
    var parentRunName: String?
    var configJSON: String
}

nonisolated struct CacheFileRecord: Equatable, Identifiable, Sendable {
    var id: Int
    var checkpointID: Int
    var filmID: String
    var modelKey: String
    var relPath: String
    var condition: String?
    var addedAt: String
}

nonisolated struct RunTreeNode: Identifiable, Equatable, Sendable {
    var run: RunRecord
    var checkpoints: [CheckpointRecord]
    var cacheCount: Int
    var children: [RunTreeNode]

    var id: Int { run.id }
}

nonisolated struct DatabaseCounts: Sendable {
    var sources: Int
    var train: Int
    var val: Int
    var masters: Int
    var variants: Int
    var runs: Int
    var checkpoints: Int
    var cacheFiles: Int
}

nonisolated struct MetricPoint: Equatable, Sendable {
    var step: Int
    var split: String
    var name: String
    var value: Double
    var recordedAt: String
}

nonisolated struct TrainingPair: Equatable, Sendable {
    var sourceID: Int
    var name: String
    var frames: Int
    var split: String
    var rung: String
    var origRelPath: String
    var width: Int
    var height: Int
    var preset: String
    var codec: String
    var lowRelPath: String
}

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

/// Shared SQLite database for the app and the trainer. WAL mode, so both can use it at once.
nonisolated final class WhoodeoDB: @unchecked Sendable {
    static let schemaVersion = 1

    static let shared: WhoodeoDB? = {
        do {
            return try openDefault()
        } catch {
            Log.model.error("Datenbank: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }()

    private let db: OpaquePointer
    private let lock = NSLock()

    static func openDefault() throws -> WhoodeoDB {
        let opened = try open(at: WhoodeoPaths.databaseURL)
        try opened.bootstrap()
        return opened
    }

    static func open(at url: URL) throws -> WhoodeoDB {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let status = url.path.withCString { sqlite3_open_v2($0, &handle, flags, nil) }
        guard status == SQLITE_OK, let handle else {
            if let handle { sqlite3_close(handle) }
            throw WhoodeoDBError("Datenbank lässt sich nicht öffnen: \(url.path)")
        }
        let db = WhoodeoDB(db: handle)
        do {
            try db.exec("PRAGMA journal_mode=WAL;")
            try db.exec("PRAGMA foreign_keys=ON;")
            try db.exec("PRAGMA busy_timeout=5000;")
            try db.migrate()
        } catch {
            sqlite3_close(handle)
            throw error
        }
        return db
    }

    private init(db: OpaquePointer) {
        self.db = db
    }

    deinit {
        sqlite3_close(db)
    }

    /// Creates the schema, fills defaults, imports the catalog once, and adopts packages not yet in the database.
    func bootstrap() throws {
        try lock.withLock {
            if try settingLocked("dataset_root") == nil {
                try setSettingLocked(WhoodeoPaths.defaultDatasetRootPath, forKey: "dataset_root")
            }
            if try settingLocked("schema_version") == nil {
                try setSettingLocked(String(Self.schemaVersion), forKey: "schema_version")
            }
            if try settingLocked("holdout_mode") == nil {
                try setSettingLocked("title", forKey: "holdout_mode")
                try setSettingLocked("0.1", forKey: "holdout_fraction")
                try setSettingLocked(
                    "The PyTorch trainer held out 10% of the frames inside every film (two spans at 30% and 70%). The database uses a stable 10% title split instead.",
                    forKey: "holdout_note"
                )
            }
        }
        if try count("sources") == 0, let root = try datasetRoot(), FileManager.default.fileExists(atPath: root.appendingPathComponent("catalog.json").path) {
            _ = try importCatalog(root: root)
        }
        _ = try adoptOrphanModels()
        _ = try associateExistingCaches(at: try cacheRoot())
    }

    func datasetRoot() throws -> URL? {
        try lock.withLock {
            try settingLocked("dataset_root").map { URL(fileURLWithPath: $0) }
        }
    }

    func cacheRoot() throws -> URL {
        try lock.withLock {
            if let stored = try settingLocked("cache_root"), !stored.isEmpty {
                return URL(fileURLWithPath: stored)
            }
            let resolved = WhoodeoPaths.resolvedCacheRoot()
            try setSettingLocked(resolved.path, forKey: "cache_root")
            return resolved
        }
    }

    func setting(_ key: String) throws -> String? {
        try lock.withLock { try settingLocked(key) }
    }

    func setSetting(_ value: String, forKey key: String) throws {
        try lock.withLock { try setSettingLocked(value, forKey: key) }
    }

    func deleteSetting(_ key: String) throws {
        try lock.withLock { try deleteSettingLocked(key) }
    }

    static func checkpointRequestKey(_ runID: Int) -> String {
        "checkpoint_request.\(runID)"
    }

    func requestCheckpoint(runID: Int) throws {
        try setSetting("1", forKey: Self.checkpointRequestKey(runID))
    }

    /// Returns true and clears the key if a request was pending for this run.
    func consumeCheckpointRequest(runID: Int) throws -> Bool {
        try lock.withLock {
            let key = Self.checkpointRequestKey(runID)
            guard let value = try settingLocked(key), !value.isEmpty, value != "0" else { return false }
            try deleteSettingLocked(key)
            return true
        }
    }

    static func stopRequestKey(_ runID: Int) -> String {
        "stop_request.\(runID)"
    }

    func requestStop(runID: Int) throws {
        try setSetting("1", forKey: Self.stopRequestKey(runID))
    }

    func hasStopRequest(runID: Int) throws -> Bool {
        try lock.withLock {
            let key = Self.stopRequestKey(runID)
            guard let value = try settingLocked(key), !value.isEmpty, value != "0" else { return false }
            return true
        }
    }

    func clearStopRequest(runID: Int) throws {
        try deleteSetting(Self.stopRequestKey(runID))
    }

    /// Runs whose `parent_checkpoint_id` points at a checkpoint of `id`.
    func childRuns(ofRunID id: Int) throws -> [RunRecord] {
        try lock.withLock {
            let owned = Set(try checkpointsLocked(visibleOnly: false).filter { $0.runID == id }.map(\.id))
            return try runsLocked().filter { run in
                guard let parent = run.parentCheckpointID else { return false }
                return owned.contains(parent)
            }
        }
    }

    func runningRuns() throws -> [RunRecord] {
        try lock.withLock { try runsLocked().filter { $0.status == "running" } }
    }

    func counts() throws -> DatabaseCounts {
        try lock.withLock {
            let train = try queryInt("SELECT COUNT(*) FROM sources WHERE split = 'train'")
            let val = try queryInt("SELECT COUNT(*) FROM sources WHERE split = 'val'")
            return DatabaseCounts(
                sources: train + val,
                train: train,
                val: val,
                masters: try queryInt("SELECT COUNT(*) FROM source_masters"),
                variants: try queryInt("SELECT COUNT(*) FROM source_variants"),
                runs: try queryInt("SELECT COUNT(*) FROM runs"),
                checkpoints: try queryInt("SELECT COUNT(*) FROM checkpoints"),
                cacheFiles: try queryInt("SELECT COUNT(*) FROM cache_files")
            )
        }
    }

    func sources() throws -> [SourceRecord] {
        try lock.withLock {
            let stmt = try prepare("SELECT id, name, origin_path, frames, split, enabled, notes, added_at FROM sources ORDER BY name")
            defer { sqlite3_finalize(stmt) }
            var rows: [SourceRecord] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                rows.append(SourceRecord(
                    id: int(stmt, 0),
                    name: text(stmt, 1) ?? "",
                    originPath: text(stmt, 2),
                    frames: int(stmt, 3),
                    split: text(stmt, 4) ?? "train",
                    enabled: int(stmt, 5) != 0,
                    notes: text(stmt, 6),
                    addedAt: text(stmt, 7) ?? ""
                ))
            }
            return rows
        }
    }

    func visibleCheckpoints() throws -> [CheckpointRecord] {
        try lock.withLock { try checkpointsLocked(visibleOnly: true) }
    }

    func runTree() throws -> [RunTreeNode] {
        try lock.withLock {
            let runs = try runsLocked()
            let checkpoints = try checkpointsLocked(visibleOnly: false)
            var byRun: [Int: [CheckpointRecord]] = [:]
            for checkpoint in checkpoints {
                byRun[checkpoint.runID, default: []].append(checkpoint)
            }
            var cacheCounts: [Int: Int] = [:]
            let countStmt = try prepare("SELECT checkpoint_id, COUNT(*) FROM cache_files GROUP BY checkpoint_id")
            defer { sqlite3_finalize(countStmt) }
            while sqlite3_step(countStmt) == SQLITE_ROW {
                cacheCounts[int(countStmt, 0)] = int(countStmt, 1)
            }
            func node(for run: RunRecord) -> RunTreeNode {
                let kids = runs.filter { child in
                    guard let parent = child.parentCheckpointID else { return false }
                    return (byRun[run.id] ?? []).contains { $0.id == parent }
                }
                return RunTreeNode(
                    run: run,
                    checkpoints: (byRun[run.id] ?? []).sorted { $0.step < $1.step },
                    cacheCount: (byRun[run.id] ?? []).reduce(0) { $0 + (cacheCounts[$1.id] ?? 0) },
                    children: kids.map(node(for:)).sorted { $0.run.startedAt < $1.run.startedAt }
                )
            }
            let roots = runs.filter { run in
                guard let parent = run.parentCheckpointID else { return true }
                return !checkpoints.contains { $0.id == parent }
            }
            return roots
                .sorted { $0.startedAt < $1.startedAt }
                .map(node(for:))
        }
    }

    /// Records an archive folder under the cache root. Existing cache files are never renamed or deleted.
    func registerCache(checkpointID: Int, filmID: String, modelKey: String, relPath: String, condition: String?) throws {
        try lock.withLock {
            try registerCacheLocked(
                checkpointID: checkpointID,
                filmID: filmID,
                modelKey: modelKey,
                relPath: relPath,
                condition: condition
            )
        }
    }

    func registerCache(forModel url: URL, filmID: String, condition: String?) throws {
        let rel = ModelCacheKey.relativePath(filmID: filmID, model: url, condition: condition)
        let key = ModelCacheKey.modelKey(of: url)
        try lock.withLock {
            guard let checkpointID = try checkpointIDLocked(forModel: url) else { return }
            try registerCacheLocked(
                checkpointID: checkpointID,
                filmID: filmID,
                modelKey: key,
                relPath: rel,
                condition: condition
            )
        }
    }

    func cacheFiles(forCheckpoint id: Int) throws -> [CacheFileRecord] {
        try lock.withLock {
            let stmt = try prepare(
                "SELECT id, checkpoint_id, film_id, model_key, rel_path, condition, added_at FROM cache_files WHERE checkpoint_id = ? ORDER BY rel_path"
            )
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int(stmt, 1, Int32(id))
            var rows: [CacheFileRecord] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                rows.append(CacheFileRecord(
                    id: int(stmt, 0),
                    checkpointID: int(stmt, 1),
                    filmID: text(stmt, 2) ?? "",
                    modelKey: text(stmt, 3) ?? "",
                    relPath: text(stmt, 4) ?? "",
                    condition: text(stmt, 5),
                    addedAt: text(stmt, 6) ?? ""
                ))
            }
            return rows
        }
    }

    /// Deletes a checkpoint row and, when `removeFiles` is true, its package, optimizer, and recorded cache folders.
    /// Only paths under the models folder or the cache root are touched. Existing files are never touched by import.
    func deleteCheckpoint(id: Int, removeFiles: Bool) throws {
        let cacheRoot = try cacheRoot()
        let models = WhoodeoPaths.modelsDirectory
        try lock.withLock {
            let caches = try cacheRowsLocked(checkpointID: id)
            var package: URL?
            var optimizer: URL?
            var valRun: String?
            var valStep: Int?
            let stmt = try prepare(
                """
                SELECT c.rel_path, c.optimizer_path, c.step, r.name
                FROM checkpoints c JOIN runs r ON r.id = c.run_id
                WHERE c.id = ?
                """
            )
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int(stmt, 1, Int32(id))
            if sqlite3_step(stmt) == SQLITE_ROW {
                if let rel = text(stmt, 0) {
                    package = models.appendingPathComponent(rel)
                }
                if let rel = text(stmt, 1) {
                    optimizer = models.appendingPathComponent(rel)
                }
                valStep = int(stmt, 2)
                valRun = text(stmt, 3)
            }
            if removeFiles {
                for row in caches {
                    let url = cacheRoot.appendingPathComponent(row.relPath)
                    try removeIfSafe(url, under: cacheRoot)
                }
                if let optimizer { try removeIfSafe(optimizer, under: models) }
                if let valRun, let valStep {
                    try ValPreview.remove(runName: valRun, step: valStep)
                }
                if let package {
                    try removeIfSafe(package, under: models)
                    let parent = package.deletingLastPathComponent()
                    if parent.standardizedFileURL.resolvingSymlinksInPath()
                        != models.standardizedFileURL.resolvingSymlinksInPath()
                    {
                        let leftover = (try? FileManager.default.contentsOfDirectory(atPath: parent.path)) ?? ["."]
                        if leftover.filter({ !$0.hasPrefix(".") }).isEmpty {
                            try removeIfSafe(parent, under: models)
                        }
                    }
                }
            }
            try exec("DELETE FROM checkpoints WHERE id = \(id)")
        }
    }

    func importCatalog(root: URL) throws -> ImportReport {
        let catalogURL = root.appendingPathComponent("catalog.json")
        let data = try Data(contentsOf: catalogURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let titles = json["titles"] as? [String: Any]
        else {
            throw WhoodeoDBError("catalog.json hat kein titles-Objekt: \(catalogURL.path)")
        }
        let now = Self.timestamp()
        var report = ImportReport()
        try lock.withLock {
            try exec("BEGIN IMMEDIATE")
            do {
                try setSettingLocked(root.path, forKey: "dataset_root")
                for (name, raw) in titles.sorted(by: { $0.key < $1.key }) {
                    if name.hasPrefix("._") { continue }
                    guard let entry = raw as? [String: Any] else { continue }
                    let frames = Self.intValue(entry["frames"]) ?? 0
                    guard frames > 0 else {
                        report.notes.append("übersprungen \(name): keine frames")
                        continue
                    }
                    let origin = entry["origin"] as? String
                    let split = Self.titleSplit(name)
                    let sourceID = try insertSourceLocked(
                        name: name,
                        origin: origin,
                        frames: frames,
                        split: split,
                        addedAt: now
                    )
                    if sourceID.inserted { report.sourcesInserted += 1 }
                    let masters = entry["masters"] as? [String: Any] ?? [:]
                    for (rung, masterRaw) in masters.sorted(by: { $0.key < $1.key }) {
                        guard let master = masterRaw as? [String: Any] else { continue }
                        let rel = "orig/\(name).\(rung).mkv"
                        if rel.split(separator: "/").contains(where: { $0.hasPrefix("._") }) { continue }
                        let width = Self.intValue(master["width"]) ?? 0
                        let height = Self.intValue(master["height"]) ?? 0
                        let masterAbs = root.appendingPathComponent(rel)
                        if !FileManager.default.fileExists(atPath: masterAbs.path) {
                            report.missingFiles.append(rel)
                        }
                        let masterID = try insertMasterLocked(
                            sourceID: sourceID.id,
                            rung: rung,
                            relPath: rel,
                            width: width,
                            height: height
                        )
                        if masterID.inserted { report.mastersInserted += 1 }
                        let lows = master["low"] as? [Any] ?? []
                        var seen = Set<String>()
                        for item in lows {
                            guard let preset = item as? String, !preset.isEmpty, !preset.hasPrefix("._"),
                                  !preset.contains("/"), preset != ".", preset != ".."
                            else { continue }
                            if !seen.insert(preset).inserted { continue }
                            let variantRel = "low/\(preset)/\(name).\(rung).mkv"
                            let variantAbs = root.appendingPathComponent(variantRel)
                            if !FileManager.default.fileExists(atPath: variantAbs.path) {
                                report.missingFiles.append(variantRel)
                            }
                            let inserted = try insertVariantLocked(
                                masterID: masterID.id,
                                preset: preset,
                                codec: Self.codec(forPreset: preset),
                                relPath: variantRel
                            )
                            if inserted { report.variantsInserted += 1 }
                        }
                    }
                }
                try exec("COMMIT")
            } catch {
                try? exec("ROLLBACK")
                throw error
            }
        }
        report.notes.append("Katalog importiert: \(root.path)")
        return report
    }

    /// Imports `.mlxmodel` packages in `Models/` that are not yet in the database.
    func adoptOrphanModels() throws -> ImportReport {
        let folder = WhoodeoPaths.modelsDirectory
        let packages = Self.collectPackages(in: folder)
        var report = ImportReport()
        var groups: [String: [URL]] = [:]
        for url in packages.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            groups[Self.runName(forPackage: url), default: []].append(url)
        }
        try lock.withLock {
            for (runName, group) in groups.sorted(by: { $0.key < $1.key }) {
                var existingRun: Int?
                for url in group {
                    if let id = try checkpointIDLocked(forModel: url) {
                        existingRun = try runIDLocked(ofCheckpoint: id)
                        continue
                    }
                    let meta = try Self.packageMeta(url)
                    let runID: Int
                    if let existingRun {
                        runID = existingRun
                    } else if let found = try runIDLocked(named: runName) {
                        runID = found
                        existingRun = found
                    } else {
                        let started = Self.startedAt(fromRunName: runName) ?? Self.timestamp()
                        runID = try insertRunLocked(
                            name: runName,
                            parentCheckpointID: nil,
                            channels: meta.channels,
                            blocksJSON: meta.blocksJSON,
                            dtype: meta.dtype,
                            cropSize: nil,
                            configJSON: meta.configJSON,
                            status: "imported",
                            startedAt: started,
                            finishedAt: nil,
                            notes: "Importiertes Modell aus der App-Bibliothek"
                        )
                        existingRun = runID
                        report.runsInserted += 1
                        try attachAllSourcesLocked(runID: runID)
                    }
                    _ = try insertCheckpointLocked(
                        runID: runID,
                        step: meta.step,
                        relPath: Self.relativeModelPath(of: url),
                        exportDtype: meta.dtype,
                        createdAt: Self.fileTimestamp(url) ?? Self.timestamp()
                    )
                    report.checkpointsInserted += 1
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
        try lock.withLock {
            try setSettingLocked(root.path, forKey: "cache_root")
            let packages = try allCheckpointPackagesLocked()
            var keyToCheckpoint: [String: Int] = [:]
            for (id, url) in packages {
                keyToCheckpoint[ModelCacheKey.modelKey(of: url)] = id
            }
            guard !keyToCheckpoint.isEmpty else { return }
            let films = (try? FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            for film in films {
                let filmName = film.lastPathComponent
                if filmName.hasPrefix("._") { continue }
                if filmName.hasSuffix(".deleting") { continue }
                guard ModelCacheKey.isFilmFolderName(filmName) else { continue }
                guard (try? film.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
                let archives = (try? FileManager.default.contentsOfDirectory(
                    at: film,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                )) ?? []
                for archive in archives {
                    let name = archive.lastPathComponent
                    if name.hasPrefix("._") || name.hasSuffix(".deleting") { continue }
                    guard (try? archive.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
                    guard let parsed = ModelCacheKey.parseArchiveName(name),
                          let checkpointID = keyToCheckpoint[parsed.modelKey]
                    else { continue }
                    let rel = "\(filmName)/\(name)"
                    let inserted = try registerCacheLocked(
                        checkpointID: checkpointID,
                        filmID: filmName,
                        modelKey: parsed.modelKey,
                        relPath: rel,
                        condition: parsed.condition
                    )
                    if inserted { report.cacheFilesInserted += 1 }
                }
            }
        }
        report.notes.append("Cache zugeordnet unter \(root.path)")
        return report
    }

    func checkpointID(forModel url: URL) throws -> Int? {
        try lock.withLock { try checkpointIDLocked(forModel: url) }
    }

    // MARK: - Schema

    private func migrate() throws {
        let version = try queryInt("PRAGMA user_version")
        if version < 1 {
            try exec("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    origin_path TEXT,
                    frames INTEGER NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('train','val')),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    notes TEXT,
                    added_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_masters (
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    rung TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    UNIQUE(source_id, rung)
                );
                CREATE TABLE IF NOT EXISTS source_variants (
                    id INTEGER PRIMARY KEY,
                    master_id INTEGER NOT NULL REFERENCES source_masters(id) ON DELETE CASCADE,
                    preset TEXT NOT NULL,
                    codec TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(master_id, preset)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    parent_checkpoint_id INTEGER,
                    channels INTEGER NOT NULL,
                    blocks_json TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    crop_size INTEGER,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('created','running','paused','finished','failed','imported')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS run_sources (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    source_id INTEGER NOT NULL REFERENCES sources(id),
                    split TEXT NOT NULL,
                    PRIMARY KEY (run_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    step INTEGER NOT NULL,
                    rel_path TEXT NOT NULL,
                    export_dtype TEXT NOT NULL,
                    optimizer_path TEXT,
                    val_loss REAL,
                    val_psnr REAL,
                    val_ssim REAL,
                    visible_in_app INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, step)
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    step INTEGER NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('train','val')),
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, step, split, name)
                );
                CREATE TABLE IF NOT EXISTS cache_files (
                    id INTEGER PRIMARY KEY,
                    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
                    film_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    condition TEXT,
                    added_at TEXT NOT NULL,
                    UNIQUE(rel_path)
                );
                CREATE INDEX IF NOT EXISTS cache_files_checkpoint ON cache_files(checkpoint_id);
                CREATE INDEX IF NOT EXISTS cache_files_model_key ON cache_files(model_key);
                PRAGMA user_version = 1;
                """)
        }
    }

    // MARK: - Locked helpers

    private func settingLocked(_ key: String) throws -> String? {
        let stmt = try prepare("SELECT value FROM settings WHERE key = ?")
        defer { sqlite3_finalize(stmt) }
        bind(stmt, 1, key)
        return sqlite3_step(stmt) == SQLITE_ROW ? text(stmt, 0) : nil
    }

    private func setSettingLocked(_ value: String, forKey key: String) throws {
        let stmt = try prepare("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        defer { sqlite3_finalize(stmt) }
        bind(stmt, 1, key)
        bind(stmt, 2, value)
        try stepDone(stmt)
    }

    private func deleteSettingLocked(_ key: String) throws {
        let stmt = try prepare("DELETE FROM settings WHERE key = ?")
        defer { sqlite3_finalize(stmt) }
        bind(stmt, 1, key)
        try stepDone(stmt)
    }

    private func insertSourceLocked(name: String, origin: String?, frames: Int, split: String, addedAt: String) throws -> (id: Int, inserted: Bool) {
        if let existing = try queryIntOptional("SELECT id FROM sources WHERE name = ?", name) {
            return (existing, false)
        }
        let stmt = try prepare(
            "INSERT INTO sources(name, origin_path, frames, split, enabled, notes, added_at) VALUES(?, ?, ?, ?, 1, NULL, ?)"
        )
        defer { sqlite3_finalize(stmt) }
        bind(stmt, 1, name)
        bind(stmt, 2, origin)
        sqlite3_bind_int(stmt, 3, Int32(frames))
        bind(stmt, 4, split)
        bind(stmt, 5, addedAt)
        try stepDone(stmt)
        return (Int(sqlite3_last_insert_rowid(db)), true)
    }

    private func insertMasterLocked(sourceID: Int, rung: String, relPath: String, width: Int, height: Int) throws -> (id: Int, inserted: Bool) {
        if let existing = try queryIntOptional("SELECT id FROM source_masters WHERE source_id = ? AND rung = ?", sourceID, rung) {
            return (existing, false)
        }
        let stmt = try prepare(
            "INSERT INTO source_masters(source_id, rung, rel_path, width, height) VALUES(?, ?, ?, ?, ?)"
        )
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(sourceID))
        bind(stmt, 2, rung)
        bind(stmt, 3, relPath)
        sqlite3_bind_int(stmt, 4, Int32(width))
        sqlite3_bind_int(stmt, 5, Int32(height))
        try stepDone(stmt)
        return (Int(sqlite3_last_insert_rowid(db)), true)
    }

    private func insertVariantLocked(masterID: Int, preset: String, codec: String, relPath: String) throws -> Bool {
        if try queryIntOptional("SELECT id FROM source_variants WHERE master_id = ? AND preset = ?", masterID, preset) != nil {
            return false
        }
        let stmt = try prepare(
            "INSERT INTO source_variants(master_id, preset, codec, rel_path, enabled) VALUES(?, ?, ?, ?, 1)"
        )
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(masterID))
        bind(stmt, 2, preset)
        bind(stmt, 3, codec)
        bind(stmt, 4, relPath)
        try stepDone(stmt)
        return true
    }

    private func insertRunLocked(
        name: String,
        parentCheckpointID: Int?,
        channels: Int,
        blocksJSON: String,
        dtype: String,
        cropSize: Int?,
        configJSON: String,
        status: String,
        startedAt: String,
        finishedAt: String?,
        notes: String?
    ) throws -> Int {
        if let existing = try runIDLocked(named: name) { return existing }
        let stmt = try prepare(
            """
            INSERT INTO runs(name, parent_checkpoint_id, channels, blocks_json, dtype, crop_size, config_json, status, started_at, finished_at, notes)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        defer { sqlite3_finalize(stmt) }
        bind(stmt, 1, name)
        if let parentCheckpointID {
            sqlite3_bind_int(stmt, 2, Int32(parentCheckpointID))
        } else {
            sqlite3_bind_null(stmt, 2)
        }
        sqlite3_bind_int(stmt, 3, Int32(channels))
        bind(stmt, 4, blocksJSON)
        bind(stmt, 5, dtype)
        if let cropSize {
            sqlite3_bind_int(stmt, 6, Int32(cropSize))
        } else {
            sqlite3_bind_null(stmt, 6)
        }
        bind(stmt, 7, configJSON)
        bind(stmt, 8, status)
        bind(stmt, 9, startedAt)
        bind(stmt, 10, finishedAt)
        bind(stmt, 11, notes)
        try stepDone(stmt)
        return Int(sqlite3_last_insert_rowid(db))
    }

    private func insertCheckpointLocked(runID: Int, step: Int, relPath: String, exportDtype: String, createdAt: String) throws -> Int {
        if let existing = try queryIntOptional("SELECT id FROM checkpoints WHERE run_id = ? AND step = ?", runID, step) {
            return existing
        }
        let stmt = try prepare(
            """
            INSERT INTO checkpoints(run_id, step, rel_path, export_dtype, optimizer_path, val_loss, val_psnr, val_ssim, visible_in_app, created_at)
            VALUES(?, ?, ?, ?, NULL, NULL, NULL, NULL, 1, ?)
            """
        )
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(runID))
        sqlite3_bind_int(stmt, 2, Int32(step))
        bind(stmt, 3, relPath)
        bind(stmt, 4, exportDtype)
        bind(stmt, 5, createdAt)
        try stepDone(stmt)
        return Int(sqlite3_last_insert_rowid(db))
    }

    private func attachAllSourcesLocked(runID: Int) throws {
        try exec(
            "INSERT OR IGNORE INTO run_sources(run_id, source_id, split) SELECT \(runID), id, split FROM sources"
        )
    }

    @discardableResult
    private func registerCacheLocked(checkpointID: Int, filmID: String, modelKey: String, relPath: String, condition: String?) throws -> Bool {
        if try queryIntOptional("SELECT id FROM cache_files WHERE rel_path = ?", relPath) != nil {
            return false
        }
        let stmt = try prepare(
            "INSERT INTO cache_files(checkpoint_id, film_id, model_key, rel_path, condition, added_at) VALUES(?, ?, ?, ?, ?, ?)"
        )
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(checkpointID))
        bind(stmt, 2, filmID)
        bind(stmt, 3, modelKey)
        bind(stmt, 4, relPath)
        bind(stmt, 5, condition)
        bind(stmt, 6, Self.timestamp())
        try stepDone(stmt)
        return true
    }

    static func relativeModelPath(of url: URL) -> String {
        let models = WhoodeoPaths.modelsDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let file = url.standardizedFileURL.resolvingSymlinksInPath()
        let root = models.path.hasSuffix("/") ? models.path : models.path + "/"
        if file.path.hasPrefix(root) {
            return String(file.path.dropFirst(root.count))
        }
        return file.lastPathComponent
    }

    private func checkpointIDLocked(forModel url: URL) throws -> Int? {
        let rel = Self.relativeModelPath(of: url)
        if let id = try queryIntOptional("SELECT id FROM checkpoints WHERE rel_path = ?", rel) {
            return id
        }
        let name = url.standardizedFileURL.lastPathComponent
        if name != rel {
            return try queryIntOptional("SELECT id FROM checkpoints WHERE rel_path = ?", name)
        }
        return nil
    }

    private func runIDLocked(named name: String) throws -> Int? {
        try queryIntOptional("SELECT id FROM runs WHERE name = ?", name)
    }

    private func runIDLocked(ofCheckpoint id: Int) throws -> Int? {
        try queryIntOptional("SELECT run_id FROM checkpoints WHERE id = ?", id)
    }

    private func allCheckpointPackagesLocked() throws -> [(Int, URL)] {
        let root = WhoodeoPaths.modelsDirectory
        let stmt = try prepare("SELECT id, rel_path FROM checkpoints")
        defer { sqlite3_finalize(stmt) }
        var rows: [(Int, URL)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = int(stmt, 0)
            guard let rel = text(stmt, 1) else { continue }
            rows.append((id, root.appendingPathComponent(rel)))
        }
        return rows
    }

    private func cacheRowsLocked(checkpointID: Int) throws -> [CacheFileRecord] {
        let stmt = try prepare(
            "SELECT id, checkpoint_id, film_id, model_key, rel_path, condition, added_at FROM cache_files WHERE checkpoint_id = ?"
        )
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(checkpointID))
        var rows: [CacheFileRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(CacheFileRecord(
                id: int(stmt, 0),
                checkpointID: int(stmt, 1),
                filmID: text(stmt, 2) ?? "",
                modelKey: text(stmt, 3) ?? "",
                relPath: text(stmt, 4) ?? "",
                condition: text(stmt, 5),
                addedAt: text(stmt, 6) ?? ""
            ))
        }
        return rows
    }

    private func runsLocked() throws -> [RunRecord] {
        let stmt = try prepare(
            "SELECT id, name, parent_checkpoint_id, channels, blocks_json, dtype, crop_size, config_json, status, started_at, finished_at, notes FROM runs"
        )
        defer { sqlite3_finalize(stmt) }
        var rows: [RunRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(RunRecord(
                id: int(stmt, 0),
                name: text(stmt, 1) ?? "",
                parentCheckpointID: sqlite3_column_type(stmt, 2) == SQLITE_NULL ? nil : int(stmt, 2),
                channels: int(stmt, 3),
                blocksJSON: text(stmt, 4) ?? "",
                dtype: text(stmt, 5) ?? "",
                cropSize: sqlite3_column_type(stmt, 6) == SQLITE_NULL ? nil : int(stmt, 6),
                configJSON: text(stmt, 7) ?? "",
                status: text(stmt, 8) ?? "",
                startedAt: text(stmt, 9) ?? "",
                finishedAt: text(stmt, 10),
                notes: text(stmt, 11)
            ))
        }
        return rows
    }

    private func checkpointsLocked(visibleOnly: Bool) throws -> [CheckpointRecord] {
        let sql = """
            SELECT c.id, c.run_id, c.step, c.rel_path, c.export_dtype, c.optimizer_path,
                   c.val_loss, c.val_psnr, c.val_ssim, c.visible_in_app, c.created_at,
                   r.name, r.channels, r.blocks_json, r.parent_checkpoint_id, r.config_json,
                   p.step, pr.name
            FROM checkpoints c
            JOIN runs r ON r.id = c.run_id
            LEFT JOIN checkpoints p ON p.id = r.parent_checkpoint_id
            LEFT JOIN runs pr ON pr.id = p.run_id
            \(visibleOnly ? "WHERE c.visible_in_app = 1" : "")
            ORDER BY c.created_at DESC, c.step DESC
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        var rows: [CheckpointRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            rows.append(CheckpointRecord(
                id: int(stmt, 0),
                runID: int(stmt, 1),
                step: int(stmt, 2),
                relPath: text(stmt, 3) ?? "",
                exportDtype: text(stmt, 4) ?? "",
                optimizerPath: text(stmt, 5),
                valLoss: sqlite3_column_type(stmt, 6) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 6),
                valPSNR: sqlite3_column_type(stmt, 7) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 7),
                valSSIM: sqlite3_column_type(stmt, 8) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 8),
                visibleInApp: int(stmt, 9) != 0,
                createdAt: text(stmt, 10) ?? "",
                runName: text(stmt, 11) ?? "",
                channels: int(stmt, 12),
                blocksJSON: text(stmt, 13) ?? "",
                parentCheckpointID: sqlite3_column_type(stmt, 14) == SQLITE_NULL ? nil : int(stmt, 14),
                parentStep: sqlite3_column_type(stmt, 16) == SQLITE_NULL ? nil : int(stmt, 16),
                parentRunName: text(stmt, 17),
                configJSON: text(stmt, 15) ?? ""
            ))
        }
        return rows
    }

    // MARK: - SQLite primitives

    private func exec(_ sql: String) throws {
        var error: UnsafeMutablePointer<CChar>?
        let status = sqlite3_exec(db, sql, nil, nil, &error)
        if let error {
            let message = String(cString: error)
            sqlite3_free(error)
            if status != SQLITE_OK { throw WhoodeoDBError(message) }
        } else if status != SQLITE_OK {
            throw WhoodeoDBError(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        var stmt: OpaquePointer?
        let status = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard status == SQLITE_OK, let stmt else {
            throw WhoodeoDBError(String(cString: sqlite3_errmsg(db)))
        }
        return stmt
    }

    private func stepDone(_ stmt: OpaquePointer) throws {
        let status = sqlite3_step(stmt)
        guard status == SQLITE_DONE else {
            throw WhoodeoDBError(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func bind(_ stmt: OpaquePointer, _ index: Int32, _ value: String?) {
        if let value {
            sqlite3_bind_text(stmt, index, value, -1, Self.transient)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func text(_ stmt: OpaquePointer, _ index: Int32) -> String? {
        guard sqlite3_column_type(stmt, index) != SQLITE_NULL,
              let pointer = sqlite3_column_text(stmt, index)
        else { return nil }
        return String(cString: pointer)
    }

    private func int(_ stmt: OpaquePointer, _ index: Int32) -> Int {
        Int(sqlite3_column_int64(stmt, index))
    }

    private func queryInt(_ sql: String) throws -> Int {
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return int(stmt, 0)
    }

    private func count(_ table: String) throws -> Int {
        try queryInt("SELECT COUNT(*) FROM \(table)")
    }

    private func queryIntOptional(_ sql: String, _ strings: String...) throws -> Int? {
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        for (index, value) in strings.enumerated() {
            bind(stmt, Int32(index + 1), value)
        }
        return sqlite3_step(stmt) == SQLITE_ROW ? int(stmt, 0) : nil
    }

    private func queryIntOptional(_ sql: String, _ first: Int, _ second: String) throws -> Int? {
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(first))
        bind(stmt, 2, second)
        return sqlite3_step(stmt) == SQLITE_ROW ? int(stmt, 0) : nil
    }

    private func queryIntOptional(_ sql: String, _ first: Int, _ second: Int) throws -> Int? {
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(first))
        sqlite3_bind_int(stmt, 2, Int32(second))
        return sqlite3_step(stmt) == SQLITE_ROW ? int(stmt, 0) : nil
    }

    private func queryIntOptional(_ sql: String, _ value: Int) throws -> Int? {
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int(stmt, 1, Int32(value))
        return sqlite3_step(stmt) == SQLITE_ROW ? int(stmt, 0) : nil
    }

    private func removeIfSafe(_ url: URL, under root: URL) throws {
        let file = url.standardizedFileURL.resolvingSymlinksInPath()
        let folder = root.standardizedFileURL.resolvingSymlinksInPath()
        let filePath = file.path
        let folderPath = folder.path
        let prefix = folderPath.hasSuffix("/") ? folderPath : folderPath + "/"
        guard filePath == folderPath || filePath.hasPrefix(prefix) else {
            throw WhoodeoDBError("Löschen außerhalb des erlaubten Ordners: \(filePath)")
        }
        if FileManager.default.fileExists(atPath: filePath) {
            try FileManager.default.removeItem(at: file)
        }
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)


    // MARK: - Training writes

    func run(named name: String) throws -> RunRecord? {
        try lock.withLock {
            guard let id = try runIDLocked(named: name) else { return nil }
            return try runsLocked().first { $0.id == id }
        }
    }

    func run(id: Int) throws -> RunRecord? {
        try lock.withLock { try runsLocked().first { $0.id == id } }
    }

    func run(idOrName text: String) throws -> RunRecord? {
        if let id = Int(text), let run = try run(id: id) { return run }
        return try run(named: text)
    }

    func checkpoint(id: Int) throws -> CheckpointRecord? {
        try lock.withLock { try checkpointsLocked(visibleOnly: false).first { $0.id == id } }
    }

    func latestCheckpoint(runID: Int) throws -> CheckpointRecord? {
        try lock.withLock {
            try checkpointsLocked(visibleOnly: false)
                .filter { $0.runID == runID }
                .max(by: { $0.step < $1.step })
        }
    }

    @discardableResult
    func createRun(
        name: String,
        parentCheckpointID: Int?,
        channels: Int,
        blocksJSON: String,
        dtype: String,
        cropSize: Int?,
        configJSON: String,
        status: String,
        notes: String?
    ) throws -> Int {
        try lock.withLock {
            if try runIDLocked(named: name) != nil {
                throw WhoodeoDBError("Run existiert bereits: \(name)")
            }
            let id = try insertRunLocked(
                name: name,
                parentCheckpointID: parentCheckpointID,
                channels: channels,
                blocksJSON: blocksJSON,
                dtype: dtype,
                cropSize: cropSize,
                configJSON: configJSON,
                status: status,
                startedAt: Self.timestamp(),
                finishedAt: nil,
                notes: notes
            )
            try attachAllSourcesLocked(runID: id)
            return id
        }
    }

    /// Resume may change `.always` parameters; the run keeps the latest config.
    func updateRunConfig(_ id: Int, configJSON: String, cropSize: Int?) throws {
        try lock.withLock {
            let stmt = try prepare("UPDATE runs SET config_json = ?, crop_size = ? WHERE id = ?")
            defer { sqlite3_finalize(stmt) }
            bind(stmt, 1, configJSON)
            if let cropSize { sqlite3_bind_int(stmt, 2, Int32(cropSize)) } else { sqlite3_bind_null(stmt, 2) }
            sqlite3_bind_int(stmt, 3, Int32(id))
            try stepDone(stmt)
        }
    }

    func updateRunStatus(_ id: Int, status: String, finished: Bool) throws {
        try lock.withLock {
            if finished {
                let stmt = try prepare("UPDATE runs SET status = ?, finished_at = ? WHERE id = ?")
                defer { sqlite3_finalize(stmt) }
                bind(stmt, 1, status)
                bind(stmt, 2, Self.timestamp())
                sqlite3_bind_int(stmt, 3, Int32(id))
                try stepDone(stmt)
            } else {
                let stmt = try prepare("UPDATE runs SET status = ?, finished_at = NULL WHERE id = ?")
                defer { sqlite3_finalize(stmt) }
                bind(stmt, 1, status)
                sqlite3_bind_int(stmt, 2, Int32(id))
                try stepDone(stmt)
            }
        }
    }

    @discardableResult
    func insertCheckpoint(
        runID: Int,
        step: Int,
        relPath: String,
        exportDtype: String,
        optimizerPath: String?,
        valLoss: Double?,
        valPSNR: Double?,
        valSSIM: Double?,
        visibleInApp: Bool
    ) throws -> Int {
        try lock.withLock {
            if let existing = try queryIntOptional("SELECT id FROM checkpoints WHERE run_id = ? AND step = ?", runID, step) {
                let stmt = try prepare(
                    """
                    UPDATE checkpoints SET rel_path = ?, export_dtype = ?, optimizer_path = ?,
                        val_loss = ?, val_psnr = ?, val_ssim = ?, visible_in_app = ? WHERE id = ?
                    """
                )
                defer { sqlite3_finalize(stmt) }
                bind(stmt, 1, relPath)
                bind(stmt, 2, exportDtype)
                bind(stmt, 3, optimizerPath)
                if let valLoss { sqlite3_bind_double(stmt, 4, valLoss) } else { sqlite3_bind_null(stmt, 4) }
                if let valPSNR { sqlite3_bind_double(stmt, 5, valPSNR) } else { sqlite3_bind_null(stmt, 5) }
                if let valSSIM { sqlite3_bind_double(stmt, 6, valSSIM) } else { sqlite3_bind_null(stmt, 6) }
                sqlite3_bind_int(stmt, 7, visibleInApp ? 1 : 0)
                sqlite3_bind_int(stmt, 8, Int32(existing))
                try stepDone(stmt)
                return existing
            }
            let id = try insertCheckpointLocked(
                runID: runID, step: step, relPath: relPath, exportDtype: exportDtype, createdAt: Self.timestamp()
            )
            let stmt = try prepare(
                "UPDATE checkpoints SET optimizer_path = ?, val_loss = ?, val_psnr = ?, val_ssim = ?, visible_in_app = ? WHERE id = ?"
            )
            defer { sqlite3_finalize(stmt) }
            bind(stmt, 1, optimizerPath)
            if let valLoss { sqlite3_bind_double(stmt, 2, valLoss) } else { sqlite3_bind_null(stmt, 2) }
            if let valPSNR { sqlite3_bind_double(stmt, 3, valPSNR) } else { sqlite3_bind_null(stmt, 3) }
            if let valSSIM { sqlite3_bind_double(stmt, 4, valSSIM) } else { sqlite3_bind_null(stmt, 4) }
            sqlite3_bind_int(stmt, 5, visibleInApp ? 1 : 0)
            sqlite3_bind_int(stmt, 6, Int32(id))
            try stepDone(stmt)
            return id
        }
    }

    func setCheckpointVisible(id: Int, visible: Bool) throws {
        try lock.withLock {
            let stmt = try prepare("UPDATE checkpoints SET visible_in_app = ? WHERE id = ?")
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int(stmt, 1, visible ? 1 : 0)
            sqlite3_bind_int(stmt, 2, Int32(id))
            try stepDone(stmt)
        }
    }

    func insertMetrics(runID: Int, step: Int, split: String, values: [String: Double]) throws {
        let now = Self.timestamp()
        try lock.withLock {
            for (name, value) in values {
                let stmt = try prepare(
                    """
                    INSERT INTO metrics(run_id, step, split, name, value, recorded_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, step, split, name) DO UPDATE SET value = excluded.value, recorded_at = excluded.recorded_at
                    """
                )
                defer { sqlite3_finalize(stmt) }
                sqlite3_bind_int(stmt, 1, Int32(runID))
                sqlite3_bind_int(stmt, 2, Int32(step))
                bind(stmt, 3, split)
                bind(stmt, 4, name)
                sqlite3_bind_double(stmt, 5, value)
                bind(stmt, 6, now)
                try stepDone(stmt)
            }
        }
    }

    func allCheckpoints() throws -> [CheckpointRecord] {
        try lock.withLock { try checkpointsLocked(visibleOnly: false) }
    }

    func lastMetrics(runID: Int, split: String) throws -> [String: Double] {
        try lock.withLock {
            let probe = try prepare("SELECT MAX(step) FROM metrics WHERE run_id = ? AND split = ?")
            defer { sqlite3_finalize(probe) }
            sqlite3_bind_int(probe, 1, Int32(runID))
            bind(probe, 2, split)
            guard sqlite3_step(probe) == SQLITE_ROW, sqlite3_column_type(probe, 0) != SQLITE_NULL else {
                return [:]
            }
            let step = int(probe, 0)
            let stmt = try prepare("SELECT name, value FROM metrics WHERE run_id = ? AND step = ? AND split = ?")
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int(stmt, 1, Int32(runID))
            sqlite3_bind_int(stmt, 2, Int32(step))
            bind(stmt, 3, split)
            var values: [String: Double] = [:]
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let name = text(stmt, 0) {
                    values[name] = sqlite3_column_double(stmt, 1)
                }
            }
            values["_step"] = Double(step)
            return values
        }
    }


    func runSourceCounts(runID: Int) throws -> (train: Int, val: Int) {
        try lock.withLock {
            let train = try queryIntOptional("SELECT COUNT(*) FROM run_sources WHERE run_id = ? AND split = ?", runID, "train") ?? 0
            let val = try queryIntOptional("SELECT COUNT(*) FROM run_sources WHERE run_id = ? AND split = ?", runID, "val") ?? 0
            return (train, val)
        }
    }

    func metrics(runID: Int) throws -> [MetricPoint] {
        try lock.withLock {
            let stmt = try prepare(
                "SELECT step, split, name, value, recorded_at FROM metrics WHERE run_id = ? ORDER BY step, split, name"
            )
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int(stmt, 1, Int32(runID))
            var rows: [MetricPoint] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                rows.append(MetricPoint(
                    step: int(stmt, 0),
                    split: text(stmt, 1) ?? "",
                    name: text(stmt, 2) ?? "",
                    value: sqlite3_column_double(stmt, 3),
                    recordedAt: text(stmt, 4) ?? ""
                ))
            }
            return rows
        }
    }

    func cacheCountsByCheckpoint() throws -> [Int: Int] {
        try lock.withLock {
            let stmt = try prepare("SELECT checkpoint_id, COUNT(*) FROM cache_files GROUP BY checkpoint_id")
            defer { sqlite3_finalize(stmt) }
            var found: [Int: Int] = [:]
            while sqlite3_step(stmt) == SQLITE_ROW {
                found[int(stmt, 0)] = int(stmt, 1)
            }
            return found
        }
    }

    func cacheBytesByCheckpoint() throws -> [Int: Int64] {

        let root = try cacheRoot()
        let rows = try lock.withLock { () -> [(Int, String)] in
            let stmt = try prepare("SELECT checkpoint_id, rel_path FROM cache_files")
            defer { sqlite3_finalize(stmt) }
            var found: [(Int, String)] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                found.append((int(stmt, 0), text(stmt, 1) ?? ""))
            }
            return found
        }
        var sizes: [Int: Int64] = [:]
        for (id, rel) in rows {
            sizes[id, default: 0] += Self.folderSize(root.appendingPathComponent(rel))
        }
        return sizes
    }

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


    func deleteRun(id: Int, removeFiles: Bool) throws {
        guard let run = try run(id: id) else {
            throw WhoodeoDBError("Lauf \(id) nicht gefunden")
        }
        if run.status == "running" {
            throw WhoodeoDBError("Lauf \(run.name) läuft noch; zuerst stoppen")
        }
        let children = try childRuns(ofRunID: id)
        if !children.isEmpty {
            let names = children.map(\.name).joined(separator: ", ")
            throw WhoodeoDBError("Lauf \(run.name) hat Kind-Läufe (\(names)); zuerst die Kind-Läufe löschen")
        }
        let checkpoints = try lock.withLock {
            try checkpointsLocked(visibleOnly: false).filter { $0.runID == id }
        }
        for checkpoint in checkpoints {
            try deleteCheckpoint(id: checkpoint.id, removeFiles: removeFiles)
        }
        if removeFiles {
            try ValPreview.removeAll(runName: run.name)
            // train.log from the app, then the run folder once it is empty.
            if !run.name.isEmpty, !run.name.contains("/"), !run.name.hasPrefix(".") {
                let folder = WhoodeoPaths.modelsDirectory.appendingPathComponent(run.name, isDirectory: true)
                let fm = FileManager.default
                try? fm.removeItem(at: folder.appendingPathComponent("train.log"))
                try? fm.removeItem(at: folder.appendingPathComponent(".DS_Store"))
                if let rest = try? fm.contentsOfDirectory(atPath: folder.path), rest.isEmpty {
                    try? fm.removeItem(at: folder)
                }
            }
        }
        try lock.withLock {
            try exec("DELETE FROM runs WHERE id = \(id)")
        }
    }

    func trainingPairs(split: String, rungs: [String], presets: [String]) throws -> [TrainingPair] {
        try lock.withLock {
            var sql = """
                SELECT s.id, s.name, s.frames, s.split, m.rung, m.rel_path, m.width, m.height,
                       v.preset, v.codec, v.rel_path
                FROM sources s
                JOIN source_masters m ON m.source_id = s.id
                JOIN source_variants v ON v.master_id = m.id
                WHERE s.enabled = 1 AND v.enabled = 1 AND s.split = ?
                """
            if !rungs.isEmpty {
                let list = rungs.map { "'\($0.replacingOccurrences(of: "'", with: "''"))'" }.joined(separator: ",")
                sql += " AND m.rung IN (\(list))"
            }
            if !presets.isEmpty {
                let list = presets.map { "'\($0.replacingOccurrences(of: "'", with: "''"))'" }.joined(separator: ",")
                sql += " AND v.preset IN (\(list))"
            }
            sql += " ORDER BY s.name, m.rung, v.preset"
            let stmt = try prepare(sql)
            defer { sqlite3_finalize(stmt) }
            bind(stmt, 1, split)
            var rows: [TrainingPair] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                rows.append(TrainingPair(
                    sourceID: int(stmt, 0),
                    name: text(stmt, 1) ?? "",
                    frames: int(stmt, 2),
                    split: text(stmt, 3) ?? split,
                    rung: text(stmt, 4) ?? "",
                    origRelPath: text(stmt, 5) ?? "",
                    width: int(stmt, 6),
                    height: int(stmt, 7),
                    preset: text(stmt, 8) ?? "",
                    codec: text(stmt, 9) ?? "",
                    lowRelPath: text(stmt, 10) ?? ""
                ))
            }
            return rows
        }
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

    static func codec(forPreset preset: String) -> String {
        let prefix = preset.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: true).first.map(String.init) ?? preset
        switch prefix {
        case "x264", "h264", "avc": return "h264"
        case "mpeg4": return "mpeg4"
        case "mpeg2": return "mpeg2"
        default: return prefix
        }
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

    private struct PackageMeta {
        var step: Int
        var channels: Int
        var blocksJSON: String
        var dtype: String
        var configJSON: String
        var inFrames: Int
        var conditionSize: Int
        var blocks: Int
    }

    private static func packageMeta(_ url: URL) throws -> PackageMeta {
        let configURL = url.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MLXAlignModel.Config.self, from: data)
        let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let channels = config.block_widths.first ?? 0
        var blocks = config.encoder_blocks
        blocks.append(config.blocks)
        let blocksJSON: String
        if let encoded = try? JSONSerialization.data(withJSONObject: blocks),
           let text = String(data: encoded, encoding: .utf8)
        {
            blocksJSON = text
        } else {
            blocksJSON = "\(blocks)"
        }
        let dtype: String
        switch (config.compute ?? (raw["compute"] as? String))?.lowercased() {
        case "float16", "fp16": dtype = "fp16"
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
        let configJSON: String
        if let encoded = try? JSONSerialization.data(withJSONObject: stored, options: [.sortedKeys]),
           let text = String(data: encoded, encoding: .utf8)
        {
            configJSON = text
        } else {
            configJSON = "{}"
        }
        return PackageMeta(
            step: config.step ?? 0,
            channels: channels,
            blocksJSON: blocksJSON,
            dtype: dtype,
            configJSON: configJSON,
            inFrames: config.in_frames,
            conditionSize: config.condition_size,
            blocks: config.blocks
        )
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

    static func timestamp(_ date: Date = Date()) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    private static func intValue(_ raw: Any?) -> Int? {
        if let value = raw as? Int { return value }
        if let value = raw as? Double { return Int(value) }
        if let value = raw as? String { return Int(value) }
        return nil
    }
}

