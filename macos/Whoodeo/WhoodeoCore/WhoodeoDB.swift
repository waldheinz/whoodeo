import Darwin
import Foundation
import GRDB
import os

nonisolated struct WhoodeoDBError: Error, LocalizedError, CustomStringConvertible {
    var description: String
    var errorDescription: String? { description }
    init(_ description: String) { self.description = description }
}

// MARK: - Database

/// Shared SQLite database for the app and the trainer.
///
/// A GRDB `DatabasePool` in WAL mode: reads never block the writer, and the app sees what a
/// separate `whoodeo-train` process commits as soon as it commits. Writes from both processes
/// wait up to 5 s for each other (`busy_timeout`).
nonisolated final class WhoodeoDB: Sendable {
    static let schemaVersion = 1

    static let shared: WhoodeoDB? = {
        do {
            return try openDefault()
        } catch {
            Log.model.error("Datenbank: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }()

    let pool: DatabasePool

    static func openDefault() throws -> WhoodeoDB {
        let opened = try open(at: WhoodeoPaths.databaseURL)
        try opened.bootstrap()
        return opened
    }

    static func open(at url: URL) throws -> WhoodeoDB {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var config = Configuration()
        config.foreignKeysEnabled = true
        config.busyMode = .timeout(5)
        config.journalMode = .wal
        let pool = try DatabasePool(path: url.path, configuration: config)
        if try pool.read({ try Int.fetchOne($0, sql: "PRAGMA user_version") ?? 0 }) < schemaVersion {
            try pool.write(migrate)
        }
        return WhoodeoDB(pool: pool)
    }

    private init(pool: DatabasePool) {
        self.pool = pool
    }

    /// Fills default settings, imports the catalog once, and adopts packages and caches not yet in the database.
    /// Writes only when something is missing.
    func bootstrap() throws {
        let defaults: [(String, String)] = [
            ("dataset_root", WhoodeoPaths.defaultDatasetRootPath),
            ("schema_version", String(Self.schemaVersion)),
            ("holdout_mode", "title"),
            ("holdout_fraction", "0.1"),
            ("holdout_note", "The PyTorch trainer held out 10% of the frames inside every film (two spans at 30% and 70%). The database uses a stable 10% title split instead."),
        ]
        let missing = try pool.read { db in try defaults.filter { try Self.setting(db, $0.0) == nil } }
        if !missing.isEmpty {
            try pool.write { db in
                for (key, value) in missing {
                    try db.execute(sql: "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", arguments: [key, value])
                }
            }
        }
        let sources = try pool.read { try Int.fetchOne($0, sql: "SELECT COUNT(*) FROM sources") ?? 0 }
        if sources == 0, let root = try datasetRoot(),
           FileManager.default.fileExists(atPath: root.appendingPathComponent("catalog.json").path)
        {
            _ = try importCatalog(root: root)
        }
        _ = try adoptOrphanModels()
        _ = try associateExistingCaches(at: try cacheRoot())
    }

    // MARK: Settings and requests

    func setting(_ key: String) throws -> String? {
        try pool.read { try Self.setting($0, key) }
    }

    func setSetting(_ value: String, forKey key: String) throws {
        try pool.write { try Self.setSetting($0, value, forKey: key) }
    }

    func deleteSetting(_ key: String) throws {
        try pool.write { try $0.execute(sql: "DELETE FROM settings WHERE key = ?", arguments: [key]) }
    }

    static func setting(_ db: Database, _ key: String) throws -> String? {
        try String.fetchOne(db, sql: "SELECT value FROM settings WHERE key = ?", arguments: [key])
    }

    static func setSetting(_ db: Database, _ value: String, forKey key: String) throws {
        try db.execute(
            sql: "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            arguments: [key, value]
        )
    }

    func datasetRoot() throws -> URL? {
        try setting("dataset_root").map { URL(fileURLWithPath: $0) }
    }

    func cacheRoot() throws -> URL {
        if let stored = try setting("cache_root"), !stored.isEmpty {
            return URL(fileURLWithPath: stored)
        }
        let resolved = WhoodeoPaths.resolvedCacheRoot()
        try setSetting(resolved.path, forKey: "cache_root")
        return resolved
    }

    static func checkpointRequestKey(_ runID: Int) -> String { "checkpoint_request.\(runID)" }
    static func stopRequestKey(_ runID: Int) -> String { "stop_request.\(runID)" }

    func requestCheckpoint(runID: Int) throws {
        try setSetting("1", forKey: Self.checkpointRequestKey(runID))
    }

    /// Returns true and clears the key if a request was pending for this run.
    func consumeCheckpointRequest(runID: Int) throws -> Bool {
        let key = Self.checkpointRequestKey(runID)
        guard try flag(key) else { return false }
        try deleteSetting(key)
        return true
    }

    func requestStop(runID: Int) throws {
        try setSetting("1", forKey: Self.stopRequestKey(runID))
    }

    func hasStopRequest(runID: Int) throws -> Bool {
        try flag(Self.stopRequestKey(runID))
    }

    func clearStopRequest(runID: Int) throws {
        try deleteSetting(Self.stopRequestKey(runID))
    }

    private func flag(_ key: String) throws -> Bool {
        guard let value = try setting(key) else { return false }
        return !value.isEmpty && value != "0"
    }

    static func runPidKey(_ runID: Int) -> String { "run_pid.\(runID)" }

    func setRunPid(_ pid: pid_t, runID: Int) throws {
        try setSetting(String(Int(pid)), forKey: Self.runPidKey(runID))
    }

    func runPid(runID: Int) throws -> pid_t? {
        guard let text = try setting(Self.runPidKey(runID)), let pid = pid_t(text), pid > 0 else {
            return nil
        }
        return pid
    }

    func clearRunPid(runID: Int) throws {
        try deleteSetting(Self.runPidKey(runID))
    }

    /// `true` if a process with this PID exists. `EPERM` means it lives but we may not signal it.
    static func processExists(_ pid: pid_t) -> Bool {
        if pid <= 0 { return false }
        if kill(pid, 0) == 0 { return true }
        return errno != ESRCH
    }

    /// Marks `running` runs whose stored trainer PID is gone as `failed`.
    /// A run without a stored PID is left alone (legacy binaries, including a still-running older trainer).
    /// Writes only when at least one dead run is found.
    @discardableResult
    func reapDeadRuns() throws -> [RunRecord] {
        let running = try runningRuns()
        var dead: [RunRecord] = []
        for run in running {
            guard let pid = try runPid(runID: run.id) else { continue }
            if !Self.processExists(pid) {
                dead.append(run)
            }
        }
        guard !dead.isEmpty else { return [] }
        try pool.write { db in
            let finishedAt = Self.timestamp()
            for run in dead {
                try db.execute(
                    sql: "UPDATE runs SET status = ?, finished_at = ? WHERE id = ? AND status = 'running'",
                    arguments: ["failed", finishedAt, run.id]
                )
                try db.execute(sql: "DELETE FROM settings WHERE key = ?", arguments: [Self.runPidKey(run.id)])
            }
        }
        return dead
    }

    // MARK: Reads

    func run(id: Int) throws -> RunRecord? {
        try pool.read { try RunRecord.fetchOne($0, sql: "SELECT * FROM runs WHERE id = ?", arguments: [id]) }
    }

    func run(named name: String) throws -> RunRecord? {
        try pool.read { try RunRecord.fetchOne($0, sql: "SELECT * FROM runs WHERE name = ?", arguments: [name]) }
    }

    func run(idOrName text: String) throws -> RunRecord? {
        if let id = Int(text), let run = try run(id: id) { return run }
        return try run(named: text)
    }

    func runningRuns() throws -> [RunRecord] {
        try pool.read { try RunRecord.fetchAll($0, sql: "SELECT * FROM runs WHERE status = 'running'") }
    }

    /// Runs whose `parent_checkpoint_id` points at a checkpoint of `id`.
    func childRuns(ofRunID id: Int) throws -> [RunRecord] {
        try pool.read {
            try RunRecord.fetchAll($0, sql: """
                SELECT r.* FROM runs r JOIN checkpoints c ON c.id = r.parent_checkpoint_id
                WHERE c.run_id = ?
                """, arguments: [id])
        }
    }

    /// Newest first.
    func allCheckpoints() throws -> [CheckpointRecord] {
        try pool.read { try Self.checkpoints($0) }
    }

    func checkpoint(id: Int) throws -> CheckpointRecord? {
        try pool.read { try Self.checkpoints($0, "WHERE c.id = ?", [id]).first }
    }

    func latestCheckpoint(runID: Int) throws -> CheckpointRecord? {
        try pool.read { try Self.checkpoints($0, "WHERE c.run_id = ?", [runID]).max { $0.step < $1.step } }
    }

    private static func checkpoints(_ db: Database, _ filter: String = "", _ arguments: StatementArguments = []) throws -> [CheckpointRecord] {
        try CheckpointRecord.fetchAll(
            db,
            sql: "\(CheckpointRecord.selectSQL) \(filter) ORDER BY c.created_at DESC, c.step DESC",
            arguments: arguments
        )
    }

    func counts() throws -> DatabaseCounts {
        try pool.read { db in
            func count(_ sql: String) throws -> Int { try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM \(sql)") ?? 0 }
            let train = try count("sources WHERE split = 'train'")
            let val = try count("sources WHERE split = 'val'")
            return DatabaseCounts(
                sources: train + val,
                train: train,
                val: val,
                masters: try count("source_masters"),
                variants: try count("source_variants"),
                runs: try count("runs"),
                checkpoints: try count("checkpoints"),
                cacheFiles: try count("cache_files")
            )
        }
    }

    /// Root runs (no parent, or a parent that is gone) oldest first, each with its checkpoints and child runs.
    func runTree() throws -> [RunTreeNode] {
        let (runs, checkpoints, cacheCounts) = try pool.read { db in
            (
                try RunRecord.fetchAll(db, sql: "SELECT * FROM runs ORDER BY started_at"),
                try Self.checkpoints(db),
                try Self.cacheCounts(db)
            )
        }
        let byRun = Dictionary(grouping: checkpoints, by: \.runID)
        let runOfCheckpoint = Dictionary(uniqueKeysWithValues: checkpoints.map { ($0.id, $0.runID) })
        func parentRun(_ run: RunRecord) -> Int? { run.parentCheckpointID.flatMap { runOfCheckpoint[$0] } }
        var children: [Int: [RunRecord]] = [:]
        for run in runs {
            if let parent = parentRun(run) { children[parent, default: []].append(run) }
        }
        func node(for run: RunRecord) -> RunTreeNode {
            let own = (byRun[run.id] ?? []).sorted { $0.step < $1.step }
            return RunTreeNode(
                run: run,
                checkpoints: own,
                cacheCount: own.reduce(0) { $0 + (cacheCounts[$1.id] ?? 0) },
                children: (children[run.id] ?? []).map(node(for:))
            )
        }
        return runs
            .filter { parentRun($0) == nil }
            .map(node(for:))
    }

    func lastMetrics(runID: Int, split: String) throws -> [String: Double] {
        try pool.read { db in
            guard let step = try Int.fetchOne(
                db, sql: "SELECT MAX(step) FROM metrics WHERE run_id = ? AND split = ?", arguments: [runID, split]
            ) else { return [:] }
            var values: [String: Double] = ["_step": Double(step)]
            for row in try Row.fetchAll(
                db, sql: "SELECT name, value FROM metrics WHERE run_id = ? AND step = ? AND split = ?",
                arguments: [runID, step, split]
            ) {
                values[row["name"]] = row["value"]
            }
            return values
        }
    }

    func metrics(runID: Int) throws -> [MetricPoint] {
        try pool.read {
            try MetricPoint.fetchAll($0, sql: "SELECT * FROM metrics WHERE run_id = ? ORDER BY step, split, name", arguments: [runID])
        }
    }

    func runSourceCounts(runID: Int) throws -> (train: Int, val: Int) {
        try pool.read { db in
            func count(_ split: String) throws -> Int {
                try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM run_sources WHERE run_id = ? AND split = ?", arguments: [runID, split]) ?? 0
            }
            return (try count("train"), try count("val"))
        }
    }

    func cacheCountsByCheckpoint() throws -> [Int: Int] {
        try pool.read(Self.cacheCounts)
    }

    private static func cacheCounts(_ db: Database) throws -> [Int: Int] {
        var found: [Int: Int] = [:]
        for row in try Row.fetchAll(db, sql: "SELECT checkpoint_id, COUNT(*) FROM cache_files GROUP BY checkpoint_id") {
            found[row[0]] = row[1]
        }
        return found
    }

    func cacheBytesByCheckpoint() throws -> [Int: Int64] {
        let root = try cacheRoot()
        let rows = try pool.read { db in
            try Row.fetchAll(db, sql: "SELECT checkpoint_id, rel_path FROM cache_files").map { ($0[0] as Int, $0[1] as String) }
        }
        var sizes: [Int: Int64] = [:]
        for (id, rel) in rows {
            sizes[id, default: 0] += Self.folderSize(root.appendingPathComponent(rel))
        }
        return sizes
    }

    func trainingPairs(split: String, rungs: [String], presets: [String]) throws -> [TrainingPair] {
        var sql = """
            SELECT s.id AS source_id, s.name, s.frames, s.split, m.rung, m.rel_path AS orig_rel_path, m.width, m.height,
                   v.preset, v.codec, v.rel_path AS low_rel_path
            FROM sources s
            JOIN source_masters m ON m.source_id = s.id
            JOIN source_variants v ON v.master_id = m.id
            WHERE s.enabled = 1 AND v.enabled = 1 AND s.split = ?
            """
        var arguments: StatementArguments = [split]
        if !rungs.isEmpty {
            sql += " AND m.rung IN (\(databaseQuestionMarks(count: rungs.count)))"
            arguments += StatementArguments(rungs)
        }
        if !presets.isEmpty {
            sql += " AND v.preset IN (\(databaseQuestionMarks(count: presets.count)))"
            arguments += StatementArguments(presets)
        }
        sql += " ORDER BY s.name, m.rung, v.preset"
        return try pool.read { [sql, arguments] in try TrainingPair.fetchAll($0, sql: sql, arguments: arguments) }
    }

    // MARK: Training writes

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
        try pool.write { db in
            if try Int.fetchOne(db, sql: "SELECT id FROM runs WHERE name = ?", arguments: [name]) != nil {
                throw WhoodeoDBError("Run existiert bereits: \(name)")
            }
            return try Self.insertRun(
                db, name: name, parentCheckpointID: parentCheckpointID, channels: channels, blocksJSON: blocksJSON,
                dtype: dtype, cropSize: cropSize, configJSON: configJSON, status: status,
                startedAt: Self.timestamp(), notes: notes
            )
        }
    }

    /// Inserts a run and attaches every source with its split.
    static func insertRun(
        _ db: Database, name: String, parentCheckpointID: Int?, channels: Int, blocksJSON: String, dtype: String,
        cropSize: Int?, configJSON: String, status: String, startedAt: String, notes: String?
    ) throws -> Int {
        try db.execute(sql: """
            INSERT INTO runs(name, parent_checkpoint_id, channels, blocks_json, dtype, crop_size, config_json, status, started_at, notes)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, arguments: [name, parentCheckpointID, channels, blocksJSON, dtype, cropSize, configJSON, status, startedAt, notes])
        let id = Int(db.lastInsertedRowID)
        try db.execute(sql: "INSERT OR IGNORE INTO run_sources(run_id, source_id, split) SELECT ?, id, split FROM sources", arguments: [id])
        return id
    }

    /// Resume may change `.always` parameters; the run keeps the latest config.
    func updateRunConfig(_ id: Int, configJSON: String, cropSize: Int?) throws {
        try pool.write {
            try $0.execute(sql: "UPDATE runs SET config_json = ?, crop_size = ? WHERE id = ?", arguments: [configJSON, cropSize, id])
        }
    }

    func updateRunStatus(_ id: Int, status: String, finished: Bool) throws {
        let finishedAt = finished ? Self.timestamp() : nil
        try pool.write { db in
            try db.execute(sql: "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?", arguments: [status, finishedAt, id])
            if status != "running" {
                try db.execute(sql: "DELETE FROM settings WHERE key = ?", arguments: [Self.runPidKey(id)])
            }
        }
    }

    /// Inserts the checkpoint, or updates everything but `created_at` if the run already has one at this step.
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
        try pool.write { db in
            try db.execute(sql: """
                INSERT INTO checkpoints(run_id, step, rel_path, export_dtype, optimizer_path, val_loss, val_psnr, val_ssim, visible_in_app, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step) DO UPDATE SET
                    rel_path = excluded.rel_path, export_dtype = excluded.export_dtype, optimizer_path = excluded.optimizer_path,
                    val_loss = excluded.val_loss, val_psnr = excluded.val_psnr, val_ssim = excluded.val_ssim,
                    visible_in_app = excluded.visible_in_app
                """, arguments: [runID, step, relPath, exportDtype, optimizerPath, valLoss, valPSNR, valSSIM, visibleInApp, Self.timestamp()])
            guard let id = try Int.fetchOne(db, sql: "SELECT id FROM checkpoints WHERE run_id = ? AND step = ?", arguments: [runID, step]) else {
                throw WhoodeoDBError("Checkpoint \(runID)/\(step) fehlt nach dem Schreiben")
            }
            return id
        }
    }

    func setCheckpointVisible(id: Int, visible: Bool) throws {
        try pool.write { try $0.execute(sql: "UPDATE checkpoints SET visible_in_app = ? WHERE id = ?", arguments: [visible, id]) }
    }

    func insertMetrics(runID: Int, step: Int, split: String, values: [String: Double]) throws {
        let now = Self.timestamp()
        try pool.write { db in
            for (name, value) in values {
                try db.execute(sql: """
                    INSERT INTO metrics(run_id, step, split, name, value, recorded_at) VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, step, split, name) DO UPDATE SET value = excluded.value, recorded_at = excluded.recorded_at
                    """, arguments: [runID, step, split, name, value, now])
            }
        }
    }

    // MARK: Caches

    /// Records an archive folder under the cache root. Existing cache files are never renamed or deleted.
    func registerCache(forModel url: URL, filmID: String, condition: String?) throws {
        let rel = ModelCacheKey.relativePath(filmID: filmID, model: url, condition: condition)
        let key = ModelCacheKey.modelKey(of: url)
        try pool.write { db in
            guard let checkpointID = try Self.checkpointID(db, forModel: url) else { return }
            try Self.registerCache(db, checkpointID: checkpointID, filmID: filmID, modelKey: key, relPath: rel, condition: condition)
        }
    }

    /// Returns false if the folder was already recorded.
    @discardableResult
    static func registerCache(_ db: Database, checkpointID: Int, filmID: String, modelKey: String, relPath: String, condition: String?) throws -> Bool {
        try db.execute(sql: """
            INSERT OR IGNORE INTO cache_files(checkpoint_id, film_id, model_key, rel_path, condition, added_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """, arguments: [checkpointID, filmID, modelKey, relPath, condition, timestamp()])
        return db.changesCount > 0
    }

    static func checkpointID(_ db: Database, forModel url: URL) throws -> Int? {
        let rel = relativeModelPath(of: url)
        let name = url.standardizedFileURL.lastPathComponent
        for candidate in rel == name ? [rel] : [rel, name] {
            if let id = try Int.fetchOne(db, sql: "SELECT id FROM checkpoints WHERE rel_path = ?", arguments: [candidate]) {
                return id
            }
        }
        return nil
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

    // MARK: Deleting

    /// Deletes a checkpoint row and, when `removeFiles` is true, its package, optimizer, validation previews,
    /// and recorded cache folders. Only paths under the models folder or the cache root are touched.
    func deleteCheckpoint(id: Int, removeFiles: Bool) throws {
        if removeFiles {
            let cacheRoot = try cacheRoot()
            let models = WhoodeoPaths.modelsDirectory
            let (caches, info) = try pool.read { db in
                (
                    try String.fetchAll(db, sql: "SELECT rel_path FROM cache_files WHERE checkpoint_id = ?", arguments: [id]),
                    try Row.fetchOne(db, sql: """
                        SELECT c.rel_path, c.optimizer_path, c.step, r.name
                        FROM checkpoints c JOIN runs r ON r.id = c.run_id WHERE c.id = ?
                        """, arguments: [id]).map { ($0[0] as String, $0[1] as String?, $0[2] as Int, $0[3] as String) }
                )
            }
            for rel in caches {
                try Self.removeIfSafe(cacheRoot.appendingPathComponent(rel), under: cacheRoot)
            }
            if let (packageRel, optimizerRel, step, runName) = info {
                if let optimizerRel { try Self.removeIfSafe(models.appendingPathComponent(optimizerRel), under: models) }
                try ValPreview.remove(runName: runName, step: step)
                let package = models.appendingPathComponent(packageRel)
                try Self.removeIfSafe(package, under: models)
                // The run folder goes once its last package is gone.
                let parent = package.deletingLastPathComponent()
                if parent.standardizedFileURL.resolvingSymlinksInPath() != models.standardizedFileURL.resolvingSymlinksInPath() {
                    let leftover = (try? FileManager.default.contentsOfDirectory(atPath: parent.path)) ?? ["."]
                    if leftover.allSatisfy({ $0.hasPrefix(".") }) {
                        try Self.removeIfSafe(parent, under: models)
                    }
                }
            }
        }
        try pool.write { try $0.execute(sql: "DELETE FROM checkpoints WHERE id = ?", arguments: [id]) }
    }

    /// Deletes a stopped run without child runs, its checkpoints, and (when `removeFiles`) their files and `train.log`.
    func deleteRun(id: Int, removeFiles: Bool) throws {
        try reapDeadRuns()
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
        let checkpointIDs = try pool.read { try Int.fetchAll($0, sql: "SELECT id FROM checkpoints WHERE run_id = ?", arguments: [id]) }
        for checkpointID in checkpointIDs {
            try deleteCheckpoint(id: checkpointID, removeFiles: removeFiles)
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
        try pool.write { db in
            try db.execute(sql: "DELETE FROM runs WHERE id = ?", arguments: [id])
            for key in [Self.runPidKey(id), Self.stopRequestKey(id), Self.checkpointRequestKey(id)] {
                try db.execute(sql: "DELETE FROM settings WHERE key = ?", arguments: [key])
            }
        }
    }

    static func timestamp(_ date: Date = Date()) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    // MARK: - Schema

    /// Creates the tables of schema version 1 on a new database. Existing databases are left as they are.
    private static func migrate(_ db: Database) throws {
        guard try Int.fetchOne(db, sql: "PRAGMA user_version") ?? 0 < schemaVersion else { return }
        try db.execute(sql: """
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
