import Foundation
import GRDB

/// Rows of `WhoodeoDB`, decoded by column name.
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

/// A checkpoint with the fields of its run and of the checkpoint it was fine-tuned from.
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
    /// Low frame k trains against master frame `(k * frameRatioNum + frameRatioDen / 2) / frameRatioDen`.
    var frameRatioNum: Int = 1
    var frameRatioDen: Int = 1
    var lowRelPath: String

    /// Frames the low file has for `masterFrames` master frames.
    func lowFrames(masterFrames: Int) -> Int {
        masterFrames * frameRatioDen / frameRatioNum
    }

    /// Master frame that is the target of low frame `lowIndex`.
    func masterIndex(lowIndex: Int) -> Int {
        (lowIndex * frameRatioNum + frameRatioDen / 2) / frameRatioDen
    }
}

nonisolated extension RunRecord: FetchableRecord {
    init(row: Row) {
        self.init(
            id: row["id"], name: row["name"], parentCheckpointID: row["parent_checkpoint_id"],
            channels: row["channels"], blocksJSON: row["blocks_json"], dtype: row["dtype"],
            cropSize: row["crop_size"], configJSON: row["config_json"], status: row["status"],
            startedAt: row["started_at"], finishedAt: row["finished_at"], notes: row["notes"]
        )
    }
}

nonisolated extension CheckpointRecord: FetchableRecord {
    /// Columns for `init(row:)`; append a WHERE clause.
    static let selectSQL = """
        SELECT c.*, r.name AS run_name, r.channels, r.blocks_json, r.parent_checkpoint_id, r.config_json,
               p.step AS parent_step, pr.name AS parent_run_name
        FROM checkpoints c
        JOIN runs r ON r.id = c.run_id
        LEFT JOIN checkpoints p ON p.id = r.parent_checkpoint_id
        LEFT JOIN runs pr ON pr.id = p.run_id
        """

    init(row: Row) {
        self.init(
            id: row["id"], runID: row["run_id"], step: row["step"], relPath: row["rel_path"],
            exportDtype: row["export_dtype"], optimizerPath: row["optimizer_path"],
            valLoss: row["val_loss"], valPSNR: row["val_psnr"], valSSIM: row["val_ssim"],
            visibleInApp: row["visible_in_app"], createdAt: row["created_at"],
            runName: row["run_name"], channels: row["channels"], blocksJSON: row["blocks_json"],
            parentCheckpointID: row["parent_checkpoint_id"], parentStep: row["parent_step"],
            parentRunName: row["parent_run_name"], configJSON: row["config_json"]
        )
    }
}

nonisolated extension MetricPoint: FetchableRecord {
    init(row: Row) {
        self.init(step: row["step"], split: row["split"], name: row["name"], value: row["value"], recordedAt: row["recorded_at"])
    }
}

nonisolated extension TrainingPair: FetchableRecord {
    init(row: Row) {
        self.init(
            sourceID: row["source_id"], name: row["name"], frames: row["frames"], split: row["split"],
            rung: row["rung"], origRelPath: row["orig_rel_path"], width: row["width"], height: row["height"],
            preset: row["preset"], codec: row["codec"],
            frameRatioNum: row["frame_ratio_num"] ?? 1, frameRatioDen: row["frame_ratio_den"] ?? 1,
            lowRelPath: row["low_rel_path"]
        )
    }
}
