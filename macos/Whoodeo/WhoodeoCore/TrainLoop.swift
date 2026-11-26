import Darwin
import Foundation
import MLX

/// From-scratch, fine-tune, and resume training. Writes metrics and FP16 checkpoints the app can load.
enum TrainLoop {
    private static let slot = BatchSlot()

    @discardableResult
    static func run(db: WhoodeoDB, config raw: TrainConfig, resumeID: Int? = nil) throws -> TrainResult {
        let config = try raw.validated()
        Memory.cacheLimit = 48 << 30
        seed(config.seed)

        let trainData = try TrainDataset(db: db, split: "train", config: config, seed: config.seed)
        let valData = try TrainDataset(db: db, split: "val", config: config, seed: config.seed)
        let prefetch: TrainPrefetcher?
        if config.workers > 0 {
            prefetch = trainData.prefetcher(workers: config.workers, batch: config.batch)
            fputs("prefetch \(config.workers) Worker, Fast-Seek\n", stderr)
        } else {
            prefetch = nil
            fputs("prefetch aus, Fast-Seek\n", stderr)
        }
        defer { prefetch?.cancel() }

        let runID: Int
        var startStep = 0
        var model: MLXAlignModel
        let optimizer = TrainAdamW(learningRate: config.learningRate, weightDecay: 0)

        if let resumeID {
            guard let run = try db.run(id: resumeID) else {
                throw CLIError("Lauf \(resumeID) nicht gefunden")
            }
            runID = run.id
            guard let last = try db.latestCheckpoint(runID: run.id) else {
                throw CLIError("Lauf \(run.name) hat keinen Checkpoint zum Fortsetzen")
            }
            let package = WhoodeoPaths.modelsDirectory.appendingPathComponent(last.relPath)
            model = try TrainModel.load(from: package, train: config)
            if let opt = last.optimizerPath {
                let optURL = WhoodeoPaths.modelsDirectory.appendingPathComponent(opt)
                if FileManager.default.fileExists(atPath: optURL.path) {
                    try optimizer.load(from: optURL)
                    fputs("optimizer geladen, \(optimizer.momentCount) Momente, davon \(optimizer.spynetMomentCount) SPyNet\n", stderr)
                }
            }
            startStep = last.step
            try db.updateRunConfig(runID, configJSON: config.configJSON(mergingInto: run.configJSON), cropSize: config.cropSize)
            try db.updateRunStatus(runID, status: "running", finished: false)
            fputs("resume \(run.name) ab Schritt \(startStep)\n", stderr)
        } else if let parent = config.parentCheckpointID {
            guard let checkpoint = try db.checkpoint(id: parent) else {
                throw CLIError("Eltern-Checkpoint \(parent) nicht gefunden")
            }
            let package = WhoodeoPaths.modelsDirectory.appendingPathComponent(checkpoint.relPath)
            model = try TrainModel.load(from: package, train: config)
            runID = try db.createRun(
                name: config.name,
                parentCheckpointID: parent,
                channels: config.channels,
                blocksJSON: config.blocksJSON,
                dtype: config.dtype,
                cropSize: config.cropSize,
                configJSON: config.configJSON,
                status: "running",
                notes: "Fine-Tune von Checkpoint \(parent) (Schritt \(checkpoint.step))"
            )
            fputs("fine-tune von Checkpoint \(parent) → \(config.name)\n", stderr)
        } else {
            model = try TrainModel.fromScratch(config)
            runID = try db.createRun(
                name: config.name,
                parentCheckpointID: nil,
                channels: config.channels,
                blocksJSON: config.blocksJSON,
                dtype: config.dtype,
                cropSize: config.cropSize,
                configJSON: config.configJSON,
                status: "running",
                notes: "from scratch"
            )
            fputs("from scratch \(config.name)  \(config.channels) Kanäle  crop \(config.cropSize.map(String.init) ?? "voll")\n", stderr)
        }

        try db.setRunPid(getpid(), runID: runID)
        model.trainSpynet = config.trainSpynet
        try db.clearStopRequest(runID: runID)
        stepDecodes = 0
        fputs("Loss aktiv: \(config.activeLossText)\n", stderr)
        fputs(config.pixelLossActive ? "VAE-Decode im Trainingsschritt: an\n" : "VAE-Decode im Trainingsschritt: übersprungen (nur Latent)\n", stderr)
        let lossFn = try TrainLoss(config: config)
        let vae = try MLXVAEEncoder()
        installStopHandler()

        let keys = TrainModel.trainableKeys(model.p, trainSpynet: config.trainSpynet)
        var params = keys.map { model.p[$0]! }
        let frozen = model.p.filter { !keys.contains($0.key) }
        if config.trainSpynet {
            for key in keys where key.hasPrefix(TrainModel.spynetPrefix) {
                optimizer.lrScale[key] = config.spynetLrScale
            }
        }
        let spyBefore = model.p.filter { $0.key.hasPrefix(TrainModel.spynetPrefix) }
        eval(Array(spyBefore.values))
        let spyTrainable = keys.filter { $0.hasPrefix(TrainModel.spynetPrefix) }.count
        fputs("\(keys.count) trainierbare Tensoren, davon \(spyTrainable) SPyNet\n", stderr)
        if config.trainSpynet {
            fputs(
                String(format: "SPyNet wird mittrainiert, lr %.2e (%.2g×)\n", config.spynetLearningRate, config.spynetLrScale),
                stderr
            )
        } else {
            fputs("SPyNet eingefroren\n", stderr)
        }
        let vg = valueAndGrad({ arrays -> [MLXArray] in
            var merged = frozen
            for (key, array) in zip(keys, arrays) { merged[key] = array }
            model.replaceWeights(merged)
            return [computeLoss(model: model, vae: vae, lossFn: lossFn, slot: slot)]
        }, argumentNumbers: Array(keys.indices))

        var lastTrain: [String: Float] = [:]
        var lastVal: [String: Float] = [:]
        var ran = 0
        let endStep = startStep + config.steps
        var status = "finished"

        fputs(
            "train \(config.steps) Schritte  batch \(config.batch) × accum \(config.gradAccum)  lr \(config.learningRate)  dtype \(config.dtype)\n",
            stderr
        )

        step: for step in (startStep + 1)...endStep {
            let t0 = Date()
            var accumGrads: [MLXArray]?
            var accumScalars: [String: Float] = [:]
            for _ in 0..<config.gradAccum {
                let data = try prefetch?.next() ?? trainData.batch(config.batch)
                slot.low = data.low
                slot.high = data.high
                slot.condition = data.condition
                slot.accum = config.gradAccum
                let (values, grads) = vg(params)
                eval(values)
                let total = values[0].item(Float.self)
                if !total.isFinite {
                    try db.updateRunStatus(runID, status: "failed", finished: true)
                    throw CLIError("NaN oder Inf im Loss bei Schritt \(step)")
                }
                for (name, value) in slot.scalars {
                    accumScalars[name, default: 0] += value / Float(config.gradAccum)
                }
                if accumGrads == nil {
                    accumGrads = grads
                } else {
                    for index in grads.indices {
                        accumGrads![index] = accumGrads![index] + grads[index]
                    }
                }
            }
            var grads = accumGrads!
            eval(grads)
            if config.schedule == "cosine" {
                let t = Float(step - startStep) / Float(max(config.steps, 1))
                optimizer.learningRate = config.learningRate * 0.5 * (1 + cos(Float.pi * min(t, 1)))
            }
            optimizer.update(keys: keys, params: &params, grads: grads)
            var merged = frozen
            for (key, array) in zip(keys, params) { merged[key] = array }
            model.replaceWeights(merged)
            lastTrain = accumScalars
            let elapsed = Date().timeIntervalSince(t0)
            lastTrain["seconds"] = Float(elapsed)
            lastTrain["lr"] = optimizer.learningRate
            if config.trainSpynet {
                lastTrain["spynet_lr"] = optimizer.learningRate * config.spynetLrScale
            }
            try db.insertMetrics(runID: runID, step: step, split: "train", values: lastTrain.mapValues { Double($0) })

            if step % max(config.logEvery, 1) == 0 {
                // Only the active terms; a term with weight 0 is not computed and not logged.
                var line = String(format: "step %d  loss %.4f", step, lastTrain["total"] ?? 0)
                for (key, label) in [("latent_l1", "latent"), ("oklab", "oklab"), ("ms_ssim", "ms-ssim"), ("vgg", "vgg")] {
                    if let value = lastTrain[key] { line += String(format: "  %@ %.4f", label, value) }
                }
                line += String(format: "  %.2fs\n", elapsed)
                fputs(line, stderr)
            }

            let doVal = config.valEvery > 0 && step % config.valEvery == 0
            let doSave = config.saveEvery > 0 && step % config.saveEvery == 0
            let requested = try TrainSignals.takeCheckpoint() || db.consumeCheckpointRequest(runID: runID)
            if !TrainSignals.stopRequested, try db.hasStopRequest(runID: runID) {
                TrainSignals.requestStopFromDB()
            }
            if doVal {
                lastVal = try validate(
                    model: model, vae: vae, lossFn: lossFn, data: valData,
                    count: config.valCount, batch: config.batch
                )
                try db.insertMetrics(runID: runID, step: step, split: "val", values: lastVal.mapValues { Double($0) })
                try? ValPreview.write(
                    runName: config.name, step: step, model: model, vae: vae, data: valData, lossFn: lossFn
                )
                fputs(
                    String(
                        format: "  val  loss %.4f  psnr %.2f  ssim %.4f\n",
                        lastVal["total"] ?? 0, lastVal["psnr"] ?? 0, lastVal["ssim"] ?? 0
                    ),
                    stderr
                )
            }
            if doSave || TrainSignals.stopRequested || requested {
                if !doVal && (doSave || TrainSignals.stopRequested) {
                    lastVal = (try? validate(
                        model: model, vae: vae, lossFn: lossFn, data: valData,
                        count: min(config.valCount, 4), batch: config.batch
                    )) ?? lastVal
                    try? ValPreview.write(
                        runName: config.name, step: step, model: model, vae: vae, data: valData, lossFn: lossFn
                    )
                }
                try saveCheckpoint(
                    db: db, runID: runID, step: step, model: model, optimizer: optimizer,
                    config: config, val: lastVal,
                    pause: TrainSignals.stopRequested && step < endStep && !requested,
                    visible: requested ? true : nil
                )
            }
            ran = step
            if TrainSignals.stopRequested {
                status = "paused"
                try db.updateRunStatus(runID, status: "paused", finished: false)
                try db.clearStopRequest(runID: runID)
                fputs("pausiert nach Schritt \(step)\n", stderr)
                break step
            }
        }

        if status == "finished" {
            if ran > startStep, config.saveEvery == 0 || ran % max(config.saveEvery, 1) != 0 {
                try saveCheckpoint(
                    db: db, runID: runID, step: ran, model: model, optimizer: optimizer,
                    config: config, val: lastVal, pause: false
                )
            }
            try db.updateRunStatus(runID, status: "finished", finished: true)
        }
        fputs("VAE-Decodes in Trainingsschritten: \(stepDecodes)\n", stderr)
        let spyDelta = TrainModel.spynetL2Delta(before: spyBefore, after: model.p)
        fputs(String(format: "SPyNet L2-Delta %.6g  %@\n", spyDelta, config.trainSpynet ? "trainiert" : "eingefroren"), stderr)
        if spyDelta.isNaN {
            try db.updateRunStatus(runID, status: "failed", finished: true)
            throw CLIError("NaN in den SPyNet-Gewichten")
        }

        return TrainResult(runID: runID, lastStep: ran, train: lastTrain, val: lastVal, status: status)
    }

    /// VAE decodes inside training steps (not validation). Stays 0 in latent-only training.
    nonisolated(unsafe) private static var stepDecodes = 0

    private static func computeLoss(model: MLXAlignModel, vae: MLXVAEEncoder, lossFn: TrainLoss, slot: BatchSlot) -> MLXArray {
        guard let low = slot.low, let high = slot.high, let condition = slot.condition else {
            return MLXArray(Float(0))
        }
        let config = lossFn.config
        let height = low.dim(1), width = low.dim(2)
        let padded = padTo(low, 4)
        let z = model.code(padded.asType(model.dtype), padded, condition.asType(model.dtype))
        // The target latent is only needed for the latent term, the decode only for pixel terms.
        let target = config.latentWeight > 0 ? stopGradient(vae.normalizedCode(high)).asType(.float32) : nil
        var predRGB: MLXArray?
        if config.pixelLossActive {
            let shift = vae.shift.reshaped([1, 1, 1, TrainModel.latent]).asType(z.dtype)
            let scale = vae.scale.reshaped([1, 1, 1, TrainModel.latent]).asType(z.dtype)
            let rgb = model.decode(z / scale + shift)
            predRGB = rgb[0..., 0 ..< 2 * height, 0 ..< 2 * width].asType(.float32)
            stepDecodes += 1
        }
        let (total, values) = lossFn.terms(
            predZ: z.asType(.float32), targetZ: target,
            predRGB: predRGB, targetRGB: high
        )
        slot.scalars = lossFn.scalars(values)
        return total / Float(max(slot.accum, 1))
    }

    private static func validate(
        model: MLXAlignModel, vae: MLXVAEEncoder, lossFn: TrainLoss, data: TrainDataset,
        count: Int, batch: Int
    ) throws -> [String: Float] {
        var sums: [String: Float] = [:]
        var seen = 0
        var psnrSum: Float = 0
        var ssimSum: Float = 0
        while seen < count {
            let need = min(batch, count - seen)
            let batchData = try data.batch(need)
            let height = batchData.low.dim(1), width = batchData.low.dim(2)
            let padded = padTo(batchData.low, 4)
            let z = model.code(padded.asType(model.dtype), padded, batchData.condition.asType(model.dtype))
            let target = vae.normalizedCode(batchData.high)
            let shift = vae.shift.reshaped([1, 1, 1, TrainModel.latent]).asType(z.dtype)
            let scale = vae.scale.reshaped([1, 1, 1, TrainModel.latent]).asType(z.dtype)
            let rgb = model.decode(z / scale + shift)
            let predRGB = rgb[0..., 0 ..< 2 * height, 0 ..< 2 * width].asType(.float32)
            let (_, values) = lossFn.terms(
                predZ: z.asType(.float32), targetZ: target.asType(.float32),
                predRGB: predRGB, targetRGB: batchData.high, metrics: true
            )
            let scalars = lossFn.scalars(values)
            for (name, value) in scalars { sums[name, default: 0] += value * Float(need) }
            psnrSum += TrainLoss.psnr(predRGB, batchData.high) * Float(need)
            if let ms = scalars["ms_ssim"] { ssimSum += (1 - ms) * Float(need) }
            seen += need
        }
        var mean = Dictionary(uniqueKeysWithValues: sums.map { ($0.key, $0.value / Float(seen)) })
        mean["psnr"] = psnrSum / Float(seen)
        mean["ssim"] = ssimSum / Float(seen)
        return mean
    }

    private static func saveCheckpoint(
        db: WhoodeoDB, runID: Int, step: Int, model: MLXAlignModel, optimizer: TrainAdamW,
        config: TrainConfig, val: [String: Float], pause: Bool, visible: Bool? = nil
    ) throws {
        let rel = String(format: "%@/step-%06d.mlxmodel", config.name, step)
        let package = WhoodeoPaths.modelsDirectory.appendingPathComponent(rel)
        try TrainModel.exportFP16(model: model, to: package, step: step, train: config)
        let optRel = String(format: "%@/step-%06d.optimizer.safetensors", config.name, step)
        let optURL = WhoodeoPaths.modelsDirectory.appendingPathComponent(optRel)
        try optimizer.save(to: optURL)
        _ = try db.insertCheckpoint(
            runID: runID,
            step: step,
            relPath: rel,
            exportDtype: "fp16",
            optimizerPath: optRel,
            valLoss: val["total"].map(Double.init),
            valPSNR: val["psnr"].map(Double.init),
            valSSIM: val["ssim"].map(Double.init),
            visibleInApp: visible ?? (pause ? false : !config.hidden)
        )
        let hidden = !(visible ?? (pause ? false : !config.hidden))
        var mark = ""
        if visible == true { mark = " (auf Anfrage)" }
        else if hidden { mark = " (versteckt)" }
        fputs("checkpoint \(rel)\(mark)\n", stderr)
    }

    private static func installStopHandler() {
        TrainSignals.reset()
        signal(SIGINT, TrainSignals.handleSigint)
        signal(SIGTERM, TrainSignals.handleSigterm)
        signal(SIGUSR1, TrainSignals.handleSigusr1)
    }
}

struct TrainResult {
    var runID: Int
    var lastStep: Int
    var train: [String: Float]
    var val: [String: Float]
    var status: String
}

/// Current micro-batch for the valueAndGrad closure.
private final class BatchSlot: @unchecked Sendable {
    var low: MLXArray?
    var high: MLXArray?
    var condition: MLXArray?
    var accum: Int = 1
    var scalars: [String: Float] = [:]
}

private enum TrainSignals {
    nonisolated(unsafe) static var stop: Int32 = 0
    nonisolated(unsafe) static var checkpoint: Int32 = 0
    nonisolated(unsafe) static var sigint: Int32 = 0

    static var stopRequested: Bool { stop != 0 }

    /// Same first-SIGINT path, used when the app or `whoodeo-train stop` sets `stop_request.<run_id>`.
    static func requestStopFromDB() {
        if stop != 0 { return }
        stop = 1
        let msg: StaticString = "Stop angefordert, beende nach diesem Schritt …\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
    }

    static func takeCheckpoint() -> Bool {
        if checkpoint != 0 {
            checkpoint = 0
            return true
        }
        return false
    }

    static func reset() {
        stop = 0
        checkpoint = 0
        sigint = 0
    }

    static let handleSigint: @convention(c) (Int32) -> Void = { _ in
        if TrainSignals.sigint != 0 {
            _exit(130)
        }
        TrainSignals.sigint = 1
        TrainSignals.stop = 1
        let msg: StaticString = "Stop angefordert, beende nach diesem Schritt … (nochmal Ctrl+C bricht sofort ab)\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
    }

    static let handleSigterm: @convention(c) (Int32) -> Void = { _ in
        TrainSignals.stop = 1
        let msg: StaticString = "Stop angefordert, beende nach diesem Schritt …\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
    }

    static let handleSigusr1: @convention(c) (Int32) -> Void = { _ in
        TrainSignals.checkpoint = 1
        let msg: StaticString = "Checkpoint angefordert\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
    }
}
