import Foundation
import MLX

/// One packed training example: 3 low frames, the master at t, and the codec vector.
struct TrainSampleMeta: Sendable {
    var name: String
    var frame: Int
    var rung: String
    var preset: String
    var codec: String
}

struct TrainSample {
    var low: MLXArray   // (1, h, w, 9) in [0, 1]
    var high: MLXArray  // (1, 2h, 2w, 3) in [0, 1]
    var condition: MLXArray // (1, condition_size)
    var unknown: Bool
    var meta: TrainSampleMeta
}

/// Reads orig/low pairs from the catalog database, cuts 3-frame crops, and
/// builds batches in the same [0, 1] NHWC layout as inference.
final class TrainDataset {
    let split: String
    let config: TrainConfig
    private let root: URL
    private let pairs: [TrainingPair]
    private let seed: UInt64
    private let unknownChance: Float
    private var syncCursor: TrainCursor?

    var count: Int { pairs.count }

    init(db: WhoodeoDB, split: String, config: TrainConfig, seed: UInt64) throws {
        guard let root = try db.datasetRoot() else {
            throw CLIError("kein dataset_root in der Datenbank")
        }
        self.root = root
        self.split = split
        self.config = config
        self.unknownChance = split == "train" ? config.codecUnknown : 0
        self.seed = seed &+ (split == "val" ? 1 : 0)
        let catalog = try db.trainingPairs(split: split, rungs: config.rungs, presets: config.presets)
        var kept: [TrainingPair] = []
        var skipped = 0
        for pair in catalog {
            let orig = root.appendingPathComponent(pair.origRelPath)
            let low = root.appendingPathComponent(pair.lowRelPath)
            if !FileManager.default.fileExists(atPath: orig.path) || !FileManager.default.fileExists(atPath: low.path) {
                skipped += 1
                continue
            }
            if let crop = config.cropSize, pair.width < crop || pair.height < crop {
                skipped += 1
                continue
            }
            if pair.frames < 1 {
                skipped += 1
                continue
            }
            kept.append(pair)
        }
        if kept.isEmpty {
            throw CLIError("keine \(split)-Paare für rungs \(config.rungs.joined(separator: ","))")
        }
        self.pairs = kept
        fputs("dataset \(split): \(kept.count) Paare, \(skipped) übersprungen\n", stderr)
    }

    func makeCursor(seed: UInt64? = nil) -> TrainCursor {
        TrainCursor(
            root: root, pairs: pairs, config: config,
            unknownChance: unknownChance, split: split, seed: seed ?? self.seed
        )
    }

    func batch(_ size: Int) throws -> (low: MLXArray, high: MLXArray, condition: MLXArray) {
        if syncCursor == nil { syncCursor = makeCursor() }
        return try syncCursor!.batch(size)
    }

    /// First `count` val (or train) samples from a fresh cursor, so the set is deterministic.
    func previewSamples(_ count: Int) throws -> [TrainSample] {
        try makeCursor().previewSamples(count)
    }

    func prefetcher(workers: Int, batch: Int, capacity: Int = 5) -> TrainPrefetcher {
        TrainPrefetcher(dataset: self, workers: workers, batchSize: batch, capacity: capacity, seed: seed)
    }

    static func packLow(_ planes: [[Float]], width: Int, height: Int) -> [Float] {
        let plane = width * height
        var out = [Float](repeating: 0, count: height * width * 9)
        for y in 0..<height {
            for x in 0..<width {
                let pix = y * width + x
                var channel = 0
                for frame in planes {
                    for c in 0..<3 {
                        out[(pix * 9) + channel] = frame[c * plane + pix]
                        channel += 1
                    }
                }
            }
        }
        return out
    }

    static func planarToNHWC(_ planar: [Float], width: Int, height: Int) -> [Float] {
        let plane = width * height
        var out = [Float](repeating: 0, count: plane * 3)
        for y in 0..<height {
            for x in 0..<width {
                let pix = y * width + x
                out[pix * 3 + 0] = planar[0 * plane + pix]
                out[pix * 3 + 1] = planar[1 * plane + pix]
                out[pix * 3 + 2] = planar[2 * plane + pix]
            }
        }
        return out
    }
}

/// One independent reader: own RNG and own FFmpeg handles (contexts are not thread-safe).
final class TrainCursor {
    private let root: URL
    private let pairs: [TrainingPair]
    private let config: TrainConfig
    private let unknownChance: Float
    private let split: String
    private var rng: SplitMix64
    private var handles = LRUHandleCache(limit: 8)

    init(root: URL, pairs: [TrainingPair], config: TrainConfig, unknownChance: Float, split: String, seed: UInt64) {
        self.root = root
        self.pairs = pairs
        self.config = config
        self.unknownChance = unknownChance
        self.split = split
        self.rng = SplitMix64(seed: seed)
    }

    func batch(_ size: Int) throws -> (low: MLXArray, high: MLXArray, condition: MLXArray) {
        var lows: [MLXArray] = []
        var highs: [MLXArray] = []
        var conds: [MLXArray] = []
        var attempts = 0
        while lows.count < size {
            attempts += 1
            if attempts > size * 20 {
                throw CLIError("keine gültigen \(split)-Crops gefunden")
            }
            guard let sample = try nextSample() else { continue }
            lows.append(sample.low)
            highs.append(sample.high)
            conds.append(sample.condition)
        }
        return (concatenated(lows, axis: 0), concatenated(highs, axis: 0), concatenated(conds, axis: 0))
    }

    func previewSamples(_ count: Int) throws -> [TrainSample] {
        let want = max(count, 0)
        var out: [TrainSample] = []
        var attempts = 0
        while out.count < want {
            attempts += 1
            if attempts > max(want, 1) * 20 {
                throw CLIError("keine gültigen \(split)-Crops für die Val-Vorschau")
            }
            guard let sample = try nextSample() else { continue }
            out.append(sample)
        }
        return out
    }

    private func nextSample() throws -> TrainSample? {
        let pair = pairs[Int(rng.next() % UInt64(pairs.count))]
        let origURL = root.appendingPathComponent(pair.origRelPath)
        let lowURL = root.appendingPathComponent(pair.lowRelPath)
        let orig = try handles.open(origURL, frames: pair.frames)
        let low = try handles.open(lowURL, frames: pair.frames)
        let frames = min(orig.count, low.count, pair.frames)
        if frames < 1 { return nil }
        let t = Int(rng.next() % UInt64(frames))
        let prev = max(t - 1, 0)
        let next = min(t + 1, frames - 1)
        var ox = 0, oy = 0, cropW = orig.width, cropH = orig.height
        if let crop = config.cropSize {
            guard let origin = randomOrigin(width: orig.width, height: orig.height, crop: crop, rng: &rng) else {
                return nil
            }
            ox = origin.x
            oy = origin.y
            cropW = crop
            cropH = crop
        } else {
            cropW = orig.width - orig.width % 2
            cropH = orig.height - orig.height % 2
        }
        let lx = ox / 2, ly = oy / 2, lw = cropW / 2, lh = cropH / 2
        if low.width < lx + lw || low.height < ly + lh { return nil }
        // Sequential t-1, t, t+1 on the same low handle: one seek, then decode forward.
        let lowPlanes = [
            try low.crop(frame: prev, x: lx, y: ly, width: lw, height: lh),
            try low.crop(frame: t, x: lx, y: ly, width: lw, height: lh),
            try low.crop(frame: next, x: lx, y: ly, width: lw, height: lh),
        ]
        let highPlane = try orig.crop(frame: t, x: ox, y: oy, width: cropW, height: cropH)
        let lowNHWC = TrainDataset.packLow(lowPlanes, width: lw, height: lh)
        let highNHWC = TrainDataset.planarToNHWC(highPlane, width: cropW, height: cropH)
        let detected = SourceCodec.detect(ffmpegName: pair.codec)
        let unknown = unknownChance > 0 && rng.unit() < unknownChance
        let codec: SourceCodec = unknown ? .unknown : detected
        let vector = codec.vector(size: config.conditionSize)
        return TrainSample(
            low: MLXArray(lowNHWC, [1, lh, lw, 9]),
            high: MLXArray(highNHWC, [1, cropH, cropW, 3]),
            condition: MLXArray(vector, [1, config.conditionSize]),
            unknown: unknown,
            meta: TrainSampleMeta(
                name: pair.name, frame: t, rung: pair.rung, preset: pair.preset, codec: pair.codec
            )
        )
    }

    private func randomOrigin(width: Int, height: Int, crop: Int, rng: inout SplitMix64) -> (x: Int, y: Int)? {
        if width < crop || height < crop { return nil }
        let maxX = (width - crop) / 2
        let maxY = (height - crop) / 2
        let x = Int(rng.next() % UInt64(maxX + 1)) * 2
        let y = Int(rng.next() % UInt64(maxY + 1)) * 2
        return (x, y)
    }
}

/// Bounded background batch queue. Each worker has its own FFmpeg cache and derived seed.
final class TrainPrefetcher: @unchecked Sendable {
    typealias Batch = (low: MLXArray, high: MLXArray, condition: MLXArray)

    private let lock = NSCondition()
    private var items: [Result<Batch, Error>] = []
    private let capacity: Int
    private var cancelled = false
    private var active = 0
    private var threads: [Thread] = []

    init(dataset: TrainDataset, workers: Int, batchSize: Int, capacity: Int, seed: UInt64) {
        self.capacity = max(capacity, 1)
        let count = max(workers, 1)
        active = count
        for index in 0..<count {
            let cursor = dataset.makeCursor(seed: seed &+ UInt64(index) &* 0x9E3779B97F4A7C15)
            let thread = Thread { [weak self] in
                self?.run(cursor: cursor, batchSize: batchSize)
            }
            thread.name = "whoodeo.prefetch.\(index)"
            thread.qualityOfService = .userInitiated
            threads.append(thread)
            thread.start()
        }
    }

    func next() throws -> Batch {
        lock.lock()
        defer { lock.unlock() }
        while items.isEmpty && !cancelled && active > 0 {
            lock.wait()
        }
        if let first = items.first {
            items.removeFirst()
            lock.broadcast()
            return try first.get()
        }
        if cancelled {
            throw CLIError("prefetch abgebrochen")
        }
        throw CLIError("prefetch leer")
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.broadcast()
        lock.unlock()
    }

    private func run(cursor: TrainCursor, batchSize: Int) {
        while true {
            lock.lock()
            if cancelled {
                active -= 1
                lock.broadcast()
                lock.unlock()
                return
            }
            while items.count >= capacity && !cancelled {
                lock.wait()
            }
            if cancelled {
                active -= 1
                lock.broadcast()
                lock.unlock()
                return
            }
            lock.unlock()
            let result: Result<Batch, Error>
            do {
                result = .success(try cursor.batch(batchSize))
            } catch {
                result = .failure(error)
            }
            lock.lock()
            if cancelled {
                active -= 1
                lock.broadcast()
                lock.unlock()
                return
            }
            items.append(result)
            if case .failure = result {
                cancelled = true
                active -= 1
                lock.broadcast()
                lock.unlock()
                return
            }
            lock.broadcast()
            lock.unlock()
        }
    }
}

/// Small LRU of open FFmpeg readers, so a batch does not reopen the same files.
final class LRUHandleCache {
    private let limit: Int
    private var order: [String] = []
    private var maps: [String: SourceHandle] = [:]

    init(limit: Int) { self.limit = limit }

    func open(_ url: URL, frames: Int? = nil) throws -> SourceHandle {
        let key = url.path
        if let hit = maps[key] {
            order.removeAll { $0 == key }
            order.append(key)
            return hit
        }
        let handle = try SourceHandle(url: url, expectedCount: frames)
        maps[key] = handle
        order.append(key)
        while maps.count > limit, let oldest = order.first {
            order.removeFirst()
            maps[oldest]?.close()
            maps.removeValue(forKey: oldest)
        }
        return handle
    }

    deinit {
        for handle in maps.values { handle.close() }
    }
}

/// Thin wrapper around `source_open_fast` / `source_frame` for training.
final class SourceHandle {
    let url: URL
    let width: Int
    let height: Int
    let count: Int
    private var reader: OpaquePointer?
    private var frames: [Int: [Float]] = [:]
    private var frameOrder: [Int] = []
    private let frameLimit = 6

    init(url: URL, expectedCount: Int? = nil) throws {
        self.url = url
        var message = [CChar](repeating: 0, count: 512)
        let opened = message.withUnsafeMutableBufferPointer { buffer in
            url.path.withCString { source_open_fast($0, buffer.baseAddress, Int32(buffer.count)) }
        }
        guard let opened else {
            let detail = String(cString: message)
            throw CLIError(detail.isEmpty ? "kein Bild in \(url.lastPathComponent)" : detail)
        }
        var count = Int(source_count(opened))
        if let expected = expectedCount, expected > 0 {
            _ = source_set_count(opened, Int32(expected))
            count = expected
        }
        let width = Int(source_width(opened))
        let height = Int(source_height(opened))
        guard count > 0, width > 0, height > 0 else {
            source_close(opened)
            throw CLIError("keine Bilder in \(url.lastPathComponent)")
        }
        self.reader = opened
        self.width = width
        self.height = height
        self.count = count
    }

    func crop(frame: Int, x: Int, y: Int, width cropW: Int, height cropH: Int) throws -> [Float] {
        let rgb = try decode(frame)
        let plane = width * height
        var out = [Float](repeating: 0, count: cropW * cropH * 3)
        let outPlane = cropW * cropH
        for c in 0..<3 {
            for row in 0..<cropH {
                let src = c * plane + (y + row) * width + x
                let dst = c * outPlane + row * cropW
                out.replaceSubrange(dst ..< dst + cropW, with: rgb[src ..< src + cropW])
            }
        }
        return out
    }

    private func decode(_ index: Int) throws -> [Float] {
        let clamped = min(max(index, 0), max(count - 1, 0))
        if let hit = frames[clamped] { return hit }
        var rgb = [Float](repeating: 0, count: width * height * 3)
        var outW: Int32 = 0
        var outH: Int32 = 0
        guard let reader else { throw CLIError("Quelle schon geschlossen: \(url.lastPathComponent)") }
        let status = rgb.withUnsafeMutableBufferPointer { buffer -> Int32 in
            guard let base = buffer.baseAddress else { return -1 }
            return source_frame(reader, Int32(clamped), base, Int32(buffer.count), &outW, &outH)
        }
        guard status == 0, Int(outW) == width, Int(outH) == height else {
            throw CLIError(String(cString: source_error(reader)))
        }
        frames[clamped] = rgb
        frameOrder.append(clamped)
        if frameOrder.count > frameLimit, let oldest = frameOrder.first {
            frameOrder.removeFirst()
            frames.removeValue(forKey: oldest)
        }
        return rgb
    }

    func close() {
        if let reader {
            source_close(reader)
            self.reader = nil
        }
        frames.removeAll()
    }

    deinit { close() }
}

/// Small reproducible RNG for dataset sampling (seed 42 by default).
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
