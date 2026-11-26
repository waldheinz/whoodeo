import Foundation
import os

/// What happens to the decoded film before the upscaler. Stored per film; every stage is off by default.
/// A new stage is one more field here plus its part of the frame map or the picture, and a cache tag.
/// Engine, export and model only ever see the film the stages make.
nonisolated struct PreprocessConfig: Codable, Hashable, Sendable {
    var decimate: DecimateMode = .off

    static let none = PreprocessConfig()

    init(decimate: DecimateMode = .off) {
        self.decimate = decimate
    }

    /// Missing keys keep their default, so older lists and later stages read alike.
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        decimate = (try? values.decodeIfPresent(DecimateMode.self, forKey: .decimate)) ?? .off
    }
}

/// Repeated frames a transcode put in to reach its rate: one of every `cycle` frames is a copy.
nonisolated enum DecimateMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case off, auto, cycle6, cycle5

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: "Aus"
        case .auto: "Automatisch"
        case .cycle6: "6er-Takt (30 → 25 fps)"
        case .cycle5: "5er-Takt (30 → 24 fps)"
        }
    }

    var cycle: Int? {
        switch self {
        case .cycle6: 6
        case .cycle5: 5
        case .off, .auto: nil
        }
    }

    static func parse(_ text: String) -> DecimateMode? {
        switch text.lowercased() {
        case "off", "aus", "none": .off
        case "auto", "automatisch": .auto
        case "6", "cycle6", "6er": .cycle6
        case "5", "cycle5", "5er": .cycle5
        default: nil
        }
    }
}

/// Which source frames make the film the app sees. Index `i` is source frame `raw[i]`.
nonisolated struct FrameMap: Equatable, Sendable {
    /// Frames of the file the map was made for.
    let rawCount: Int
    /// Source frames per cycle; one of them is dropped.
    let cycle: Int
    let raw: [Int]

    var count: Int { raw.count }

    /// Suffix of the frame cache, so a decimated film never shares an archive with the plain one.
    var cacheTag: String { "dec\(cycle)" }

    func fps(source: Double) -> Double {
        source * Double(cycle - 1) / Double(cycle)
    }

    /// The kept frame at or after a source frame.
    func index(forSource source: Int) -> Int {
        var low = 0
        var high = raw.count
        while low < high {
            let mid = (low + high) / 2
            if raw[mid] < source { low = mid + 1 } else { high = mid }
        }
        return min(low, max(raw.count - 1, 0))
    }
}

/// Finds the repeated frames. A scan measures how much each frame differs from the one before;
/// detection and the frame map then work on those numbers alone, so they are cached per film.
nonisolated enum Decimate {
    /// Below this mean luma difference (0…255) two frames count as the same picture with codec noise.
    /// Motion has to reach it before a cycle decides where its repeat is.
    static let noiseFloor: Float = 1.0
    /// The repeat must differ at most half as much as the next-closest pair of its cycle.
    static let clearRatio: Float = 0.5
    private static let fileName = "decimate.bin"
    private static let magic: UInt32 = 0x3143_4457 // "WDC1"

    /// Mean absolute luma difference of every source frame to the one before, on a coarse grid.
    /// The first frame has none and gets `.infinity`. Splits the film into stretches decoded in parallel.
    static func scan(
        url: URL,
        progress: @escaping @Sendable (Double) -> Void,
        cancelled: @escaping @Sendable () -> Bool
    ) throws -> [Float] {
        let (count, step) = try {
            let probe = try Film.open(url)
            return (probe.source.count, max(1, probe.source.width / 160))
        }()
        let parts = max(1, min(8, ProcessInfo.processInfo.activeProcessorCount / 2, count / 1000))
        let result = Diffs(count: count)
        let done = OSAllocatedUnfairLock(initialState: 0)
        let failure = OSAllocatedUnfairLock<Error?>(initialState: nil)
        DispatchQueue.concurrentPerform(iterations: parts) { part in
            let lower = count * part / parts
            let upper = count * (part + 1) / parts
            do {
                let film = try Film.open(url)
                var previous = [UInt8]()
                var current = [UInt8]()
                var pixels = 0
                if lower > 0 {
                    let size = try film.luma(source: lower - 1, step: step, into: &previous)
                    pixels = size.width * size.height
                }
                for index in lower..<upper {
                    if index % 64 == 0 {
                        if cancelled() { throw CancellationError() }
                        if failure.withLock({ $0 != nil }) { return }
                    }
                    let size = try film.luma(source: index, step: step, into: &current)
                    pixels = size.width * size.height
                    if index == 0 {
                        result.set(0, .infinity)
                    } else {
                        result.set(index, meanDifference(previous, current, pixels))
                    }
                    swap(&previous, &current)
                    if index % 500 == 499 {
                        let total = done.withLock { value -> Int in
                            value += 500
                            return value
                        }
                        progress(Double(total) / Double(count))
                    }
                }
            } catch {
                failure.withLock { if $0 == nil { $0 = error } }
            }
        }
        if let error = failure.withLock({ $0 }) { throw error }
        progress(1)
        return result.values
    }

    private static func meanDifference(_ a: [UInt8], _ b: [UInt8], _ pixels: Int) -> Float {
        guard pixels > 0 else { return 0 }
        var sum = 0
        a.withUnsafeBufferPointer { left in
            b.withUnsafeBufferPointer { right in
                for i in 0..<pixels {
                    sum += abs(Int(left[i]) - Int(right[i]))
                }
            }
        }
        return Float(sum) / Float(pixels)
    }

    /// The cycle of the repeats, or nil when the film has no steady one. Each cycle-long block whose
    /// repeat is clear votes; a real cadence keeps its position from block to block, a wrong
    /// cycle length makes it drift. Cuts that shift the position cost a single vote.
    static func detect(_ diffs: [Float]) -> Int? {
        var best: (cycle: Int, score: Double)?
        for cycle in [6, 5] {
            var clear = 0
            var steady = 0
            var last: Int?
            for block in 0..<(diffs.count / cycle) {
                guard let phase = clearPhase(diffs, block: block, cycle: cycle) else { continue }
                clear += 1
                if phase == last { steady += 1 }
                last = phase
            }
            guard clear >= 20 else { continue }
            let score = Double(steady) / Double(clear - 1)
            if score >= 0.7, score > (best?.score ?? 0) { best = (cycle, score) }
        }
        return best?.cycle
    }

    /// Drops one frame per cycle. A block with clear motion picks its own repeat and sets the phase;
    /// a still or unclear block keeps the phase of the last clear one.
    static func map(_ diffs: [Float], cycle: Int) -> FrameMap {
        let count = diffs.count
        let blocks = count / cycle
        var votes = [Int](repeating: 0, count: cycle)
        for block in 0..<blocks {
            if let phase = clearPhase(diffs, block: block, cycle: cycle) { votes[phase] += 1 }
        }
        var phase = votes.indices.max { votes[$0] < votes[$1] } ?? 0
        if votes[phase] == 0 { phase = cycle - 1 }
        var raw: [Int] = []
        raw.reserveCapacity(count - blocks)
        for block in 0...blocks {
            let base = block * cycle
            let length = min(cycle, count - base)
            guard length > 0 else { break }
            if length == cycle, let clear = clearPhase(diffs, block: block, cycle: cycle) {
                phase = clear
            }
            for offset in 0..<length where offset != phase {
                raw.append(base + offset)
            }
        }
        return FrameMap(rawCount: count, cycle: cycle, raw: raw)
    }

    /// The position of the repeat in a block, when it stands out from the motion around it.
    private static func clearPhase(_ diffs: [Float], block: Int, cycle: Int) -> Int? {
        let base = block * cycle
        var first = Float.infinity
        var second = Float.infinity
        var phase = 0
        for offset in 0..<cycle {
            let value = diffs[base + offset]
            if value < first {
                second = first
                first = value
                phase = offset
            } else if value < second {
                second = value
            }
        }
        guard second.isFinite, second >= noiseFloor, first <= clearRatio * second else { return nil }
        return phase
    }

    /// The map for a mode, and a short note on what automatic found. Nil when nothing is dropped.
    static func resolve(_ mode: DecimateMode, diffs: [Float]) -> (map: FrameMap?, note: String) {
        switch mode {
        case .off:
            return (nil, "")
        case .cycle6, .cycle5:
            return (map(diffs, cycle: mode.cycle!), "")
        case .auto:
            guard let cycle = detect(diffs) else { return (nil, "kein Takt erkannt") }
            return (map(diffs, cycle: cycle), "\(cycle)er-Takt erkannt")
        }
    }

    // MARK: Cache

    private static func file(filmID: String) -> URL {
        FrameStore.cacheRoot
            .appendingPathComponent(filmID, isDirectory: true)
            .appendingPathComponent(fileName)
    }

    /// The scan of a film, if one was made. `count` rejects a scan of another version of the file.
    static func cached(filmID: String, count: Int? = nil) -> [Float]? {
        guard let data = try? Data(contentsOf: file(filmID: filmID)), data.count >= 8 else { return nil }
        let header = data.withUnsafeBytes { ($0.loadUnaligned(fromByteOffset: 0, as: UInt32.self),
                                             $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) }
        let stored = Int(UInt32(littleEndian: header.1))
        guard UInt32(littleEndian: header.0) == magic, data.count == 8 + stored * 4 else { return nil }
        if let count, count != stored { return nil }
        var values = [Float](repeating: 0, count: stored)
        values.withUnsafeMutableBytes { out in
            data.withUnsafeBytes { raw in
                out.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[8...]))
            }
        }
        return values
    }

    static func store(_ diffs: [Float], filmID: String) throws {
        let url = file(filmID: filmID)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = Data(capacity: 8 + diffs.count * 4)
        withUnsafeBytes(of: magic.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(diffs.count).littleEndian) { data.append(contentsOf: $0) }
        diffs.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url, options: .atomic)
    }

    /// Frame differences written from several threads, each to its own stretch.
    nonisolated private final class Diffs: @unchecked Sendable {
        private var storage: UnsafeMutableBufferPointer<Float>

        init(count: Int) {
            storage = .allocate(capacity: count)
            storage.initialize(repeating: 0)
        }

        deinit { storage.deallocate() }

        func set(_ index: Int, _ value: Float) { storage[index] = value }

        var values: [Float] { Array(storage) }
    }
}
