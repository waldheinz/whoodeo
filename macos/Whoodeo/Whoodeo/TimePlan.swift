import CryptoKit
import Foundation
import os

// MARK: Choice

/// How the app reads the time structure of a film before the upscaler.
/// Automatic and manual both follow a time plan; automatic only takes it when the pattern is clear.
nonisolated enum TimingMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case off, auto, duplicates, manual

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: "Aus"
        case .auto: "Automatisch"
        case .duplicates: "Nur doppelte Bilder"
        case .manual: "Manuell"
        }
    }

    static func parse(_ text: String) -> TimingMode? {
        switch text.lowercased() {
        case "off", "aus", "none": .off
        case "auto", "automatisch": .auto
        case "duplicates", "doppelt", "dup", "decimate": .duplicates
        case "manual", "manuell": .manual
        default: nil
        }
    }
}

/// A correction of the plan for a stretch of source frames `from ..< to`.
nonisolated struct TimeOverride: Codable, Hashable, Sendable {
    var from: Int
    var to: Int
    /// Show the originals there, nothing is reconstructed.
    var original: Bool = false
    /// Mixing groups there start at this phase (source frame modulo 6).
    var phase: Int?

    func covers(_ frame: Int) -> Bool { frame >= from && frame < to }
}

// MARK: Features

/// What the analysis pass measures per source frame, on luma of about 320 pixels width (0…255).
nonisolated struct TimeFeatures: Sendable {
    /// Frames of the file as the reader counts them.
    var count: Int
    /// Frames the header claims, 0 when it names none.
    var declared: Int
    /// Frame rate of the file.
    var fps: Double = 0
    /// The reader stopped its count at the last readable frame.
    var clipped: Bool
    /// First frame that did not decode, nil when all did.
    var failedAt: Int?
    /// Mean |Y(t) − Y(t−1)|, infinite for the first frame.
    var d: [Float]
    /// Mean |O2 − 2·O3 + 2·O4 − 2·O5 + O6| with O2 = frame t; infinite near the end.
    /// Zero where the frames t+1, t+2, t+3 are 50/50 mixes of their neighbors' sources (type A).
    var r: [Float]
    /// Mean squared gradient: which of a few frames is the sharpest.
    var sharp: [Float]

    /// Frames up to the first one that did not decode.
    var usable: Int { min(count, failedAt ?? count) }
}

nonisolated enum TimeAnalysis {
    static let fileName = "timefeat.bin"
    private static let magic: UInt32 = 0x3146_5457 // "WTF1"
    static let version: UInt32 = 1
    private static let header = 32

    /// Decodes every frame once, in parallel stretches like the decimate scan.
    static func scan(
        url: URL,
        progress: @escaping @Sendable (Double) -> Void,
        cancelled: @escaping @Sendable () -> Bool
    ) throws -> TimeFeatures {
        let probe = try Film.open(url)
        let count = probe.source.count
        let width = probe.source.width
        let height = probe.source.height
        let declared = probe.source.declaredCount
        let clipped = probe.source.clipped
        let step = max(1, width / 320)
        let gw = width / step
        let gh = height / step
        let parts = max(1, min(8, ProcessInfo.processInfo.activeProcessorCount / 2, count / 1000))
        let out = Columns(count: count)
        let done = OSAllocatedUnfairLock(initialState: 0)
        let failure = OSAllocatedUnfairLock<Error?>(initialState: nil)
        let broken = OSAllocatedUnfairLock<Int?>(initialState: nil)
        DispatchQueue.concurrentPerform(iterations: parts) { part in
            let lower = count * part / parts
            let upper = count * (part + 1) / parts
            do {
                let film = try Film.open(url)
                var full = [UInt8]()
                var ring = [[Float]](repeating: [Float](repeating: 0, count: gw * gh), count: 5)
                let first = max(0, lower - 1)
                let last = min(count, upper + 4)
                var counted = 0
                for frame in first..<last {
                    if frame % 64 == 0 {
                        if cancelled() { throw CancellationError() }
                        if failure.withLock({ $0 != nil }) { return }
                    }
                    if let bad = broken.withLock({ $0 }), frame >= bad { break }
                    do {
                        _ = try film.luma(source: frame, step: 1, into: &full)
                    } catch {
                        broken.withLock { $0 = min($0 ?? frame, frame) }
                        break
                    }
                    let slot = frame % 5
                    shrink(full, width: width, step: step, into: &ring[slot], gw: gw, gh: gh)
                    if frame >= lower && frame < upper {
                        out.sharp[frame] = sharpness(ring[slot], gw, gh)
                        out.d[frame] = frame > first ? meanDifference(ring[slot], ring[(frame - 1) % 5]) : .infinity
                        counted += 1
                    }
                    let start = frame - 4
                    if start >= lower && start < upper {
                        out.r[start] = mixResidual(ring, start: start)
                    }
                    if counted == 500 {
                        counted = 0
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
        let failedAt = broken.withLock { $0 }
        var features = TimeFeatures(
            count: count, declared: declared, fps: probe.source.fps, clipped: clipped, failedAt: failedAt,
            d: out.values(out.d), r: out.values(out.r), sharp: out.values(out.sharp)
        )
        if let failedAt {
            for frame in max(0, failedAt - 4)..<count { features.r[frame] = .infinity }
        }
        return features
    }

    /// Box mean over `step × step` pixels.
    private static func shrink(_ full: [UInt8], width: Int, step: Int, into out: inout [Float], gw: Int, gh: Int) {
        let scale = 1 / Float(step * step)
        full.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<gh {
                    for x in 0..<gw {
                        var sum = 0
                        for dy in 0..<step {
                            let row = (y * step + dy) * width + x * step
                            for dx in 0..<step { sum += Int(src[row + dx]) }
                        }
                        dst[y * gw + x] = Float(sum) * scale
                    }
                }
            }
        }
    }

    private static func meanDifference(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in a.indices { sum += abs(a[i] - b[i]) }
        return sum / Float(max(a.count, 1))
    }

    private static func mixResidual(_ ring: [[Float]], start: Int) -> Float {
        let o2 = ring[start % 5], o3 = ring[(start + 1) % 5], o4 = ring[(start + 2) % 5]
        let o5 = ring[(start + 3) % 5], o6 = ring[(start + 4) % 5]
        var sum: Float = 0
        for i in o2.indices {
            sum += abs(o2[i] - 2 * o3[i] + 2 * o4[i] - 2 * o5[i] + o6[i])
        }
        return sum / Float(max(o2.count, 1))
    }

    private static func sharpness(_ a: [Float], _ w: Int, _ h: Int) -> Float {
        guard w > 1, h > 1 else { return 0 }
        var sum: Float = 0
        for y in 0..<(h - 1) {
            for x in 0..<(w - 1) {
                let i = y * w + x
                let gx = a[i + 1] - a[i]
                let gy = a[i + w] - a[i]
                sum += gx * gx + gy * gy
            }
        }
        return sum / Float((w - 1) * (h - 1))
    }

    // MARK: Cache

    private static func file(filmID: String) -> URL {
        FrameStore.cacheRoot.appendingPathComponent(filmID, isDirectory: true).appendingPathComponent(fileName)
    }

    static func cached(filmID: String, count: Int? = nil) -> TimeFeatures? {
        guard let data = try? Data(contentsOf: file(filmID: filmID)), data.count >= header else { return nil }
        let word = { (at: Int) in data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: at, as: UInt32.self)) } }
        guard word(0) == magic, word(4) == version else { return nil }
        let stored = Int(word(8))
        guard data.count == header + stored * 12 else { return nil }
        if let count, count != stored { return nil }
        let failed = Int(Int32(bitPattern: word(16)))
        let column = { (index: Int) -> [Float] in
            var values = [Float](repeating: 0, count: stored)
            values.withUnsafeMutableBytes { out in
                data.withUnsafeBytes { raw in
                    let start = header + index * stored * 4
                    out.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[start..<(start + stored * 4)]))
                }
            }
            return values
        }
        let fps = data.withUnsafeBytes { Double(bitPattern: UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 24, as: UInt64.self))) }
        return TimeFeatures(
            count: stored, declared: Int(word(12)), fps: fps, clipped: word(20) != 0,
            failedAt: failed >= 0 ? failed : nil,
            d: column(0), r: column(1), sharp: column(2)
        )
    }

    static func store(_ features: TimeFeatures, filmID: String) throws {
        _ = try CacheSettings.availableRoot()
        let url = file(filmID: filmID)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = Data(capacity: header + features.count * 12)
        for word in [magic, version, UInt32(features.count), UInt32(features.declared),
                     UInt32(bitPattern: Int32(features.failedAt ?? -1)), features.clipped ? 1 : 0] {
            withUnsafeBytes(of: word.littleEndian) { data.append(contentsOf: $0) }
        }
        withUnsafeBytes(of: features.fps.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        for column in [features.d, features.r, features.sharp] {
            column.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        try data.write(to: url, options: .atomic)
    }

    /// Three feature columns written from several threads, each to its own stretch.
    nonisolated private final class Columns: @unchecked Sendable {
        let d: UnsafeMutableBufferPointer<Float>
        let r: UnsafeMutableBufferPointer<Float>
        let sharp: UnsafeMutableBufferPointer<Float>

        init(count: Int) {
            d = .allocate(capacity: count)
            r = .allocate(capacity: count)
            sharp = .allocate(capacity: count)
            d.initialize(repeating: .infinity)
            r.initialize(repeating: .infinity)
            sharp.initialize(repeating: 0)
        }

        deinit {
            d.deallocate()
            r.deallocate()
            sharp.deallocate()
        }

        func values(_ column: UnsafeMutableBufferPointer<Float>) -> [Float] { Array(column) }
    }
}

// MARK: Plan

/// A stretch of the film with one way of reading it.
nonisolated struct TimeSegment: Codable, Equatable, Hashable, Sendable, Identifiable {
    enum Kind: String, Codable, Sendable {
        /// Type A mixes, reconstructed.
        case blend = "mischA"
        /// No pattern the solver knows, or switched off: the sharpest originals.
        case original
    }

    /// Source frames `from ..< to`.
    var from: Int
    var to: Int
    var kind: Kind
    /// Start of the groups modulo 6, nil for originals.
    var phase: Int?
    /// 0…1 from how clearly the right phase stands out from the others.
    var confidence: Double
    /// Groups that copy because their mix did not fit.
    var fallbacks: Int
    /// An override set this stretch.
    var forced: Bool

    var id: Int { from }
}

/// What the planner made of a film: which groups are reconstructed, which stretches stay original.
nonisolated struct TimePlan: Codable, Equatable, Sendable {
    static let currentVersion = 1

    var version: Int
    /// Frames of the file and how many of them decode.
    var count: Int
    var usable: Int
    var declared: Int
    var sourceFps: Double
    /// Starts (O0) of the type A groups, and the ones among them that copy instead.
    var groups: [Int]
    var fallback: [Int]
    var segments: [TimeSegment]
    /// Share of the readable frames inside reconstructed groups.
    var coverage: Double
    /// Mean confidence of the mixing stretches, weighted by length.
    var confidence: Double
    /// Phase changes between neighboring mixing stretches.
    var jumps: Int
    /// The pattern is clear enough for automatic (or manual forced it).
    var accepted: Bool
    var note: String
    var hash: String
    /// What the solver measured on this film. Part of the hash, so a refit never reuses older pictures.
    var solver: SolverParams = .standard

    /// The film is cut short: fewer frames decode than the header claims.
    var truncated: Bool { declared > usable || usable < count }
}

nonisolated enum TimePlanner {
    struct Parameters {
        /// Cost of a frame outside a group, in units of the local noise level L.
        var gapCost: Double = 2.0
        /// A group whose residual exceeds this many L copies instead of reconstructing.
        var fallbackQ: Double = 3.0
        /// Type A when the residual is at most this share of the motion across the group
        /// (measured: type A 0.14…0.25, a ramp mix 0.30…0.39, as medians over ±25 groups).
        var maxRatio: Double = 0.27
        /// Automatic takes the plan when this share of the film is reconstructed.
        var minCoverage: Double = 0.2
        /// Mixing stretches shorter than this many groups go back to the originals.
        var minRun: Int = 4
        /// Gaps up to this many frames inside mixing are phase jumps, longer ones original stretches.
        var maxJump: Int = 12
    }

    static let fileName = "timeplan.json"

    /// Viterbi over group starts: a 6-frame group costs its residual, a frame outside one costs
    /// `gapCost · L`. Phases follow from where the cheapest chain puts its groups; a frame repeat
    /// or a cut shows up as a short gap, a stretch without the pattern as a long one.
    static func solve(
        _ features: TimeFeatures,
        fps: Double,
        overrides: [TimeOverride] = [],
        force: Bool = false,
        solver: SolverParams = .standard,
        parameters: Parameters = Parameters()
    ) -> TimePlan {
        let n = features.usable
        var rg = [Double](repeating: .infinity, count: max(n, 0))
        for start in 0..<n where start + 6 < n {
            let value = features.r[start + 2]
            if value.isFinite { rg[start] = Double(value) }
        }
        let level = localLevel(rg)
        var allowed = [Bool](repeating: true, count: n)
        var forcedBlend = [Bool](repeating: false, count: n)
        for override in overrides {
            for start in max(0, override.from - 6)..<min(n, override.to) {
                if override.original {
                    if start + 6 >= override.from { allowed[start] = false }
                } else if let phase = override.phase, override.covers(start) {
                    if start % 6 != ((phase % 6) + 6) % 6 { allowed[start] = false } else { forcedBlend[start] = true }
                }
            }
        }

        // Dynamic programming over frame positions.
        var best = [Double](repeating: .infinity, count: n + 1)
        var back = [Int32](repeating: 0, count: n + 1)
        var group = [Bool](repeating: false, count: n + 1)
        if n > 0 { best[0] = 0 }
        for t in 0..<n where best[t].isFinite {
            if allowed[t], t + 6 <= n, rg[t].isFinite {
                let cost = best[t] + rg[t]
                if cost < best[t + 6] {
                    best[t + 6] = cost
                    back[t + 6] = Int32(t)
                    group[t + 6] = true
                }
            }
            let cost = best[t] + parameters.gapCost * level[t]
            if cost < best[t + 1] {
                best[t + 1] = cost
                back[t + 1] = Int32(t)
                group[t + 1] = false
            }
        }
        var starts: [Int] = []
        var t = n
        while t > 0 {
            let previous = Int(back[t])
            if group[t] { starts.append(previous) }
            t = previous
        }
        starts.reverse()

        // Is it really type A? There the residual of the right phase is codec noise and does not
        // grow with motion; with other mixes (a ramp, field blends) it does. So the residual over
        // the motion across the group (sum of the four frame differences) decides, as a median
        // over the groups with motion within ±25 groups. Stretches without motion take the
        // decision of the nearest stretch with motion in the same chain.
        let motionFloor = 6.0
        var ratio = [Double](repeating: .nan, count: starts.count)
        for (index, start) in starts.enumerated() where start + 6 < n {
            var motion = 0.0
            for k in 3...6 where start + k < n {
                let value = Double(features.d[start + k])
                if value.isFinite { motion += value }
            }
            if motion > motionFloor { ratio[index] = rg[start] / motion }
        }
        var smooth = [Double](repeating: .nan, count: starts.count)
        for index in starts.indices {
            let window = ratio[max(0, index - 25)...min(starts.count - 1, index + 25)].filter { !$0.isNaN }
            if window.count >= 5 { smooth[index] = median(Array(window)) }
        }
        // Fill the stretches without motion from both sides, within a chain without long gaps.
        var filled = smooth
        var lastSeen: Double?
        for index in starts.indices {
            if index > 0, starts[index] - starts[index - 1] - 6 > parameters.maxJump { lastSeen = nil }
            if smooth[index].isNaN { if let lastSeen { filled[index] = lastSeen } } else { lastSeen = smooth[index] }
        }
        lastSeen = nil
        for index in starts.indices.reversed() {
            if index + 1 < starts.count, starts[index + 1] - starts[index] - 6 > parameters.maxJump { lastSeen = nil }
            if smooth[index].isNaN {
                if let lastSeen, filled[index].isNaN || lastSeen > filled[index] { filled[index] = lastSeen }
            } else {
                lastSeen = smooth[index]
            }
        }
        smooth = filled
        var keep = starts.indices.map { forcedBlend[starts[$0]] || (!smooth[$0].isNaN && smooth[$0] <= parameters.maxRatio) }
        // Runs of kept groups that are too short to trust go back to the originals.
        var first = 0
        while first < starts.count {
            guard keep[first] else {
                first += 1
                continue
            }
            var last = first
            while last + 1 < starts.count, keep[last + 1],
                  starts[last + 1] - starts[last] - 6 <= parameters.maxJump {
                last += 1
            }
            if last - first + 1 < parameters.minRun, !(first...last).contains(where: { forcedBlend[starts[$0]] }) {
                for k in first...last { keep[k] = false }
            }
            first = last + 1
        }

        var kept: [Int] = []
        var fallback: [Int] = []
        var keptSmooth: [Double] = []
        for index in starts.indices where keep[index] {
            let start = starts[index]
            kept.append(start)
            keptSmooth.append(smooth[index])
            if rg[start] / level[start] >= parameters.fallbackQ { fallback.append(start) }
        }

        // Segments: a new mixing stretch at every phase change, original stretches between.
        var segments: [TimeSegment] = []
        let fallbackSet = Set(fallback)
        func confidence(_ value: Double) -> Double {
            value.isNaN ? 0.5 : min(max((0.30 - value) / 0.15, 0), 1)
        }
        func forced(_ from: Int, _ to: Int) -> Bool {
            overrides.contains { $0.from < to && $0.to > from }
        }
        var cursor = 0
        var index = 0
        var jumps = 0
        while index < kept.count {
            var end = index
            while end + 1 < kept.count, kept[end + 1] == kept[end] + 6 { end += 1 }
            let from = kept[index]
            let to = kept[end] + 6
            let gap = from - cursor
            var segmentFrom = from
            if gap > 0 {
                if gap <= parameters.maxJump, let last = segments.indices.last, segments[last].kind == .blend {
                    segmentFrom = cursor
                    jumps += 1
                } else {
                    segments.append(TimeSegment(from: cursor, to: from, kind: .original, phase: nil, confidence: 0, fallbacks: 0, forced: forced(cursor, from)))
                }
            }
            let values = keptSmooth[index...end].filter { !$0.isNaN }
            segments.append(TimeSegment(
                from: segmentFrom, to: to, kind: .blend, phase: from % 6,
                confidence: confidence(values.isEmpty ? .nan : median(values)),
                fallbacks: kept[index...end].filter { fallbackSet.contains($0) }.count,
                forced: forced(segmentFrom, to)
            ))
            cursor = to
            index = end + 1
        }
        if cursor < n {
            segments.append(TimeSegment(from: cursor, to: n, kind: .original, phase: nil, confidence: 0, fallbacks: 0, forced: forced(cursor, n)))
        }

        let coverage = n > 0 ? Double(kept.count * 6) / Double(n) : 0
        let blendFrames = segments.filter { $0.kind == .blend }.reduce(0) { $0 + $1.to - $1.from }
        let meanConfidence = blendFrames > 0
            ? segments.filter { $0.kind == .blend }.reduce(0.0) { $0 + $1.confidence * Double($1.to - $1.from) } / Double(blendFrames)
            : 0
        let rateFits = abs(fps - 30) < 0.1 || abs(fps - 30000.0 / 1001) < 0.1
        let accepted = force ? !kept.isEmpty : rateFits && coverage >= parameters.minCoverage && kept.count >= 100
        var notes: [String] = []
        if !rateFits { notes.append("Mischmuster nur bei 30 fps") }
        if kept.isEmpty {
            notes.append("kein Mischmuster")
        } else {
            let share = Int((coverage * 100).rounded())
            notes.append("Mischmuster A in \(share) % des Films")
            if accepted, coverage < 0.95 { notes.append("Rest ohne bekanntes Muster: Originale") }
            if !accepted, !force { notes.append("zu unsicher für Automatik") }
        }
        if features.declared > n || features.failedAt != nil {
            notes.append("Datei endet nach Bild \(n)" + (features.declared > n ? " (laut Kopf \(features.declared))" : ""))
        }
        var plan = TimePlan(
            version: TimePlan.currentVersion, count: features.count, usable: n, declared: features.declared,
            sourceFps: fps, groups: kept, fallback: fallback, segments: segments,
            coverage: coverage, confidence: meanConfidence, jumps: jumps,
            accepted: accepted, note: notes.joined(separator: " · "), hash: "", solver: solver
        )
        plan.hash = hash(of: plan)
        return plan
    }

    /// The 12th percentile of the group residuals within ±150 frames, every 10 frames, at least 0.3.
    static func localLevel(_ rg: [Double]) -> [Double] {
        let n = rg.count
        var level = [Double](repeating: 0.3, count: n)
        var k = 0
        while k < n {
            let window = rg[max(0, k - 150)..<min(n, k + 150)].filter { $0.isFinite }.sorted()
            var value = 0.3
            if !window.isEmpty {
                let position = 0.12 * Double(window.count - 1)
                let low = Int(position)
                let high = min(low + 1, window.count - 1)
                value = window[low] + (window[high] - window[low]) * (position - Double(low))
            }
            for i in k..<min(n, k + 10) { level[i] = max(value, 0.3) }
            k += 10
        }
        return level
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    private static func hash(of plan: TimePlan) -> String {
        var data = Data()
        for value in [plan.version, plan.count, plan.usable] + plan.groups + [-1] + plan.fallback + [-2] + plan.solver.hashWords {
            withUnsafeBytes(of: Int64(value).littleEndian) { data.append(contentsOf: $0) }
        }
        return SHA256.hash(data: data).prefix(4).map { String(format: "%02x", $0) }.joined()
    }

    /// Output frames up to source frame `e`: five of every six, rounded half to even like Python.
    static func target(_ e: Int) -> Int {
        let whole = e * 5 / 6
        let rest = e * 5 % 6
        return rest > 3 || (rest == 3 && whole % 2 == 1) ? whole + 1 : whole
    }

    /// The film the app sees: five frames per group (O0, O1, O2, S3, S4; a fallback group
    /// O0, O1, O2, O2, O6), elsewhere the sharpest five of every six originals.
    static func map(_ plan: TimePlan, sharp: [Float]) -> TimeMap {
        let n = plan.usable
        var kind = [UInt8](repeating: 0, count: n)
        for start in plan.groups where start + 6 < n { kind[start] = 1 }
        for start in plan.fallback where start + 6 < n { kind[start] = 2 }
        var raw: [Int] = []
        var ops: [UInt8] = []
        raw.reserveCapacity(target(plan.count) + 8)
        ops.reserveCapacity(target(plan.count) + 8)
        func emit(_ frame: Int, _ op: UInt8) {
            raw.append(frame)
            ops.append(op)
        }
        var t = 0
        while t < n {
            if kind[t] != 0 {
                let rebuilt = kind[t] == 1
                let last: (Int, UInt8) = rebuilt ? (t + 5, TimeSolver.opS4) : (t + 6, 0)
                let fourth: (Int, UInt8) = rebuilt ? (t + 3, TimeSolver.opS3) : (t + 2, 0)
                let outs: [(Int, UInt8)] = [(t, 0), (t + 1, 0), (t + 2, 0), fourth, last]
                let need = target(t + 6) - raw.count
                for out in outs.prefix(max(0, need)) { emit(out.0, out.1) }
                if need > 5 { for _ in 0..<(need - 5) { emit(outs[4].0, outs[4].1) } }
                t += 6
                continue
            }
            var end = t
            while end < n, kind[end] == 0 { end += 1 }
            var chunk = t
            while chunk < end {
                let chunkEnd = min(chunk + 6, end)
                let need = target(chunkEnd) - raw.count
                if need > 0 {
                    let frames = Array(chunk..<chunkEnd)
                    let picked = frames.sorted { (sharp.indices.contains($0) ? sharp[$0] : 0) > (sharp.indices.contains($1) ? sharp[$1] : 0) }
                        .prefix(min(need, frames.count)).sorted()
                    for frame in picked { emit(frame, 0) }
                    if need > picked.count { for _ in 0..<(need - picked.count) { emit(chunkEnd - 1, 0) } }
                }
                chunk = chunkEnd
            }
            t = end
        }
        let need = target(plan.count) - raw.count
        if need > 0 { for _ in 0..<need { emit(max(n - 1, 0), 0) } }
        let reconstructs = ops.contains { $0 != 0 }
        return TimeMap(rawCount: plan.count, cycle: 6, raw: raw, ops: reconstructs ? ops : [], plan: plan.hash, solver: plan.solver)
    }

    // MARK: Cache

    private static func file(filmID: String) -> URL {
        FrameStore.cacheRoot.appendingPathComponent(filmID, isDirectory: true).appendingPathComponent(fileName)
    }

    static func store(_ plan: TimePlan, filmID: String) throws {
        _ = try CacheSettings.availableRoot()
        let url = file(filmID: filmID)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(plan).write(to: url, options: .atomic)
    }

    static func cachedURL(filmID: String) -> URL { file(filmID: filmID) }
}
