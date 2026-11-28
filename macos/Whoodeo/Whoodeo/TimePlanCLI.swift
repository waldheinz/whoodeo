import CoreMedia
import Foundation
import os

/// `--timeplan` and `--timeplan-selftest`: the time plan without the window.
nonisolated extension CLI {
    static let timePlanUsage = """
    Whoodeo --timeplan video [--force] [--rescan] [--refit] [--json] [--compare reference offset [blocks]]
    Whoodeo --timeplan-selftest

    --timeplan measures the time structure of a film (or reuses the app's measurement), solves the
    plan with the overrides stored for the film, writes timeplan.json into the film's cache and
    measures the solver's κ and σ² on the film once (--refit repeats it) and prints what the pipeline panel would show. --force takes the plan even when automatic would not.
    --compare decodes frames of the planned film and of a reference video (its frame 0 is planned
    frame `offset`) in `blocks` stretches of 60 frames and prints the PSNR.
    """

    static func timePlan(_ args: [String]) throws {
        var rest = args
        func flag(_ name: String) -> Bool {
            guard let at = rest.firstIndex(of: name) else { return false }
            rest.remove(at: at)
            return true
        }
        let force = flag("--force")
        let rescan = flag("--rescan")
        let refit = flag("--refit")
        let json = flag("--json")
        var compare: (url: URL, offset: Int, blocks: Int)?
        if let at = rest.firstIndex(of: "--compare") {
            guard at + 2 < rest.count, let offset = Int(rest[at + 2]) else { throw CLIError(timePlanUsage) }
            var blocks = 10
            var end = at + 3
            if end < rest.count, let parsed = Int(rest[end]) {
                blocks = parsed
                end += 1
            }
            compare = (URL(fileURLWithPath: (rest[at + 1] as NSString).expandingTildeInPath), offset, blocks)
            rest.removeSubrange(at..<end)
        }
        guard rest.count == 1 else { throw CLIError(timePlanUsage) }
        let url = URL(fileURLWithPath: (rest[0] as NSString).expandingTildeInPath).standardizedFileURL
        let entry = FilmLibrary.load().films.first { $0.path == url.path }
        guard entry != nil || FileManager.default.fileExists(atPath: url.path) else { throw CLIError("Die Datei fehlt.") }
        let id = entry?.id ?? FrameStore.filmDigest(of: url)
        let film = try Film.open(url)
        let started = CFAbsoluteTimeGetCurrent()
        let features: TimeFeatures
        if !rescan, let cached = TimeAnalysis.cached(filmID: id, count: film.source.count) {
            features = cached
            fputs("Messung aus dem Cache\n", stdout)
        } else {
            let last = OSAllocatedUnfairLock(initialState: -1)
            features = try TimeAnalysis.scan(url: url, progress: { fraction in
                let percent = Int(fraction * 100)
                let report = last.withLock { value -> Bool in
                    guard percent / 10 != value / 10 else { return false }
                    value = percent
                    return true
                }
                if report { fputs("  \(percent) %\n", stderr) }
            }, cancelled: { false })
            try TimeAnalysis.store(features, filmID: id)
            let seconds = CFAbsoluteTimeGetCurrent() - started
            fputs(String(format: "Messung %d Bilder in %.1f s (%.0f Bilder/s)\n", features.count, seconds, Double(features.count) / seconds), stdout)
        }
        let overrides = entry?.preprocess?.overrides ?? []
        let solveStart = CFAbsoluteTimeGetCurrent()
        var solver = refit ? nil : SolverParams.cached(filmID: id)
        var plan = TimePlanner.solve(features, fps: film.source.fps, overrides: overrides, force: force || entry?.preprocess?.timing == .manual, solver: solver ?? .standard)
        if solver == nil, plan.accepted {
            let fitStart = CFAbsoluteTimeGetCurrent()
            let fit = try SolverFit.estimate(url: url, plan: plan)
            try SolverParams.store(fit.params, filmID: id, samples: fit.groups)
            solver = fit.params
            fputs(String(format: "Solver-Parameter aus %d Gruppen in %.1f s: κ %.3f, σ² %.3f%@\n", fit.groups, CFAbsoluteTimeGetCurrent() - fitStart, fit.params.kappa, fit.params.sigma2, fit.groups == 0 ? " (Standard)" : ""), stdout)
            plan = TimePlanner.solve(features, fps: film.source.fps, overrides: overrides, force: force || entry?.preprocess?.timing == .manual, solver: fit.params)
        } else if let solver {
            fputs(String(format: "Solver-Parameter aus dem Cache: κ %.3f, σ² %.3f\n", solver.kappa, solver.sigma2), stdout)
        }
        let solveMs = (CFAbsoluteTimeGetCurrent() - solveStart) * 1000
        try TimePlanner.store(plan, filmID: id)
        let map = TimePlanner.map(plan, sharp: features.sharp)
        let blends = plan.segments.filter { $0.kind == .blend }
        let originals = plan.segments.filter { $0.kind == .original }
        let originalFrames = originals.reduce(0) { $0 + $1.to - $1.from }
        fputs("Datei: \(features.count) Bilder lesbar \(plan.usable), laut Kopf \(features.declared)\(features.clipped ? ", abgeschnitten" : "")\(features.failedAt.map { ", Bild \($0) dekodiert nicht" } ?? "")\n", stdout)
        fputs(String(format: "Plan %@ in %.0f ms: %@\n", plan.hash, solveMs, plan.note), stdout)
        fputs("  Gruppen \(plan.groups.count), Abdeckung \(String(format: "%.1f", plan.coverage * 100)) %, Zuversicht \(String(format: "%.2f", plan.confidence))\n", stdout)
        fputs("  Abschnitte: \(blends.count) Mischung, \(originals.count) Original (\(originalFrames) Bilder), Phasensprünge \(plan.jumps)\n", stdout)
        if plan.accepted {
            fputs("  übernommen: \(film.source.count) → \(map.count) Bilder, \(film.source.fps) → \(map.fps(source: film.source.fps)) fps, \(map.ops.filter { $0 != 0 }.count) rekonstruiert, Cache-Tag \(map.cacheTag)\n", stdout)
        } else {
            fputs("  nicht übernommen (Automatisch bleibt aus): \(film.source.count) Bilder, \(film.source.fps) fps\n", stdout)
        }
        for segment in plan.segments.prefix(json ? Int.max : 12) {
            fputs("  \(clock(segment.from, film.source.fps))–\(clock(segment.to, film.source.fps))  \(segment.kind.rawValue)\(segment.phase.map { " Phase \($0)" } ?? "")  Zuversicht \(String(format: "%.2f", segment.confidence))\(segment.forced ? "  (manuell)" : "")\n", stdout)
        }
        if json { fputs("\(TimePlanner.cachedURL(filmID: id).path)\n", stdout) }
        guard plan.accepted else { return }
        film.apply(map)
        let end = film.time(for: film.facts.count).seconds
        let sourceEnd = CMTimeAdd(film.time(for: 0), CMTimeMultiply(film.sourceDuration, multiplier: Int32(film.source.count))).seconds
        fputs(String(format: "  Dauer %.3f s (Quelle %.3f s)\n", end, sourceEnd), stdout)
        if let compare { try comparePlan(film, reference: compare.url, offset: compare.offset, blocks: compare.blocks) }
    }

    private static func clock(_ frame: Int, _ fps: Double) -> String {
        let seconds = Double(frame) / max(fps, 1)
        return String(format: "%d:%02d.%02d", Int(seconds) / 60, Int(seconds) % 60, Int((seconds - floor(seconds)) * 100))
    }

    /// PSNR of planned frames against a reference video, split by how the frame was made.
    private static func comparePlan(_ film: Film, reference: URL, offset: Int, blocks: Int) throws {
        let ref = try Film.open(reference)
        let total = min(ref.facts.count, film.facts.count - offset)
        guard total > 60 else { throw CLIError("Vergleich: zu wenige Bilder") }
        var byOp: [UInt8: [Double]] = [:]
        var better = 0
        var poor = 0
        var poorFrames: [String] = []
        var worst: (Double, Int) = (.infinity, 0)
        let started = CFAbsoluteTimeGetCurrent()
        for block in 0..<blocks {
            let first = (total - 60) * block / max(blocks - 1, 1)
            for j in first..<(first + 60) {
                let ours = try film.decode(at: j + offset)
                let theirs = try ref.decode(at: j)
                let value = psnr(ours.rgb, theirs.rgb)
                let op = film.map?.op(at: j + offset) ?? 0
                byOp[op, default: []].append(value)
                if value < worst.0 { worst = (value, j) }
                if value < 35 {
                    poor += 1
                    if poorFrames.count < 40 { poorFrames.append("\(j)/\(film.sourceIndex(j + offset))\(op == 0 ? "" : "r")") }
                    // Off by one frame?
                    let neighbor = try film.decode(at: j + offset + 1)
                    if psnr(neighbor.rgb, theirs.rgb) > value + 3 { better += 1 }
                }
            }
        }
        if !poorFrames.isEmpty { fputs("  unter 35 dB (Referenzbild/Quellbild): \(poorFrames.joined(separator: " "))\n", stdout) }
        let names: [UInt8: String] = [0: "Kopie", TimeSolver.opS3: "S3", TimeSolver.opS4: "S4"]
        for (op, values) in byOp.sorted(by: { $0.key < $1.key }) {
            let sorted = values.sorted()
            fputs(String(format: "  PSNR %@: %d Bilder, Mittel %.2f dB, Median %.2f, Min %.2f\n",
                         names[op] ?? "?", values.count, values.reduce(0, +) / Double(values.count), sorted[sorted.count / 2], sorted[0]), stdout)
        }
        fputs(String(format: "  schlechtestes Bild %d (%.2f dB), unter 35 dB: %d, davon passt der Nachbar besser: %d; %.1f s\n", worst.1, worst.0, poor, better, CFAbsoluteTimeGetCurrent() - started), stdout)
    }

    private static func psnr(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var sum = 0.0
        for i in 0..<n {
            let d = Double(a[i] - b[i])
            sum += d * d
        }
        let mse = sum / Double(n)
        return mse > 0 ? 10 * log10(1 / mse) : 99
    }

    // MARK: Self test

    /// Planner and map on made-up measurements. Throws on the first failed check.
    static func timePlanSelfTest() throws {
        var failures: [String] = []
        func check(_ condition: Bool, _ text: String) {
            fputs("\(condition ? "ok  " : "FEHLER") \(text)\n", stdout)
            if !condition { failures.append(text) }
        }
        var generator = PlanNoise(seed: 7)
        func features(count: Int, residual: (Int) -> Float) -> TimeFeatures {
            var r = [Float](repeating: .infinity, count: count)
            for t in 0..<(count - 4) { r[t] = residual(t) * (0.85 + 0.3 * generator.unit()) }
            let sharp = (0..<count).map { _ in Float(10 + generator.unit()) }
            let d = (0..<count).map { $0 == 0 ? Float.infinity : Float(2.5 + generator.unit()) }
            return TimeFeatures(count: count, declared: count, fps: 30, clipped: false, failedAt: nil, d: d, r: r, sharp: sharp)
        }

        // 1. Type A with a +1 jump at 3001 and a −1 jump at 6000 (a repeat, then a skip).
        let count = 9000
        func phase(_ group: Int) -> Int { group < 3001 ? 0 : (group < 6000 ? 1 : 0) }
        let typeA = features(count: count) { t in
            let start = t - 2
            return start >= 0 && start % 6 == phase(start) ? 0.6 : 6
        }
        let plan = TimePlanner.solve(typeA, fps: 30)
        let wrong = plan.groups.filter { $0 % 6 != phase($0) }.count
        check(plan.accepted, "Typ A erkannt (Abdeckung \(String(format: "%.3f", plan.coverage)))")
        check(wrong == 0, "alle Gruppen in der richtigen Phase (falsch: \(wrong))")
        check(plan.coverage > 0.97, "Abdeckung über 97 %")
        check(plan.jumps == 2, "zwei Phasensprünge (\(plan.jumps))")
        let map = TimePlanner.map(plan, sharp: typeA.sharp)
        check(map.count == TimePlanner.target(count), "Bildzahl \(map.count) = 5/6 von \(count)")
        check(zip(map.raw, map.raw.dropFirst()).allSatisfy { $0 <= $1 }, "Quellbilder laufen vorwärts")
        let s3 = map.ops.filter { $0 == TimeSolver.opS3 }.count
        check(s3 == plan.groups.count, "je Gruppe ein S3 (\(s3))")
        check(map.cacheTag == "tp\(TimeSolver.version)m\(TimePlanner.layout)-\(plan.hash)", "Cache-Tag tp\(TimeSolver.version)m\(TimePlanner.layout)-\(plan.hash)")
        var drift = 0
        for (index, source) in map.raw.enumerated() { drift = max(drift, abs(source - index * 6 / 5)) }
        check(drift <= 6, "höchstens 6 Quellbilder Versatz zur Zeitachse (\(drift))")

        // 2. A stretch without the pattern in the middle runs as originals.
        let mixed = features(count: count) { t in
            let start = t - 2
            if start >= 3000 && start < 6000 { return 4 }
            return start >= 0 && start % 6 == 0 ? 0.6 : 6
        }
        let mixedPlan = TimePlanner.solve(mixed, fps: 30)
        let inside = mixedPlan.groups.filter { $0 >= 3100 && $0 < 5900 }.count
        check(inside == 0, "ohne Muster keine Gruppen (\(inside))")
        check(mixedPlan.segments.contains { $0.kind == .original && $0.from <= 3100 && $0.to >= 5900 }, "Original-Abschnitt in der Mitte")

        // 3. No pattern at all: automatic stays off.
        let plain = features(count: count) { _ in 4 }
        let plainPlan = TimePlanner.solve(plain, fps: 30)
        check(!plainPlan.accepted, "ohne Muster nicht übernommen (\(plainPlan.groups.count) Gruppen)")
        let rated = TimePlanner.solve(typeA, fps: 25)
        check(!rated.accepted, "25 fps nicht übernommen")

        // 4. Overrides: original stretch and forced phase.
        let forcedOriginal = TimePlanner.solve(typeA, fps: 30, overrides: [TimeOverride(from: 1200, to: 2400, original: true)])
        check(!forcedOriginal.groups.contains { $0 + 6 >= 1200 && $0 < 2400 }, "Override Original ohne Gruppen")
        let shifted = TimePlanner.solve(typeA, fps: 30, overrides: [TimeOverride(from: 600, to: 1800, phase: 1)])
        let shiftedGroups = shifted.groups.filter { $0 >= 600 && $0 < 1800 }
        check(!shiftedGroups.isEmpty && shiftedGroups.allSatisfy { $0 % 6 == 1 }, "Override Phase 1 (\(shiftedGroups.count) Gruppen)")

        // 5. A file that stops decoding.
        var cut = typeA
        cut.failedAt = 7000
        cut.declared = 9000
        let cutPlan = TimePlanner.solve(cut, fps: 30)
        check(cutPlan.usable == 7000 && cutPlan.truncated, "abgeschnittene Datei erkannt")
        check(cutPlan.groups.allSatisfy { $0 + 6 < 7000 }, "keine Gruppe hinter dem Ende")
        check(TimePlanner.map(cutPlan, sharp: cut.sharp).raw.allSatisfy { $0 < 7000 }, "Map liest nichts hinter dem Ende")

        // 6. Solver: an exact type A group comes back exactly, a mismatch stays mixed, values stay in 0…1.
        let w = 16, h = 12, plane = w * h
        func mix(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map { ($0 + $1) / 2 } }
        func maxAbs(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map { abs($0 - $1) }.max() ?? 1 }
        let sources = (0..<6).map { k in (0..<(plane * 3)).map { i in Float((i * 7 + k * 31) % 200 + 20) / 255 } }
        let group = [sources[2], mix(sources[2], sources[3]), mix(sources[3], sources[4]), mix(sources[4], sources[5]), sources[5]]
        let got3 = TimeSolver.reconstruct(group, op: TimeSolver.opS3, width: w, height: h)
        let got4 = TimeSolver.reconstruct(group, op: TimeSolver.opS4, width: w, height: h)
        let e3 = maxAbs(got3, sources[3])
        let e4 = maxAbs(got4, sources[4])
        check(e3 < 1e-6 && e4 < 1e-6, String(format: "S3/S4 exakt aus Typ-A-Mischung (Fehler %.2e / %.2e)", e3, e4))
        func constant(_ value: Float) -> [Float] { [Float](repeating: value, count: plane * 3) }
        let mismatch = [constant(0.1), constant(0.8), constant(0.1), constant(0.8), constant(0.1)]
        let stay3 = TimeSolver.reconstruct(mismatch, op: TimeSolver.opS3, width: w, height: h)
        let stay4 = TimeSolver.reconstruct(mismatch, op: TimeSolver.opS4, width: w, height: h)
        check(maxAbs(stay3, mismatch[1]) < 1e-6 && maxAbs(stay4, mismatch[3]) < 1e-6, "bei großem Residuum bleibt das Mischbild")
        // True S2=0, S3=1, S4=0, S5=0; O3 a hair off, so 2·O3−O2 exceeds 1 and must stay at 1.
        let overflow = [constant(0), constant(0.502), constant(0.5), constant(0), constant(0)]
        let clipped = TimeSolver.reconstruct(overflow, op: TimeSolver.opS3, width: w, height: h)
        check(clipped.allSatisfy { $0 >= 0 && $0 <= 1 } && clipped.allSatisfy { abs($0 - 1) < 1e-6 }, "Überlauf wird auf 1 begrenzt")
        let clippedLow = TimeSolver.reconstruct([constant(1), constant(0.998), constant(0.5), constant(1), constant(1)], op: TimeSolver.opS3, width: w, height: h)
        check(clippedLow.allSatisfy { $0 >= 0 && $0 <= 1 }, "Unterlauf bleibt im Bereich")

        // 6b. A moving soft edge with a codec error that grows near the edge. The weighted solution must
        // come closer to the truth than the plain 2·O3−O2 / 2·O5−O6, and with no error it stays exact.
        let ew = 64, eh = 48, eplane = ew * eh
        func scene(_ k: Int) -> [Float] {
            var out = [Float](repeating: 0, count: eplane * 3)
            let edge = Float(20 + 4 * k)
            for y in 0..<eh {
                for x in 0..<ew {
                    let t = min(max((Float(x) - edge) / 2 + 0.5, 0), 1)        // soft step, about 2 px wide
                    let value = 0.2 + 0.6 * t + 0.1 * Float(y) / Float(eh)
                    for c in 0..<3 { out[c * eplane + y * ew + x] = value * (c == 0 ? 1 : c == 1 ? 0.96 : 0.9) }
                }
            }
            return out
        }
        let truth = (0..<6).map(scene)
        let exact = [truth[2], mix(truth[2], truth[3]), mix(truth[3], truth[4]), mix(truth[4], truth[5]), truth[5]]
        let calm = TimeSolver.reconstructPair(exact, width: ew, height: eh)
        check(maxAbs(calm.s3, truth[3]) < 1e-5 && maxAbs(calm.s4, truth[4]) < 1e-5,
              String(format: "ohne Störung exakt (Fehler %.2e / %.2e)", maxAbs(calm.s3, truth[3]), maxAbs(calm.s4, truth[4])))
        var seed: UInt32 = 12345
        func noise() -> Float {
            seed = seed &* 1664525 &+ 1013904223
            return Float(seed >> 8) / Float(1 << 24) * 2 - 1
        }
        let disturbed = exact.enumerated().map { k, plane -> [Float] in
            var out = plane
            for y in 0..<eh {
                for x in 0..<ew {
                    // Distance to the nearest edge of the two sources the frame shows.
                    let near = min(abs(Float(x) - Float(28 + 4 * max(k - 1, 0))), abs(Float(x) - Float(28 + 4 * min(k, 3))))
                    let amount = 0.02 * exp(-near * near / 18)
                    for c in 0..<3 { out[c * eplane + y * ew + x] += amount * noise() }
                }
            }
            return out
        }
        let rough = TimeSolver.reconstructPair(disturbed, width: ew, height: eh)
        func rms(_ a: [Float], _ b: [Float]) -> Float { (zip(a, b).map { ($0 - $1) * ($0 - $1) }.reduce(0, +) / Float(a.count)).squareRoot() * 255 }
        let plain3 = zip(disturbed[1], disturbed[0]).map { min(max(2 * $0 - $1, 0), 1) }
        let plain4 = zip(disturbed[3], disturbed[4]).map { min(max(2 * $0 - $1, 0), 1) }
        let (p3, p4) = (rms(plain3, truth[3]), rms(plain4, truth[4]))
        let (r3, r4) = (rms(rough.s3, truth[3]), rms(rough.s4, truth[4]))
        check(r3 < p3 * 0.9 && r4 < p4 * 0.9, String(format: "Kantenfehler sinkt: S3 %.2f → %.2f, S4 %.2f → %.2f (Stufen von 255)", p3, r3, p4, r4))

        let refit = TimePlanner.solve(cut, fps: 30, solver: SolverParams(kappa: 0.9, sigma2: 0.6))
        check(refit.hash != TimePlanner.solve(cut, fps: 30).hash, "andere Solver-Parameter ergeben einen anderen Plan-Hash (neuer Cache)")

        // Time per pair at film size, the preview's cost.
        let fw = 640, fh = 480, fplane = fw * fh
        let big = (0..<5).map { k in (0..<(fplane * 3)).map { i in Float((i * 13 + k * 17) % 211 + 20) / 255 } }
        _ = TimeSolver.reconstructPair(big, width: fw, height: fh)
        let timer = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 { _ = TimeSolver.reconstructPair(big, width: fw, height: fh) }
        fputs(String(format: "  Info: S3 + S4 aus 5 Bildern bei 640×480: %.1f ms\n", (CFAbsoluteTimeGetCurrent() - timer) * 200), stdout)

        // 7. Film list: old entries, new entries, round trip.
        let decoder = JSONDecoder()
        let old = try decoder.decode(PreprocessConfig.self, from: Data(#"{"decimate":"auto"}"#.utf8))
        check(old.timing == .duplicates && old.decimateInForce == .auto, "alter Eintrag decimate=auto liest als „Nur doppelte Bilder“")
        let empty = try decoder.decode(PreprocessConfig.self, from: Data("{}".utf8))
        check(empty == .none && empty.decimateInForce == .off, "leerer Eintrag ist aus")
        let planned = PreprocessConfig(timing: .manual, overrides: [TimeOverride(from: 10, to: 20, phase: 3)])
        let back = try decoder.decode(PreprocessConfig.self, from: JSONEncoder().encode(planned))
        check(back == planned && back.plansTime && back.decimateInForce == .off, "neuer Eintrag mit Korrektur übersteht Speichern und Laden")

        // 8. Times beyond 32 bits.
        let tick = CMTime(value: 1001, timescale: 30000)
        let far = Film.scaled(tick, by: 3_000_000 * 6, over: 5)
        check(abs(far.seconds - 3_000_000 * 1.2 * 1001 / 30000) < 1e-6, String(format: "Zeit bei 3 Mio. Bildern exakt (%.6f s)", far.seconds))

        if !failures.isEmpty { throw CLIError("\(failures.count) Prüfungen fehlgeschlagen") }
        fputs("alle Prüfungen bestanden\n", stdout)
    }
}

/// Small deterministic generator for the self test.
nonisolated struct PlanNoise {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
}
