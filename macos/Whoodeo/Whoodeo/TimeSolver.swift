import Foundation

// MARK: Parameters

/// What the solver learns from a film: how fast the codec error grows with the local edge strength.
/// `kappa` and `sigma2` come from the film itself (see `SolverFit`), stored beside the plan.
nonisolated struct SolverParams: Codable, Equatable, Hashable, Sendable {
    /// Relative growth of the error variance of a source frame per luma step of local gradient (0…255 scale).
    var kappa: Double
    /// Variance of the residual in a flat area, per unit of the weights (0…255 scale, squared).
    var sigma2: Double

    /// Measured on Private Einblicke Vol. II; used until a film has its own fit.
    static let standard = SolverParams(kappa: 0.586, sigma2: 0.629)

    /// What the plan hash keeps: thousandths.
    var hashWords: [Int] { [Int((kappa * 1000).rounded()), Int((sigma2 * 1000).rounded())] }

    static let fileName = "timesolver.json"

    private static func file(filmID: String) -> URL {
        FrameStore.cacheRoot.appendingPathComponent(filmID, isDirectory: true).appendingPathComponent(fileName)
    }

    /// The fit stored for a film, if there is one made by this solver version.
    static func cached(filmID: String) -> SolverParams? {
        guard let data = try? Data(contentsOf: file(filmID: filmID)),
              let stored = try? JSONDecoder().decode(Stored.self, from: data),
              stored.solver == TimeSolver.version, stored.kappa.isFinite, stored.sigma2.isFinite
        else { return nil }
        return SolverParams(kappa: stored.kappa, sigma2: stored.sigma2)
    }

    static func store(_ params: SolverParams, filmID: String, samples: Int) throws {
        _ = try CacheSettings.availableRoot()
        let url = file(filmID: filmID)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let stored = Stored(solver: TimeSolver.version, kappa: params.kappa, sigma2: params.sigma2, groups: samples)
        try encoder.encode(stored).write(to: url, options: .atomic)
    }

    private struct Stored: Codable {
        var solver: Int
        var kappa: Double
        var sigma2: Double
        var groups: Int
    }
}

// MARK: Solver

/// Rebuilds the two mixed frames of a type A group. In a group of six source pictures S0…S5 the
/// file shows O0 = S0, O1 = S1, O2 = S2, O3 = (S2+S3)/2, O4 = (S3+S4)/2, O5 = (S4+S5)/2, O6 = S5.
/// Read exactly, S3 = 2·O3 − O2 and S4 = 2·O5 − O6. But the codec made each O a little wrong, mostly
/// at moving edges, and the doubling shows that as a hem. Five observations O2…O6 of four unknowns
/// S2…S5 leave one check, the residual r = O4 − O3 − O5 + (O2 + O6)/2, which would be zero without
/// error. It is handed back to the five frames in proportion to how much each is likely to be wrong
/// (variance 1 + κ·local gradient, weights v = ½, −1, 1, −1, ½ of the residual). That is the weighted
/// least squares solution, applied pixel by pixel: no picture is filtered. Then
/// S3 = 2·O3′ − O2′ and S4 = 2·O5′ − O6′, in 0…1.
/// Where r is far larger than the codec explains (a wrong group, an occlusion) the mixed frame stays;
/// that decision uses the residual, normalized by its expected size, blurred only for the decision.
nonisolated enum TimeSolver {
    static let opS3: UInt8 = 1
    static let opS4: UInt8 = 2
    /// Bumped when the reconstructed picture changes, so the frame cache does not keep an older result.
    static let version = 3
    /// Source frames a reconstruction reads: O2 … O6.
    static let inputs = 5
    /// Normalized residual (blurred) where the mixed frame starts to take over, and where it has fully.
    static let zLow: Float = 3
    static let zHigh: Float = 6
    /// Weights of the residual onto O2…O6.

    /// The first of the five source frames for a recipe whose frame stands at `main`.
    static func firstSource(main: Int, op: UInt8) -> Int? {
        switch op {
        case opS3: max(main - 1, 0)
        case opS4: max(main - 3, 0)
        default: nil
        }
    }

    /// `planes` are O2 … O6 as planar RGB 0…1.
    static func reconstruct(_ planes: [[Float]], op: UInt8, width: Int, height: Int, params: SolverParams = .standard) -> [Float] {
        let pair = reconstructPair(planes, width: width, height: height, params: params)
        return op == opS3 ? pair.s3 : pair.s4
    }

    /// S3 and S4 of one group; both read the same five frames, so one pass makes both.
    static func reconstructPair(_ planes: [[Float]], width: Int, height: Int, params: SolverParams = .standard) -> (s3: [Float], s4: [Float]) {
        let plane = width * height
        precondition(planes.count == inputs && planes.allSatisfy { $0.count >= plane * 3 })
        let kappa = Float(params.kappa)
        let sigma2 = Float(params.sigma2)

        func make(_ count: Int) -> UnsafeMutablePointer<Float> {
            let pointer = UnsafeMutablePointer<Float>.allocate(capacity: count)
            pointer.initialize(repeating: 0, count: count)
            return pointer
        }
        let o = (0..<inputs).map { k -> UnsafeMutablePointer<Float> in
            let pointer = UnsafeMutablePointer<Float>.allocate(capacity: plane * 3)
            planes[k].withUnsafeBufferPointer { _ = memcpy(pointer, $0.baseAddress!, plane * 3 * MemoryLayout<Float>.size) }
            return pointer
        }
        let spread = (0..<inputs).map { _ in make(plane) }   // luma first, then 1 + κ·gradient
        let scratchA = make(plane)
        let scratchB = make(plane)
        let denom = make(plane)
        let coef3 = make(plane)
        let coef4 = make(plane)
        let zmap = make(plane)
        let ztemp = make(plane)
        let out3 = UnsafeMutablePointer<Float>.allocate(capacity: plane * 3)
        let out4 = UnsafeMutablePointer<Float>.allocate(capacity: plane * 3)
        defer {
            for pointer in o + spread + [scratchA, scratchB, denom, coef3, coef4, zmap, ztemp, out3, out4] { pointer.deallocate() }
        }

        // Luma, 0…255, then the local gradient level, then the variance of each frame.
        for k in 0..<inputs {
            let src = o[k]
            let luma = spread[k]
            rows(height) { y0, y1 in
                for y in y0..<y1 {
                    let base = y * width
                    for x in 0..<width {
                        let i = base + x
                        luma[i] = 255 * (0.299 * src[i] + 0.587 * src[plane + i] + 0.114 * src[2 * plane + i])
                    }
                }
            }
            gradientLevel(luma, into: scratchB, scratch: scratchA, width: width, height: height)
            let level = scratchB
            rows(height) { y0, y1 in
                for i in (y0 * width)..<(y1 * width) { luma[i] = 1 + kappa * level[i] }
            }
        }
        let (o2, o3, o4, o5, o6) = (o[0], o[1], o[2], o[3], o[4])
        let (g0, g1, g2, g3, g4) = (spread[0], spread[1], spread[2], spread[3], spread[4])
        rows(height) { y0, y1 in
            for i in (y0 * width)..<(y1 * width) {
                // D = Σ v²·var with v = ½, −1, 1, −1, ½; the weights are w = var·v / D.
                let d = 0.25 * g0[i] + g1[i] + g2[i] + g3[i] + 0.25 * g4[i]
                denom[i] = d
                coef3[i] = (-2 * g1[i] - 0.5 * g0[i]) / d      // 2·w1 − w0
                coef4[i] = (-2 * g3[i] - 0.5 * g4[i]) / d      // 2·w3 − w4
                // Normalized luma residual for the decision.
                let rr = o4[i] - o3[i] - o5[i] + (o2[i] + o6[i]) * 0.5
                let rg = o4[plane + i] - o3[plane + i] - o5[plane + i] + (o2[plane + i] + o6[plane + i]) * 0.5
                let rb = o4[2 * plane + i] - o3[2 * plane + i] - o5[2 * plane + i] + (o2[2 * plane + i] + o6[2 * plane + i]) * 0.5
                let rl = 0.299 * rr + 0.587 * rg + 0.114 * rb
                zmap[i] = abs(255 * rl) / (sigma2 * d).squareRoot()
            }
        }
        blur121(zmap, temp: ztemp, width: width, height: height)
        let span = zHigh - zLow
        rows(height) { y0, y1 in
            for c in 0..<3 {
                let offset = c * plane
                for i in (y0 * width)..<(y1 * width) {
                    let j = offset + i
                    let r = o4[j] - o3[j] - o5[j] + (o2[j] + o6[j]) * 0.5
                    let alpha = min(max((zmap[i] - zLow) / span, 0), 1)
                    let s3 = 2 * o3[j] - o2[j] - coef3[i] * r
                    let s4 = 2 * o5[j] - o6[j] - coef4[i] * r
                    out3[j] = min(max((1 - alpha) * s3 + alpha * o3[j], 0), 1)
                    out4[j] = min(max((1 - alpha) * s4 + alpha * o5[j], 0), 1)
                }
            }
        }
        return (Array(UnsafeBufferPointer(start: out3, count: plane * 3)), Array(UnsafeBufferPointer(start: out4, count: plane * 3)))
    }

    /// √(box5(gx² + gy²)) of a plane: central differences (zero on the border), then a 5×5 mean with the
    /// edge repeated. `out` and `scratch` hold width × height values.
    static func gradientLevel(
        _ luma: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>, scratch: UnsafeMutablePointer<Float>,
        width: Int, height: Int
    ) {
        rows(height) { y0, y1 in
            for y in y0..<y1 {
                let base = y * width
                for x in 0..<width {
                    var gx: Float = 0
                    var gy: Float = 0
                    if x > 0 && x < width - 1 { gx = (luma[base + x + 1] - luma[base + x - 1]) * 0.5 }
                    if y > 0 && y < height - 1 { gy = (luma[base + x + width] - luma[base + x - width]) * 0.5 }
                    out[base + x] = gx * gx + gy * gy
                }
            }
        }
        rows(height) { y0, y1 in
            for y in y0..<y1 {
                let base = y * width
                for x in 0..<width {
                    var sum: Float = 0
                    for d in -2...2 { sum += out[base + min(max(x + d, 0), width - 1)] }
                    scratch[base + x] = sum / 5
                }
            }
        }
        rows(height) { y0, y1 in
            for y in y0..<y1 {
                let base = y * width
                for x in 0..<width {
                    var sum: Float = 0
                    for d in -2...2 { sum += scratch[min(max(y + d, 0), height - 1) * width + x] }
                    out[base + x] = max(sum / 5, 0).squareRoot()
                }
            }
        }
    }

    /// [1 2 1]/4 across, then down; the border rows and columns stay. Used on the decision map only.
    private static func blur121(_ values: UnsafeMutablePointer<Float>, temp: UnsafeMutablePointer<Float>, width: Int, height: Int) {
        guard width > 2, height > 2 else { return }
        rows(height) { y0, y1 in
            for y in y0..<y1 {
                let base = y * width
                temp[base] = values[base]
                temp[base + width - 1] = values[base + width - 1]
                for x in 1..<(width - 1) {
                    temp[base + x] = 0.25 * values[base + x - 1] + 0.5 * values[base + x] + 0.25 * values[base + x + 1]
                }
            }
        }
        rows(height) { y0, y1 in
            for y in y0..<y1 {
                let base = y * width
                if y == 0 || y == height - 1 {
                    for x in 0..<width { values[base + x] = temp[base + x] }
                } else {
                    for x in 0..<width {
                        values[base + x] = 0.25 * temp[base - width + x] + 0.5 * temp[base + x] + 0.25 * temp[base + width + x]
                    }
                }
            }
        }
    }

    /// Runs `body` over bands of rows on all cores.
    @inline(__always)
    private static func rows(_ height: Int, _ body: (Int, Int) -> Void) {
        let bands = min(8, max(1, height / 24))
        if bands == 1 { body(0, height); return }
        DispatchQueue.concurrentPerform(iterations: bands) { band in
            body(height * band / bands, height * (band + 1) / bands)
        }
    }
}

// MARK: Fit

/// Measures `SolverParams` on the film: on evenly spread, steady groups the residual is regressed on
/// the local gradient of the five frames, E[r²] = σ²·(Σv² + κ·Σv²·G). A few dozen groups are enough;
/// the result is stored with the film and enters the plan hash.
nonisolated enum SolverFit {
    static let groupCount = 120
    static let step = 3
    /// Squared residual beyond which a pixel counts as this much only (10 grey steps): codec noise sets the
    /// fit, not the occlusions and cuts that would otherwise swing it from one choice of groups to the next.
    static let residualCap: Float = 100

    static func estimate(url: URL, plan: TimePlan, progress: ((Double) -> Void)? = nil) throws -> (params: SolverParams, groups: Int) {
        let candidates = plan.groups.filter { $0 + 6 < plan.usable }
        guard candidates.count >= 8 else { return (.standard, 0) }
        let film = try Film.open(url)
        let wanted = min(groupCount, candidates.count)
        var n = 0.0, sX = 0.0, sXX = 0.0, sY = 0.0, sXY = 0.0
        var used = 0
        let weights: [Float] = [0.5, -1, 1, -1, 0.5]
        for index in 0..<wanted {
            let start = candidates[(index * candidates.count + candidates.count / 2) / wanted]
            var lumas: [[Float]] = []
            var width = 0, height = 0
            for k in 0..<TimeSolver.inputs {
                let decoded = try film.decodeRaw(source: start + 2 + k)
                width = decoded.width
                height = decoded.height
                let plane = width * height
                var luma = [Float](repeating: 0, count: plane)
                for i in 0..<plane {
                    luma[i] = 255 * (0.299 * decoded.rgb[i] + 0.587 * decoded.rgb[plane + i] + 0.114 * decoded.rgb[2 * plane + i])
                }
                lumas.append(luma)
            }
            let plane = width * height
            var levels: [[Float]] = []
            var out = [Float](repeating: 0, count: plane)
            var scratch = [Float](repeating: 0, count: plane)
            for k in 0..<TimeSolver.inputs {
                lumas[k].withUnsafeBufferPointer { src in
                    out.withUnsafeMutableBufferPointer { dst in
                        scratch.withUnsafeMutableBufferPointer { tmp in
                            TimeSolver.gradientLevel(src.baseAddress!, into: dst.baseAddress!, scratch: tmp.baseAddress!, width: width, height: height)
                        }
                    }
                }
                levels.append(out)
            }
            var y = 4
            while y < height - 4 {
                var x = 4
                while x < width - 4 {
                    let i = y * width + x
                    let r = lumas[2][i] - lumas[1][i] - lumas[3][i] + (lumas[0][i] + lumas[4][i]) * 0.5
                    var g: Float = 0
                    for k in 0..<TimeSolver.inputs { g += weights[k] * weights[k] * levels[k][i] }
                    let value = Double(min(r * r, residualCap))
                    let regressor = Double(g)
                    n += 1; sX += regressor; sXX += regressor * regressor; sY += value; sXY += regressor * value
                    x += step
                }
                y += step
            }
            used += 1
            progress?(Double(index + 1) / Double(wanted))
        }
        let det = n * sXX - sX * sX
        guard n > 100, det > 1e-9 else { return (.standard, used) }
        let a = (sY * sXX - sX * sXY) / det
        let b = (n * sXY - sX * sY) / det
        let sigma2 = a / 3.5
        guard sigma2 > 0.05, sigma2 < 4, b > 0 else { return (.standard, used) }
        let kappa = b / sigma2
        guard kappa > 0.05, kappa < 3 else { return (.standard, used) }
        let rounded = SolverParams(kappa: (kappa * 1000).rounded() / 1000, sigma2: (sigma2 * 1000).rounded() / 1000)
        return (rounded, used)
    }
}
