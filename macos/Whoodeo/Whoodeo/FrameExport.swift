import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// What a diagnosis export needs to know about the film on screen besides the film itself.
nonisolated struct FrameExportContext: @unchecked Sendable {
    var filmID: String
    var plan: TimePlan?
    var preprocess: PreprocessConfig
    /// The archive of the model on screen. Only read; a missing frame is never computed.
    var store: FrameStore?
    var modelPath: String?
    var modelDetail: String?
    var runName: String?
    var step: Int?
    /// Condition and map part of the archive folder name, as the session builds it.
    var cacheTag: String?
    var gradeActive = false
    var origin = "app"
}

/// "Aktuelles Bild exportieren" and `--export-frame`: the frame as the preview shows it, its upscale
/// when it is cached, and the raw source frames it comes from, with one info file.
/// The mapping is the one of the preview: `Film.decode(at:)` through the time map.
nonisolated enum FrameExport {
    static var rootFolder: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Downloads/whoodeo-diag/export", isDirectory: true)
    }

    /// The map the film's stored preprocessing makes, read from the app's caches. Nothing is measured here.
    static func resolve(url: URL, filmID: String, preprocess: PreprocessConfig) throws -> (film: Film, plan: TimePlan?, note: String) {
        let film = try Film.open(url)
        let count = film.source.count
        switch preprocess.timing {
        case .off:
            return (film, nil, "Zeitstruktur aus")
        case .duplicates:
            guard let scan = Decimate.cached(filmID: filmID, count: count) else {
                return (film, nil, "Wiederholungs-Messung fehlt, Film ohne Vorverarbeitung")
            }
            let resolved = Decimate.resolve(preprocess.decimateInForce, diffs: scan)
            film.apply(resolved.map)
            return (film, nil, resolved.note)
        case .auto, .manual:
            guard let features = TimeAnalysis.cached(filmID: filmID, count: count) else {
                return (film, nil, "Zeitmessung fehlt, Film ohne Vorverarbeitung")
            }
            let solver = SolverParams.cached(filmID: filmID) ?? .standard
            let plan = TimePlanner.solve(features, fps: features.fps, overrides: preprocess.overrides, force: preprocess.timing == .manual, solver: solver)
            guard plan.accepted else { return (film, plan, "Plan nicht angenommen: \(plan.note)") }
            film.apply(TimePlanner.map(plan, sharp: features.sharp))
            return (film, plan, plan.note)
        }
    }

    /// Writes the export for the 0-based frame `index` and returns its folder.
    @discardableResult
    static func export(film: Film, index requested: Int, title: String, context: FrameExportContext, into root: URL? = nil) throws -> URL {
        let total = film.facts.count
        guard total > 0 else { throw CLIError("Der Film hat keine Bilder.") }
        let index = min(max(requested, 0), total - 1)
        let folder = (root ?? rootFolder).appendingPathComponent("\(safe(title))_\(index + 1)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let main = film.sourceIndex(index)
        let op = film.map?.op(at: index) ?? 0
        let preview = try film.decode(at: index)
        try writePNG(planar: preview.rgb, width: preview.width, height: preview.height, to: folder.appendingPathComponent("vorschau.png"))

        // Raw sources: the five solver inputs for a reconstruction, otherwise the mapped frame ±3.
        var wanted: [Int]
        if op != 0, let first = TimeSolver.firstSource(main: main, op: op) {
            wanted = (0..<TimeSolver.inputs).map { first + $0 }
        } else {
            wanted = Array((main - 3)...(main + 3))
        }
        var sources: [[String: Any]] = []
        var seen = Set<Int>()
        for number in wanted {
            let clamped = min(max(number, 0), film.source.count - 1)
            guard seen.insert(clamped).inserted else { continue }
            let decoded = try film.decodeRaw(source: clamped)
            let name = "quelle_\(clamped).png"
            try writePNG(planar: decoded.rgb, width: decoded.width, height: decoded.height, to: folder.appendingPathComponent(name))
            sources.append([
                "datei": name,
                "quellindex": clamped,
                "zeit_s": film.sourceTime(clamped).seconds,
                "gemappt": clamped == main,
                "o_nummer": op != 0 ? clamped - (wanted.first ?? clamped) + 2 : NSNull(),
            ])
        }

        // The cached upscale, never computed here.
        var upscale: [String: Any] = ["aus_cache": false]
        if let store = context.store, store.contains(index) {
            do {
                let buffer = try store.image(at: index)
                let width = CVPixelBufferGetWidth(buffer)
                let height = CVPixelBufferGetHeight(buffer)
                let planar = try Pixels.planar(buffer)
                try writePNG(planar: planar, width: width, height: height, to: folder.appendingPathComponent("upscale.png"))
                upscale = ["aus_cache": true, "datei": "upscale.png", "breite": width, "hoehe": height]
            } catch {
                upscale = ["aus_cache": false, "hinweis": "Cache lesen fehlgeschlagen: \(error.localizedDescription)"]
            }
        } else if context.store == nil {
            upscale["hinweis"] = "kein Modell-Cache geöffnet"
        } else {
            upscale["hinweis"] = "Bild nicht im Cache, nicht neu gerechnet"
        }

        var info: [String: Any] = [
            "anzeigeindex": index + 1,
            "index0": index,
            "gesamt": total,
            "quellindex": main,
            "op": op == 0 ? "0" : (op == TimeSolver.opS3 ? "S3" : "S4"),
            "op_code": Int(op),
            "rekonstruiert": op != 0,
            "zeit_anzeige_s": film.time(for: index).seconds,
            "zeit_quelle_s": film.sourceTime(main).seconds,
            "film": film.facts.url.path,
            "film_id": context.filmID,
            "quell_bilder_im_film": film.source.count,
            "quell_fps": film.source.fps,
            "anzeige_fps": film.facts.fps,
            "zeitstruktur": context.preprocess.timing.rawValue,
            "solver": ["version": TimeSolver.version, "kappa": (film.map?.solver ?? .standard).kappa, "sigma2": (film.map?.solver ?? .standard).sigma2,
                       "aus_film_gemessen": SolverParams.cached(filmID: context.filmID) != nil] as [String: Any],
            "plan_hash": film.map?.plan ?? NSNull(),
            "cache_tag": context.cacheTag ?? film.map?.cacheTag ?? NSNull(),
            "vorschau": [
                "datei": "vorschau.png",
                "farbe_und_licht": false,
                "farbe_und_licht_aktiv_in_app": context.gradeActive,
                "hinweis": "Eingang nach Zeitstruktur, vor dem Upscaler, ohne Farbe & Licht, 16 Bit RGB, Werte 0…1 wie dekodiert",
            ] as [String: Any],
            "upscale": upscale,
            "quellbilder": sources,
            "modell": context.modelPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? NSNull(),
            "modell_anzeige": context.modelDetail ?? NSNull(),
            "lauf": context.runName ?? NSNull(),
            "schritt": context.step ?? NSNull(),
            "zeitstempel": ISO8601DateFormatter().string(from: Date()),
            "ausgeloest_durch": context.origin,
        ]
        if let plan = context.plan ?? nil {
            info["plan"] = planInfo(plan, main: main, op: op)
        } else {
            info["plan"] = NSNull()
        }
        let data = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: folder.appendingPathComponent("info.json"), options: .atomic)
        return folder
    }

    /// Where the frame sits in the plan: its group start, and whether that group is reconstructed.
    private static func planInfo(_ plan: TimePlan, main: Int, op: UInt8) -> [String: Any] {
        func start(of group: [Int], containing frame: Int) -> Int? {
            var low = 0
            var high = group.count
            while low < high {
                let mid = (low + high) / 2
                if group[mid] <= frame { low = mid + 1 } else { high = mid }
            }
            return low > 0 ? group[low - 1] : nil
        }
        var result: [String: Any] = [
            "hash": plan.hash,
            "angenommen": plan.accepted,
            "gruppen": plan.groups.count,
        ]
        if let group = start(of: plan.groups, containing: main), main <= group + 6 {
            result["gruppenstart"] = group
            result["gruppenstart_o_nummer"] = main - group
            result["gruppe_typ"] = "rekonstruiert"
        } else {
            result["gruppenstart"] = NSNull()
            result["gruppe_typ"] = "keine Gruppe (Originale)"
        }
        if let segment = plan.segments.first(where: { main >= $0.from && main < $0.to }) {
            result["segment"] = ["von": segment.from, "bis": segment.to, "art": segment.kind.rawValue]
        }
        return result
    }

    static func safe(_ name: String) -> String {
        let bad = CharacterSet(charactersIn: "/:\\\0")
        return name.components(separatedBy: bad).joined(separator: "-")
    }

    /// 16-bit PNG of a planar RGB picture.
    static func writePNG(planar rgb: [Float], width: Int, height: Int, to url: URL) throws {
        guard let image = PlaygroundImage.image16(planar: rgb, width: width, height: height),
              let target = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CLIError("PNG \(url.lastPathComponent) lässt sich nicht anlegen") }
        CGImageDestinationAddImage(target, image, nil)
        guard CGImageDestinationFinalize(target) else { throw CLIError("PNG \(url.lastPathComponent) nicht geschrieben") }
    }
}

nonisolated extension CLI {
    static let exportFrameUsage = """
    Whoodeo --export-frame video index|g<group>|h<group> outdir [model.mlxmodel]

    --export-frame writes the diagnosis export of one frame (index as the window shows it, counted
    from 1) into outdir/<film>_<index>/: vorschau.png, quelle_<n>.png, info.json, and upscale.png when
    the given model has the frame in its cache. It uses the film's stored preprocessing, so the mapping
    is the preview's. Nothing is computed on the GPU.
    """

    static func exportFrame(_ args: [String]) throws {
        // `g<group>` / `h<group>`: the S4 / S3 picture of the group that starts at that source frame.
        let groupSpec: (group: Int, op: UInt8)? = {
            guard args.count > 1, let lead = args[1].first, lead == "g" || lead == "h", let group = Int(args[1].dropFirst()) else { return nil }
            return (group, lead == "g" ? TimeSolver.opS4 : TimeSolver.opS3)
        }()
        guard args.count == 3 || args.count == 4, groupSpec != nil || (Int(args[1]) ?? 0) >= 1 else {
            throw CLIError(exportFrameUsage)
        }
        var display = Int(args[1]) ?? 0
        let url = URL(fileURLWithPath: (args[0] as NSString).expandingTildeInPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else { throw CLIError("Die Datei fehlt.") }
        let entry = FilmLibrary.load().films.first { $0.path == url.path }
        let id = entry?.id ?? FrameStore.filmDigest(of: url)
        let preprocess = entry?.preprocess ?? .none
        let (film, plan, note) = try FrameExport.resolve(url: url, filmID: id, preprocess: preprocess)
        fputs("Zeitstruktur \(preprocess.timing.rawValue): \(note)\n", stdout)
        fputs("Film \(film.facts.count) Bilder (Quelle \(film.source.count))\n", stdout)
        if let groupSpec {
            let main = groupSpec.group + (groupSpec.op == TimeSolver.opS4 ? 5 : 3)
            guard let found = (0..<film.facts.count).first(where: { film.sourceIndex($0) == main && film.map?.op(at: $0) == groupSpec.op }) else {
                throw CLIError("Gruppe \(groupSpec.group) steht nicht als Rekonstruktion im Film.")
            }
            display = found + 1
        }
        var context = FrameExportContext(filmID: id, plan: plan, preprocess: preprocess, origin: "cli")
        if args.count == 4 {
            let package = URL(fileURLWithPath: (args[3] as NSString).expandingTildeInPath)
            let model = LibraryModel.read(package)
            let codec: SourceCodec? = model.conditionInput ? wantedCodec(.auto, film: film.facts.codec) : nil
            let condition = model.conditionInput ? codec?.cacheTag : nil
            let tag = [condition, film.map?.cacheTag].compactMap { $0 }.joined(separator: "-")
            let store = try FrameStore.open(filmID: id, model: package, count: film.facts.count, condition: tag.isEmpty ? nil : tag)
            context.store = store
            context.modelPath = package.path
            context.modelDetail = model.detail
            context.runName = model.runName
            context.step = model.step
            context.cacheTag = tag.isEmpty ? nil : tag
        }
        let outRoot = URL(fileURLWithPath: (args[2] as NSString).expandingTildeInPath, isDirectory: true)
        let folder = try FrameExport.export(
            film: film, index: display - 1, title: url.deletingPathExtension().lastPathComponent,
            context: context, into: outRoot
        )
        context.store?.flush()
        let sourceIndex = film.sourceIndex(display - 1)
        let op = film.map?.op(at: display - 1) ?? 0
        fputs("Bild \(display) → Quellbild \(sourceIndex), op \(op == 0 ? "0" : (op == TimeSolver.opS3 ? "S3" : "S4"))\n", stdout)
        fputs("\(folder.path)\n", stdout)
    }
}
