import Foundation

@main
enum TrainCLI {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        do {
            try run(args)
        } catch {
            fputs("whoodeo-train: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static let commands = """
        whoodeo-train — Training und Modelldatenbank (Swift/MLX, nicht das Python-Kommando)

        Befehle:
          start [--name <name>] [Optionen]  neuer Lauf, from scratch oder Fine-Tune mit --from <ckpt-id>
          resume <run-id|name> [Optionen]   Lauf ab dem letzten Checkpoint fortsetzen
          tree                             Stammbaum der Läufe und Checkpoints
          models                           Modelle, die die App anzeigt
          checkpoint show|hide <ckpt-id>   Sichtbarkeit in der App umschalten
          checkpoint delete <ckpt-id> --yes Checkpoint und Dateien löschen
          checkpoint now [run-id|name]    Checkpoint im laufenden Training anfordern
          stop [run-id|name]              Lauf nach dem aktuellen Schritt pausieren
          val-preview <ckpt-id>           4 Val-Beispiele für einen Checkpoint schreiben
          run delete <run-id|name> --yes   Lauf, seine Checkpoints und Dateien löschen
          status                           wie tree
          help                             diese Hilfe

        start:
          --name <name>      Lauf-Name und Ordner unter Models/, leer = train-YYYYMMDD-HHMMSS
          --from <ckpt-id>   Fine-Tune von diesem Checkpoint (neuer Stammbaum-Ast)
          --spynet-lr <x>    absolute SPyNet-Lernrate statt --spynet-lr-scale

        Architektur-Optionen gelten nur für einen neuen Lauf. Bei --from und resume kommen
        sie aus dem Eltern-Checkpoint bzw. dem Lauf. Alle anderen Optionen gehen auch bei resume.
        Ein Loss-Gewicht 0 schaltet den Term ganz ab (nicht berechnet); sind oklab, ms-ssim und vgg 0,
        entfällt der VAE-Decode im Trainingsschritt. Mindestens ein Gewicht muss größer als 0 sein.
        """

    /// Command overview plus the parameter sections, generated from `TrainParameterSpec`.
    static var usage: String {
        var text = commands + "\n"
        for section in TrainParameterSection.allCases {
            let specs = TrainParameterSpec.specs(in: section)
            let suffix = specs.first?.scope == .newOnly ? " (nur neuer Lauf)" : ""
            text += "\n\(section.title)\(suffix):\n"
            for spec in specs {
                let flag: String
                switch spec.kind {
                case .int: flag = "--\(spec.flag) <n>"
                case .float: flag = "--\(spec.flag) <x>"
                case .bool: flag = "--\(spec.flag) | --no-\(spec.flag)"
                case .choice(let options): flag = "--\(spec.flag) \(options.joined(separator: "|"))"
                case .list: flag = "--\(spec.flag) <a,b>"
                }
                let padded = flag.count < 34 ? flag.padding(toLength: 34, withPad: " ", startingAt: 0) : flag + "  "
                let fallback = spec.defaultText.isEmpty ? "leer" : spec.defaultText
                text += "  \(padded)\(spec.label): \(spec.help) [\(fallback)]\n"
            }
        }
        return text
    }

    /// Reads every `TrainParameterSpec` flag. With `locked`, architecture flags are an error.
    static func applyParameters(_ flags: inout FlagParser, to config: inout TrainConfig, locked: String?) throws {
        for spec in TrainParameterSpec.all {
            var text: String?
            if case .bool = spec.kind {
                if flags.flag(spec.flag) { text = "true" }
                if flags.flag("no-" + spec.flag) { text = "false" }
            } else {
                text = flags.string(spec.flag)
            }
            guard let text else { continue }
            if let locked, spec.scope == .newOnly {
                throw CLIError("--\(spec.flag) gehört zur Architektur und kommt \(locked); nur bei einem neuen Lauf änderbar")
            }
            try spec.write(&config, text)
        }
    }

    static func run(_ args: [String]) throws {
        let command = args.first ?? "tree"
        let rest = Array(args.dropFirst())
        if (command == "start" || command == "resume"), rest.contains("--help") || rest.contains("-h") {
            fputs(usage + "\n", stdout)
            return
        }
        switch command {
        case "start":
            try start(rest)
        case "resume":
            try resume(rest)
        case "tree", "status":
            try printTree()
        case "models":
            try printModels()
        case "checkpoint":
            try checkpoint(rest)
        case "stop":
            try stopCommand(rest)
        case "val-preview":
            try valPreview(rest)
        case "run":
            try runCommand(rest)
        case "-h", "--help", "help":
            fputs(usage + "\n", stdout)
        default:
            fputs("unbekanntes Kommando: \(command)\n\(usage)\n", stderr)
            exit(2)
        }
    }

    static func start(_ args: [String]) throws {
        var flags = FlagParser(args)
        let name = flags.string("name") ?? TrainConfig.automaticName()
        let parent = flags.string("from")
        let db = try WhoodeoDB.openDefault()
        var config: TrainConfig
        if let parent {
            guard let id = Int(parent) else { throw CLIError("--from braucht eine Checkpoint-ID, nicht „\(parent)“") }
            guard let checkpoint = try db.checkpoint(id: id) else {
                throw CLIError("Eltern-Checkpoint \(id) nicht gefunden")
            }
            guard let run = try db.run(id: checkpoint.runID) else {
                throw CLIError("Lauf \(checkpoint.runID) von Checkpoint \(id) nicht gefunden")
            }
            config = TrainConfig.fineTuneBase(parent: run, checkpointID: id, name: name)
            try applyParameters(&flags, to: &config, locked: "vom Eltern-Checkpoint \(id)")
        } else {
            config = TrainConfig(name: name)
            try applyParameters(&flags, to: &config, locked: nil)
        }
        if let spyLR = flags.float("spynet-lr") {
            config.spynetLrScale = spyLR / config.learningRate
        }
        try flags.finish()
        if name.isEmpty || name.contains("/") || name.hasPrefix(".") {
            throw CLIError("ungültiger Lauf-Name: „\(name)“")
        }
        if try db.run(idOrName: name) != nil {
            throw CLIError("Lauf „\(name)“ existiert schon; für weitere Schritte: resume \(name)")
        }
        config = try config.validated()
        let result = try TrainLoop.run(db: db, config: config)
        fputs("fertig  run \(result.runID)  status \(result.status)  Schritt \(result.lastStep)\n", stderr)
    }

    static func resume(_ args: [String]) throws {
        guard let name = args.first, !name.hasPrefix("--") else {
            throw CLIError("resume braucht zuerst eine Lauf-ID oder einen Namen")
        }
        var flags = FlagParser(Array(args.dropFirst()))
        let db = try WhoodeoDB.openDefault()
        guard let run = try db.run(idOrName: name) else {
            throw CLIError("Lauf nicht gefunden: \(name)")
        }
        var config = TrainConfig(run: run)
        try applyParameters(&flags, to: &config, locked: "aus dem Lauf \(run.name)")
        if let spyLR = flags.float("spynet-lr") {
            config.spynetLrScale = spyLR / config.learningRate
        }
        try flags.finish()
        config = try config.validated()
        let result = try TrainLoop.run(db: db, config: config, resumeID: run.id)
        fputs("fertig  run \(result.runID)  status \(result.status)  Schritt \(result.lastStep)\n", stderr)
    }

    static func config(from run: RunRecord) throws -> TrainConfig {
        TrainConfig(run: run)
    }

    static func checkpoint(_ args: [String]) throws {
        guard let action = args.first else {
            throw CLIError("checkpoint show|hide|delete <ckpt-id>  oder  checkpoint now [run]")
        }
        if action == "now" {
            try checkpointNow(Array(args.dropFirst()))
            return
        }
        guard args.count >= 2, let id = Int(args[1]) else {
            throw CLIError("checkpoint show|hide|delete <ckpt-id>  oder  checkpoint now [run]")
        }
        let db = try WhoodeoDB.openDefault()
        guard let record = try db.checkpoint(id: id) else {
            throw CLIError("Checkpoint \(id) nicht gefunden")
        }
        switch action {
        case "show":
            try db.setCheckpointVisible(id: id, visible: true)
            fputs("Checkpoint \(id) (\(record.relPath)) ist in der App sichtbar\n", stdout)
        case "hide":
            try db.setCheckpointVisible(id: id, visible: false)
            fputs("Checkpoint \(id) (\(record.relPath)) ist in der App ausgeblendet\n", stdout)
        case "delete":
            guard args.contains("--yes") else {
                throw CLIError("checkpoint delete braucht --yes (löscht auch die Dateien)")
            }
            try db.deleteCheckpoint(id: id, removeFiles: true)
            fputs("Checkpoint \(id) (\(record.relPath)) gelöscht\n", stdout)
        default:
            throw CLIError("checkpoint show|hide|delete <ckpt-id>")
        }
    }

    static func checkpointNow(_ args: [String]) throws {
        let db = try WhoodeoDB.openDefault()
        let run = try runningRun(from: args, db: db)
        try db.requestCheckpoint(runID: run.id)
        fputs("Checkpoint angefordert für \(run.name) (id \(run.id))\n", stdout)
    }

    static func stopCommand(_ args: [String]) throws {
        let db = try WhoodeoDB.openDefault()
        let run = try runningRun(from: args, db: db)
        try db.requestStop(runID: run.id)
        fputs("Stop angefordert für \(run.name) (id \(run.id))\n", stdout)
    }

    private static func runningRun(from args: [String], db: WhoodeoDB) throws -> RunRecord {
        let run: RunRecord
        if let name = args.first, name != "--yes" {
            guard let found = try db.run(idOrName: name) else {
                throw CLIError("Lauf nicht gefunden: \(name)")
            }
            run = found
        } else {
            let running = try db.runningRuns()
            if running.isEmpty {
                throw CLIError("kein laufendes Training; übergib eine Lauf-ID oder einen Namen")
            }
            if running.count > 1 {
                let names = running.map { "\($0.name) (id \($0.id))" }.joined(separator: ", ")
                throw CLIError("mehrere Läufe laufen, bitte einen nennen: \(names)")
            }
            run = running[0]
        }
        if run.status != "running" {
            throw CLIError("Lauf \(run.name) läuft nicht (Status \(run.status))")
        }
        return run
    }

    static func valPreview(_ args: [String]) throws {
        guard let text = args.first, let id = Int(text) else {
            throw CLIError("val-preview braucht eine Checkpoint-ID")
        }
        let db = try WhoodeoDB.openDefault()
        guard let checkpoint = try db.checkpoint(id: id) else {
            throw CLIError("Checkpoint \(id) nicht gefunden")
        }
        guard let run = try db.run(id: checkpoint.runID) else {
            throw CLIError("Lauf \(checkpoint.runID) nicht gefunden")
        }
        var config = try config(from: run)
        switch config.dtype.lowercased() {
        case "fp16", "float16":
            config.dtype = "fp32"
        default:
            break
        }
        config = try config.validated()
        let package = WhoodeoPaths.modelsDirectory.appendingPathComponent(checkpoint.relPath)
        fputs("val-preview \(run.name)  Schritt \(checkpoint.step)  \(package.lastPathComponent)\n", stderr)
        let model = try TrainModel.load(from: package, train: config)
        model.trainSpynet = false
        let vae = try MLXVAEEncoder()
        let lossFn = try TrainLoss(config: config)
        let data = try TrainDataset(db: db, split: "val", config: config, seed: config.seed)
        try ValPreview.write(runName: run.name, step: checkpoint.step, model: model, vae: vae, data: data, lossFn: lossFn)
        fputs("geschrieben unter Models/\(run.name)/val/step-\(String(format: "%06d", checkpoint.step))\n", stdout)
    }

    static func runCommand(_ args: [String]) throws {
        guard args.first == "delete", args.count >= 2 else {
            throw CLIError("run delete <run-id|name> --yes")
        }
        guard args.contains("--yes") else {
            throw CLIError("run delete braucht --yes (löscht Checkpoints und Dateien)")
        }
        let db = try WhoodeoDB.openDefault()
        guard let run = try db.run(idOrName: args[1]) else {
            throw CLIError("Lauf nicht gefunden: \(args[1])")
        }
        try db.deleteRun(id: run.id, removeFiles: true)
        fputs("Lauf \(run.name) (id \(run.id)) gelöscht\n", stdout)
    }

    static func printTree() throws {
        let db = try WhoodeoDB.openDefault()
        let counts = try db.counts()
        let tree = try db.runTree()
        fputs("Datenbank \(WhoodeoPaths.databaseURL.path)\n", stdout)
        if let root = try db.datasetRoot() {
            fputs("Trainingsdaten \(root.path)\n", stdout)
        }
        fputs("Cache \(try db.cacheRoot().path)\n", stdout)
        fputs(
            "sources \(counts.sources) (train \(counts.train), val \(counts.val)) · masters \(counts.masters) · variants \(counts.variants)\n",
            stdout
        )
        fputs(
            "runs \(counts.runs) · checkpoints \(counts.checkpoints) · cache-Dateien \(counts.cacheFiles)\n",
            stdout
        )
        if tree.isEmpty {
            fputs("keine Läufe\n", stdout)
            return
        }
        fputs("\n", stdout)
        for node in tree {
            try writeNode(node, indent: "", db: db)
        }
    }

    static func writeNode(_ node: RunTreeNode, indent: String, db: WhoodeoDB) throws {
        let run = node.run
        var line = "\(indent)\(run.name)  id \(run.id)  [\(run.status)]  \(run.channels) Kanäle  \(run.dtype)"
        if let crop = run.cropSize { line += "  crop \(crop)" }
        if run.parentCheckpointID == nil {
            line += "  from scratch"
        } else if let parent = run.parentCheckpointID {
            line += "  Kind von checkpoint \(parent)"
        }
        if let notes = run.notes, !notes.isEmpty {
            line += "  — \(notes)"
        }
        fputs(line + "\n", stdout)
        fputs("\(indent)  \(TrainConfig.spynetLabel(from: run.configJSON))\n", stdout)
        if let train = try? db.lastMetrics(runID: run.id, split: "train"), let total = train["total"] {
            let step = train["_step"].map { Int($0) } ?? 0
            fputs(String(format: "%@  train Schritt %d  loss %.4f\n", indent, step, total), stdout)
        }
        if let val = try? db.lastMetrics(runID: run.id, split: "val"), let total = val["total"] {
            let step = val["_step"].map { Int($0) } ?? 0
            fputs(String(format: "%@  val Schritt %d  loss %.4f\n", indent, step, total), stdout)
        }
        for checkpoint in node.checkpoints {
            var row = "\(indent)  checkpoint \(checkpoint.id)  Schritt \(checkpoint.step)  \(checkpoint.relPath)  \(checkpoint.exportDtype)"
            if checkpoint.visibleInApp {
                row += "  sichtbar"
            } else {
                row += "  (ausgeblendet)"
            }
            fputs(row + "\n", stdout)
        }
        if node.cacheCount > 0 {
            fputs("\(indent)  cache-Ordner: \(node.cacheCount)\n", stdout)
        }
        for child in node.children {
            try writeNode(child, indent: indent + "  ", db: db)
        }
    }

    static func printModels() throws {
        _ = try WhoodeoDB.openDefault()
        let models = ModelLibrary.models()
        if models.isEmpty {
            fputs("keine Modelle\n", stdout)
            return
        }
        for model in models {
            var line = "\(WhoodeoDB.relativeModelPath(of: model.url))\n  \(model.title)\n  \(model.detail)"
            if let id = model.checkpointID { line += "\n  checkpoint-id \(id)" }
            fputs(line + "\n", stdout)
        }
    }
}

/// Tiny flag parser: `--name value`, `--hidden`, and trailing positionals.
struct FlagParser {
    private var leftover: [String]
    private var used: Set<String> = []
    private var setNames: Set<String> = []

    init(_ args: [String]) { leftover = args }

    mutating func string(_ name: String) -> String? {
        used.insert(name)
        guard let index = leftover.firstIndex(of: "--" + name), index + 1 < leftover.count else { return nil }
        let value = leftover[index + 1]
        leftover.removeSubrange(index...index + 1)
        setNames.insert(name)
        return value
    }

    mutating func int(_ name: String) -> Int? {
        guard let text = string(name) else { return nil }
        return Int(text)
    }

    mutating func uint64(_ name: String) -> UInt64? {
        guard let text = string(name) else { return nil }
        return UInt64(text)
    }

    mutating func float(_ name: String) -> Float? {
        guard let text = string(name) else { return nil }
        return Float(text)
    }

    mutating func flag(_ name: String) -> Bool {
        used.insert(name)
        if let index = leftover.firstIndex(of: "--" + name) {
            leftover.remove(at: index)
            setNames.insert(name)
            return true
        }
        return false
    }

    func wasSet(_ name: String) -> Bool { setNames.contains(name) }

    mutating func positional() -> [String] {
        leftover.filter { !$0.hasPrefix("--") }
    }

    func finish() throws {
        let unknown = leftover.filter { $0.hasPrefix("--") }
        if !unknown.isEmpty {
            throw CLIError("unbekannte Option: \(unknown.joined(separator: " "))")
        }
    }
}
