import Foundation

/// Form group in the app and section in `whoodeo-train --help`.
nonisolated enum TrainParameterSection: String, CaseIterable, Sendable {
    case architecture, optimizer, loss, data

    var title: String {
        switch self {
        case .architecture: "Architektur"
        case .optimizer: "Optimierer & Schedule"
        case .loss: "Loss-Gewichte"
        case .data: "Daten & Laufzeit"
        }
    }
}

/// `.newOnly` belongs to the weights and is fixed after a run starts (fine-tune and resume take it over).
/// `.always` can change for every run, also when resuming.
nonisolated enum TrainParameterScope: Sendable {
    case newOnly, always
}

nonisolated enum TrainParameterKind: Sendable {
    case int(ClosedRange<Int>)
    case float(ClosedRange<Double>)
    case bool
    case choice([String])
    /// Comma separated list, shown as a text field.
    case list
}

/// One training parameter. The single source for the app form, the CLI flags and `--help`,
/// and the keys in `runs.config_json`. Values travel as text, so form and command line share one parser.
nonisolated struct TrainParameterSpec: Identifiable, @unchecked Sendable {
    let key: String
    let flag: String
    let section: TrainParameterSection
    let scope: TrainParameterScope
    let label: String
    let help: String
    let kind: TrainParameterKind
    let read: (TrainConfig) -> String
    let write: (inout TrainConfig, String) throws -> Void
    let json: (TrainConfig) -> Any
    let readJSON: (inout TrainConfig, Any) -> Void

    var id: String { key }

    /// Text of the default value.
    var defaultText: String { read(TrainConfig(name: "")) }

    var isLossWeight: Bool { section == .loss }

    static func spec(key: String) -> TrainParameterSpec? { all.first { $0.key == key } }
    static func specs(in section: TrainParameterSection) -> [TrainParameterSpec] { all.filter { $0.section == section } }

    static let all: [TrainParameterSpec] = [
        // Architektur
        int("channels", "channels", .architecture, .newOnly, "Kanäle",
            "Breite der Feature-Maps", \.channels, 8...512),
        intList("encoder_blocks", "encoder-blocks", .architecture, .newOnly, "Encoder-Blöcke",
                "Residual-Blöcke je Encoder-Stufe, z. B. 2,4", \.encoderBlocks),
        int("blocks", "blocks", .architecture, .newOnly, "Mittlere Blöcke",
            "Residual-Blöcke in der Mitte", \.midBlocks, 1...64),
        choice("dtype", "dtype", .architecture, .newOnly, "Rechen-Datentyp",
               "fp32 oder bf16 im Training", \.dtype, ["fp32", "bf16"]),

        // Optimierer & Schedule
        int("steps", "steps", .optimizer, .always, "Schritte",
            "Optimierer-Schritte; beim Fortsetzen zusätzlich zum letzten Checkpoint", \.steps, 1...10_000_000),
        float("lr", "lr", .optimizer, .always, "Lernrate",
              "Start-Lernrate (AdamW)", \.learningRate, 1e-8...1),
        choice("schedule", "schedule", .optimizer, .always, "LR-Verlauf",
               "constant oder cosine (fällt über die Schritte auf 0)", \.schedule, ["constant", "cosine"]),
        int("batch", "batch", .optimizer, .always, "Batch",
            "Beispiele pro Mikro-Batch", \.batch, 1...64),
        int("grad_accum", "grad-accum", .optimizer, .always, "Gradient-Akkumulation",
            "Mikro-Batches pro Optimierer-Schritt", \.gradAccum, 1...256),
        bool("train_spynet", "train-spynet", .optimizer, .always, "SPyNet mittrainieren",
             "Sonst bleibt der optische Fluss eingefroren", \.trainSpynet),
        float("spynet_lr_scale", "spynet-lr-scale", .optimizer, .always, "SPyNet-LR-Faktor",
              "SPyNet lernt mit Lernrate × Faktor", \.spynetLrScale, 1e-6...10),

        // Loss-Gewichte
        float("w_latent", "w-latent", .loss, .always, "Latent L1",
              "L1 auf den VAE-Latents, braucht keinen Decode", \.latentWeight, 0...1000),
        float("w_oklab", "w-oklab", .loss, .always, "Oklab",
              "L1 im Oklab-Farbraum auf dem decodierten Bild", \.oklabWeight, 0...1000),
        float("w_ms_ssim", "w-ms-ssim", .loss, .always, "MS-SSIM",
              "Struktur auf der Oklab-Helligkeit, 5 Skalen", \.msSSIMWeight, 0...1000),
        float("w_vgg", "w-vgg", .loss, .always, "VGG",
              "Wahrnehmung über VGG19 relu2_2 und relu3_4 (Faktor auf die Layer-Gewichte)", \.vggWeight, 0...1000),

        // Daten & Laufzeit
        TrainParameterSpec(
            key: "crop_size", flag: "crop", section: .data, scope: .always, label: "Crop",
            help: "Seitenlänge des Master-Ausschnitts in Pixeln, 0 = Vollbild, sonst durch 32 teilbar",
            kind: .int(0...4096),
            read: { String($0.cropSize ?? 0) },
            write: { config, text in
                guard let value = Int(text.trimmingCharacters(in: .whitespaces)), value >= 0 else {
                    throw CLIError("--crop: ganze Zahl ab 0 erwartet, nicht „\(text)“")
                }
                config.cropSize = value == 0 ? nil : value
            },
            json: { $0.cropSize ?? 0 },
            readJSON: { config, any in
                if let value = intValue(any) { config.cropSize = value == 0 ? nil : value }
            }
        ),
        stringList("rungs", "rungs", .data, .always, "Stufen",
                   "Eingangs-Auflösungen, z. B. 480,720", \.rungs),
        stringList("presets", "presets", .data, .always, "Presets",
                   "Codec-Presets, leer = alle", \.presets),
        float("codec_unknown", "codec-unknown", .data, .always, "Codec unbekannt",
              "Anteil der Trainingsbeispiele mit Codec-Bedingung „unknown“", \.codecUnknown, 0...1),
        int("workers", "workers", .data, .always, "Prefetch-Threads",
            "0 = synchron ohne Hintergrund-Queue", \.workers, 0...32),
        TrainParameterSpec(
            key: "seed", flag: "seed", section: .data, scope: .always, label: "Seed",
            help: "Startwert für Zufall und Datenreihenfolge", kind: .int(0...Int.max),
            read: { String($0.seed) },
            write: { config, text in
                guard let value = UInt64(text.trimmingCharacters(in: .whitespaces)) else {
                    throw CLIError("--seed: ganze Zahl ab 0 erwartet, nicht „\(text)“")
                }
                config.seed = value
            },
            json: { Int(clamping: $0.seed) },
            readJSON: { config, any in
                if let value = intValue(any), value >= 0 { config.seed = UInt64(value) }
            }
        ),
        int("val_every", "val-every", .data, .always, "Validierung alle",
            "Schritte zwischen Validierungen mit Vorschau, 0 = aus", \.valEvery, 0...1_000_000),
        int("val_count", "val-count", .data, .always, "Val-Beispiele",
            "Beispiele pro Validierung", \.valCount, 1...1000),
        int("save_every", "save-every", .data, .always, "Checkpoint alle",
            "Schritte zwischen Checkpoints, 0 = nur am Ende", \.saveEvery, 0...1_000_000),
        int("log_every", "log-every", .data, .always, "Log alle",
            "Schritte zwischen Zeilen im Log", \.logEvery, 1...100_000),
        bool("hidden", "hidden", .data, .always, "Checkpoints ausblenden",
             "Neue Checkpoints erscheinen nicht in der Modell-Liste der App", \.hidden),
    ]

    // MARK: Builders

    private static func int(
        _ key: String, _ flag: String, _ section: TrainParameterSection, _ scope: TrainParameterScope,
        _ label: String, _ help: String, _ path: WritableKeyPath<TrainConfig, Int>, _ range: ClosedRange<Int>
    ) -> TrainParameterSpec {
        TrainParameterSpec(
            key: key, flag: flag, section: section, scope: scope, label: label, help: help, kind: .int(range),
            read: { String($0[keyPath: path]) },
            write: { config, text in
                guard let value = Int(text.trimmingCharacters(in: .whitespaces)), range.contains(value) else {
                    throw CLIError("--\(flag): ganze Zahl von \(range.lowerBound) bis \(range.upperBound) erwartet, nicht „\(text)“")
                }
                config[keyPath: path] = value
            },
            json: { $0[keyPath: path] },
            readJSON: { config, any in
                if let value = intValue(any) { config[keyPath: path] = value }
            }
        )
    }

    private static func float(
        _ key: String, _ flag: String, _ section: TrainParameterSection, _ scope: TrainParameterScope,
        _ label: String, _ help: String, _ path: WritableKeyPath<TrainConfig, Float>, _ range: ClosedRange<Double>
    ) -> TrainParameterSpec {
        TrainParameterSpec(
            key: key, flag: flag, section: section, scope: scope, label: label, help: help, kind: .float(range),
            read: { floatText($0[keyPath: path]) },
            write: { config, text in
                let clean = text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
                guard let value = Double(clean), value.isFinite, range.contains(value) else {
                    throw CLIError("--\(flag): Zahl von \(floatText(Float(range.lowerBound))) bis \(floatText(Float(range.upperBound))) erwartet, nicht „\(text)“")
                }
                config[keyPath: path] = Float(value)
            },
            // Through text, so 1e-4 is stored as 0.0001 and not as 9.9999997e-05.
            json: { NSDecimalNumber(string: floatText($0[keyPath: path])) },
            readJSON: { config, any in
                if let value = doubleValue(any) { config[keyPath: path] = Float(value) }
            }
        )
    }

    private static func bool(
        _ key: String, _ flag: String, _ section: TrainParameterSection, _ scope: TrainParameterScope,
        _ label: String, _ help: String, _ path: WritableKeyPath<TrainConfig, Bool>
    ) -> TrainParameterSpec {
        TrainParameterSpec(
            key: key, flag: flag, section: section, scope: scope, label: label, help: help, kind: .bool,
            read: { $0[keyPath: path] ? "true" : "false" },
            write: { config, text in
                switch text.lowercased() {
                case "true", "1", "yes", "ja", "on": config[keyPath: path] = true
                case "false", "0", "no", "nein", "off": config[keyPath: path] = false
                default: throw CLIError("--\(flag): true oder false erwartet, nicht „\(text)“")
                }
            },
            json: { $0[keyPath: path] },
            readJSON: { config, any in
                if let value = any as? Bool { config[keyPath: path] = value }
                else if let value = intValue(any) { config[keyPath: path] = value != 0 }
            }
        )
    }

    private static func choice(
        _ key: String, _ flag: String, _ section: TrainParameterSection, _ scope: TrainParameterScope,
        _ label: String, _ help: String, _ path: WritableKeyPath<TrainConfig, String>, _ options: [String]
    ) -> TrainParameterSpec {
        TrainParameterSpec(
            key: key, flag: flag, section: section, scope: scope, label: label, help: help, kind: .choice(options),
            read: { $0[keyPath: path] },
            write: { config, text in
                let value = text.trimmingCharacters(in: .whitespaces).lowercased()
                guard options.contains(value) else {
                    throw CLIError("--\(flag): \(options.joined(separator: " oder ")) erwartet, nicht „\(text)“")
                }
                config[keyPath: path] = value
            },
            json: { $0[keyPath: path] },
            readJSON: { config, any in
                if let value = any as? String { config[keyPath: path] = value }
            }
        )
    }

    private static func intList(
        _ key: String, _ flag: String, _ section: TrainParameterSection, _ scope: TrainParameterScope,
        _ label: String, _ help: String, _ path: WritableKeyPath<TrainConfig, [Int]>
    ) -> TrainParameterSpec {
        TrainParameterSpec(
            key: key, flag: flag, section: section, scope: scope, label: label, help: help, kind: .list,
            read: { $0[keyPath: path].map(String.init).joined(separator: ",") },
            write: { config, text in
                let parts = TrainConfig.intList(text)
                let values = parts.compactMap(Int.init)
                guard !values.isEmpty, values.count == parts.count, values.allSatisfy({ $0 >= 1 }) else {
                    throw CLIError("--\(flag): Liste positiver ganzer Zahlen erwartet, nicht „\(text)“")
                }
                config[keyPath: path] = values
            },
            json: { $0[keyPath: path] },
            readJSON: { config, any in
                if let values = any as? [Int], !values.isEmpty { config[keyPath: path] = values }
            }
        )
    }

    private static func stringList(
        _ key: String, _ flag: String, _ section: TrainParameterSection, _ scope: TrainParameterScope,
        _ label: String, _ help: String, _ path: WritableKeyPath<TrainConfig, [String]>
    ) -> TrainParameterSpec {
        TrainParameterSpec(
            key: key, flag: flag, section: section, scope: scope, label: label, help: help, kind: .list,
            read: { $0[keyPath: path].joined(separator: ",") },
            write: { config, text in config[keyPath: path] = TrainConfig.intList(text) },
            json: { $0[keyPath: path] },
            readJSON: { config, any in
                if let values = any as? [String] { config[keyPath: path] = values }
            }
        )
    }

    static func floatText(_ value: Float) -> String {
        if value == value.rounded(), abs(value) < 1e6 { return String(Int(value)) }
        return "\(value)"
    }

    static func intValue(_ any: Any) -> Int? {
        if let value = any as? Int { return value }
        if let value = any as? Double, value == value.rounded() { return Int(value) }
        return nil
    }

    static func doubleValue(_ any: Any) -> Double? {
        if let value = any as? Double { return value }
        if let value = any as? Int { return Double(value) }
        return nil
    }
}

// MARK: - TrainConfig binding

nonisolated extension TrainConfig {
    /// Text values of every parameter, keyed by `TrainParameterSpec.key`.
    var parameterValues: [String: String] {
        Dictionary(uniqueKeysWithValues: TrainParameterSpec.all.map { ($0.key, $0.read(self)) })
    }

    /// Applies text values (form or command line). Unknown keys are ignored.
    mutating func apply(_ values: [String: String], scope: TrainParameterScope? = nil) throws {
        for spec in TrainParameterSpec.all {
            if let scope, spec.scope != scope { continue }
            if let text = values[spec.key] { try spec.write(&self, text) }
        }
    }

    /// Takes over the `.newOnly` parameters from the parent (fine-tune) or the run itself (resume).
    mutating func inheritArchitecture(from other: TrainConfig) {
        for spec in TrainParameterSpec.all where spec.scope == .newOnly {
            try? spec.write(&self, spec.read(other))
        }
    }

    /// Config of an existing run: table columns first, then every key in `config_json`.
    /// Old runs without the newer keys keep the defaults.
    init(run: RunRecord) {
        self.init(name: run.name)
        channels = run.channels
        dtype = run.dtype
        cropSize = run.cropSize
        parentCheckpointID = run.parentCheckpointID
        if let data = run.configJSON.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            for spec in TrainParameterSpec.all {
                if let value = object[spec.key] { spec.readJSON(&self, value) }
            }
        }
    }

    /// Architecture to fine-tune from: the parent run, with an imported fp16 export trained on in fp32.
    static func fineTuneBase(parent run: RunRecord, checkpointID: Int, name: String) -> TrainConfig {
        var parent = TrainConfig(run: run)
        if parent.dtype != "bf16" { parent.dtype = "fp32" }
        var config = parent
        config.name = name
        config.parentCheckpointID = checkpointID
        config.inheritArchitecture(from: parent)
        return config
    }

    /// `train-YYYYMMDD-HHMMSS` in local time, the same pattern the library parses.
    static func automaticName(date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "'train-'yyyyMMdd-HHmmss"
        return formatter.string(from: date)
    }

    var pixelLossActive: Bool { oklabWeight > 0 || msSSIMWeight > 0 || vggWeight > 0 }

    /// Short German list of the active loss terms, for the log and the app.
    var activeLossText: String {
        let parts: [(String, Float)] = [
            ("latent", latentWeight), ("oklab", oklabWeight), ("ms-ssim", msSSIMWeight), ("vgg", vggWeight),
        ]
        let active = parts.filter { $0.1 > 0 }.map { "\($0.0) \(TrainParameterSpec.floatText($0.1))" }
        return active.isEmpty ? "keine" : active.joined(separator: ", ")
    }
}
