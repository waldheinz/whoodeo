import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// On-disk val triples (`low` / `pred` / `orig` PNGs plus `meta.json`) under
/// `Models/<run>/val/step-XXXXXX/`. Written during validation and by `whoodeo-train val-preview`.
enum ValPreview {
    static let sampleCount = 4

    struct Meta: Codable, Equatable, Sendable {
        var step: Int
        var run: String
        var samples: [Sample]
    }

    struct Sample: Codable, Equatable, Identifiable, Sendable {
        var index: Int
        var name: String
        var frame: Int
        var rung: String
        var preset: String
        var psnr: Double?
        var ssim: Double?

        var id: Int { index }

        var caption: String {
            var parts = [name]
            if !preset.isEmpty { parts.append(preset) }
            if !rung.isEmpty { parts.append("rung \(rung)") }
            parts.append("Bild \(frame + 1)")
            if let psnr { parts.append(String(format: "PSNR %.2f", psnr)) }
            if let ssim { parts.append(String(format: "SSIM %.3f", ssim)) }
            return parts.joined(separator: " · ")
        }
    }

    struct Page: Equatable, Sendable {
        var meta: Meta
        var folder: URL

        func url(sample: Int, kind: Kind) -> URL {
            folder.appendingPathComponent(String(format: "sample-%02d-%@.png", sample, kind.rawValue))
        }
    }

    enum Kind: String, CaseIterable, Sendable {
        case low, pred, orig

        var title: String {
            switch self {
            case .low: "Eingabe"
            case .pred: "Modell"
            case .orig: "Original"
            }
        }
    }

    static func directory(runName: String, step: Int) -> URL {
        WhoodeoPaths.modelsDirectory
            .appendingPathComponent(runName, isDirectory: true)
            .appendingPathComponent("val", isDirectory: true)
            .appendingPathComponent(String(format: "step-%06d", step), isDirectory: true)
    }

    static func steps(forRun name: String) -> [Int] {
        let root = WhoodeoPaths.modelsDirectory
            .appendingPathComponent(name, isDirectory: true)
            .appendingPathComponent("val", isDirectory: true)
        let items = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        var found: [Int] = []
        for item in items {
            guard item.hasPrefix("step-"), let step = Int(item.dropFirst(5)) else { continue }
            let meta = root.appendingPathComponent(item, isDirectory: true).appendingPathComponent("meta.json")
            if FileManager.default.fileExists(atPath: meta.path) {
                found.append(step)
            }
        }
        return found.sorted()
    }

    static func load(runName: String, step: Int) -> Page? {
        let folder = directory(runName: runName, step: step)
        let metaURL = folder.appendingPathComponent("meta.json")
        guard let data = try? Data(contentsOf: metaURL),
              let meta = try? JSONDecoder().decode(Meta.self, from: data)
        else { return nil }
        return Page(meta: meta, folder: folder)
    }

    /// Writes the first `sampleCount` deterministic val samples as PNG triples.
    static func write(
        runName: String, step: Int, model: MLXAlignModel, vae: MLXVAEEncoder,
        data: TrainDataset, lossFn: TrainLoss?
    ) throws {
        let samples = try data.previewSamples(Self.sampleCount)
        let folder = directory(runName: runName, step: step)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var rows: [Sample] = []
        for (index, sample) in samples.enumerated() {
            let (pred, psnr, ssim) = try predict(sample, model: model, vae: vae, lossFn: lossFn)
            let lowH = sample.low.dim(1), lowW = sample.low.dim(2)
            let highH = sample.high.dim(1), highW = sample.high.dim(2)
            eval(sample.low, sample.high, pred)
            let low = sample.low.asArray(Float.self)
            let high = sample.high.asArray(Float.self)
            let out = pred.asArray(Float.self)
            guard let lowImage = upscaledInput(low, width: lowW, height: lowH, targetWidth: highW, targetHeight: highH),
                  let predImage = imageNHWC(out, width: highW, height: highH),
                  let origImage = imageNHWC(high, width: highW, height: highH)
            else {
                throw CLIError("Val-Bild \(index) lässt sich nicht schreiben")
            }
            try PlaygroundImage.savePNG(lowImage, to: folder.appendingPathComponent(String(format: "sample-%02d-low.png", index)))
            try PlaygroundImage.savePNG(predImage, to: folder.appendingPathComponent(String(format: "sample-%02d-pred.png", index)))
            try PlaygroundImage.savePNG(origImage, to: folder.appendingPathComponent(String(format: "sample-%02d-orig.png", index)))
            let info = sample.meta
            rows.append(Sample(
                index: index, name: info.name, frame: info.frame, rung: info.rung, preset: info.preset,
                psnr: psnr, ssim: ssim
            ))
        }
        let meta = Meta(step: step, run: runName, samples: rows)
        let dataJSON = try JSONEncoder().encode(meta)
        try dataJSON.write(to: folder.appendingPathComponent("meta.json"), options: .atomic)
        fputs("val-bilder \(runName)/val/step-\(String(format: "%06d", step))  \(rows.count) Samples\n", stderr)
    }

    /// Removes `Models/<run>/val/step-XXXXXX` and a now-empty `val/` folder. Only under the models root.
    static func remove(runName: String, step: Int) throws {
        try removeIfSafe(directory(runName: runName, step: step))
        let valRoot = WhoodeoPaths.modelsDirectory
            .appendingPathComponent(runName, isDirectory: true)
            .appendingPathComponent("val", isDirectory: true)
        let leftover = (try? FileManager.default.contentsOfDirectory(atPath: valRoot.path)) ?? ["."]
        if leftover.filter({ !$0.hasPrefix(".") }).isEmpty {
            try removeIfSafe(valRoot)
        }
    }

    static func removeAll(runName: String) throws {
        let valRoot = WhoodeoPaths.modelsDirectory
            .appendingPathComponent(runName, isDirectory: true)
            .appendingPathComponent("val", isDirectory: true)
        try removeIfSafe(valRoot)
    }

    private static func predict(
        _ sample: TrainSample, model: MLXAlignModel, vae: MLXVAEEncoder, lossFn: TrainLoss?
    ) throws -> (pred: MLXArray, psnr: Double?, ssim: Double?) {
        let height = sample.low.dim(1), width = sample.low.dim(2)
        let padded = padTo(sample.low, 4)
        let z = model.code(padded.asType(model.dtype), padded, sample.condition.asType(model.dtype))
        let shift = vae.shift.reshaped([1, 1, 1, TrainModel.latent]).asType(z.dtype)
        let scale = vae.scale.reshaped([1, 1, 1, TrainModel.latent]).asType(z.dtype)
        let rgb = model.decode(z / scale + shift)
        let pred = clip(rgb[0..., 0 ..< 2 * height, 0 ..< 2 * width].asType(.float32), min: Float(0), max: Float(1))
        eval(pred)
        var psnr: Double? = Double(TrainLoss.psnr(pred, sample.high))
        var ssim: Double?
        if let lossFn {
            let target = vae.normalizedCode(sample.high)
            let (_, values) = lossFn.terms(
                predZ: z.asType(.float32), targetZ: target.asType(.float32),
                predRGB: pred, targetRGB: sample.high, metrics: true
            )
            let scalars = lossFn.scalars(values)
            if let ms = scalars["ms_ssim"] { ssim = Double(1 - ms) }
        }
        return (pred, psnr, ssim)
    }

    /// Middle low frame (channels 3–5 of the 9-channel stack), drawn to the master size.
    private static func upscaledInput(_ lowNHWC: [Float], width: Int, height: Int, targetWidth: Int, targetHeight: Int) -> CGImage? {
        let plane = width * height
        var planar = [Float](repeating: 0, count: plane * 3)
        for i in 0..<plane {
            planar[i] = lowNHWC[i * 9 + 3]
            planar[plane + i] = lowNHWC[i * 9 + 4]
            planar[2 * plane + i] = lowNHWC[i * 9 + 5]
        }
        guard let small = PlaygroundImage.image(planar: planar, width: width, height: height) else { return nil }
        if targetWidth == width, targetHeight == height { return small }
        let space = small.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: targetWidth, height: targetHeight, bitsPerComponent: 8,
            bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        context.interpolationQuality = .high
        context.draw(small, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        return context.makeImage()
    }

    private static func imageNHWC(_ values: [Float], width: Int, height: Int) -> CGImage? {
        let plane = width * height
        guard values.count >= plane * 3 else { return nil }
        var planar = [Float](repeating: 0, count: plane * 3)
        for i in 0..<plane {
            planar[i] = values[i * 3]
            planar[plane + i] = values[i * 3 + 1]
            planar[2 * plane + i] = values[i * 3 + 2]
        }
        return PlaygroundImage.image(planar: planar, width: width, height: height)
    }

    private static func removeIfSafe(_ url: URL) throws {
        let file = url.standardizedFileURL.resolvingSymlinksInPath()
        let folder = WhoodeoPaths.modelsDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let filePath = file.path
        let folderPath = folder.path
        let prefix = folderPath.hasSuffix("/") ? folderPath : folderPath + "/"
        guard filePath.hasPrefix(prefix), filePath != folderPath else { return }
        if FileManager.default.fileExists(atPath: filePath) {
            try FileManager.default.removeItem(at: file)
        }
    }
}
