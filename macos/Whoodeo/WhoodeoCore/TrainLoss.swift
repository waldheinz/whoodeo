import Foundation
import MLX

/// Official falign loss: latent L1, Oklab, 5-scale MS-SSIM on Oklab L, VGG19 relu2_2 + relu3_4.
/// Every term is optional: weight 0 skips it.
/// Image terms compare the unclamped vae2 decode with the master in [0, 1].
/// Faces and scene_cut are skipped in this first Swift trainer.
nonisolated final class TrainLoss {
    static let oklabM1: [Float] = [
        0.4122214708, 0.5363325363, 0.0514459929,
        0.2119034982, 0.6806995451, 0.1073969566,
        0.0883024619, 0.2817188376, 0.6299787005,
    ]
    static let oklabM2: [Float] = [
        0.2104542553, 0.7936177850, -0.0040720468,
        1.9779984951, -2.4285922050, 0.4505937099,
        0.0259040371, 0.7827717662, -0.8086757660,
    ]
    static let oklabEps: Float = 1e-6
    static let cbrtRoot: Float = 6.0 / 29.0
    static let cbrtKnee: Float = (6.0 / 29.0) * (6.0 / 29.0) * (6.0 / 29.0)
    static let msWeights: [Float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    static let ssimWindow = 11
    static let ssimSigma: Float = 1.5
    static let ssimC1: Float = 0.01 * 0.01
    static let ssimC2: Float = 0.03 * 0.03
    static let vggIndexes: [String: Int] = ["relu2_2": 8, "relu3_4": 17]
    static let vggMean: [Float] = [0.485, 0.456, 0.406]
    static let vggStd: [Float] = [0.229, 0.224, 0.225]

    let config: TrainConfig
    /// `nil` when `w_vgg` is 0: VGG19 is not even loaded.
    private let vgg: [String: MLXArray]?
    private let window: MLXArray

    init(config: TrainConfig) throws {
        self.config = config
        if config.vggWeight > 0 {
            let url = WhoodeoPaths.trainDirectory.appendingPathComponent("vgg19.safetensors")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw CLIError("VGG19-Gewichte fehlen: \(url.path)")
            }
            let loaded = try loadArrays(url: url)
            eval(Array(loaded.values))
            vgg = loaded
            fputs("VGG19 geladen\n", stderr)
        } else {
            vgg = nil
            fputs("VGG19 nicht geladen (w-vgg 0)\n", stderr)
        }
        window = Self.makeWindow()
        eval(window)
    }

    var vggLoaded: Bool { vgg != nil }

    /// Raw terms plus the weighted total. Keys are ready for the `metrics` table.
    /// A term whose weight is 0 is not computed at all. `predRGB` may be `nil` when no pixel term is active
    /// (latent-only training skips the VAE decode), `targetZ` may be `nil` when the latent term is off.
    /// `metrics` (validation) also computes the cheap Oklab and MS-SSIM terms with weight 0, so PSNR/SSIM
    /// and the curves stay comparable; they do not enter the total. VGG runs only when it is loaded.
    func terms(
        predZ: MLXArray, targetZ: MLXArray?, predRGB: MLXArray?, targetRGB: MLXArray, metrics: Bool = false
    ) -> (total: MLXArray, values: [String: MLXArray]) {
        var total: MLXArray?
        var values: [String: MLXArray] = [:]
        func add(_ term: MLXArray, _ weight: Float) {
            guard weight > 0 else { return }
            let part = term * weight
            total = total.map { $0 + part } ?? part
        }
        if let targetZ, config.latentWeight > 0 || metrics {
            let latent = mean(abs(predZ - targetZ))
            values["latent_l1"] = latent
            add(latent, config.latentWeight)
        }
        if let predRGB {
            let wantLab = config.oklabWeight > 0 || config.msSSIMWeight > 0 || metrics
            if wantLab {
                let predLab = oklabFull(predRGB)
                let highLab = oklabFull(targetRGB)
                if config.oklabWeight > 0 || metrics {
                    let oklab = oklabLoss(predLab, highLab)
                    values["oklab"] = oklab
                    add(oklab, config.oklabWeight)
                }
                if config.msSSIMWeight > 0 || metrics {
                    let ms = msSSIMLoss(predLab, highLab)
                    values["ms_ssim"] = ms
                    add(ms, config.msSSIMWeight)
                }
            }
            if vgg != nil, config.vggWeight > 0 {
                var vggSum: MLXArray?
                for (name, raw) in vggLayers(predRGB, targetRGB) {
                    values["vgg_\(name)"] = raw
                    let part = raw * (config.vggLayers[name] ?? 0)
                    vggSum = vggSum.map { $0 + part } ?? part
                }
                if let vggSum {
                    values["vgg"] = vggSum
                    add(vggSum, config.vggWeight)
                }
            }
        }
        let sum = total ?? MLXArray(Float(0))
        values["total"] = sum
        return (sum, values)
    }

    func scalars(_ values: [String: MLXArray]) -> [String: Float] {
        var found: [String: Float] = [:]
        for (name, value) in values {
            eval(value)
            found[name] = value.item(Float.self)
        }
        return found
    }

    // MARK: Oklab

    func srgbLinear(_ image: MLXArray) -> MLXArray {
        let size = abs(image)
        let curve = pow((maximum(size, Float(0.04045)) + Float(0.055)) / Float(1.055), Float(2.4))
        return which(lessEqual(size, Float(0.04045)), image / Float(12.92), sign(image) * curve)
    }

    func safeCbrt(_ value: MLXArray) -> MLXArray {
        let root = pow(maximum(value, Self.cbrtKnee), Float(1.0 / 3.0))
        let line = Self.cbrtRoot + (value - Self.cbrtKnee) / (Float(3) * Self.cbrtRoot * Self.cbrtRoot)
        return which(greater(value, Self.cbrtKnee), root, line)
    }

    func oklabFull(_ image: MLXArray) -> MLXArray {
        let linear = srgbLinear(image)
        let m1 = MLXArray(Self.oklabM1, [3, 3])
        let m2 = MLXArray(Self.oklabM2, [3, 3])
        let lms = matmul(linear, m1.T)
        return matmul(safeCbrt(lms), m2.T)
    }

    func oklabLoss(_ pred: MLXArray, _ high: MLXArray) -> MLXArray {
        let channels = config.oklabChannels
        let weight = MLXArray(channels, [1, 1, 1, channels.count])
        let diff = abs(pred - high) * weight
        return mean(diff) * Float(3) / Float(channels.reduce(0, +))
    }

    // MARK: MS-SSIM

    static func makeWindow() -> MLXArray {
        let half = ssimWindow / 2
        let coords = MLXArray((-half...half).map { Float($0) })
        let gauss = exp(-coords * coords / (Float(2) * ssimSigma * ssimSigma))
        return gauss / gauss.sum()
    }

    func blur(_ image: MLXArray) -> MLXArray {
        let kernelW = window.reshaped([1, 1, Self.ssimWindow, 1])
        let kernelH = window.reshaped([1, Self.ssimWindow, 1, 1])
        let wide = conv2d(image, kernelW, padding: IntOrPair(integerLiteral: 0))
        return conv2d(wide, kernelH, padding: IntOrPair(integerLiteral: 0))
    }

    func ssimParts(_ x: MLXArray, _ y: MLXArray) -> (ssim: MLXArray, cs: MLXArray) {
        let muX = blur(x)
        let muY = blur(y)
        let varX = blur(x * x) - muX * muX
        let varY = blur(y * y) - muY * muY
        let cov = blur(x * y) - muX * muY
        let cs = (Float(2) * cov + Self.ssimC2) / (varX + varY + Self.ssimC2)
        let luminance = (Float(2) * muX * muY + Self.ssimC1) / (muX * muX + muY * muY + Self.ssimC1)
        return (spatialMean(luminance * cs), spatialMean(cs))
    }

    func spatialMean(_ x: MLXArray) -> MLXArray {
        x.mean(axes: [1, 2, 3])
    }

    func avgPool2Pad(_ x: MLXArray) -> MLXArray {
        var image = x
        let h = image.dim(1), w = image.dim(2)
        if h % 2 != 0 {
            image = concatenated([image, image[0..., h - 1 ..< h, 0..., 0...]], axis: 1)
        }
        if w % 2 != 0 {
            image = concatenated([image, image[0..., 0..., w - 1 ..< w, 0...]], axis: 2)
        }
        let s = image.shape
        return image.reshaped([s[0], s[1] / 2, 2, s[2] / 2, 2, s[3]]).mean(axes: [2, 4])
    }

    func msSSIM(_ predL: MLXArray, _ highL: MLXArray) -> MLXArray {
        let height = predL.dim(1), width = predL.dim(2)
        var usable = 0
        while usable < config.msSSIMScales && min(height, width) >> usable >= Self.ssimWindow {
            usable += 1
        }
        if usable == 0 {
            return MLXArray(Float(0))
        }
        var weights = MLXArray(Array(Self.msWeights.prefix(usable)))
        weights = weights / weights.sum()
        var x = predL
        var y = highL
        var factors: [MLXArray] = []
        for level in 0..<usable {
            let parts = ssimParts(x, y)
            factors.append(level == usable - 1 ? parts.ssim : parts.cs)
            if level < usable - 1 {
                x = avgPool2Pad(x)
                y = avgPool2Pad(y)
            }
        }
        // factors are (B,) per scale -> (B, scales)
        let stacked = stacked(factors, axis: 1)
        let powered = pow(maximum(stacked, Float(1e-4)), weights.reshaped([1, usable]))
        return mean(product(powered, axis: 1))
    }

    func msSSIMLoss(_ predLab: MLXArray, _ highLab: MLXArray) -> MLXArray {
        let predL = predLab[0..., 0..., 0..., 0 ..< 1]
        let highL = highLab[0..., 0..., 0..., 0 ..< 1]
        return Float(1) - msSSIM(predL, highL)
    }

    // MARK: VGG19

    func vggConv(_ index: Int, _ x: MLXArray) -> MLXArray {
        var y = conv2d(x, vgg!["features.\(index).weight"]!, padding: IntOrPair(integerLiteral: 1))
        if let bias = vgg?["features.\(index).bias"] { y = y + bias }
        return y
    }

    func maxPool2(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        return x.reshaped([s[0], s[1] / 2, 2, s[2] / 2, 2, s[3]]).max(axes: [2, 4])
    }

    func vggMaps(_ image: MLXArray) -> [String: MLXArray] {
        let mean = MLXArray(Self.vggMean, [1, 1, 1, 3])
        let std = MLXArray(Self.vggStd, [1, 1, 1, 3])
        var hidden = (clip(image, min: Float(0), max: Float(1)) - mean) / std
        var found: [String: MLXArray] = [:]
        let wanted = Set(config.vggLayers.keys.compactMap { Self.vggIndexes[$0] })
        // VGG19 features 0..17: conv, relu, pool as in torchvision.
        let layers: [(Int, String)] = [
            (0, "conv"), (1, "relu"), (2, "conv"), (3, "relu"), (4, "pool"),
            (5, "conv"), (6, "relu"), (7, "conv"), (8, "relu"), (9, "pool"),
            (10, "conv"), (11, "relu"), (12, "conv"), (13, "relu"),
            (14, "conv"), (15, "relu"), (16, "conv"), (17, "relu"),
        ]
        for (index, kind) in layers {
            switch kind {
            case "conv": hidden = vggConv(index, hidden)
            case "relu": hidden = maximum(hidden, Float(0))
            default: hidden = maxPool2(hidden)
            }
            if wanted.contains(index), let name = Self.vggIndexes.first(where: { $0.value == index })?.key {
                found[name] = hidden
            }
        }
        return found
    }

    func vggLayers(_ pred: MLXArray, _ high: MLXArray) -> [String: MLXArray] {
        let predMaps = vggMaps(pred)
        let highMaps = vggMaps(high)
        var found: [String: MLXArray] = [:]
        for name in config.vggLayers.keys {
            if let a = predMaps[name], let b = highMaps[name] {
                found[name] = mean(abs(a - b))
            }
        }
        return found
    }

    static func psnr(_ pred: MLXArray, _ high: MLXArray) -> Float {
        let mse = mean(pow(clip(pred, min: Float(0), max: Float(1)) - clip(high, min: Float(0), max: Float(1)), Float(2)))
        eval(mse)
        let value = mse.item(Float.self)
        if value <= 0 { return 99 }
        return -10 * log10(value)
    }
}
