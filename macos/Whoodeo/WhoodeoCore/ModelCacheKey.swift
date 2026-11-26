import CryptoKit
import Foundation

/// Identity of a `.mlxmodel` package in the frame cache.
///
/// The digest is of the package path plus the size and mtime of `model.safetensors`
/// and `config.json`, so a package with different weights does not share a cache.
/// Archive folders are `{digest}-gpu` or `{digest}-gpu-{tag}`, where `tag` is the
/// codec condition and optional preprocessing (for example `h264` or `h264-dec6`).
/// The `gpu` suffix is kept so existing caches stay valid.
nonisolated enum ModelCacheKey {
    static let unitSuffix = "gpu"

    /// SHA-256 of the package identity, used as the prefix of every archive folder.
    static func modelKey(of url: URL) -> String {
        digest(identity(url))
    }

    /// Folder name of a model's archive, optionally plus the condition of a `-cond` package.
    static func folderName(model: URL, condition: String? = nil) -> String {
        var name = "\(modelKey(of: model))-\(unitSuffix)"
        if let condition, !condition.isEmpty {
            name += "-\(condition)"
        }
        return name
    }

    /// Relative path of an archive under the cache root: `{filmID}/{folderName}`.
    static func relativePath(filmID: String, model: URL, condition: String? = nil) -> String {
        "\(filmID)/\(folderName(model: model, condition: condition))"
    }

    /// `{modelKey, condition}` from an archive folder name, or nil when the name is not one.
    static func parseArchiveName(_ name: String) -> (modelKey: String, condition: String?)? {
        let base = name.hasSuffix(".deleting") ? String(name.dropLast(".deleting".count)) : name
        let needle = "-\(unitSuffix)"
        guard let range = base.range(of: needle) else { return nil }
        let key = String(base[..<range.lowerBound])
        guard key.count == 64, key.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) else { return nil }
        let rest = String(base[range.upperBound...])
        if rest.isEmpty { return (key, nil) }
        guard rest.hasPrefix("-") else { return nil }
        let condition = String(rest.dropFirst())
        return condition.isEmpty ? nil : (key, condition)
    }

    /// A film-cache folder is a 64-character lowercase hex digest, or the same with `.deleting`.
    static func isFilmFolderName(_ name: String) -> Bool {
        let base = name.hasSuffix(".deleting") ? String(name.dropLast(".deleting".count)) : name
        return base.count == 64 && base.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    /// Path, size and mtime of the package. Must stay bit-identical to the former `FrameStore` helper.
    static func identity(_ url: URL) -> String {
        let file = url.standardizedFileURL
        let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .isDirectoryKey])
        if values?.isDirectory == true, file.pathExtension == MLXAlignModel.fileExtension {
            let weights = file.appendingPathComponent("model.safetensors")
            let config = file.appendingPathComponent("config.json")
            return "\(file.path)\n\(stamp(weights))\n\(stamp(config))"
        }
        return "\(file.path)\n\(values?.fileSize ?? 0)\n\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
    }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func stamp(_ url: URL) -> String {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        return "\(values?.fileSize ?? 0)-\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
    }
}
