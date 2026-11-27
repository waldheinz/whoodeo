import Foundation

/// Application Support folder, the SQLite database, and the model packages.
nonisolated enum WhoodeoPaths {
    static let appDefaultsSuite = "meins.whoodeo.Whoodeo"
    static let cacheDirectoryKey = "cacheDirectory"
    static let defaultDatasetRootPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("storinator/home/whoodeo/train", isDirectory: true)
        .path

    static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("Whoodeo", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var databaseURL: URL {
        supportDirectory.appendingPathComponent("whoodeo.sqlite")
    }

    static var modelsDirectory: URL {
        let url = supportDirectory.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var defaultCacheRoot: URL {
        supportDirectory.appendingPathComponent("Cache", isDirectory: true)
    }

    /// Frozen VAE, VGG19, and SPyNet assets for the Swift trainer.
    static var trainDirectory: URL {
        let url = supportDirectory.appendingPathComponent("Train", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// The app's cache folder from its defaults, otherwise the default under Application Support.
    static func resolvedCacheRoot() -> URL {
        if let path = UserDefaults.standard.string(forKey: cacheDirectoryKey), !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        if let path = UserDefaults(suiteName: appDefaultsSuite)?.string(forKey: cacheDirectoryKey), !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return defaultCacheRoot
    }
}
