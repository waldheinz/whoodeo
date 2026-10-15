import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

struct FilmFacts: Sendable {
    var url: URL
    var width: Int
    var height: Int
    var count: Int
    var fps: Double
    var bits: Int
}

struct AudioFacts: Sendable {
    var channels: Int
    var sampleRate: Double
    var subtype: AudioFormatID
    var dataRate: Double
}

struct Decoded {
    var index: Int
    var time: CMTime
    var width: Int
    var height: Int
    var rgb: [Float]
}

final class Film {
    let facts: FilmFacts
    let asset: AVURLAsset
    let video: AVAssetTrack
    let audio: [(track: AVAssetTrack, facts: AudioFacts)]
    let start: CMTime
    let frameDuration: CMTime
    let timeRange: CMTimeRange
    /// Sample index and its real timestamp, where the picture stayed up longer than one interval.
    private let breaks: [(index: Int, time: CMTime)]

    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var nextIndex: Int?
    private var held: [Int: Decoded] = [:]
    private var heldOrder: [Int] = []
    private var pixelFormat: OSType

    static func load(_ url: URL) async throws -> Film {
        // AVFoundation does not read Matroska. A stream copy keeps the pictures.
        let media = try await readableVideo(url)
        let asset = AVURLAsset(url: media, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let videos = try await asset.loadTracks(withMediaType: .video)
        guard let video = videos.first else { throw CLIError("kein Bild in \(url.lastPathComponent)") }
        let range = try await video.load(.timeRange)
        let rate = try await video.load(.nominalFrameRate)
        let natural = try await video.load(.naturalSize)
        let descriptions = try await video.load(.formatDescriptions)
        var bits = 8
        if let raw = descriptions.first {
            let description = raw as CMFormatDescription
            if let number = CMFormatDescriptionGetExtension(description, extensionKey: kCMFormatDescriptionExtension_BitsPerComponent) as? NSNumber {
                bits = number.intValue
            }
        }
        guard let cursor = video.makeSampleCursor(presentationTimeStamp: range.start) else {
            throw CLIError("keine Bilder in \(url.lastPathComponent)")
        }
        let start = cursor.presentationTimeStamp
        var duration = CMTime(value: 1, timescale: rate > 0 ? CMTimeScale(rate.rounded()) : 30)
        var count = 1
        // A frame that stays up longer than the first interval shifts every later
        // timestamp. The picture is still the next sample, so remember where the clock jumped.
        var breaks: [(index: Int, time: CMTime)] = []
        if cursor.stepInPresentationOrder(byCount: 1) == 1 {
            duration = CMTimeSubtract(cursor.presentationTimeStamp, start)
            count = 2
            var expected = cursor.presentationTimeStamp
            let slack = duration.seconds * 0.45
            while cursor.stepInPresentationOrder(byCount: 1) == 1 {
                count += 1
                expected = CMTimeAdd(expected, duration)
                let stamp = cursor.presentationTimeStamp
                if abs(CMTimeSubtract(stamp, expected).seconds) > slack {
                    breaks.append((count - 1, stamp))
                    expected = stamp
                }
            }
        }
        var audios: [(AVAssetTrack, AudioFacts)] = []
        for track in try await asset.loadTracks(withMediaType: .audio) {
            let formats = try await track.load(.formatDescriptions)
            guard let raw = formats.first else { continue }
            let description = raw as CMFormatDescription
            let subtype = CMFormatDescriptionGetMediaSubType(description)
            let basic = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee
            let dataRate = try await track.load(.estimatedDataRate)
            audios.append((track, AudioFacts(
                channels: Int(basic?.mChannelsPerFrame ?? 2),
                sampleRate: basic?.mSampleRate ?? 48_000,
                subtype: subtype,
                dataRate: Double(dataRate)
            )))
        }
        let fps = duration.seconds > 0 ? 1 / duration.seconds : Double(rate > 0 ? rate : 30)
        let facts = FilmFacts(
            url: url,
            width: Int(abs(natural.width).rounded()),
            height: Int(abs(natural.height).rounded()),
            count: count,
            fps: fps,
            bits: bits
        )
        return Film(
            facts: facts,
            asset: asset,
            video: video,
            audio: audios,
            start: start,
            frameDuration: duration,
            timeRange: range,
            breaks: breaks,
            pixelFormat: bits > 8 ? kCVPixelFormatType_64RGBALE : kCVPixelFormatType_32BGRA
        )
    }

    private init(
        facts: FilmFacts,
        asset: AVURLAsset,
        video: AVAssetTrack,
        audio: [(AVAssetTrack, AudioFacts)],
        start: CMTime,
        frameDuration: CMTime,
        timeRange: CMTimeRange,
        breaks: [(index: Int, time: CMTime)],
        pixelFormat: OSType
    ) {
        self.facts = facts
        self.asset = asset
        self.video = video
        self.audio = audio
        self.start = start
        self.frameDuration = frameDuration
        self.timeRange = timeRange
        self.breaks = breaks
        self.pixelFormat = pixelFormat
    }

    /// The file AVFoundation can actually read. Matroska is copied into a QuickTime cache.
    private static func readableVideo(_ url: URL) async throws -> URL {
        if await hasVideo(url) { return url }
        guard let ffmpeg = ffmpegURL() else {
            throw CLIError("AVFoundation öffnet \(url.pathExtension) nicht, und ffmpeg ist nicht installiert.")
        }
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("whoodeo", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let size = values.fileSize ?? 0
        let stamp = Int(values.contentModificationDate?.timeIntervalSince1970 ?? 0)
        let base = safeName(url.deletingPathExtension().lastPathComponent)
        let dest = cache.appendingPathComponent("\(base)-\(size)-\(stamp).mov")
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        // The extension has to be .mov. ffmpeg refuses a file that ends in .partial.
        let partial = cache.appendingPathComponent(".\(base)-\(size)-\(stamp).partial.mov")
        if try ffmpegCopy(ffmpeg, url: url, output: partial, audio: true) != nil {
            if let reason = try ffmpegCopy(ffmpeg, url: url, output: partial, audio: false) {
                throw CLIError("ffmpeg konnte \(url.lastPathComponent) nicht nach QuickTime umpacken. \(reason)")
            }
        }
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.moveItem(at: partial, to: dest)
        return dest
    }

    private static func hasVideo(_ url: URL) async -> Bool {
        let asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        do {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            return !tracks.isEmpty
        } catch {
            return false
        }
    }

    private static func safeName(_ name: String) -> String {
        let scalars = name.unicodeScalars.map { scalar -> Character in
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_" {
                return Character(scalar)
            }
            return "-"
        }
        let cleaned = String(scalars)
        return cleaned.isEmpty ? "film" : cleaned
    }

    private static func ffmpegURL() -> URL? {
        for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
        where FileManager.default.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        return nil
    }

    /// Nil when the copy landed. Otherwise the last ffmpeg error line.
    private static func ffmpegCopy(_ ffmpeg: URL, url: URL, output: URL, audio: Bool) throws -> String? {
        if FileManager.default.fileExists(atPath: output.path) {
            try FileManager.default.removeItem(at: output)
        }
        var args = ["-hide_banner", "-loglevel", "error", "-y", "-i", url.path, "-map", "0:v:0"]
        if audio {
            args.append("-map")
            args.append("0:a?")
        }
        args.append(contentsOf: ["-c", "copy", "-movflags", "+faststart", "-f", "mov", output.path])
        let proc = Process()
        proc.executableURL = ffmpeg
        proc.arguments = args
        let log = FileManager.default.temporaryDirectory.appendingPathComponent("whoodeo-ffmpeg-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        proc.standardError = handle
        proc.standardOutput = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        try handle.close()
        defer { try? FileManager.default.removeItem(at: log) }
        if proc.terminationStatus == 0 { return nil }
        let text = (try? String(contentsOf: log, encoding: .utf8)) ?? ""
        try? FileManager.default.removeItem(at: output)
        let line = text.split(whereSeparator: \.isNewline).last.map(String.init) ?? "exit \(proc.terminationStatus)"
        return line
    }

    func time(for index: Int) -> CMTime {
        var originIndex = 0
        var origin = start
        for item in breaks where item.index <= index {
            originIndex = item.index
            origin = item.time
        }
        return CMTimeAdd(origin, CMTimeMultiply(frameDuration, multiplier: Int32(index - originIndex)))
    }

    func duration(at index: Int) -> CMTime {
        guard index + 1 < facts.count else { return frameDuration }
        return CMTimeSubtract(time(for: index + 1), time(for: index))
    }

    func frame(at index: Int) throws -> Decoded {
        let clamped = min(max(index, 0), max(facts.count - 1, 0))
        if let hit = held[clamped] { return hit }
        if nextIndex != clamped { try openReader(at: clamped) }
        let expected = time(for: clamped).seconds
        let half = frameDuration.seconds * 0.45
        while true {
            guard let sample = output?.copyNextSampleBuffer(),
                  let buffer = CMSampleBufferGetImageBuffer(sample) else {
                throw CLIError("Bild \(clamped) fehlt")
            }
            let stamp = CMSampleBufferGetPresentationTimeStamp(sample)
            let delta = stamp.seconds - expected
            if delta > half {
                throw CLIError("Bild \(clamped) kommt bei \(stamp.seconds) s, erwartet \(expected) s")
            }
            if delta < -half { continue }
            let decoded = Decoded(
                index: clamped,
                time: stamp,
                width: CVPixelBufferGetWidth(buffer),
                height: CVPixelBufferGetHeight(buffer),
                rgb: try Pixels.planar(buffer)
            )
            nextIndex = clamped + 1
            remember(decoded)
            return decoded
        }
    }

    /// Previous, current, next. The same edge repeat as `iter_reconstructions`.
    func window(at index: Int, frames: Int) throws -> (stack: [Float], current: Decoded) {
        let radius = frames / 2
        var planes: [[Float]] = []
        var current: Decoded?
        for dt in -radius...radius {
            let slot = min(max(index + dt, 0), max(facts.count - 1, 0))
            let decoded = try frame(at: slot)
            planes.append(decoded.rgb)
            if dt == 0 { current = decoded }
        }
        guard let current else { throw CLIError("Bild \(index) fehlt") }
        return (Pixels.stack(planes), current)
    }

    private func openReader(at index: Int) throws {
        reader?.cancelReading()
        reader = nil
        output = nil
        nextIndex = nil
        let reader = try AVAssetReader(asset: asset)
        let begin = time(for: index)
        var end = timeRange.end
        if CMTimeCompare(end, begin) <= 0 {
            end = CMTimeAdd(begin, frameDuration)
        }
        reader.timeRange = CMTimeRange(start: begin, end: end)
        let output = AVAssetReaderTrackOutput(track: video, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw CLIError("Decoder lehnt das Bildformat ab") }
        reader.add(output)
        if !reader.startReading() {
            if pixelFormat != kCVPixelFormatType_32BGRA {
                pixelFormat = kCVPixelFormatType_32BGRA
                try openReader(at: index)
                return
            }
            throw reader.error ?? CLIError("Decoder startet nicht")
        }
        self.reader = reader
        self.output = output
        nextIndex = index
    }

    private func remember(_ decoded: Decoded) {
        if held[decoded.index] == nil { heldOrder.append(decoded.index) }
        held[decoded.index] = decoded
        while heldOrder.count > 8 {
            held.removeValue(forKey: heldOrder.removeFirst())
        }
    }
}
