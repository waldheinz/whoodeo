import CoreMedia
import Foundation

nonisolated struct FilmFacts: Sendable {
    var url: URL
    var width: Int
    var height: Int
    var count: Int
    var fps: Double
    var bits: Int
}

nonisolated struct AudioFacts: Sendable {
    var channels: Int
    var sampleRate: Double
    var dataRate: Double
    /// Which audio stream this is, in the order libav lists them.
    var track: Int
}

nonisolated struct Decoded {
    var index: Int
    var time: CMTime
    var width: Int
    var height: Int
    var rgb: [Float]
}

nonisolated final class Film {
    let facts: FilmFacts
    let audio: [AudioFacts]
    let frameDuration: CMTime
    private let reader: OpaquePointer
    private var held: [Int: Decoded] = [:]
    private var heldOrder: [Int] = []

    @concurrent
    static func load(_ url: URL) async throws -> Film {
        var message = [CChar](repeating: 0, count: 512)
        let opened = message.withUnsafeMutableBufferPointer { buffer in
            url.path.withCString { source_open($0, buffer.baseAddress, Int32(buffer.count)) }
        }
        guard let opened else {
            let detail = String(cString: message)
            throw CLIError(detail.isEmpty ? "kein Bild in \(url.lastPathComponent)" : detail)
        }
        let count = Int(source_count(opened))
        let width = Int(source_width(opened))
        let height = Int(source_height(opened))
        guard count > 0, width > 0, height > 0 else {
            source_close(opened)
            throw CLIError("keine Bilder in \(url.lastPathComponent)")
        }
        var audio: [AudioFacts] = []
        let tracks = Int(source_audio_count(opened))
        for track in 0..<tracks {
            var channels: Int32 = 0
            var rate = 0.0
            var bitRate = 0.0
            guard source_audio_info(opened, Int32(track), &channels, &rate, &bitRate) == 0 else { continue }
            audio.append(AudioFacts(
                channels: Int(channels),
                sampleRate: rate,
                dataRate: bitRate,
                track: track
            ))
        }
        let fps = source_fps(opened)
        let duration = Film.interval(opened, count: count, fps: fps)
        let facts = FilmFacts(
            url: url,
            width: width,
            height: height,
            count: count,
            fps: fps > 0 ? fps : 25,
            bits: Int(source_bits(opened))
        )
        return Film(facts: facts, audio: audio, frameDuration: duration, reader: opened)
    }

    private static func interval(_ file: OpaquePointer, count: Int, fps: Double) -> CMTime {
        if count >= 2 {
            let delta = CMTimeSubtract(stamp(1, file), stamp(0, file))
            if delta.seconds > 0 { return delta }
        }
        let scale = Int32(max(1, (fps > 0 ? fps : 25).rounded()))
        return CMTime(value: 1, timescale: scale)
    }

    private static func stamp(_ index: Int, _ file: OpaquePointer) -> CMTime {
        var value: Int64 = 0
        var scale: Int32 = 1
        if source_time(file, Int32(index), &value, &scale) != 0 || scale <= 0 {
            return .zero
        }
        return CMTime(value: value, timescale: scale)
    }

    private init(
        facts: FilmFacts,
        audio: [AudioFacts],
        frameDuration: CMTime,
        reader: OpaquePointer
    ) {
        self.facts = facts
        self.audio = audio
        self.frameDuration = frameDuration
        self.reader = reader
    }

    deinit {
        source_close(reader)
    }

    func time(for index: Int) -> CMTime {
        guard facts.count > 0 else { return .zero }
        if index < facts.count { return Film.stamp(max(index, 0), reader) }
        let last = Film.stamp(facts.count - 1, reader)
        let extra = index - (facts.count - 1)
        guard extra > 0, extra <= Int(Int32.max) else { return last }
        return CMTimeAdd(last, CMTimeMultiply(frameDuration, multiplier: Int32(extra)))
    }

    func duration(at index: Int) -> CMTime {
        guard index + 1 < facts.count else { return frameDuration }
        let delta = CMTimeSubtract(time(for: index + 1), time(for: index))
        return delta.seconds > 0 ? delta : frameDuration
    }

    func frame(at index: Int) async throws -> Decoded {
        let clamped = min(max(index, 0), max(facts.count - 1, 0))
        if let hit = held[clamped] { return hit }
        let plane = facts.width * facts.height
        var rgb = [Float](repeating: 0, count: plane * 3)
        var width: Int32 = 0
        var height: Int32 = 0
        let status = rgb.withUnsafeMutableBufferPointer { buffer -> Int32 in
            guard let base = buffer.baseAddress else { return -1 }
            return source_frame(reader, Int32(clamped), base, Int32(buffer.count), &width, &height)
        }
        guard status == 0, width > 0, height > 0 else {
            throw CLIError(String(cString: source_error(reader)))
        }
        let pictureWidth = Int(width)
        let pictureHeight = Int(height)
        if pictureWidth != facts.width || pictureHeight != facts.height {
            throw CLIError("Bild \(clamped + 1) ist \(pictureWidth)×\(pictureHeight), der Film \(facts.width)×\(facts.height)")
        }
        let decoded = Decoded(
            index: clamped,
            time: time(for: clamped),
            width: pictureWidth,
            height: pictureHeight,
            rgb: rgb
        )
        remember(decoded)
        return decoded
    }

    /// Previous, current, next. The same edge repeat as `iter_reconstructions`.
    func window(at index: Int, frames: Int) async throws -> (stack: [Float], current: Decoded) {
        let radius = frames / 2
        var planes: [[Float]] = []
        var current: Decoded?
        for dt in -radius...radius {
            let slot = min(max(index + dt, 0), max(facts.count - 1, 0))
            let decoded = try await frame(at: slot)
            planes.append(decoded.rgb)
            if dt == 0 { current = decoded }
        }
        guard let current else { throw CLIError("Bild \(index) fehlt") }
        return (Pixels.stack(planes), current)
    }

    private func remember(_ decoded: Decoded) {
        if held[decoded.index] == nil { heldOrder.append(decoded.index) }
        held[decoded.index] = decoded
        while heldOrder.count > 8 {
            held.removeValue(forKey: heldOrder.removeFirst())
        }
    }
}
