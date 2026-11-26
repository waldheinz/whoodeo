import CoreMedia
import Foundation

nonisolated struct FilmFacts: Sendable {
    var url: URL
    var width: Int
    var height: Int
    var count: Int
    var fps: Double
    var bits: Int
    /// FFmpeg's name of the picture codec, empty when unknown.
    var codecName: String = ""
    /// Frames and rate of the file itself. `count` and `fps` are what the app sees after preprocessing.
    var sourceCount: Int = 0
    var sourceFps: Double = 0
    /// Decimate cycle in force, nil when every source frame is used.
    var cycle: Int?

    /// What the codec condition of a latent_up with FiLM calls this film.
    var codec: SourceCodec { SourceCodec.detect(ffmpegName: codecName) }
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

/// One film as the rest of the app sees it: indexes, times and pictures after preprocessing.
/// Without a frame map that is the file itself. With one, index `i` is source frame `map.raw[i]`,
/// timed evenly at the reduced rate over the same total length, so the sound stays in place.
nonisolated final class Film {
    private(set) var facts: FilmFacts
    /// The file as it is.
    let source: FilmFacts
    let audio: [AudioFacts]
    /// Length of one source frame.
    let sourceDuration: CMTime
    private(set) var map: FrameMap?
    private let reader: OpaquePointer
    private var held: [Int: Decoded] = [:]
    private var heldOrder: [Int] = []

    @concurrent
    static func load(_ url: URL, map: FrameMap? = nil) async throws -> Film {
        try open(url, map: map)
    }

    /// Opens the file on the calling thread. A map made for another frame count is ignored.
    static func open(_ url: URL, map: FrameMap? = nil) throws -> Film {
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
            bits: Int(source_bits(opened)),
            codecName: String(cString: source_codec(opened)),
            sourceCount: count,
            sourceFps: fps > 0 ? fps : 25
        )
        let film = Film(facts: facts, audio: audio, frameDuration: duration, reader: opened)
        film.apply(map)
        return film
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
        self.source = facts
        self.audio = audio
        self.sourceDuration = frameDuration
        self.reader = reader
    }

    /// Switch the preprocessing. Nil shows every source frame again.
    func apply(_ next: FrameMap?) {
        guard let next, next.rawCount == source.count, next.count > 0 else {
            map = nil
            facts = source
            return
        }
        map = next
        var logical = source
        logical.count = next.count
        logical.fps = next.fps(source: source.fps)
        logical.cycle = next.cycle
        facts = logical
    }

    /// Length of one frame as the app sees it.
    var frameDuration: CMTime {
        guard let map else { return sourceDuration }
        return CMTimeMultiplyByRatio(sourceDuration, multiplier: Int32(map.cycle), divisor: Int32(map.cycle - 1))
    }

    /// The source frame behind a frame of the app, clamped to the film.
    func sourceIndex(_ index: Int) -> Int {
        let clamped = min(max(index, 0), max(facts.count - 1, 0))
        guard let map else { return clamped }
        return map.raw[clamped]
    }

    deinit {
        source_close(reader)
    }

    func time(for index: Int) -> CMTime {
        guard facts.count > 0 else { return .zero }
        if let map {
            // Even steps from the first source stamp: `cycle` source frames become `cycle - 1`.
            let start = Film.stamp(0, reader)
            let ticks = Int64(max(index, 0)) * Int64(map.cycle)
            guard ticks <= Int64(Int32.max) else { return start }
            let offset = CMTimeMultiplyByRatio(sourceDuration, multiplier: Int32(ticks), divisor: Int32(map.cycle - 1))
            return CMTimeAdd(start, offset)
        }
        if index < facts.count { return Film.stamp(max(index, 0), reader) }
        let last = Film.stamp(facts.count - 1, reader)
        let extra = index - (facts.count - 1)
        guard extra > 0, extra <= Int(Int32.max) else { return last }
        return CMTimeAdd(last, CMTimeMultiply(frameDuration, multiplier: Int32(extra)))
    }

    func duration(at index: Int) -> CMTime {
        if map != nil { return frameDuration }
        guard index + 1 < facts.count else { return frameDuration }
        let delta = CMTimeSubtract(time(for: index + 1), time(for: index))
        return delta.seconds > 0 ? delta : frameDuration
    }

    func frame(at index: Int) async throws -> Decoded {
        try decode(at: index)
    }

    /// Decodes on the calling thread. One film is not safe to share between threads.
    func decode(at index: Int) throws -> Decoded {
        let clamped = min(max(index, 0), max(facts.count - 1, 0))
        var decoded = try decodeSource(sourceIndex(clamped))
        decoded.index = clamped
        decoded.time = time(for: clamped)
        return decoded
    }

    /// 8-bit luma of a source frame on a grid of every `step`-th pixel, for frame comparisons.
    /// Reading the frames in order never seeks.
    func luma(source index: Int, step: Int, into buffer: inout [UInt8]) throws -> (width: Int, height: Int) {
        let needed = ((facts.width + step - 1) / step) * ((facts.height + step - 1) / step)
        if buffer.count < needed { buffer = [UInt8](repeating: 0, count: needed) }
        var width: Int32 = 0
        var height: Int32 = 0
        let status = buffer.withUnsafeMutableBufferPointer { out -> Int32 in
            guard let base = out.baseAddress else { return -1 }
            return source_luma(reader, Int32(index), base, Int32(out.count), Int32(step), &width, &height)
        }
        guard status == 0 else { throw CLIError(String(cString: source_error(reader))) }
        return (Int(width), Int(height))
    }

    /// One frame of the file, by its own index. Held by that index, so a map change keeps the hold.
    private func decodeSource(_ index: Int) throws -> Decoded {
        let clamped = min(max(index, 0), max(source.count - 1, 0))
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
            time: Film.stamp(clamped, reader),
            width: pictureWidth,
            height: pictureHeight,
            rgb: rgb
        )
        remember(decoded)
        return decoded
    }

    /// Previous, current, next. The same edge repeat as `iter_reconstructions`.
    /// `frames` names the source frame of each plane (the edge repeats the first or last one).
    /// With a frame map the neighbors are the kept frames, so a repeat never stands in for one.
    func window(at index: Int, frames: Int) async throws -> (stack: [Float], current: Decoded, frames: FrameKeys) {
        let radius = frames / 2
        var planes: [[Float]] = []
        var slots: [Int] = []
        var current: Decoded?
        for dt in -radius...radius {
            let slot = min(max(index + dt, 0), max(facts.count - 1, 0))
            let decoded = try await frame(at: slot)
            planes.append(decoded.rgb)
            slots.append(sourceIndex(slot))
            if dt == 0 { current = decoded }
        }
        guard let current else { throw CLIError("Bild \(index) fehlt") }
        return (Pixels.stack(planes), current, FrameKeys(film: facts.url, slots: slots))
    }

    private func remember(_ decoded: Decoded) {
        if held[decoded.index] == nil { heldOrder.append(decoded.index) }
        held[decoded.index] = decoded
        while heldOrder.count > 8 {
            held.removeValue(forKey: heldOrder.removeFirst())
        }
    }
}
