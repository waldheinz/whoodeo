import Foundation

/// The codec a latent_up with FiLM is told about, in the order of the condition vector's one-hot slots.
/// Unknown is all slots zero, not a slot of its own.
nonisolated enum SourceCodec: String, CaseIterable, Identifiable, Sendable {
    case h264, mpeg4, mpeg2, unknown

    var id: String { rawValue }

    /// One-hot slots of the condition vector, same order as Python `CODECS`.
    static let slots: [SourceCodec] = [.h264, .mpeg4, .mpeg2]

    var title: String {
        switch self {
        case .h264: "h264"
        case .mpeg4: "MPEG-4"
        case .mpeg2: "MPEG-2"
        case .unknown: "unbekannt"
        }
    }

    /// Condition vector of `size` (codec slots, then reserve zeros).
    func vector(size: Int) -> [Float] {
        var values = [Float](repeating: 0, count: max(size, 0))
        if self != .unknown, let index = Self.slots.firstIndex(of: self), index < values.count {
            values[index] = 1
        }
        return values
    }

    /// Suffix of a `-cond` package's frame cache for this codec vector.
    var cacheTag: String { rawValue }

    /// FFmpeg's codec name of the picture stream. MPEG-4 Part 2 (DivX, XviD) and the
    /// older Microsoft MPEG-4 (DivX 3) are mpeg4, MPEG-1 and MPEG-2 are mpeg2.
    static func detect(ffmpegName name: String) -> SourceCodec {
        switch name.lowercased() {
        case "h264": .h264
        case "mpeg4", "msmpeg4v1", "msmpeg4v2", "msmpeg4v3", "msmpeg4": .mpeg4
        case "mpeg1video", "mpeg2video": .mpeg2
        default: .unknown
        }
    }
}

/// The codec menu: automatic follows the open film, the rest is fixed.
nonisolated enum CodecChoice: String, CaseIterable, Identifiable, Sendable {
    case auto, h264, mpeg4, mpeg2, unknown

    var id: String { rawValue }

    var fixed: SourceCodec? { SourceCodec(rawValue: rawValue) }

    var title: String { fixed?.title ?? "Automatisch" }

    func resolve(film: SourceCodec?) -> SourceCodec {
        fixed ?? film ?? .unknown
    }

    static func parse(_ text: String) -> CodecChoice? {
        switch text.lowercased() {
        case "auto", "automatisch": .auto
        case "h264", "avc", "x264": .h264
        case "mpeg4", "mpeg-4", "divx", "xvid": .mpeg4
        case "mpeg2", "mpeg-2", "mpeg1": .mpeg2
        case "unknown", "unbekannt": .unknown
        default: nil
        }
    }
}
