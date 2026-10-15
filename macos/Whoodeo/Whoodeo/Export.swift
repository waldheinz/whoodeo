import AVFoundation
import AudioToolbox
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

enum Codec: String, CaseIterable, Identifiable {
    case hevc, prores, prores4444

    var id: String { rawValue }

    var title: String {
        switch self {
        case .hevc: "HEVC 10-bit"
        case .prores: "ProRes 422 HQ"
        case .prores4444: "ProRes 4444"
        }
    }
}

enum Export {
    static func write(
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        url: URL,
        codec: Codec,
        range: Range<Int>,
        progress: @escaping (Int, Int) -> Void,
        stored: @escaping (Int) -> Void,
        cancelled: @escaping () -> Bool
    ) throws {
        if range.isEmpty { throw CLIError("der Bereich ist leer") }
        let size = try dimensions(film: film, predictor: predictor, store: store, index: range.lowerBound, stored: stored)
        if size.width % 2 != 0 || size.height % 2 != 0 {
            throw CLIError("Ausgabe \(size.width)×\(size.height) ist ungerade")
        }
        if url.exists { try FileManager.default.removeItem(at: url) }
        if codec == .prores4444 {
            try writePassthrough(
                film: film, predictor: predictor, store: store, url: url, range: range,
                progress: progress, stored: stored, cancelled: cancelled
            )
            return
        }
        try writeEncoded(
            film: film, predictor: predictor, store: store, url: url, codec: codec,
            width: size.width, height: size.height, range: range,
            progress: progress, stored: stored, cancelled: cancelled
        )
    }

    private static func dimensions(
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        index: Int,
        stored: (Int) -> Void
    ) throws -> (width: Int, height: Int) {
        if let size = store.dimensions() { return size }
        _ = try materialize(index, film: film, predictor: predictor, store: store, stored: stored)
        guard let size = store.dimensions() else { throw CLIError("Ausgabe ohne Größe") }
        return size
    }

    /// Floats when the frame was just computed, so a transcode can skip the decode.
    private static func materialize(
        _ index: Int,
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        stored: (Int) -> Void
    ) throws -> (data: Data, rgb: [Float]?) {
        if let data = try store.data(at: index) { return (data, nil) }
        let window = try film.window(at: index, frames: predictor.info.inFrames)
        let predicted = try predictor.predict(
            stack: window.stack,
            width: window.current.width,
            height: window.current.height
        )
        let data = try store.put(rgb: predicted.image, width: predicted.width, height: predicted.height, index: index)
        stored(index)
        return (data, predicted.image)
    }

    private static func writePassthrough(
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        url: URL,
        range: Range<Int>,
        progress: @escaping (Int, Int) -> Void,
        stored: @escaping (Int) -> Void,
        cancelled: @escaping () -> Bool
    ) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: try store.videoFormat())
        guard writer.canAdd(video) else { throw CLIError("der Schreiber nimmt das Bild nicht an") }
        let pictures = writer.inputReceiver(for: video)
        let audioPipes = try audioInputs(film: film, writer: writer, range: range)
        try writer.start()
        writer.startSession(atSourceTime: film.time(for: range.lowerBound))
        let pulse = Pulse()
        try commit(writer: writer, url: url, pulse: pulse, cancelled: cancelled) {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    try await writeCachedPictures(
                        film: film, predictor: predictor, store: store, range: range,
                        receiver: pictures, progress: progress, stored: stored,
                        pulse: pulse, cancelled: cancelled
                    )
                }
                for pipe in audioPipes {
                    let output = pipe.output
                    let reader = pipe.reader
                    let receiver = pipe.receiver
                    group.addTask {
                        try await writeAudioSamples(
                            output: output, reader: reader, receiver: receiver,
                            pulse: pulse, cancelled: cancelled
                        )
                    }
                }
                try await group.waitForAll()
            }
        }
    }

    private static func writeEncoded(
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        url: URL,
        codec: Codec,
        width: Int,
        height: Int,
        range: Range<Int>,
        progress: @escaping (Int, Int) -> Void,
        stored: @escaping (Int) -> Void,
        cancelled: @escaping () -> Bool
    ) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings(codec, width: width, height: height, fps: film.facts.fps))
        let format = pixelFormat(codec)
        guard writer.canAdd(video) else { throw CLIError("der Schreiber nimmt das Bild nicht an") }
        var attributes = CVPixelBufferCreationAttributes(
            pixelFormatType: CVPixelFormatType(rawValue: format),
            size: CVImageSize(width: width, height: height)
        )
        attributes.backing = .ioSurface
        let pictures = writer.inputPixelBufferReceiver(for: video, pixelBufferAttributes: attributes)
        let audioPipes = try audioInputs(film: film, writer: writer, range: range)
        try writer.start()
        writer.startSession(atSourceTime: film.time(for: range.lowerBound))
        let transfer = try ColorTransfer()
        let pulse = Pulse()
        try commit(writer: writer, url: url, pulse: pulse, cancelled: cancelled) {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    try await writeEncodedPictures(
                        film: film, predictor: predictor, store: store, range: range,
                        width: width, height: height, format: format, transfer: transfer,
                        receiver: pictures, progress: progress, stored: stored,
                        pulse: pulse, cancelled: cancelled
                    )
                }
                for pipe in audioPipes {
                    let output = pipe.output
                    let reader = pipe.reader
                    let receiver = pipe.receiver
                    group.addTask {
                        try await writeAudioSamples(
                            output: output, reader: reader, receiver: receiver,
                            pulse: pulse, cancelled: cancelled
                        )
                    }
                }
                try await group.waitForAll()
            }
        }
    }

    /// The receiver suspends while its track runs ahead. Separate tasks let the other track continue.
    private static func commit(
        writer: AVAssetWriter,
        url: URL,
        pulse: Pulse,
        cancelled: () -> Bool,
        _ body: @escaping () async throws -> Void
    ) throws {
        do {
            try watch(writer: writer, pulse: pulse, cancelled: cancelled, body)
            try finish(writer)
        } catch {
            if writer.status == .writing { writer.cancelWriting() }
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    private static func watch(
        writer: AVAssetWriter,
        pulse: Pulse,
        cancelled: () -> Bool,
        _ body: @escaping () async throws -> Void
    ) throws {
        let done = DispatchSemaphore(value: 0)
        let failure = Failure()
        let task = Task {
            defer { done.signal() }
            do { try await body() }
            catch { failure.set(error) }
        }
        while done.wait(timeout: .now() + .milliseconds(50)) == .timedOut {
            let stop = cancelled()
            let stalled = pulse.idle() > 45
            if stop || stalled {
                task.cancel()
                if writer.status == .writing { writer.cancelWriting() }
                done.wait()
                throw CLIError(stop ? "abgebrochen" : "Export wartet auf den Schreiber")
            }
        }
        if let error = failure.get() { throw error }
    }

    private static func finish(_ writer: AVAssetWriter) throws {
        let finished = DispatchSemaphore(value: 0)
        writer.finishWriting { finished.signal() }
        finished.wait()
        if writer.status != .completed {
            throw writer.error ?? CLIError("Export unvollständig")
        }
    }

    private static func writeCachedPictures(
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        range: Range<Int>,
        receiver: AVAssetWriterInput.SampleBufferReceiver,
        progress: (Int, Int) -> Void,
        stored: (Int) -> Void,
        pulse: Pulse,
        cancelled: () -> Bool
    ) async throws {
        let total = range.count
        var done = 0
        for index in range {
            try Task.checkCancellation()
            if cancelled() { throw CLIError("abgebrochen") }
            let time = film.time(for: index)
            let prepared = try materialize(index, film: film, predictor: predictor, store: store, stored: stored)
            let sample = try store.sampleBuffer(prepared.data, time: time, duration: film.duration(at: index))
            try await receiver.append(CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample))
            done += 1
            pulse.tick()
            progress(done, total)
        }
        receiver.finish()
    }

    private static func writeEncodedPictures(
        film: Film,
        predictor: Predictor,
        store: FrameStore,
        range: Range<Int>,
        width: Int,
        height: Int,
        format: OSType,
        transfer: ColorTransfer,
        receiver: AVAssetWriterInput.PixelBufferReceiver,
        progress: (Int, Int) -> Void,
        stored: (Int) -> Void,
        pulse: Pulse,
        cancelled: () -> Bool
    ) async throws {
        let total = range.count
        var done = 0
        for index in range {
            try Task.checkCancellation()
            if cancelled() { throw CLIError("abgebrochen") }
            let time = film.time(for: index)
            let prepared = try materialize(index, film: film, predictor: predictor, store: store, stored: stored)
            let buffer: CVPixelBuffer
            if let rgb = prepared.rgb {
                buffer = try transfer.convert(rgb, width: width, height: height, format: format, pool: nil)
            } else {
                let cached = try store.image(at: index)
                buffer = try transfer.convert(cached, format: format, pool: nil)
            }
            try await receiver.append(CVReadOnlyPixelBuffer(unsafeBuffer: buffer), with: time)
            done += 1
            pulse.tick()
            progress(done, total)
        }
        receiver.finish()
    }

    private static func writeAudioSamples(
        output: AVAssetReaderTrackOutput,
        reader: AVAssetReader,
        receiver: AVAssetWriterInput.SampleBufferReceiver,
        pulse: Pulse,
        cancelled: () -> Bool
    ) async throws {
        while true {
            try Task.checkCancellation()
            if cancelled() { throw CLIError("abgebrochen") }
            guard let sample = output.copyNextSampleBuffer() else {
                if reader.status == .failed {
                    throw reader.error ?? CLIError("Ton bricht ab")
                }
                break
            }
            try await receiver.append(CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample))
            pulse.tick()
        }
        receiver.finish()
    }

    private static func videoSettings(_ codec: Codec, width: Int, height: Int, fps: Double) -> [String: Any] {
        var compression: [String: Any] = [
            AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
            AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
            AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            AVVideoExpectedSourceFrameRateKey: fps,
        ]
        let kind: AVVideoCodecType
        switch codec {
        case .hevc:
            kind = .hevc
            compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel
            compression[AVVideoAverageBitRateKey] = width * height * 4
        case .prores:
            kind = .proRes422HQ
        case .prores4444:
            kind = .proRes4444
        }
        return [
            AVVideoCodecKey: kind,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
        ]
    }

    private static func pixelFormat(_ codec: Codec) -> OSType {
        switch codec {
        case .hevc: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        case .prores: kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
        case .prores4444: kCVPixelFormatType_4444AYpCbCr16
        }
    }

    private struct AudioPipe {
        var output: AVAssetReaderTrackOutput
        var reader: AVAssetReader
        var receiver: AVAssetWriterInput.SampleBufferReceiver
    }

    private final class Pulse: @unchecked Sendable {
        private let lock = NSLock()
        private var mark = Date()

        func tick() {
            lock.lock()
            mark = Date()
            lock.unlock()
        }

        func idle() -> TimeInterval {
            lock.lock()
            defer { lock.unlock() }
            return Date().timeIntervalSince(mark)
        }
    }

    private final class Failure: @unchecked Sendable {
        private let lock = NSLock()
        private var error: Error?

        func set(_ error: Error) {
            lock.lock()
            self.error = error
            lock.unlock()
        }

        func get() -> Error? {
            lock.lock()
            defer { lock.unlock() }
            return error
        }
    }

    private static func audioInputs(film: Film, writer: AVAssetWriter, range: Range<Int>) throws -> [AudioPipe] {
        let begin = film.time(for: range.lowerBound)
        let end = film.time(for: range.upperBound)
        let span = CMTimeRange(start: begin, end: end)
        var pipes: [AudioPipe] = []
        for (track, facts) in film.audio {
            let passthrough = facts.subtype == kAudioFormatMPEG4AAC
                || facts.subtype == kAudioFormatLinearPCM
                || facts.subtype == kAudioFormatAppleLossless
            let settings = passthrough ? nil : aacSettings(facts)
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: settings)
            guard writer.canAdd(input) else {
                throw CLIError("Audio \(fourCC(facts.subtype)) passt nicht in den Film")
            }
            let receiver = writer.inputReceiver(for: input)
            let reader = try AVAssetReader(asset: film.asset)
            reader.timeRange = span
            let outputSettings: [String: Any]? = passthrough ? nil : [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsNonInterleaved: false,
                AVSampleRateKey: facts.sampleRate,
                AVNumberOfChannelsKey: facts.channels,
            ]
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
            guard reader.canAdd(output) else { throw CLIError("Audio lässt sich nicht lesen") }
            reader.add(output)
            guard reader.startReading() else { throw reader.error ?? CLIError("Audio startet nicht") }
            pipes.append(AudioPipe(output: output, reader: reader, receiver: receiver))
        }
        return pipes
    }

    private static func aacSettings(_ facts: AudioFacts) -> [String: Any] {
        let floor = facts.channels * 64_000
        let cap = facts.channels * 256_000
        let source = Int(facts.dataRate.rounded())
        let bitrate = min(cap, max(floor, source > 0 ? source : floor))
        return [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: facts.sampleRate,
            AVNumberOfChannelsKey: facts.channels,
            AVEncoderBitRateKey: bitrate,
        ]
    }

    private static func fourCC(_ value: AudioFormatID) -> String {
        let bytes: [UInt8] = [
            UInt8((value >> 24) & 0xff),
            UInt8((value >> 16) & 0xff),
            UInt8((value >> 8) & 0xff),
            UInt8(value & 0xff),
        ]
        return String(bytes: bytes, encoding: .ascii) ?? String(value)
    }
}

/// Gamma RGB in, limited-range BT.709 YCbCr out. VideoToolbox keeps the 10 bits.
final class ColorTransfer {
    private var session: VTPixelTransferSession?

    init() throws {
        var created: VTPixelTransferSession?
        let status = VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &created)
        guard status == noErr, let created else {
            throw CLIError("Farbwandlung startet nicht (\(status))")
        }
        session = created
        try set(kVTPixelTransferPropertyKey_DestinationColorPrimaries, kCVImageBufferColorPrimaries_ITU_R_709_2)
        try set(kVTPixelTransferPropertyKey_DestinationTransferFunction, kCVImageBufferTransferFunction_ITU_R_709_2)
        try set(kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, kCVImageBufferYCbCrMatrix_ITU_R_709_2)
        try set(kVTPixelTransferPropertyKey_DownsamplingMode, kVTDownsamplingMode_Average)
    }

    deinit {
        if let session {
            VTPixelTransferSessionInvalidate(session)
        }
    }

    func convert(_ buffer: CVPixelBuffer, format: OSType, pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        guard let session else { throw CLIError("Farbwandlung ist beendet") }
        let destination = try Self.destination(
            width: CVPixelBufferGetWidth(buffer),
            height: CVPixelBufferGetHeight(buffer),
            format: format,
            pool: pool
        )
        let status = VTPixelTransferSessionTransferImage(session, from: buffer, to: destination)
        guard status == noErr else { throw CLIError("Farbwandlung (\(status))") }
        return destination
    }

    func convert(_ rgb: [Float], width: Int, height: Int, format: OSType, pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        guard let session else { throw CLIError("Farbwandlung ist beendet") }
        let source = try Pixels.rgba16(rgb, width: width, height: height)
        CVBufferRemoveAttachment(source, kCVImageBufferCGColorSpaceKey)
        CVBufferSetAttachment(source, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(source, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        let destination = try Self.destination(width: width, height: height, format: format, pool: pool)
        let status = VTPixelTransferSessionTransferImage(session, from: source, to: destination)
        guard status == noErr else { throw CLIError("Farbwandlung (\(status))") }
        return destination
    }

    private func set(_ key: CFString, _ value: CFString) throws {
        guard let session else { throw CLIError("Farbwandlung ist beendet") }
        let status = VTSessionSetProperty(session, key: key, value: value)
        guard status == noErr else { throw CLIError("Farbwandlung (\(status))") }
    }

    private static func destination(width: Int, height: Int, format: OSType, pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status: CVReturn
        if let pool {
            status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        } else {
            let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
            status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs as CFDictionary, &buffer)
        }
        guard status == kCVReturnSuccess, let buffer else {
            throw CLIError("Pixelpuffer \(width)×\(height) fehlt (\(status))")
        }
        return buffer
    }
}

private extension URL {
    var exists: Bool { FileManager.default.fileExists(atPath: path) }
}
