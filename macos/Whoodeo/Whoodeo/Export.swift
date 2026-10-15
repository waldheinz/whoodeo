import AVFoundation
import AudioToolbox
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

nonisolated enum Codec: String, CaseIterable, Identifiable {
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

nonisolated enum Export {
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
        stored: @escaping (Int) -> Void
    ) throws -> (width: Int, height: Int) {
        if let size = store.dimensions() { return size }
        _ = try wait { try await materialize(index, film: film, predictor: predictor, store: store, stored: stored) }
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
    ) async throws -> (data: Data, rgb: [Float]?) {
        if let data = try store.data(at: index) { return (data, nil) }
        let window = try await film.window(at: index, frames: predictor.info.inFrames)
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
                    let reader = pipe.reader
                    let receiver = pipe.receiver
                    group.addTask {
                        try await writeAudioSamples(
                            reader: reader, receiver: receiver,
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
                    let reader = pipe.reader
                    let receiver = pipe.receiver
                    group.addTask {
                        try await writeAudioSamples(
                            reader: reader, receiver: receiver,
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

    @concurrent
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
            let prepared = try await materialize(index, film: film, predictor: predictor, store: store, stored: stored)
            let sample = try store.sampleBuffer(prepared.data, time: time, duration: film.duration(at: index))
            try await receiver.append(CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample))
            done += 1
            pulse.tick()
            progress(done, total)
        }
        receiver.finish()
    }

    @concurrent
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
            let prepared = try await materialize(index, film: film, predictor: predictor, store: store, stored: stored)
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

    @concurrent
    private static func writeAudioSamples(
        reader: AudioReader,
        receiver: AVAssetWriterInput.SampleBufferReceiver,
        pulse: Pulse,
        cancelled: () -> Bool
    ) async throws {
        while true {
            try Task.checkCancellation()
            if cancelled() { throw CLIError("abgebrochen") }
            guard let sample = try reader.next() else { break }
            try await receiver.append(sample)
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
        var reader: AudioReader
        var receiver: AVAssetWriterInput.SampleBufferReceiver
    }

    /// PCM from libav, one film file of its own so it does not share the picture decoder.
    private final class AudioReader: @unchecked Sendable {
        private let audio: OpaquePointer
        private let channels: Int
        private let rate: Double
        private var done = false

        init(film: Film, track: AudioFacts, start: Double, end: Double) throws {
            var message = [CChar](repeating: 0, count: 512)
            let opened = message.withUnsafeMutableBufferPointer { buffer in
                film.facts.url.path.withCString {
                    source_audio_begin($0, Int32(track.track), start, end, buffer.baseAddress, Int32(buffer.count))
                }
            }
            guard let opened else {
                let detail = String(cString: message)
                throw CLIError(detail.isEmpty ? "Ton lässt sich nicht öffnen" : detail)
            }
            audio = opened
            channels = Int(source_audio_channels(opened))
            rate = source_audio_rate(opened)
        }

        deinit {
            source_audio_close(audio)
        }

        func next() throws -> CMReadySampleBuffer<CMSampleBuffer.DynamicContent>? {
            if done { return nil }
            let limit = 8192
            var samples = [Int16](repeating: 0, count: limit * channels)
            var frames: Int32 = 0
            var pts = 0.0
            let status = samples.withUnsafeMutableBufferPointer { buffer -> Int32 in
                guard let base = buffer.baseAddress else { return -1 }
                return source_audio_read(audio, base, Int32(limit), &frames, &pts)
            }
            if status < 0 { throw CLIError(String(cString: source_audio_error(audio))) }
            if status > 0 || frames == 0 {
                done = true
                return nil
            }
            return try Self.buffer(samples: samples, frames: Int(frames), channels: channels, rate: rate, pts: pts)
        }

        private static func buffer(
            samples: [Int16],
            frames: Int,
            channels: Int,
            rate: Double,
            pts: Double
        ) throws -> CMReadySampleBuffer<CMSampleBuffer.DynamicContent> {
            var description = AudioStreamBasicDescription(
                mSampleRate: rate,
                mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
                mBytesPerPacket: UInt32(channels * 2),
                mFramesPerPacket: 1,
                mBytesPerFrame: UInt32(channels * 2),
                mChannelsPerFrame: UInt32(channels),
                mBitsPerChannel: 16,
                mReserved: 0
            )
            var format: CMAudioFormatDescription?
            let made = CMAudioFormatDescriptionCreate(
                allocator: kCFAllocatorDefault,
                asbd: &description,
                layoutSize: 0,
                layout: nil,
                magicCookieSize: 0,
                magicCookie: nil,
                extensions: nil,
                formatDescriptionOut: &format
            )
            guard made == noErr, let format else { throw CLIError("Tonformat fehlt (\(made))") }
            let bytes = frames * channels * MemoryLayout<Int16>.size
            var block: CMBlockBuffer?
            var status = CMBlockBufferCreateWithMemoryBlock(
                allocator: kCFAllocatorDefault,
                memoryBlock: nil,
                blockLength: bytes,
                blockAllocator: kCFAllocatorDefault,
                customBlockSource: nil,
                offsetToData: 0,
                dataLength: bytes,
                flags: 0,
                blockBufferOut: &block
            )
            guard status == kCMBlockBufferNoErr, let block else { throw CLIError("Tonpuffer fehlt") }
            status = samples.withUnsafeBytes { raw in
                CMBlockBufferReplaceDataBytes(
                    with: raw.baseAddress!,
                    blockBuffer: block,
                    offsetIntoDestination: 0,
                    dataLength: bytes
                )
            }
            guard status == kCMBlockBufferNoErr else { throw CLIError("Tonpuffer fehlt") }
            let scale = CMTimeScale(max(1, rate.rounded()))
            var timing = CMSampleTimingInfo(
                duration: CMTime(value: 1, timescale: scale),
                presentationTimeStamp: CMTime(value: CMTimeValue((pts * rate).rounded()), timescale: scale),
                decodeTimeStamp: .invalid
            )
            var sample: CMSampleBuffer?
            status = CMSampleBufferCreate(
                allocator: kCFAllocatorDefault,
                dataBuffer: block,
                dataReady: true,
                makeDataReadyCallback: nil,
                refcon: nil,
                formatDescription: format,
                sampleCount: frames,
                sampleTimingEntryCount: 1,
                sampleTimingArray: &timing,
                sampleSizeEntryCount: 0,
                sampleSizeArray: nil,
                sampleBufferOut: &sample
            )
            guard status == noErr, let sample else { throw CLIError("Tonpuffer fehlt (\(status))") }
            return CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample)
        }
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
        let begin = film.time(for: range.lowerBound).seconds
        let end = film.time(for: range.upperBound).seconds
        var pipes: [AudioPipe] = []
        for facts in film.audio {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: aacSettings(facts))
            guard writer.canAdd(input) else { throw CLIError("Audio passt nicht in den Film") }
            let receiver = writer.inputReceiver(for: input)
            let reader = try AudioReader(film: film, track: facts, start: begin, end: end)
            pipes.append(AudioPipe(reader: reader, receiver: receiver))
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

    private static func wait<T>(_ body: @escaping () async throws -> T) throws -> T {
        var result: Result<T, Error>?
        let gate = DispatchSemaphore(value: 0)
        Task {
            do { result = .success(try await body()) }
            catch { result = .failure(error) }
            gate.signal()
        }
        gate.wait()
        return try result!.get()
    }

}

/// Gamma RGB in, limited-range BT.709 YCbCr out. VideoToolbox keeps the 10 bits.
nonisolated final class ColorTransfer {
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

private nonisolated extension URL {
    var exists: Bool { FileManager.default.fileExists(atPath: path) }
}
