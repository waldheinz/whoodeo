import AppKit
import AVFoundation
import CoreMedia
import os

/// Which picture the screen shows: the upscale, the original, or the upscale of the comparison model.
nonisolated enum ScreenSource: Sendable {
    case upscale, original, comparison
}

/// What the screen can show for one film, timed on the film clock: the stored upscale, the decoded
/// original, and the stored upscale of a comparison model.
nonisolated struct Reel: Sendable {
    let store: FrameStore
    let originals: Originals?
    /// The archive of the comparison model. The session computes a missing frame into it before showing it.
    let comparison: FrameStore?
    let fps: Double
    let count: Int

    func has(_ frame: Int, source: ScreenSource) -> Bool {
        guard frame >= 0, frame < count else { return false }
        switch source {
        case .upscale: return store.contains(frame)
        case .original: return originals != nil
        case .comparison: return comparison?.contains(frame) == true
        }
    }

    func sample(_ frame: Int, source: ScreenSource) throws -> CMSampleBuffer {
        let time = Self.time(frame, fps: fps)
        let duration = Self.time(1, fps: fps)
        switch source {
        case .original:
            guard let originals else { throw CLIError("Original fehlt") }
            return try originals.sample(at: frame, time: time, duration: duration)
        case .comparison:
            guard let comparison else { throw CLIError("Vergleichsmodell fehlt") }
            return try comparison.sample(at: frame, time: time, duration: duration)
        case .upscale:
            return try store.sample(at: frame, time: time, duration: duration)
        }
    }

    static func time(_ frame: Int, fps: Double) -> CMTime {
        let ticks = (Double(frame) * 600 / max(fps, 1)).rounded()
        return CMTime(value: CMTimeValue(ticks), timescale: 600)
    }
}

/// The low-res film at its own size. It has its own reader, so it never waits for the model queue.
nonisolated final class Originals: @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()
    private var film: Film?

    init(url: URL) {
        self.url = url
    }

    /// One original frame as 16-bit RGB, tagged BT.709 like the cache. The layer scales it.
    /// The color profile matters: without it the layer renders 709 darker than the ProRes upscales,
    /// whose decoder attaches the same profile.
    func sample(at index: Int, time: CMTime, duration: CMTime) throws -> CMSampleBuffer {
        lock.lock()
        defer { lock.unlock() }
        if film == nil { film = try Film.open(url) }
        let decoded = try film!.decode(at: index)
        let buffer = try Pixels.rgba16(decoded.rgb, width: decoded.width, height: decoded.height)
        CVBufferRemoveAllAttachments(buffer)
        CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        if let tags = CVBufferCopyAttachments(buffer, .shouldPropagate),
           let space = CVImageBufferCreateColorSpaceFromAttachments(tags)?.takeRetainedValue() {
            CVBufferSetAttachment(buffer, kCVImageBufferCGColorSpaceKey, space, .shouldPropagate)
        }
        var format: CMVideoFormatDescription?
        var status = CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: buffer,
            formatDescriptionOut: &format
        )
        guard status == noErr, let format else { throw CLIError("Original \(index + 1) ohne Bildformat (\(status))") }
        var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        status = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: buffer,
            formatDescription: format,
            sampleTiming: &timing,
            sampleBufferOut: &sample
        )
        guard status == noErr, let sample else { throw CLIError("Original \(index + 1) (\(status))") }
        return sample
    }
}

/// The only way the picture reaches the screen. Stills and playback both go through the system
/// video display: a still is one sample, playback a stream on a clock. The layer decodes ProRes
/// and scales the picture. It shows one source at a time, the upscale, the original or the comparison;
/// switching flushes and feeds the other source from the frame on screen.
/// The clock waits until the renderer has decoded a cushion. A realtime clock that starts
/// earlier drops frames that are already late and still advances at the film's speed.
@MainActor
final class FrameScreen {
    enum Event {
        case show(Int)
        case needs(Int)
        case ended
        case failed(String)
    }

    private let host = VideoHost()
    private let display = AVSampleBufferDisplayLayer()
    private let synchronizer = AVSampleBufferRenderSynchronizer()
    private var receiver: AVSampleBufferVideoRenderer.Receiver?
    private var observer: Any?
    private var feed: Task<Void, Never>?
    nonisolated private let tokenLock = OSAllocatedUnfairLock(initialState: 0)
    nonisolated private let cursorLock = OSAllocatedUnfairLock(initialState: FeedCursor())
    nonisolated private let failures = OSAllocatedUnfairLock(initialState: 0)
    private weak var view: PictureView?
    private var onEvent: ((Event) -> Void)?
    private var playing = false
    private var reel: Reel?
    /// The original or the comparison instead of the upscale, while the mouse button is down.
    private(set) var source = ScreenSource.upscale
    /// The still on screen, or where playback started.
    private var frame = 0
    private var clockStarted = false
    /// Ignore timebase callbacks until they land on this play. A seek leaves the old time in place
    /// until setRate applies, and that stale time would pause the new clock.
    private var awaitingAnchor = false
    private var holding = false
    private var fps = 25.0
    private var count = 0
    private var startIndex = 0
    private var enqueued = 0
    private var lastShown = 0

    func attach(to view: PictureView) {
        self.view = view
        if host.superview !== view {
            view.addSubview(host)
        }
        if display.superlayer == nil {
            host.wantsLayer = true
            host.layer?.backgroundColor = nil
            host.layer?.isOpaque = false
            display.videoGravity = .resize
            display.actions = [
                "bounds": NSNull(),
                "position": NSNull(),
                "frame": NSNull(),
                "hidden": NSNull(),
                "contents": NSNull(),
                "opacity": NSNull(),
            ]
            host.layer?.addSublayer(display)
        }
        if receiver == nil {
            receiver = synchronizer.sampleBufferReceiver(adding: display.sampleBufferRenderer)
            synchronizer.delaysRateChangeUntilHasSufficientMediaData = true
        }
        layout(in: view.bounds)
        updateScale()
        setSharp(view.sharp)
    }

    func layout(in bounds: CGRect) {
        guard host.frame != bounds || display.frame != host.bounds else { return }
        host.frame = bounds
        display.frame = host.bounds
        updateScale()
    }

    func updateScale() {
        let scale = view?.window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
        if host.layer?.contentsScale != scale {
            host.layer?.contentsScale = scale
        }
        if display.contentsScale != scale {
            display.contentsScale = scale
        }
    }

    func setSharp(_ sharp: Bool) {
        let filter: CALayerContentsFilter = sharp ? .nearest : .linear
        display.magnificationFilter = filter
        display.minificationFilter = filter
    }

    /// Switch the source at the frame on screen. Playback keeps running.
    /// `reel` is the current one; the comparison may have been chosen after this picture went up.
    func setSource(_ next: ScreenSource, reel current: Reel) {
        guard next != source else { return }
        source = next
        guard reel != nil else { return }
        reel = current
        if playing {
            start(at: currentFrame(), keepPicture: true)
        } else {
            show(frame, reel: current)
        }
    }

    /// One frame as a still, from `source` when given. A frame the source does not have leaves the picture as it is.
    func show(_ index: Int, reel: Reel, source next: ScreenSource? = nil) {
        if let next { source = next }
        shutdownFeed()
        self.reel = reel
        fps = max(reel.fps, 1)
        count = reel.count
        frame = index
        guard reel.has(index, source: source), let receiver else { return }
        let token = invalidate()
        let source = source
        synchronizer.setRate(0, time: Reel.time(index, fps: fps))
        let previous = feed
        feed = Task.detached {
            await previous?.value
            guard let sample = try? reel.sample(index, source: source) else { return }
            Self.displayImmediately(sample)
            await receiver.flush(removingDisplayedImage: false)
            guard self.tokenMatches(token) else { return }
            _ = try? await receiver.enqueue(CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample))
        }
    }

    /// Start playback at `index`. Samples use film time, not the order they were stored.
    func play(from index: Int, reel: Reel, on event: @escaping (Event) -> Void) {
        onEvent = event
        self.reel = reel
        start(at: index, keepPicture: false)
    }

    /// Stop the clock and return the frame on screen. The picture stays.
    func pause() -> Int {
        guard playing else { return frame }
        let shown = currentFrame()
        shutdownFeed()
        frame = shown
        return shown
    }

    /// Seek and step. The picture stays until the next still replaces it.
    func yield() {
        guard playing else { return }
        frame = currentFrame()
        shutdownFeed()
    }

    /// Drop the picture. Used when the film or the model goes away.
    func stop() {
        shutdownFeed()
        reel = nil
        let receiver = receiver
        let previous = feed
        feed = Task { @MainActor in
            await previous?.value
            await receiver?.flush(removingDisplayedImage: true)
        }
    }

    /// The frame the clock is on. Before the clock runs, the first frame of this play.
    private func currentFrame() -> Int {
        guard playing, clockStarted else { return startIndex }
        let raw = frameIndex(synchronizer.currentTime())
        return min(max(raw, startIndex), max(enqueued - 1, startIndex))
    }

    private func start(at index: Int, keepPicture: Bool) {
        shutdownFeed()
        guard let reel, let receiver else {
            onEvent?(.failed("Die Video-Anzeige fehlt."))
            return
        }
        fps = max(reel.fps, 1)
        count = reel.count
        frame = index
        startIndex = index
        lastShown = index
        enqueued = index
        failures.withLock { $0 = 0 }
        playing = true
        display.preventsDisplaySleepDuringVideoPlayback = true
        // The first sample shows as soon as it is decoded; the clock starts after the cushion.
        synchronizer.setRate(0, time: mediaTime(index))
        let token = invalidate()
        cursorLock.withLock { state in
            state.token = token
            state.next = index
            state.wake = true
            state.signaled = false
        }
        let source = source
        let previous = feed
        feed = Task { @MainActor in
            await previous?.value
            // Drop the queued samples. A seek also drops the picture, or it flashes the old frame.
            await receiver.flush(removingDisplayedImage: !keepPicture)
            guard self.tokenMatches(token), self.playing else { return }
            self.installObserver()
            let screen = self
            self.feed = Task.detached {
                await screen.read(token: token, receiver: receiver, reel: reel, source: source, from: index)
            }
        }
    }

    private func shutdownFeed() {
        playing = false
        clockStarted = false
        awaitingAnchor = false
        holding = false
        synchronizer.rate = 0
        display.preventsDisplaySleepDuringVideoPlayback = false
        _ = invalidate()
        feed?.cancel()
        if let observer {
            synchronizer.removeTimeObserver(observer)
            self.observer = nil
        }
    }

    private func invalidate() -> Int {
        tokenLock.withLock { value in
            value += 1
            return value
        }
    }

    private nonisolated func tokenMatches(_ token: Int) -> Bool {
        tokenLock.withLock { $0 == token }
    }

    private func installObserver() {
        if let observer {
            synchronizer.removeTimeObserver(observer)
            self.observer = nil
        }
        let interval = CMTime(seconds: 1 / fps, preferredTimescale: 600)
        observer = synchronizer.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            MainActor.assumeIsolated {
                self?.consider(time)
            }
        }
    }

    private func thin(_ next: Int) -> Bool {
        next >= count || reel?.has(next, source: source) != true
    }

    /// Wait for decoded ProRes before the clock runs. Originals arrive decoded already.
    private func preroll(_ next: Int) -> Bool {
        source != .original && !thin(next)
    }

    private func startClock() {
        guard playing, !clockStarted else { return }
        enqueued = cursorLock.withLock { $0.next }
        clockStarted = true
        awaitingAnchor = true
        // A gap has nothing more to preroll. A cached span waits until the renderer has decoded enough.
        synchronizer.delaysRateChangeUntilHasSufficientMediaData = preroll(enqueued)
        synchronizer.setRate(1, time: mediaTime(startIndex))
        cursorLock.withLock { $0.wake = false }
    }

    /// Continues after a gap. A cached span waits for decoded media; a gap starts on the frame in hand.
    private func resumeTiming() {
        enqueued = cursorLock.withLock { $0.next }
        holding = false
        synchronizer.delaysRateChangeUntilHasSufficientMediaData = preroll(enqueued)
        synchronizer.rate = 1
        cursorLock.withLock { $0.wake = false }
    }

    private func consider(_ time: CMTime) {
        guard playing, clockStarted else { return }
        let end = mediaTime(count)
        if CMTimeCompare(time, end) >= 0 {
            onEvent?(.ended)
            return
        }
        if CMTimeCompare(time, mediaTime(startIndex)) < 0 { return }
        enqueued = cursorLock.withLock { $0.next }
        let raw = frameIndex(time)
        if awaitingAnchor {
            guard raw >= startIndex, raw <= max(enqueued, startIndex) else { return }
            awaitingAnchor = false
        }
        let shown = enqueued > startIndex ? min(raw, max(enqueued - 1, startIndex)) : startIndex
        lastShown = shown
        onEvent?(.show(shown))
        if raw >= enqueued, enqueued < count {
            if synchronizer.rate > 0 {
                synchronizer.rate = 0
                holding = true
                cursorLock.withLock { $0.wake = true }
            }
            onEvent?(.needs(enqueued))
        }
    }

    /// The feed calls this when the clock still needs a nudge. The enqueue loop itself does not wait for the main actor.
    private func applyCursor(_ token: Int) {
        guard tokenMatches(token), playing else { return }
        let next = cursorLock.withLock { state -> Int in
            state.signaled = false
            return state.next
        }
        guard next >= enqueued else { return }
        enqueued = next
        let cushion = next - startIndex
        let blocked = thin(next)
        // The renderer queues only a few decoded originals, fewer than the ProRes cushion.
        let wanted = source == .original ? 2 : 8
        if !clockStarted, cushion >= wanted || (blocked && cushion >= 1) {
            startClock()
        } else if holding, next > lastShown {
            resumeTiming()
        }
    }

    private func mediaTime(_ frame: Int) -> CMTime {
        Reel.time(frame, fps: fps)
    }

    private func frameIndex(_ time: CMTime) -> Int {
        guard time.isValid, fps > 0 else { return startIndex }
        let frame = Int((time.seconds * fps + 0.001).rounded(.down))
        return min(max(frame, 0), max(count - 1, 0))
    }

    private func fail(_ message: String, token: Int) {
        guard tokenMatches(token), playing else { return }
        onEvent?(.failed(message))
    }

    /// Off the main actor. `enqueue` waits until the display wants another sample.
    private nonisolated func read(
        token: Int,
        receiver: AVSampleBufferVideoRenderer.Receiver,
        reel: Reel,
        source: ScreenSource,
        from start: Int
    ) async {
        var cursor = start
        var strikes = 0
        while !Task.isCancelled {
            if !tokenMatches(token) { return }
            if cursor >= reel.count { return }
            if !reel.has(cursor, source: source) {
                let missing = cursor
                await MainActor.run {
                    guard tokenMatches(token) else { return }
                    onEvent?(.needs(missing))
                }
                try? await Task.sleep(for: .milliseconds(5))
                continue
            }
            do {
                let sample = try reel.sample(cursor, source: source)
                let ready = CMReadySampleBuffer<CMSampleBuffer.DynamicContent>(unsafeBuffer: sample)
                let result = try await receiver.enqueue(ready)
                switch result {
                case .enqueued:
                    strikes = 0
                    cursor += 1
                    publish(cursor, token: token)
                case .enqueuedWithDecodeFailures(let errors):
                    strikes = 0
                    noteFailure(cursor, message: errors.first?.localizedDescription ?? "Decode")
                    cursor += 1
                    publish(cursor, token: token)
                case .cancelledDueToFlush, .cancelledDueToFlushRequiredToResume:
                    return
                case .cancelledDueToError(let error):
                    let message = error.localizedDescription
                    await MainActor.run { fail(message, token: token) }
                    return
                @unknown default:
                    return
                }
            } catch {
                strikes += 1
                if strikes >= 3 {
                    let message = error.localizedDescription
                    await MainActor.run { fail(message, token: token) }
                    return
                }
                try? await Task.sleep(for: .milliseconds(10))
            }
        }
    }

    /// Records the next frame for the clock. Hops to the main actor only while the clock is waiting to start or to resume.
    private nonisolated func publish(_ next: Int, token: Int) {
        let hop = cursorLock.withLock { state -> Bool in
            guard state.token == token else { return false }
            state.next = next
            guard state.wake, !state.signaled else { return false }
            state.signaled = true
            return true
        }
        if hop {
            Task { @MainActor in
                applyCursor(token)
            }
        }
    }

    private nonisolated func noteFailure(_ frame: Int, message: String) {
        let first = failures.withLock { count -> Bool in
            count += 1
            return count == 1
        }
        if first {
            Log.play.error("Bild \(frame + 1, privacy: .public) decodiert nicht: \(message, privacy: .public)")
        }
    }

    /// A still ignores the clock.
    private nonisolated static func displayImmediately(_ sample: CMSampleBuffer) {
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: true),
              CFArrayGetCount(attachments) > 0 else { return }
        let first = unsafeBitCast(CFArrayGetValueAtIndex(attachments, 0), to: CFMutableDictionary.self)
        CFDictionarySetValue(
            first,
            Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
            Unmanaged.passUnretained(kCFBooleanTrue).toOpaque()
        )
    }
}

private nonisolated struct FeedCursor: Sendable {
    var token = 0
    var next = 0
    var wake = false
    var signaled = false
}

private final class VideoHost: NSView {
    override var isFlipped: Bool { false }

    /// Clicks belong to the picture view underneath.
    override func hitTest(_: NSPoint) -> NSView? { nil }
}
