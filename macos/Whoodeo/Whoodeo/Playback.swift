import AppKit
import AVFoundation
import CoreMedia
import os

/// Plays cached ProRes 4444 by handing the access units to the system video display.
/// The layer decodes and the window compositor scans the picture out. Playback does not
/// build a 16-bit bitmap for each frame.
/// The clock waits until the renderer has decoded a cushion. A realtime clock that starts
/// earlier drops frames that are already late and still advances at the film's speed.
@MainActor
final class CacheScreen {
    enum Event {
        case show(Int)
        case needs(Int)
        case visible
        case ended
        case failed(String)
    }

    private let host = VideoHost()
    private let display = AVSampleBufferDisplayLayer()
    private let synchronizer = AVSampleBufferRenderSynchronizer()
    private var receiver: AVSampleBufferVideoRenderer.Receiver?
    private var observer: Any?
    private var readyObserver: NSObjectProtocol?
    private var feed: Task<Void, Never>?
    nonisolated private let tokenLock = OSAllocatedUnfairLock(initialState: 0)
    nonisolated private let cursorLock = OSAllocatedUnfairLock(initialState: FeedCursor())
    nonisolated private let failures = OSAllocatedUnfairLock(initialState: 0)
    private weak var view: PixelView?
    private var onEvent: ((Event) -> Void)?
    private var mode = Mode.stopped
    private var coveredByOriginal = false
    /// The last decoded video frame stays up until the still picture replaces it.
    private var holdingLastFrame = false
    private var clockStarted = false
    /// Ignore timebase callbacks until they land on this play. A seek leaves the old time in place
    /// until setRate applies, and that stale time would pause the new clock.
    private var awaitingAnchor = false
    private var holding = false
    private var announced = false
    private var fps = 25.0
    private var count = 0
    private var startIndex = 0
    private var enqueued = 0
    private var lastShown = 0
    private var store: FrameStore?

    private enum Mode {
        case stopped, playing, paused
    }

    var isRunning: Bool { mode == .playing || mode == .paused }

    /// True while the video layer, not the still bitmap, is what the user sees.
    var isCoveringStill: Bool { !host.isHidden }

    func attach(to view: PixelView) {
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
            readyObserver = NotificationCenter.default.addObserver(
                forName: .AVSampleBufferDisplayLayerReadyForDisplayDidChange,
                object: display,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.updateVisibility()
                }
            }
        }
        if receiver == nil {
            receiver = synchronizer.sampleBufferReceiver(adding: display.sampleBufferRenderer)
            synchronizer.delaysRateChangeUntilHasSufficientMediaData = true
        }
        layout(in: view.bounds)
        updateScale()
        setSharp(view.sharp)
        host.isHidden = true
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

    func setCoveredByOriginal(_ covered: Bool) {
        coveredByOriginal = covered
        updateVisibility()
    }

    /// Continue a pause, or start at `index`. Samples use film time, not the order they were stored.
    func play(from index: Int, fps: Double, store: FrameStore, count: Int, on event: @escaping (Event) -> Void) {
        onEvent = event
        if mode == .paused, feed != nil {
            mode = .playing
            display.preventsDisplaySleepDuringVideoPlayback = true
            if clockStarted {
                resumeTiming()
            } else {
                startClock()
            }
            updateVisibility()
            return
        }
        self.fps = max(fps, 1)
        self.count = count
        self.store = store
        startIndex = index
        lastShown = index
        enqueued = index
        clockStarted = false
        awaitingAnchor = false
        holding = false
        holdingLastFrame = false
        announced = false
        failures.withLock { $0 = 0 }
        mode = .playing
        synchronizer.delaysRateChangeUntilHasSufficientMediaData = true
        host.isHidden = true
        display.preventsDisplaySleepDuringVideoPlayback = true
        let token = invalidate()
        cursorLock.withLock { state in
            state.token = token
            state.next = index
            state.wake = true
            state.signaled = false
        }
        feed?.cancel()
        feed = nil
        if let observer {
            synchronizer.removeTimeObserver(observer)
            self.observer = nil
        }
        let rate = self.fps
        let total = count
        let cursor = index
        guard let receiver else {
            mode = .stopped
            onEvent?(.failed("Die Video-Anzeige fehlt."))
            return
        }
        // Drop the previous picture before new samples arrive, or a seek flashes the old frame.
        feed = Task { @MainActor in
            await receiver.flush(removingDisplayedImage: true)
            guard self.tokenMatches(token) else { return }
            guard self.mode == .playing || self.mode == .paused else { return }
            self.installObserver()
            let screen = self
            self.feed = Task.detached {
                await screen.read(
                    token: token,
                    receiver: receiver,
                    store: store,
                    fps: rate,
                    count: total,
                    from: cursor
                )
            }
        }
    }

    func pause() {
        guard mode == .playing else { return }
        mode = .paused
        synchronizer.rate = 0
        display.preventsDisplaySleepDuringVideoPlayback = false
        updateVisibility()
    }

    /// Seek and step. The last video frame stays until the still is ready.
    func yield() {
        guard mode != .stopped || holdingLastFrame else { return }
        shutdownFeed(removePicture: false)
        holdingLastFrame = display.isReadyForDisplay
        updateVisibility()
    }

    /// Drop the video frame. Used when the film or the picture goes away.
    func stop() {
        shutdownFeed(removePicture: true)
        holdingLastFrame = false
        host.isHidden = true
        view?.needsDisplay = true
    }

    /// The still picture is on screen, so the video layer gets out of its way.
    func stepAside() {
        holdingLastFrame = false
        guard mode == .stopped else { return }
        host.isHidden = true
        view?.needsDisplay = true
    }

    private func shutdownFeed(removePicture: Bool) {
        mode = .stopped
        clockStarted = false
        holding = false
        synchronizer.rate = 0
        display.preventsDisplaySleepDuringVideoPlayback = false
        _ = invalidate()
        feed?.cancel()
        feed = nil
        if let observer {
            synchronizer.removeTimeObserver(observer)
            self.observer = nil
        }
        let receiver = receiver
        if removePicture {
            Task { @MainActor in
                await receiver?.flush(removingDisplayedImage: true)
            }
        } else {
            receiver?.flush()
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

    private func startClock() {
        guard mode == .playing, !clockStarted else { return }
        enqueued = cursorLock.withLock { $0.next }
        clockStarted = true
        awaitingAnchor = true
        let thin = enqueued >= count || store?.contains(enqueued) != true
        // A gap has nothing more to preroll. A cached span waits until the renderer has decoded enough.
        synchronizer.delaysRateChangeUntilHasSufficientMediaData = !thin
        synchronizer.setRate(1, time: mediaTime(startIndex))
        cursorLock.withLock { $0.wake = false }
        updateVisibility()
    }

    /// Continues from the paused time. A cached span waits for decoded media; a gap starts on the frame in hand.
    private func resumeTiming() {
        enqueued = cursorLock.withLock { $0.next }
        holding = false
        let thin = enqueued >= count || store?.contains(enqueued) != true
        synchronizer.delaysRateChangeUntilHasSufficientMediaData = !thin
        synchronizer.rate = 1
        cursorLock.withLock { $0.wake = false }
    }

    private func consider(_ time: CMTime) {
        guard mode == .playing, clockStarted else { return }
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
        guard tokenMatches(token), mode == .playing || mode == .paused else { return }
        let next = cursorLock.withLock { state -> Int in
            state.signaled = false
            return state.next
        }
        guard next >= enqueued else { return }
        enqueued = next
        let cushion = next - startIndex
        let blocked = next >= count || store?.contains(next) != true
        if !clockStarted, mode == .playing, cushion >= 8 || (blocked && cushion >= 1) {
            startClock()
        } else if holding, next > lastShown, mode == .playing {
            resumeTiming()
        }
    }

    private func updateVisibility() {
        let show = !coveredByOriginal
            && display.isReadyForDisplay
            && (holdingLastFrame || (clockStarted && (mode == .playing || mode == .paused)))
        if host.isHidden == !show { return }
        host.isHidden = !show
        view?.needsDisplay = true
        if show, !announced, mode == .playing || mode == .paused {
            announced = true
            onEvent?(.visible)
        }
    }

    private func mediaTime(_ frame: Int) -> CMTime {
        let ticks = (Double(frame) * 600 / fps).rounded()
        return CMTime(value: CMTimeValue(ticks), timescale: 600)
    }

    private func frameIndex(_ time: CMTime) -> Int {
        guard time.isValid, fps > 0 else { return startIndex }
        let frame = Int((time.seconds * fps + 0.001).rounded(.down))
        return min(max(frame, 0), max(count - 1, 0))
    }

    private func fail(_ message: String, token: Int) {
        guard tokenMatches(token), mode == .playing else { return }
        onEvent?(.failed(message))
    }

    /// Off the main actor. `enqueue` waits until the display wants another sample.
    private nonisolated func read(
        token: Int,
        receiver: AVSampleBufferVideoRenderer.Receiver,
        store: FrameStore,
        fps: Double,
        count: Int,
        from start: Int
    ) async {
        var cursor = start
        var strikes = 0
        while !Task.isCancelled {
            if !tokenMatches(token) { return }
            if cursor >= count { return }
            if !store.contains(cursor) {
                let missing = cursor
                await MainActor.run {
                    guard tokenMatches(token) else { return }
                    onEvent?(.needs(missing))
                }
                try? await Task.sleep(for: .milliseconds(5))
                continue
            }
            do {
                let sample = try store.sample(at: cursor, time: Self.time(cursor, fps: fps), duration: Self.time(1, fps: fps))
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

    private nonisolated static func time(_ frame: Int, fps: Double) -> CMTime {
        let ticks = (Double(frame) * 600 / max(fps, 1)).rounded()
        return CMTime(value: CMTimeValue(ticks), timescale: 600)
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
