import CoreGraphics
import Foundation

nonisolated struct Shown {
    var index: Int
    var recon: Picture?
    var source: Picture?
    var milliseconds: Double
    var width: Int
    var height: Int
    var fromCache: Bool
    var stored: Bool
    /// Further cached frames decoded in the same playback trip.
    var ahead: [Shown] = []
}

nonisolated enum FrameNeed {
    /// Show this frame, from the archive or by running the model. Includes the original.
    case display
    /// Playback picture only. A cached frame does not decode the source film.
    case play
    /// Store this frame and skip the preview pictures.
    case fill
    /// Show this frame only when the archive already has it.
    case cached
}

/// The main actor only enqueues work. `queue` runs the model, `playback` decodes cached frames, and `lock` covers the request slot.
nonisolated final class Engine: @unchecked Sendable {
    private let queue = DispatchQueue(label: "whoodeo.engine")
    /// Cached playback. Kept off `queue`, so a running model is not the thing that decodes the blue span.
    private let playback = DispatchQueue(label: "whoodeo.playback")
    private let lock = NSLock()
    private var wanted: Int?
    private var handler: ((Result<Shown, Error>) -> Void)?
    private var store: FrameStore?
    private var need = FrameNeed.display
    private var runLimit = 1
    private var busy = false
    private var hold = false
    private var film: Film?
    private var predictor: Predictor?
    private var cancelExport = false

    func openFilm(_ url: URL, done: @escaping (Result<FilmFacts, Error>) -> Void) {
        queue.async {
            do {
                let film = try Self.wait { try await Film.load(url) }
                self.film = film
                done(.success(film.facts))
            } catch {
                done(.failure(error))
            }
        }
    }

    func openModel(_ url: URL, unit: Compute, done: @escaping (Result<PackageInfo, Error>) -> Void) {
        queue.async {
            do {
                let predictor = try Predictor(url: url, unit: unit)
                self.predictor = predictor
                done(.success(predictor.info))
            } catch {
                done(.failure(error))
            }
        }
    }

    func setUnit(_ unit: Compute, url: URL, done: @escaping (Result<PackageInfo, Error>) -> Void) {
        openModel(url, unit: unit, done: done)
    }

    func request(_ index: Int, store: FrameStore?, need: FrameNeed, run: Int = 1, done: @escaping (Result<Shown, Error>) -> Void) {
        lock.lock()
        // A request that has not started yet is replaced. Tell its caller, or the
        // session keeps believing that fill is still running and never asks again.
        let displaced = handler
        wanted = index
        handler = done
        self.store = store
        self.need = need
        self.runLimit = max(run, 1)
        let start = !busy && !hold
        if start { busy = true }
        lock.unlock()
        if let displaced {
            displaced(.failure(CancellationError()))
        }
        if start {
            queue.async { self.drain() }
        }
    }

    func export(
        url: URL,
        codec: Codec,
        range: Range<Int>,
        store: FrameStore,
        progress: @escaping (Int, Int) -> Void,
        stored: @escaping (Int) -> Void,
        done: @escaping (Result<Void, Error>) -> Void
    ) {
        queue.async {
            self.lock.lock()
            self.hold = true
            self.cancelExport = false
            self.lock.unlock()
            defer {
                self.lock.lock()
                self.hold = false
                let resume = self.wanted != nil && !self.busy
                if resume { self.busy = true }
                self.lock.unlock()
                if resume { self.drain() }
            }
            guard let film = self.film, let predictor = self.predictor else {
                done(.failure(CLIError("Modell und Film fehlen")))
                return
            }
            do {
                try Export.write(
                    film: film,
                    predictor: predictor,
                    store: store,
                    url: url,
                    codec: codec,
                    range: range,
                    progress: progress,
                    stored: stored,
                    cancelled: {
                        self.lock.lock()
                        let stop = self.cancelExport
                        self.lock.unlock()
                        return stop
                    }
                )
                done(.success(()))
            } catch {
                done(.failure(error))
            }
        }
    }

    func cancelExportNow() {
        lock.lock()
        cancelExport = true
        lock.unlock()
    }

    private func drain() {
        while true {
            lock.lock()
            let index = wanted
            let done = handler
            let store = store
            let need = need
            let run = runLimit
            wanted = nil
            handler = nil
            if index == nil || done == nil {
                busy = false
                lock.unlock()
                return
            }
            lock.unlock()
            let result: Result<Shown, Error>
            do {
                let shown = try make(index!, store: store, need: need)
                result = .success(cushion(shown, index: index!, store: store, need: need, run: run))
            } catch {
                result = .failure(error)
            }
            done?(result)
        }
    }

    /// Decode a cached run for playback. Does not use the model queue or the request slot.
    func decodeCached(from index: Int, store: FrameStore, limit: Int, done: @escaping (Result<[Shown], Error>) -> Void) {
        let count = max(limit, 1)
        playback.async {
            do {
                var frames: [Shown] = []
                var cursor = index
                while frames.count < count {
                    guard store.contains(cursor) else { break }
                    let started = CFAbsoluteTimeGetCurrent()
                    let buffer = try store.image(at: cursor)
                    let picture = try Pixels.picture(buffer)
                    frames.append(Shown(
                        index: cursor,
                        recon: picture,
                        source: nil,
                        milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000,
                        width: picture.width,
                        height: picture.height,
                        fromCache: true,
                        stored: true
                    ))
                    cursor += 1
                }
                done(.success(frames))
            } catch {
                done(.failure(error))
            }
        }
    }

    /// The enlarged original for the frame on screen. Runs on the engine queue.
    func original(index: Int, width: Int, height: Int, done: @escaping (CGImage?) -> Void) {
        queue.async {
            guard let film = self.film, width > 0, height > 0 else {
                done(nil)
                return
            }
            done(try? self.sourcePicture(film: film, index: index, width: width, height: height).image)
        }
    }

    /// One playback trip decodes the cached run, up to `run` frames.
    /// A hole stops the run. The frame being watched is still computed when it is the first one.
    private func cushion(_ first: Shown, index: Int, store: FrameStore?, need: FrameNeed, run: Int) -> Shown {
        guard need == .play, run > 1, first.fromCache, first.recon != nil else { return first }
        var ahead: [Shown] = []
        var cursor = index + 1
        while ahead.count + 1 < run {
            guard let store, store.contains(cursor) else { break }
            do {
                let next = try make(cursor, store: store, need: .play)
                guard next.recon != nil else { break }
                ahead.append(next)
            } catch {
                break
            }
            cursor += 1
        }
        guard !ahead.isEmpty else { return first }
        var first = first
        first.ahead = ahead
        return first
    }

    private func make(_ index: Int, store: FrameStore?, need: FrameNeed) throws -> Shown {
        guard let film else { throw CLIError("kein Film") }
        let started = CFAbsoluteTimeGetCurrent()
        let withSource = need == .display
        if store?.contains(index) == true {
            if need == .fill {
                return Shown(index: index, recon: nil, source: nil, milliseconds: 0, width: 0, height: 0, fromCache: true, stored: true)
            }
            let image = try store!.image(at: index)
            let recon = try Pixels.picture(image)
            let source = withSource
                ? try sourcePicture(film: film, index: index, width: recon.width, height: recon.height)
                : nil
            return Shown(
                index: index,
                recon: recon,
                source: source,
                milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000,
                width: recon.width,
                height: recon.height,
                fromCache: true,
                stored: true
            )
        }
        if need == .cached {
            return Shown(index: index, recon: nil, source: nil, milliseconds: 0, width: 0, height: 0, fromCache: false, stored: false)
        }
        guard let predictor else { throw CLIError("kein Modell") }
        let window = try film.window(at: index, frames: predictor.info.inFrames)
        let predicted = try predictor.predict(
            stack: window.stack,
            width: window.current.width,
            height: window.current.height
        )
        var stored = false
        if let store {
            try store.put(rgb: predicted.image, width: predicted.width, height: predicted.height, index: index)
            stored = true
        }
        if need == .fill {
            return Shown(
                index: index,
                recon: nil,
                source: nil,
                milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000,
                width: predicted.width,
                height: predicted.height,
                fromCache: false,
                stored: stored
            )
        }
        let source = withSource
            ? try sourcePicture(film: film, index: index, width: predicted.width, height: predicted.height)
            : nil
        let recon = try Pixels.picture(planar: predicted.image, width: predicted.width, height: predicted.height)
        return Shown(
            index: index,
            recon: recon,
            source: source,
            milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000,
            width: predicted.width,
            height: predicted.height,
            fromCache: false,
            stored: stored
        )
    }

    private func sourcePicture(film: Film, index: Int, width: Int, height: Int) throws -> Picture {
        let current = try film.frame(at: index)
        let scale = max(width / max(current.width, 1), 1)
        let enlarged = Pixels.nearest(current.rgb, width: current.width, height: current.height, scale: scale)
        let sized = enlarged.count == width * height * 3
            ? enlarged
            : Pixels.nearest(current.rgb, width: current.width, height: current.height, scale: predictor?.info.scale ?? scale)
        return try Pixels.picture(planar: sized, width: width, height: height)
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
