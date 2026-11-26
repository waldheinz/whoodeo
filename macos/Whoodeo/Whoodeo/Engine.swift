import CoreGraphics
import Foundation

/// One frame the engine stored or found in the archive. The screen reads it from there.
nonisolated struct Shown {
    var index: Int
    var milliseconds: Double
    var width: Int
    var height: Int
    var fromCache: Bool
    var stored: Bool
}

/// The main actor only enqueues work. `queue` runs the model, and `lock` covers the request slot.
/// The queue runs at utility, below the main thread, so background fill never competes with
/// scrolling or a pinch. A frame the user asks for raises its own work to user-initiated.
nonisolated final class Engine: @unchecked Sendable {
    private let queue = DispatchQueue(label: "whoodeo.engine", qos: .utility)
    private let lock = NSLock()
    private var wanted: Int?
    private var handler: ((Result<Shown, Error>) -> Void)?
    private var store: FrameStore?
    private var busy = false
    private var hold = false
    private var film: Film?
    private var predictor: Predictor?
    private var cancelExport = false
    /// Set when the app quits: no new model work starts, so MLX is idle before `exit` tears down Metal.
    private var closing = false

    /// `map` is the preprocessing of the film; one made for another frame count is ignored.
    func openFilm(_ url: URL, filmID: String, map: FrameMap? = nil, done: @escaping (Result<FilmFacts, Error>) -> Void) {
        queue.async(qos: .userInitiated, flags: .enforceQoS) {
            do {
                let film = try Self.wait { try await Film.load(url, map: map) }
                self.film = film
                done(.success(film.facts))
            } catch {
                done(.failure(error))
            }
        }
    }

    /// Switch the preprocessing of the open film. Runs after the requests already queued.
    func applyMap(_ map: FrameMap?, done: @escaping (Result<FilmFacts, Error>) -> Void) {
        queue.async(qos: .userInitiated, flags: .enforceQoS) {
            guard let film = self.film else {
                done(.failure(CLIError("kein Film")))
                return
            }
            film.apply(map)
            done(.success(film.facts))
        }
    }

    func openModel(_ url: URL, condition: SourceCodec? = nil, done: @escaping (Result<PackageInfo, Error>) -> Void) {
        queue.async(qos: .userInitiated, flags: .enforceQoS) {
            do {
                let predictor = try Predictor(url: url)
                if let condition { predictor.setCondition(condition) }
                self.predictor = predictor
                done(.success(predictor.info))
            } catch {
                done(.failure(error))
            }
        }
    }

    /// Set the condition vector of a `-cond` package. Runs on the model queue, before later requests.
    func setCondition(_ codec: SourceCodec) {
        queue.async(qos: .userInitiated, flags: .enforceQoS) {
            self.predictor?.setCondition(codec)
        }
    }

    /// `qos` is how urgent the frame is: `.utility` for background fill, `.userInitiated` for a
    /// frame on screen. Enforced, so fill does not inherit the main thread's priority.
    func request(
        _ index: Int,
        store: FrameStore?,
        qos: DispatchQoS = .userInitiated,
        done: @escaping (Result<Shown, Error>) -> Void
    ) {
        lock.lock()
        if closing {
            lock.unlock()
            return
        }
        // A request that has not started yet is replaced. Tell its caller, or the
        // session keeps believing that fill is still running and never asks again.
        let displaced = handler
        wanted = index
        handler = done
        self.store = store
        let start = !busy && !hold
        if start { busy = true }
        lock.unlock()
        if let displaced {
            displaced(.failure(CancellationError()))
        }
        if start {
            queue.async(execute: DispatchWorkItem(qos: qos, flags: .enforceQoS) { self.drain() })
        } else if qos.qosClass.rawValue.rawValue > DispatchQoS.QoSClass.utility.rawValue.rawValue {
            // The slot is taken over by a running utility drain. An empty item at the higher QoS
            // behind it lifts that drain until it has run this request.
            queue.async(execute: DispatchWorkItem(qos: qos, flags: .enforceQoS) {})
        }
    }

    func export(
        url: URL,
        codec: Codec,
        range: Range<Int>,
        grade: ColorGrade = .neutral,
        store: FrameStore,
        progress: @escaping (Int, Int) -> Void,
        stored: @escaping (Int) -> Void,
        done: @escaping (Result<Void, Error>) -> Void
    ) {
        queue.async(qos: .userInitiated, flags: .enforceQoS) {
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
                    grade: grade,
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

    /// Stop for quitting: drop the waiting request, cancel an export, and refuse new work.
    /// Returns once the model queue has finished what it was running, or after `timeout` seconds.
    /// Call it off the main thread; work on the queue may still report to the main actor.
    @discardableResult
    func shutdown(timeout: TimeInterval) -> Bool {
        lock.lock()
        closing = true
        cancelExport = true
        wanted = nil
        handler = nil
        store = nil
        lock.unlock()
        let idle = DispatchSemaphore(value: 0)
        queue.async { idle.signal() }
        return idle.wait(timeout: .now() + timeout) == .success
    }

    func cancelExportNow() {
        lock.lock()
        cancelExport = true
        lock.unlock()
    }

    /// Forget the frame store and return once both queues have finished work that already holds one.
    /// `done` runs on the engine queue.
    func settle(done: @escaping () -> Void) {
        lock.lock()
        let displaced = handler
        wanted = nil
        handler = nil
        store = nil
        lock.unlock()
        displaced?(.failure(CancellationError()))
        queue.async { [weak self] in
            guard let self else {
                done()
                return
            }
            self.lock.lock()
            self.store = nil
            self.lock.unlock()
            done()
        }
    }

    private func drain() {
        while true {
            lock.lock()
            if closing {
                wanted = nil
                handler = nil
            }
            let index = wanted
            let done = handler
            let store = store
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
                result = .success(try make(index!, store: store))
            } catch {
                result = .failure(error)
            }
            done?(result)
        }
    }

    /// Runs the model for a frame the archive does not have yet and stores the result.
    private func make(_ index: Int, store: FrameStore?) throws -> Shown {
        guard film != nil else { throw CLIError("kein Film") }
        let started = CFAbsoluteTimeGetCurrent()
        if let store, store.contains(index) {
            let size = store.dimensions()
            return Shown(index: index, milliseconds: 0, width: size?.width ?? 0, height: size?.height ?? 0, fromCache: true, stored: true)
        }
        guard let film, let predictor else { throw CLIError("kein Modell") }
        let window = try Self.wait { try await film.window(at: index, frames: predictor.info.inFrames) }
        let predicted = try predictor.predict(
            stack: window.stack,
            width: window.current.width,
            height: window.current.height,
            frames: window.frames
        )
        var stored = false
        if let store {
            try store.put(rgb: predicted.image, width: predicted.width, height: predicted.height, index: index)
            stored = true
        }
        return Shown(
            index: index,
            milliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000,
            width: predicted.width,
            height: predicted.height,
            fromCache: false,
            stored: stored
        )
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
