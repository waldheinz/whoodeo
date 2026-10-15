import AVFoundation
import CoreMedia
import CryptoKit
import Foundation
import os
import VideoToolbox

/// Which frames of one film, model, and compute unit are already encoded.
nonisolated struct Coverage: Equatable {
    private(set) var count: Int
    private var words: [UInt64]

    init(count: Int) {
        self.count = max(count, 0)
        words = Array(repeating: 0, count: (self.count + 63) / 64)
    }

    func contains(_ index: Int) -> Bool {
        guard index >= 0, index < count else { return false }
        return words[index / 64] & (UInt64(1) << (index % 64)) != 0
    }

    mutating func insert(_ index: Int) {
        guard index >= 0, index < count, !contains(index) else { return }
        words[index / 64] |= UInt64(1) << (index % 64)
    }

    /// Frames already marked present. Bits past `count` are never set.
    var filled: Int {
        words.reduce(0) { $0 + $1.nonzeroBitCount }
    }

    /// Frames still absent in `lower..<upper`.
    func missing(from lower: Int, until upper: Int) -> Int {
        let start = max(0, lower)
        let end = min(count, upper)
        guard start < end else { return 0 }
        var missing = 0
        var index = start
        while index < end {
            let bit = index % 64
            let span = min(64 - bit, end - index)
            let mask: UInt64 = span == 64 ? .max : (UInt64(1) << span) - 1
            let present = ((words[index / 64] >> bit) & mask).nonzeroBitCount
            missing += span - present
            index += span
        }
        return missing
    }

    /// The next hole after `index`, then anything earlier. Nil when every frame is present.
    func nextMissing(after index: Int) -> Int? {
        if count == 0 { return nil }
        let start = min(max(index + 1, 0), count)
        if let hole = firstEmpty(from: start, until: count) { return hole }
        return firstEmpty(from: 0, until: min(index, count))
    }

    /// True when any frame in `lower..<upper` is present. An empty span checks `lower`.
    /// The seek bar asks this once per column, and playback redraws the thumb every frame.
    /// A `for frame in` loop paid for a generic iterator on each frame and could not keep
    /// up with the film, so this walks the bit words directly.
    func any(from lower: Int, until upper: Int) -> Bool {
        let start = max(0, lower)
        guard start < count else { return false }
        let end = min(count, max(upper, start + 1))
        guard start < end else { return false }
        var index = start
        while index < end {
            let bit = index % 64
            let span = min(64 - bit, end - index)
            let mask: UInt64 = span == 64 ? .max : (UInt64(1) << span) - 1
            if (words[index / 64] >> bit) & mask != 0 { return true }
            index += span
        }
        return false
    }

    private func firstEmpty(from lower: Int, until upper: Int) -> Int? {
        var index = max(lower, 0)
        while index < upper {
            let bit = index % 64
            let wordEnd = min(upper, index + (64 - bit))
            var bits = words[index / 64] >> bit
            var cursor = index
            while cursor < wordEnd {
                if bits & 1 == 0 { return cursor }
                bits >>= 1
                cursor += 1
            }
            index = wordEnd
        }
        return nil
    }
}

/// Append-only ProRes 4444 frames. A slot table points at each frame, so a seek can fill one hole.
/// The table in RAM follows every stored frame. `slots` on disk is updated from a patch of the
/// runs that changed, about once a minute and again when the archive closes. `frames` is synchronized
/// before that patch is renamed into place, so a published slot never points at bytes still in the cache.
/// Playback reads through its own handle and decoder, so an append on the model queue can proceed.
nonisolated final class FrameStore: @unchecked Sendable {
    let count: Int
    private(set) var width: Int
    private(set) var height: Int

    /// Unpublished slots may sit in RAM for this long. A crash recomputes that stretch.
    private static let checkpointInterval: TimeInterval = 60

    private let directory: URL
    private let lock = NSLock()
    private let publishLock = NSLock()
    private let readLock = NSLock()
    private let decodeLock = NSLock()
    private var slots: [Slot]
    /// Frame indexes stored since the last patch that reached disk.
    private var dirty: Set<Int> = []
    private var dirtySinceUptime: TimeInterval?
    /// After a failed checkpoint, wait before trying again so fill does not stall on every frame.
    private var holdUntilUptime: TimeInterval = 0
    private var dataHandle: FileHandle
    private var readHandle: FileHandle
    private var slotHandle: FileHandle
    /// Separate from `dataHandle`, so a checkpoint can synchronize `frames` while another append writes.
    private var syncHandle: FileHandle
    private var extensionsData: Data?
    private var format: CMVideoFormatDescription?
    private var encoder: VTCompressionSession?
    private var decoder: VTDecompressionSession?
    private var transfer: ColorTransfer?
    private var stamp: CMTimeValue = 0

    /// Cache gate, taken after the film-list lock. Held only around directory claims, not while a large folder is deleted.
    private static let gate = NSLock()

    static var cacheRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Whoodeo/Cache", isDirectory: true)
    }

    /// Digest of the file as it is now. Call this once, when the film is added, and store the result.
    static func filmDigest(of url: URL) -> String {
        digest(identity(url))
    }

    /// `filmID` is the digest stored on the film. Opening does not hash the file again.
    static func open(filmID: String, model: URL, unit: Compute, count: Int) throws -> FrameStore {
        guard count > 0 else { throw CLIError("der Film hat keine Bilder") }
        let root = cacheRoot
            .appendingPathComponent(filmID, isDirectory: true)
            .appendingPathComponent("\(digest(identity(model)))-\(unit.rawValue)", isDirectory: true)
        gate.lock()
        defer { gate.unlock() }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return try FrameStore(directory: root, count: count)
    }

    /// Deletes one film directory and every model archive in it.
    static func removeCache(id: String) throws {
        let directory = cacheRoot.appendingPathComponent(id, isDirectory: true)
        gate.lock()
        defer { gate.unlock() }
        var directoryFlag: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &directoryFlag),
              directoryFlag.boolValue else { return }
        try FileManager.default.removeItem(at: directory)
    }

    /// Folders whose name is not a film id are renamed aside and then deleted.
    /// The rename is the claim. The delete happens without the gate, so opening another film does not wait on it.
    static func pruneCaches(isKept: @Sendable (String) -> Bool) {
        let root = cacheRoot
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        for url in children {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
            guard values?.isDirectory != false else { continue }
            let name = url.lastPathComponent
            if name.hasSuffix(".deleting") {
                try? FileManager.default.removeItem(at: url)
                continue
            }
            gate.lock()
            var doomed: URL?
            if !isKept(name) {
                let dest = root.appendingPathComponent(name + ".deleting")
                try? FileManager.default.removeItem(at: dest)
                if (try? FileManager.default.moveItem(at: url, to: dest)) != nil {
                    doomed = dest
                }
            }
            gate.unlock()
            guard let doomed else { continue }
            try? FileManager.default.removeItem(at: doomed)
            Log.fill.info("Cache ohne Film entfernt")
        }
    }

    func coverage() -> Coverage {
        lock.lock()
        defer { lock.unlock() }
        var marks = Coverage(count: count)
        for index in 0..<count where slots[index].present {
            marks.insert(index)
        }
        return marks
    }

    func dimensions() -> (width: Int, height: Int)? {
        lock.lock()
        defer { lock.unlock() }
        guard width > 0, height > 0 else { return nil }
        return (width, height)
    }

    func contains(_ index: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard slots.indices.contains(index) else { return false }
        return slots[index].present
    }

    /// Encode one finished frame and remember it. A frame that is already stored is left as it is.
    @discardableResult
    func put(rgb: [Float], width: Int, height: Int, index: Int) throws -> Data {
        guard slots.indices.contains(index) else { throw CLIError("Bild \(index) liegt außerhalb") }
        if let hit = try cached(index) { return hit }
        let encoded = try encode(rgb, width: width, height: height)
        let stored = try remember(encoded, width: width, height: height, index: index)
        if stored.due { publish(force: false) }
        return stored.data
    }

    /// Publish the index now. Quitting and dropping the archive both come through here.
    func flush() {
        publish(force: true)
    }

    func data(at index: Int) throws -> Data? {
        lock.lock()
        guard slots.indices.contains(index), slots[index].present else {
            lock.unlock()
            return nil
        }
        let slot = slots[index]
        lock.unlock()
        return try read(slot)
    }

    /// One stored ProRes access unit, timed on the film clock. The display layer decodes it.
    /// The file read stays outside `decodeLock`; only the format description needs that lock.
    func sample(at index: Int, time: CMTime, duration: CMTime) throws -> CMSampleBuffer {
        guard let data = try data(at: index) else { throw CLIError("Bild \(index) fehlt im Cache") }
        decodeLock.lock()
        defer { decodeLock.unlock() }
        return try sampleBuffer(data, time: time, duration: duration)
    }

    /// Decode one stored frame to 16-bit RGBA for the preview and for a transcode.
    func image(at index: Int) throws -> CVPixelBuffer {
        guard let data = try data(at: index) else { throw CLIError("Bild \(index) fehlt im Cache") }
        decodeLock.lock()
        defer { decodeLock.unlock() }
        let sample = try sampleBuffer(data, time: .zero, duration: .invalid)
        let session = try decoderSession()
        let decoded = DecodedImage()
        let gate = DispatchSemaphore(value: 0)
        let status = VTDecompressionSessionDecodeFrame(session, sampleBuffer: sample, flags: [], infoFlagsOut: nil) { result, _, buffer, _, _ in
            decoded.status = result
            decoded.image = buffer
            gate.signal()
        }
        guard status == noErr else { throw CLIError("Cache liest Bild \(index) nicht (\(status))") }
        guard gate.wait(timeout: .now() + 30) == .success else {
            throw CLIError("Cache liest Bild \(index) nicht")
        }
        guard decoded.status == noErr, let image = decoded.image else {
            throw CLIError("Cache liest Bild \(index) nicht (\(decoded.status))")
        }
        guard CVPixelBufferGetPixelFormatType(image) == kCVPixelFormatType_64RGBALE else {
            throw CLIError("Cache liefert ein unbekanntes Bildformat")
        }
        return image
    }

    func videoFormat() throws -> CMVideoFormatDescription {
        try formatDescription()
    }

    func sampleBuffer(_ data: Data, time: CMTime, duration: CMTime) throws -> CMSampleBuffer {
        let format = try formatDescription()
        var block: CMBlockBuffer?
        let created = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: data.count,
            blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: data.count,
            flags: 0,
            blockBufferOut: &block
        )
        guard created == kCMBlockBufferNoErr, let block else {
            throw CLIError("Cachepuffer (\(created))")
        }
        let copied = data.withUnsafeBytes { raw in
            CMBlockBufferReplaceDataBytes(
                with: raw.baseAddress!,
                blockBuffer: block,
                offsetIntoDestination: 0,
                dataLength: data.count
            )
        }
        guard copied == kCMBlockBufferNoErr else { throw CLIError("Cachepuffer (\(copied))") }
        var size = data.count
        var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        let status = CMSampleBufferCreateReady(
            allocator: kCFAllocatorDefault,
            dataBuffer: block,
            formatDescription: format,
            sampleCount: 1,
            sampleTimingEntryCount: 1,
            sampleTimingArray: &timing,
            sampleSizeEntryCount: 1,
            sampleSizeArray: &size,
            sampleBufferOut: &sample
        )
        guard status == noErr, let sample else { throw CLIError("Cachebild (\(status))") }
        return sample
    }

    private init(directory: URL, count: Int) throws {
        self.directory = directory
        self.count = count
        let metaURL = directory.appendingPathComponent("meta.json")
        let slotURL = directory.appendingPathComponent("slots")
        let dataURL = directory.appendingPathComponent("frames")
        if let stored = try Self.meta(at: metaURL), stored.version == 1, stored.count == count {
            width = stored.width
            height = stored.height
        } else {
            width = 0
            height = 0
            try? FileManager.default.removeItem(at: slotURL)
            try? FileManager.default.removeItem(at: dataURL)
            try? FileManager.default.removeItem(at: directory.appendingPathComponent("format.plist"))
            try? FileManager.default.removeItem(at: directory.appendingPathComponent("slots.patch"))
            try? FileManager.default.removeItem(at: directory.appendingPathComponent("slots.patch.tmp"))
            try Self.write(Meta(version: 1, count: count, width: 0, height: 0), to: metaURL)
        }
        if !FileManager.default.fileExists(atPath: slotURL.path) {
            FileManager.default.createFile(atPath: slotURL.path, contents: Data(count: count * Slot.size))
        }
        if !FileManager.default.fileExists(atPath: dataURL.path) {
            FileManager.default.createFile(atPath: dataURL.path, contents: Data())
        }
        slotHandle = try FileHandle(forUpdating: slotURL)
        dataHandle = try FileHandle(forUpdating: dataURL)
        readHandle = try FileHandle(forReadingFrom: dataURL)
        syncHandle = try FileHandle(forUpdating: dataURL)
        let raw = try Data(contentsOf: slotURL)
        guard raw.count == count * Slot.size else {
            throw CLIError("Cache-Index passt nicht zum Film")
        }
        slots = (0..<count).map { Slot(raw, at: $0) }
        extensionsData = try? Data(contentsOf: directory.appendingPathComponent("format.plist"))
        try recoverIndex()
    }

    deinit {
        flush()
        if let encoder { VTCompressionSessionInvalidate(encoder) }
        decodeLock.lock()
        if let decoder { VTDecompressionSessionInvalidate(decoder) }
        decoder = nil
        decodeLock.unlock()
        try? syncHandle.close()
        try? dataHandle.close()
        try? readHandle.close()
        try? slotHandle.close()
    }

    private func writeMeta() throws {
        try Self.write(Meta(version: 1, count: count, width: width, height: height), to: directory.appendingPathComponent("meta.json"))
    }

    /// The write handle. Callers already hold `lock`, so this stays off the playback reader.
    private func readUnlocked(_ slot: Slot) throws -> Data {
        try dataHandle.seek(toOffset: slot.offset)
        guard let data = try dataHandle.read(upToCount: Int(slot.length)), data.count == slot.length else {
            throw CLIError("Cachebild ist abgeschnitten")
        }
        return data
    }

    /// Playback and export. Frames are append-only, so the bytes at `slot.offset` stay put.
    /// This does not take `lock`. The slot itself was copied under the lock.
    private func read(_ slot: Slot) throws -> Data {
        readLock.lock()
        defer { readLock.unlock() }
        try readHandle.seek(toOffset: slot.offset)
        guard let data = try readHandle.read(upToCount: Int(slot.length)), data.count == slot.length else {
            throw CLIError("Cachebild ist abgeschnitten")
        }
        return data
    }

    /// The picture bytes are a normal write. The slot stays in RAM until the next checkpoint.
    private func append(_ data: Data, index: Int) throws {
        guard data.count <= Int(UInt32.max) else {
            throw CLIError("Cachebild ist zu groß")
        }
        try dataHandle.seekToEnd()
        let offset = try dataHandle.offset()
        try dataHandle.write(contentsOf: data)
        let slot = Slot(offset: offset, length: UInt32(data.count), flags: 1)
        slots[index] = slot
        dirty.insert(index)
        if dirtySinceUptime == nil {
            dirtySinceUptime = ProcessInfo.processInfo.systemUptime
        }
    }

    private func cached(_ index: Int) throws -> Data? {
        lock.lock()
        defer { lock.unlock() }
        guard slots[index].present else { return nil }
        return try readUnlocked(slots[index])
    }

    private struct Remembered {
        var data: Data
        var due: Bool
    }

    /// Callers hold no lock. The checkpoint, when it is due, runs after this returns.
    private func remember(_ encoded: Encoded, width: Int, height: Int, index: Int) throws -> Remembered {
        lock.lock()
        defer { lock.unlock() }
        if slots[index].present {
            return Remembered(data: try readUnlocked(slots[index]), due: false)
        }
        if self.width == 0 {
            self.width = width
            self.height = height
            try writeMeta()
        } else if self.width != width || self.height != height {
            throw CLIError("Cache erwartet \(self.width)×\(self.height), Bild \(index) ist \(width)×\(height)")
        }
        try append(encoded.data, index: index)
        if extensionsData == nil {
            try encoded.extensions.write(to: directory.appendingPathComponent("format.plist"), options: .atomic)
            decodeLock.lock()
            extensionsData = encoded.extensions
            format = nil
            if let decoder {
                VTDecompressionSessionInvalidate(decoder)
                self.decoder = nil
            }
            decodeLock.unlock()
        }
        let now = ProcessInfo.processInfo.systemUptime
        let due = dirtySinceUptime.map { now - $0 >= Self.checkpointInterval && now >= holdUntilUptime } ?? false
        return Remembered(data: encoded.data, due: due)
    }

    /// Fold a leftover patch, then publish every slot that is already in RAM.
    /// `frames` is synchronized before the patch is renamed, and the patch is only the runs
    /// frozen at that moment. Frames that arrive while this is in progress stay dirty.
    private func publish(force: Bool) {
        publishLock.lock()
        defer { publishLock.unlock() }
        do {
            try foldOrDiscard()
            let runs = copySnapshot(force: force)
            guard !runs.isEmpty else { return }
            try syncHandle.synchronize()
            try writePatchAtomically(runs)
            try foldExistingPatch()
            let published = runs.reduce(0) { $0 + $1.slots.count }
            Log.fill.debug("Cache-Index veröffentlicht, \(published, privacy: .public) Bilder")
        } catch {
            lock.lock()
            holdUntilUptime = ProcessInfo.processInfo.systemUptime + Self.checkpointInterval
            lock.unlock()
            Log.fill.error("Cache-Index bleibt unveröffentlicht: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Drop an unfinished temp patch, fold a published one, then drop frame bytes no slot names.
    private func recoverIndex() throws {
        if FileManager.default.fileExists(atPath: patchTempURL.path) {
            try? FileManager.default.removeItem(at: patchTempURL)
        }
        try foldOrDiscard()
        guard let end = publishedEnd() else { return }
        let fileEnd = try dataHandle.seekToEnd()
        if fileEnd > end {
            try dataHandle.truncate(atOffset: end)
            Log.fill.info("Cache kürzt \(fileEnd - end, privacy: .public) Bytes nach dem letzten veröffentlichten Bild")
        }
    }

    private var patchURL: URL { directory.appendingPathComponent("slots.patch") }
    private var patchTempURL: URL { directory.appendingPathComponent("slots.patch.tmp") }

    /// The highest end offset named by a present slot. Nil when a slot's range overflows.
    private func publishedEnd() -> UInt64? {
        var end: UInt64 = 0
        for slot in slots where slot.present {
            let (next, overflow) = slot.offset.addingReportingOverflow(UInt64(slot.length))
            if overflow { return nil }
            if next > end { end = next }
        }
        return end
    }

    /// Copy the dirty runs. Frames appended after this returns are not in the result.
    private func copySnapshot(force: Bool) -> [SlotRun] {
        lock.lock()
        defer { lock.unlock() }
        let now = ProcessInfo.processInfo.systemUptime
        if !force {
            if now < holdUntilUptime { return [] }
            guard let dirtySinceUptime, now - dirtySinceUptime >= Self.checkpointInterval else { return [] }
        }
        guard !dirty.isEmpty else { return [] }
        var runs: [SlotRun] = []
        for index in dirty.sorted() {
            let slot = slots[index]
            if var last = runs.last, last.first + last.slots.count == index {
                last.slots.append(slot)
                runs[runs.count - 1] = last
            } else {
                runs.append(SlotRun(first: index, slots: [slot]))
            }
        }
        return runs
    }

    /// A damaged patch is removed. An I/O error leaves it in place so the next open can try again.
    private func foldOrDiscard() throws {
        do {
            try foldExistingPatch()
        } catch is PatchRejected {
            Log.fill.error("Cache-Index-Patch wird verworfen")
            try? FileManager.default.removeItem(at: patchURL)
            try? synchronizeDirectory()
        }
    }

    /// Apply `slots.patch` onto the slot file. A format error is `PatchRejected` and writes nothing.
    /// Applying the same patch twice writes the same bytes.
    private func foldExistingPatch() throws {
        guard FileManager.default.fileExists(atPath: patchURL.path) else { return }
        let data = try Data(contentsOf: patchURL)
        let runs: [SlotRun]
        do {
            runs = try SlotPatch.decode(data, count: count)
        } catch {
            throw PatchRejected()
        }
        try writeRuns(runs)
        try slotHandle.synchronize()
        install(runs)
        try FileManager.default.removeItem(at: patchURL)
        try synchronizeDirectory()
        forget(runs)
    }

    private func writePatchAtomically(_ runs: [SlotRun]) throws {
        if FileManager.default.fileExists(atPath: patchTempURL.path) {
            try FileManager.default.removeItem(at: patchTempURL)
        }
        FileManager.default.createFile(atPath: patchTempURL.path, contents: nil)
        do {
            let handle = try FileHandle(forWritingTo: patchTempURL)
            defer { try? handle.close() }
            try handle.write(contentsOf: SlotPatch.encode(runs))
            try handle.synchronize()
        } catch {
            try? FileManager.default.removeItem(at: patchTempURL)
            throw error
        }
        guard !FileManager.default.fileExists(atPath: patchURL.path) else {
            try? FileManager.default.removeItem(at: patchTempURL)
            throw CLIError("Cache-Index-Patch liegt schon vor")
        }
        try FileManager.default.moveItem(at: patchTempURL, to: patchURL)
        try synchronizeDirectory()
    }

    private func writeRuns(_ runs: [SlotRun]) throws {
        for run in runs {
            guard run.first >= 0, run.slots.count <= count, run.first <= count - run.slots.count else {
                throw CLIError("Cache-Index-Patch ist beschädigt")
            }
            var bytes = Data()
            bytes.reserveCapacity(run.slots.count * Slot.size)
            for slot in run.slots {
                bytes.append(slot.bytes)
            }
            try slotHandle.seek(toOffset: UInt64(run.first * Slot.size))
            try slotHandle.write(contentsOf: bytes)
        }
    }

    private func install(_ runs: [SlotRun]) {
        lock.lock()
        defer { lock.unlock() }
        for run in runs {
            for (offset, slot) in run.slots.enumerated() {
                slots[run.first + offset] = slot
            }
        }
    }

    /// Drop published indexes from the dirty set. A slot that no longer matches stays dirty.
    private func forget(_ runs: [SlotRun]) {
        guard !runs.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for run in runs {
            for (offset, slot) in run.slots.enumerated() {
                let index = run.first + offset
                if slots.indices.contains(index), slots[index] == slot {
                    dirty.remove(index)
                }
            }
        }
        if dirty.isEmpty { dirtySinceUptime = nil }
        holdUntilUptime = 0
    }

    private func synchronizeDirectory() throws {
        let fd = Darwin.open(directory.path, O_RDONLY)
        guard fd >= 0 else {
            throw CLIError("Cache-Ordner lässt sich nicht öffnen (\(Darwin.errno))")
        }
        defer { Darwin.close(fd) }
        guard Darwin.fsync(fd) == 0 else {
            throw CLIError("Cache-Ordner bleibt unveröffentlicht (\(Darwin.errno))")
        }
    }

    private struct Encoded {
        var data: Data
        var extensions: Data
    }

    /// ProRes wants a rising timestamp inside one session. The number is not the film time.
    private func encode(_ rgb: [Float], width: Int, height: Int) throws -> Encoded {
        let session = try encoderSession(width: width, height: height)
        let buffer = try colorTransfer().convert(
            rgb,
            width: width,
            height: height,
            format: kCVPixelFormatType_4444AYpCbCr16,
            pool: nil
        )
        stamp += 1
        let time = CMTime(value: stamp, timescale: 600)
        var encoded: Data?
        var extensions: Data?
        var encodeStatus: OSStatus = noErr
        let gate = DispatchSemaphore(value: 0)
        let status = VTCompressionSessionEncodeFrame(session, imageBuffer: buffer, presentationTimeStamp: time, duration: .invalid, frameProperties: nil, infoFlagsOut: nil) { result, _, sample in
            encodeStatus = result
            if result == noErr, let sample {
                encoded = Self.bytes(of: sample)
                if let format = CMSampleBufferGetFormatDescription(sample),
                   let raw = CMFormatDescriptionGetExtensions(format),
                   PropertyListSerialization.propertyList(raw, isValidFor: .binary),
                   let data = try? PropertyListSerialization.data(fromPropertyList: raw, format: .binary, options: 0) {
                    extensions = data
                }
            }
            gate.signal()
        }
        guard status == noErr else { throw CLIError("ProRes startet nicht (\(status))") }
        guard gate.wait(timeout: .now() + 30) == .success else { throw CLIError("ProRes antwortet nicht") }
        guard encodeStatus == noErr, let encoded, !encoded.isEmpty else {
            throw CLIError("ProRes schreibt Bild nicht (\(encodeStatus))")
        }
        let described: Data
        if let extensions {
            described = extensions
        } else if let extensionsData {
            described = extensionsData
        } else {
            throw CLIError("ProRes ohne Formatbeschreibung")
        }
        return Encoded(data: encoded, extensions: described)
    }

    private func encoderSession(width: Int, height: Int) throws -> VTCompressionSession {
        if let encoder { return encoder }
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: NSNumber(value: kCVPixelFormatType_4444AYpCbCr16),
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
        ]
        var session: VTCompressionSession?
        let status = VTCompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            width: Int32(width),
            height: Int32(height),
            codecType: kCMVideoCodecType_AppleProRes4444,
            encoderSpecification: nil,
            imageBufferAttributes: attributes as CFDictionary,
            compressedDataAllocator: nil,
            outputCallback: nil,
            refcon: nil,
            compressionSessionOut: &session
        )
        guard status == noErr, let session else { throw CLIError("ProRes startet nicht (\(status))") }
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanFalse)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ColorPrimaries, value: kCVImageBufferColorPrimaries_ITU_R_709_2)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_TransferFunction, value: kCVImageBufferTransferFunction_ITU_R_709_2)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_YCbCrMatrix, value: kCVImageBufferYCbCrMatrix_ITU_R_709_2)
        let prepared = VTCompressionSessionPrepareToEncodeFrames(session)
        guard prepared == noErr else { throw CLIError("ProRes startet nicht (\(prepared))") }
        encoder = session
        return session
    }

    private func decoderSession() throws -> VTDecompressionSession {
        if let decoder { return decoder }
        let format = try formatDescription()
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: NSNumber(value: kCVPixelFormatType_64RGBALE),
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
        ]
        var session: VTDecompressionSession?
        let status = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: format,
            decoderSpecification: nil,
            imageBufferAttributes: attributes as CFDictionary,
            outputCallback: nil,
            decompressionSessionOut: &session
        )
        guard status == noErr, let session else { throw CLIError("Decoder startet nicht (\(status))") }
        decoder = session
        return session
    }

    private func formatDescription() throws -> CMVideoFormatDescription {
        if let format { return format }
        guard let extensionsData else { throw CLIError("Cache hat noch kein Bildformat") }
        let object = try PropertyListSerialization.propertyList(from: extensionsData, options: [], format: nil)
        guard let extensions = object as? NSDictionary else {
            throw CLIError("Bildformat lässt sich nicht lesen")
        }
        var created: CMVideoFormatDescription?
        let status = CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            codecType: kCMVideoCodecType_AppleProRes4444,
            width: Int32(width),
            height: Int32(height),
            extensions: extensions,
            formatDescriptionOut: &created
        )
        guard status == noErr, let created else { throw CLIError("Bildformat (\(status))") }
        format = created
        return created
    }

    private func colorTransfer() throws -> ColorTransfer {
        if let transfer { return transfer }
        let created = try ColorTransfer()
        transfer = created
        return created
    }

    private static func bytes(of sample: CMSampleBuffer) -> Data? {
        guard let block = CMSampleBufferGetDataBuffer(sample) else { return nil }
        let length = CMBlockBufferGetDataLength(block)
        guard length > 0 else { return nil }
        var data = Data(count: length)
        let status = data.withUnsafeMutableBytes { raw in
            CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: raw.baseAddress!)
        }
        return status == kCMBlockBufferNoErr ? data : nil
    }

    private static func identity(_ url: URL) -> String {
        let file = url.standardizedFileURL
        let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .isDirectoryKey])
        if values?.isDirectory == true {
            let weights = file.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
            let model = file.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
            return "\(file.path)\n\(stamp(weights))\n\(stamp(model))"
        }
        return "\(file.path)\n\(values?.fileSize ?? 0)\n\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
    }

    private static func stamp(_ url: URL) -> String {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        return "\(values?.fileSize ?? 0)-\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
    }

    private static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private struct Meta: Codable {
        var version: Int
        var count: Int
        var width: Int
        var height: Int
    }

    private static func meta(at url: URL) throws -> Meta? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try JSONDecoder().decode(Meta.self, from: Data(contentsOf: url))
    }

    private static func write(_ meta: Meta, to url: URL) throws {
        try JSONEncoder().encode(meta).write(to: url, options: .atomic)
    }

    /// The decode callback writes here. The caller waits on a semaphore, so the buffer is not used concurrently.
    private final class DecodedImage: @unchecked Sendable {
        var image: CVPixelBuffer?
        var status: OSStatus = noErr
    }

    /// A published patch that is not a well-formed run list. The slot file has not been touched.
    private struct PatchRejected: Error {}

    private struct SlotRun {
        var first: Int
        var slots: [Slot]
    }

    private enum SlotPatch {
        static func encode(_ runs: [SlotRun]) -> Data {
            var data = Data("WSP1".utf8)
            append(UInt32(runs.count), to: &data)
            for run in runs {
                append(UInt32(run.first), to: &data)
                append(UInt32(run.slots.count), to: &data)
                for slot in run.slots {
                    data.append(slot.bytes)
                }
            }
            return data
        }

        static func decode(_ data: Data, count: Int) throws -> [SlotRun] {
            guard data.count >= 8, data.prefix(4) == Data("WSP1".utf8) else { throw PatchRejected() }
            let runCount = Int(readUInt32(data, 4))
            var cursor = 8
            var runs: [SlotRun] = []
            var previousEnd = 0
            for _ in 0..<runCount {
                guard data.count >= 8, cursor <= data.count - 8 else { throw PatchRejected() }
                let first = Int(readUInt32(data, cursor))
                let n = Int(readUInt32(data, cursor + 4))
                cursor += 8
                guard n > 0, first >= previousEnd, first <= count, n <= count - first else { throw PatchRejected() }
                let bytes = n * Slot.size
                guard cursor <= data.count - bytes else { throw PatchRejected() }
                let slice = data.subdata(in: cursor..<(cursor + bytes))
                var decoded: [Slot] = []
                decoded.reserveCapacity(n)
                for index in 0..<n {
                    let slot = Slot(slice, at: index)
                    guard slot.present else { throw PatchRejected() }
                    decoded.append(slot)
                }
                runs.append(SlotRun(first: first, slots: decoded))
                previousEnd = first + n
                cursor += bytes
            }
            guard cursor == data.count else { throw PatchRejected() }
            return runs
        }

        private static func append(_ value: UInt32, to data: inout Data) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }

        private static func readUInt32(_ data: Data, _ offset: Int) -> UInt32 {
            var value: UInt32 = 0
            for step in 0..<4 {
                value |= UInt32(data[offset + step]) << (8 * step)
            }
            return value
        }
    }

    private struct Slot: Equatable {
        static let size = 16

        var offset: UInt64
        var length: UInt32
        var flags: UInt32

        var present: Bool { flags & 1 != 0 && length > 0 }

        init(offset: UInt64, length: UInt32, flags: UInt32) {
            self.offset = offset
            self.length = length
            self.flags = flags
        }

        init(_ data: Data, at index: Int) {
            let base = index * Slot.size
            offset = Self.integer(data, base, 8)
            length = UInt32(Self.integer(data, base + 8, 4))
            flags = UInt32(Self.integer(data, base + 12, 4))
        }

        var bytes: Data {
            var data = Data(count: Slot.size)
            Self.write(offset, into: &data, at: 0, count: 8)
            Self.write(UInt64(length), into: &data, at: 8, count: 4)
            Self.write(UInt64(flags), into: &data, at: 12, count: 4)
            return data
        }

        private static func integer(_ data: Data, _ offset: Int, _ count: Int) -> UInt64 {
            guard data.count >= offset + count else { return 0 }
            var value: UInt64 = 0
            for step in 0..<count {
                value |= UInt64(data[offset + step]) << (8 * step)
            }
            return value
        }

        private static func write(_ value: UInt64, into data: inout Data, at offset: Int, count: Int) {
            for step in 0..<count {
                data[offset + step] = UInt8((value >> (8 * step)) & 0xff)
            }
        }
    }
}
