import AppKit
import os
import SwiftUI

/// Holds the picture. Its frame is the picture. The video layer inside shows every frame,
/// still or playing, upscale, original or comparison. The layer does not take clicks; this view does.
final class PictureView: NSView {
    var sharp = false {
        didSet {
            guard sharp != oldValue else { return }
            created?.setSharp(sharp)
        }
    }
    weak var stage: PictureStage?
    /// Created with the first picture.
    private var created: FrameScreen?

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }

    /// The screen, created on first use.
    func screen() -> FrameScreen {
        if let created { return created }
        let screen = FrameScreen()
        screen.attach(to: self)
        created = screen
        return screen
    }

    /// The screen if a picture has been shown, else nil.
    func screenIfCreated() -> FrameScreen? {
        created
    }

    override func layout() {
        super.layout()
        created?.layout(in: bounds)
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        created?.updateScale()
    }

    override func resetCursorRects() {
        discardCursorRects()
        let cursor: NSCursor = stage?.canPan() == true ? .openHand : .arrow
        addCursorRect(bounds, cursor: cursor)
    }

    override func mouseDown(with event: NSEvent) {
        stage?.trackMouse(event)
    }

    override func scrollWheel(with event: NSEvent) {
        stage?.scrollWheel(with: event)
    }

    override func magnify(with event: NSEvent) {
        stage?.magnify(with: event)
    }
}

/// The picture sits on the window background, which follows light and dark mode.
/// Empty space stays that plain color. Once the picture is larger than the free area,
/// dragging moves the real pixels under the sidebar and the bars.
final class PictureStage: NSView {
    let picture = PictureView()
    var zoomToFit = true
    var magnification: CGFloat = 1
    var pixelWidth = 0
    var pixelHeight = 0
    var onZoom: ((Bool, CGFloat) -> Void)?
    var onFilm: ((URL) -> Void)?
    var onHover: ((Bool) -> Void)?
    var onAlternate: ((Bool) -> Void)?
    /// A short note over the picture, top left in the free area. Nil hides it.
    var badge: String? {
        didSet {
            guard badge != oldValue else { return }
            badgeLabel.stringValue = badge ?? ""
            badgeLabel.isHidden = badge == nil
            placeBadge()
        }
    }
    private let badgeLabel = BadgeLabel()

    func screen() -> FrameScreen { picture.screen() }

    func screenIfCreated() -> FrameScreen? { picture.screenIfCreated() }

    var fallbackTop: CGFloat = 0
    var fallbackLeading: CGFloat = 0
    var fallbackBottom: CGFloat = 0
    var fallbackTrailing: CGFloat = 0

    private var pan: CGPoint?
    private var hovering = false
    private var applying = false
    private var lastReportedFit = false
    private var lastReported: CGFloat = -1
    private var fitReport: DispatchWorkItem?
    private var lastPixelWidth = 0
    private var lastPixelHeight = 0
    /// Where a pinch in progress shows the picture, as a GPU transform. Nil when the views are the truth.
    private var liveFrame: CGRect?
    /// Unsnapped zoom during a pinch; nil when no pinch is in progress.
    /// The detent at 100 % follows this raw value, not the last displayed zoom, so the picture does not jitter.
    private var rawMagnification: CGFloat?
    private var snappedToHundred = false
    /// The stage zoomed on its own and the session has not heard of it yet.
    private var zoomDirty = false
    private var zoomReport: DispatchWorkItem?
    private var lastCanPan: Bool?

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { true }
    override var wantsUpdateLayer: Bool { true }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        addSubview(picture)
        addSubview(badgeLabel)
        picture.stage = self
        registerForDraggedTypes(FilmDrop.pasteboardTypes)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        wantsLayer = true
        addSubview(picture)
        addSubview(badgeLabel)
        picture.stage = self
        registerForDraggedTypes(FilmDrop.pasteboardTypes)
    }

    /// One plain color under the picture. The layer holds it, so zooming never redraws the stage
    /// on the CPU. AppKit calls this again when light or dark mode changes.
    override func updateLayer() {
        layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        needsDisplay = true
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        lastCanPan = nil
        liftSidebar()
        layoutPicture(recenter: false)
    }

    private var inLayout = false

    override func layout() {
        inLayout = true
        defer { inLayout = false }
        super.layout()
        liftSidebar()
        layoutPicture(recenter: false)
        placeBadge()
    }

    /// The stage reaches under the sidebar and the bars, so SwiftUI overlays end up beneath the picture.
    /// The note is a subview above it instead.
    private func placeBadge() {
        guard !badgeLabel.isHidden else { return }
        let size = badgeLabel.fittingSize
        let safe = safeRect()
        badgeLabel.frame = CGRect(x: safe.minX + 10, y: safe.minY + 10, width: size.width + 16, height: size.height + 6)
    }

    override func scrollWheel(with event: NSEvent) {
        if pans(event) {
            finishLiveZoom()
            let moved = delta(of: event)
            movePicture(by: moved)
            return
        }
        guard event.momentumPhase.isEmpty else { return }
        let deltaY = event.scrollingDeltaY
        guard deltaY != 0 else { return }
        let multiplier = event.hasPreciseScrollingDeltas ? exp(deltaY * 0.008) : pow(1.15, deltaY)
        changeZoom(multiplier: multiplier, windowPoint: event.locationInWindow, live: false)
    }

    /// A pinch scales and moves the picture on the GPU only, the way NSScrollView handles live
    /// magnification. The views, the cursor and the session catch up once the fingers lift.
    /// An event without a phase has no end to wait for; it lays out at once and reports a moment later.
    override func magnify(with event: NSEvent) {
        let phase = event.phase
        let ending = phase.contains(.ended) || phase.contains(.cancelled)
        if phase.contains(.began) {
            rawMagnification = nil
            snappedToHundred = false
        }
        let multiplier = 1 + event.magnification
        if multiplier > 0.01 {
            // The last tick of a gesture still uses the live detent rules, then the real frame lands.
            changeZoom(multiplier: multiplier, windowPoint: event.locationInWindow, live: !phase.isEmpty)
        }
        if ending { finishLiveZoom() }
    }

    func trackMouse(_ event: NSEvent) {
        finishLiveZoom()
        let startWindow = event.locationInWindow
        let startOrigin = picture.frame.origin
        let movable = panAxes()
        var panning = false
        var showing = false
        if !movable.horizontal && !movable.vertical {
            showing = true
            showAlternate(true)
        }

        while true {
            let wait: Date = (showing || panning) ? .distantFuture : Date().addingTimeInterval(0.12)
            guard let next = window?.nextEvent(
                matching: [.leftMouseUp, .leftMouseDragged],
                until: wait,
                inMode: .eventTracking,
                dequeue: true
            ) else {
                if !panning && !showing {
                    showing = true
                    showAlternate(true)
                }
                continue
            }
            if next.type == .leftMouseUp { break }
            let distance = hypot(
                next.locationInWindow.x - startWindow.x,
                next.locationInWindow.y - startWindow.y
            )
            if distance > 4, movable.horizontal || movable.vertical {
                if showing {
                    showAlternate(false)
                    showing = false
                }
                if !panning {
                    panning = true
                    NSCursor.closedHand.set()
                }
                let start = convert(startWindow, from: nil)
                let now = convert(next.locationInWindow, from: nil)
                var origin = startOrigin
                if movable.horizontal { origin.x += now.x - start.x }
                if movable.vertical { origin.y += now.y - start.y }
                place(origin)
            }
        }
        if showing { showAlternate(false) }
        if panning { NSCursor.openHand.set() }
    }

    func layoutPicture(recenter: Bool) {
        guard !applying else { return }
        guard bounds.width > 1, bounds.height > 1 else { return }
        applying = true
        defer { applying = false }

        if pixelWidth != lastPixelWidth || pixelHeight != lastPixelHeight {
            pan = nil
            lastPixelWidth = pixelWidth
            lastPixelHeight = pixelHeight
        }
        guard pixelWidth > 0, pixelHeight > 0 else { return }

        if !zoomToFit { lastReportedFit = false }
        let target = placement(recenter: recenter)
        // One transaction: the transform of a pinch goes away as the real frame takes its place,
        // so the picture does not move by a pixel.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        if liveFrame != nil {
            liveFrame = nil
            picture.screenIfCreated()?.setLiveFrame(nil, from: self)
        }
        picture.setFrameSize(target.size)
        picture.setFrameOrigin(target.origin)
        picture.screenIfCreated()?.layout(in: picture.bounds)
        CATransaction.commit()
        picture.sharp = !zoomToFit && magnification >= 0.999
        updateCursorIfNeeded()
        reportFitIfNeeded(size: target.size)
    }

    /// Size and origin of the picture for the current zoom and pan, kept inside the free area.
    /// Updates `pan` to the clamped origin.
    private func placement(recenter: Bool) -> CGRect {
        let safe = safeRect()
        let size = pictureSize(in: safe)
        let ranges = ranges(for: size, safe: safe)
        let centered = CGPoint(
            x: (ranges.horizontal.low + ranges.horizontal.high) / 2,
            y: (ranges.vertical.low + ranges.vertical.high) / 2
        )
        let origin: CGPoint
        if zoomToFit {
            origin = centered
            pan = nil
        } else if recenter || pan == nil {
            origin = centered
            pan = centered
        } else {
            origin = CGPoint(
                x: min(max(pan!.x, ranges.horizontal.low), ranges.horizontal.high),
                y: min(max(pan!.y, ranges.vertical.low), ranges.vertical.high)
            )
            pan = origin
        }
        // On whole device pixels, so the pinch transform and the final frame land on the same pixels.
        let aligned = CGPoint(
            x: Self.align(origin.x, within: ranges.horizontal, scale: screenScale),
            y: Self.align(origin.y, within: ranges.vertical, scale: screenScale)
        )
        return CGRect(origin: aligned, size: size)
    }

    /// Rounds to the backing grid. A picture larger than the free area stays inside its range,
    /// so rounding never uncovers a sliver of background at an edge.
    private static func align(_ value: CGFloat, within axis: Axis, scale: CGFloat) -> CGFloat {
        var pixels = (value * scale).rounded()
        if axis.high > axis.low {
            pixels = min(max(pixels, (axis.low * scale).rounded(.up)), (axis.high * scale).rounded(.down))
        }
        return pixels / scale
    }

    /// The open hand only when there is something to drag. Cursor rects follow the frame by
    /// themselves; they are rebuilt here only when panning becomes possible or impossible.
    private func updateCursorIfNeeded() {
        let pannable = canPan()
        guard pannable != lastCanPan else { return }
        lastCanPan = pannable
        window?.invalidateCursorRects(for: picture)
    }

    /// One step of a pinch: the screen transforms the picture to where the layout would put it.
    private func showLiveZoom() {
        guard let screen = picture.screenIfCreated(), picture.frame.width > 1, picture.frame.height > 1 else {
            layoutPicture(recenter: false)
            return
        }
        let target = placement(recenter: false)
        liveFrame = target
        screen.setLiveFrame(target, from: self)
        picture.sharp = magnification >= 0.999
    }

    /// The pinch is over, or something else needs the real frame: lay out where the picture
    /// already is and tell the session.
    private func finishLiveZoom() {
        rawMagnification = nil
        snappedToHundred = false
        if liveFrame != nil { layoutPicture(recenter: false) }
        reportZoomNow()
    }

    /// The zoom factor feeds the toolbar slider and label. Reporting it sets published session
    /// state, which renders the whole window again; doing that on every step of a pinch or a
    /// scroll made zooming stutter. A pinch reports when the fingers lift; this is the fallback
    /// for events without an end and for a pinch that never sends one.
    private func scheduleZoomReport(after delay: TimeInterval) {
        zoomDirty = true
        zoomReport?.cancel()
        let report = DispatchWorkItem { [weak self] in self?.finishLiveZoom() }
        zoomReport = report
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: report)
    }

    private func reportZoomNow() {
        zoomReport?.cancel()
        zoomReport = nil
        guard zoomDirty else { return }
        zoomDirty = false
        guard !zoomToFit else { return }
        onZoom?(false, magnification)
    }

    /// The toolbar, a menu or a script sets the zoom. A pinch in progress lands where it is,
    /// and its pending report is dropped so the old value does not come back.
    func adoptZoom() {
        zoomReport?.cancel()
        zoomReport = nil
        zoomDirty = false
        rawMagnification = nil
        snappedToHundred = false
        if liveFrame != nil { layoutPicture(recenter: false) }
    }

    /// Keeps the point in the middle of the free area fixed. Used when the toolbar slider moves.
    func zoomAboutSafeCenter() {
        let safe = safeRect()
        let old = picture.frame
        let anchor = CGPoint(x: safe.midX, y: safe.midY)
        let fraction: CGPoint
        if old.width > 1, old.height > 1 {
            fraction = CGPoint(
                x: (anchor.x - old.minX) / old.width,
                y: (anchor.y - old.minY) / old.height
            )
        } else {
            fraction = CGPoint(x: 0.5, y: 0.5)
        }
        let size = pictureSize(in: safe)
        pan = CGPoint(
            x: anchor.x - fraction.x * size.width,
            y: anchor.y - fraction.y * size.height
        )
        layoutPicture(recenter: false)
    }

    func canPan() -> Bool {
        let axes = panAxes()
        return axes.horizontal || axes.vertical
    }

    private func pans(_ event: NSEvent) -> Bool {
        event.hasPreciseScrollingDeltas && canPan()
    }

    private func panAxes() -> (horizontal: Bool, vertical: Bool) {
        let safe = safeRect()
        let size = pictureSize(in: safe)
        let ranges = ranges(for: size, safe: safe)
        return (
            ranges.horizontal.high - ranges.horizontal.low > 1,
            ranges.vertical.high - ranges.vertical.low > 1
        )
    }

    private func movePicture(by delta: CGPoint) {
        guard pixelWidth > 0 else { return }
        // From the unrounded pan, so small trackpad steps add up instead of rounding away.
        let origin = zoomToFit ? (liveFrame ?? picture.frame).origin : (pan ?? picture.frame.origin)
        pan = CGPoint(x: origin.x + delta.x, y: origin.y + delta.y)
        zoomToFit = false
        layoutPicture(recenter: false)
    }

    private func place(_ origin: CGPoint) {
        pan = origin
        zoomToFit = false
        layoutPicture(recenter: false)
    }

    private func changeZoom(multiplier: CGFloat, windowPoint: NSPoint, live: Bool) {
        guard pixelWidth > 0, pixelHeight > 0, multiplier > 0 else { return }
        let scale = screenScale
        let old = liveFrame ?? picture.frame
        let natural = CGFloat(pixelWidth) / scale
        let visual = zoomToFit && natural > 1 ? old.width / natural : magnification
        let next: CGFloat
        if live {
            if rawMagnification == nil {
                snappedToHundred = !zoomToFit && abs(visual - 1) < 0.0001
            }
            let raw = min(max((rawMagnification ?? visual) * multiplier, ZoomScale.minimum), ZoomScale.maximum)
            rawMagnification = raw
            // Detent around 100 %: about 96 %–104 %. The raw value decides, not the last displayed zoom.
            if abs(log2(raw)) < 0.06 {
                next = 1
                if !snappedToHundred {
                    snappedToHundred = true
                    NSHapticFeedbackManager.defaultPerformer.perform(.levelChange, performanceTime: .now)
                }
            } else {
                snappedToHundred = false
                next = raw
            }
        } else {
            let raw = min(max(visual * multiplier, ZoomScale.minimum), ZoomScale.maximum)
            let wasHundred = !zoomToFit && abs(visual - 1) < 0.0001
            let nearHundred = abs(log2(raw)) < 0.06
            let crossedHundred = (visual < 1 && raw >= 1) || (visual > 1 && raw <= 1)
            if wasHundred {
                next = raw
            } else if nearHundred || crossedHundred {
                next = 1
                NSHapticFeedbackManager.defaultPerformer.perform(.levelChange, performanceTime: .now)
            } else {
                next = raw
            }
        }
        guard zoomToFit || abs(next - magnification) > 0.0001 else { return }
        let local = convert(windowPoint, from: nil)
        let fraction: CGPoint
        if old.width > 1, old.height > 1 {
            fraction = CGPoint(
                x: min(max((local.x - old.minX) / old.width, 0), 1),
                y: min(max((local.y - old.minY) / old.height, 0), 1)
            )
        } else {
            fraction = CGPoint(x: 0.5, y: 0.5)
        }
        zoomToFit = false
        magnification = next
        let safe = safeRect()
        let size = pictureSize(in: safe)
        pan = CGPoint(
            x: local.x - fraction.x * size.width,
            y: local.y - fraction.y * size.height
        )
        if live {
            showLiveZoom()
            scheduleZoomReport(after: 0.5)
        } else {
            layoutPicture(recenter: false)
            scheduleZoomReport(after: 0.25)
        }
    }

    private func showAlternate(_ show: Bool) {
        onAlternate?(show)
    }

    private func pictureSize(in safe: CGRect) -> CGSize {
        if zoomToFit {
            let fit = min(safe.width / CGFloat(pixelWidth), safe.height / CGFloat(pixelHeight))
            return CGSize(width: CGFloat(pixelWidth) * fit, height: CGFloat(pixelHeight) * fit)
        }
        let scale = screenScale
        return CGSize(
            width: CGFloat(pixelWidth) / scale * magnification,
            height: CGFloat(pixelHeight) / scale * magnification
        )
    }

    private func ranges(for size: CGSize, safe: CGRect) -> (horizontal: Axis, vertical: Axis) {
        (
            axis(image: size.width, safeMin: safe.minX, safeMax: safe.maxX),
            axis(image: size.height, safeMin: safe.minY, safeMax: safe.maxY)
        )
    }

    /// A picture smaller than the free area stays centered there. A larger one can slide until
    /// the free area stays covered, which is what puts pixels under the sidebar and the bars.
    private func axis(image: CGFloat, safeMin: CGFloat, safeMax: CGFloat) -> Axis {
        let safeLength = max(safeMax - safeMin, 0)
        if image <= safeLength + 0.5 {
            let center = safeMin + (safeLength - image) / 2
            return Axis(low: center, high: center)
        }
        return Axis(low: safeMax - image, high: safeMin)
    }

    private func safeRect() -> CGRect {
        let insets = resolvedInsets()
        var rect = bounds
        rect.origin.x += insets.left
        rect.origin.y += isFlipped ? insets.top : insets.bottom
        rect.size.width -= insets.left + insets.right
        rect.size.height -= insets.top + insets.bottom
        if rect.width < 2 || rect.height < 2 || rect.width > bounds.width + 1 || rect.height > bounds.height + 1 {
            return bounds
        }
        return rect.intersection(bounds)
    }

    private func resolvedInsets() -> NSEdgeInsets {
        let own = safeAreaInsets
        return NSEdgeInsets(
            top: max(own.top, fallbackTop),
            left: max(own.left, fallbackLeading),
            bottom: max(own.bottom, fallbackBottom),
            right: max(own.right, fallbackTrailing)
        )
    }

    private func reportFitIfNeeded(size: CGSize) {
        guard zoomToFit, pixelWidth > 0 else { return }
        let factor = size.width * screenScale / CGFloat(pixelWidth)
        if lastReportedFit, abs(factor - lastReported) < 0.001 { return }
        lastReportedFit = true
        lastReported = factor
        // The fit factor only feeds the zoom slider and label. Reporting it sets published
        // session state, which renders the whole window again; doing that on every step of a
        // panel or window animation made the animation stutter. So only the settled value goes out.
        fitReport?.cancel()
        let report = DispatchWorkItem { [weak self] in
            guard let self, self.zoomToFit, self.lastReportedFit else { return }
            self.onZoom?(true, self.lastReported)
        }
        fitReport = report
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25, execute: report)
    }

    private func delta(of event: NSEvent) -> CGPoint {
        let start = convert(event.locationInWindow, from: nil)
        let shifted = convert(
            NSPoint(
                x: event.locationInWindow.x + event.scrollingDeltaX,
                y: event.locationInWindow.y + event.scrollingDeltaY
            ),
            from: nil
        )
        return CGPoint(x: shifted.x - start.x, y: shifted.y - start.y)
    }

    private var screenScale: CGFloat {
        window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
    }

    private func liftSidebar() {
        guard let root = window?.contentViewController else { return }
        func walk(_ controller: NSViewController) {
            if let split = controller as? NSSplitViewController {
                for item in split.splitViewItems where item.behavior != .sidebar {
                    item.automaticallyAdjustsSafeAreaInsets = true
                }
            }
            for child in controller.children { walk(child) }
        }
        walk(root)
        watchSplits()
    }

    /// The stage spans the whole window, under the sidebar and the inspector. Opening or closing
    /// a panel only moves the safe area, which does not lay the stage out again by itself, so
    /// "Passend" kept the old size. Every resize of an enclosing split view, the animation steps
    /// included, lays it out again.
    private var watchedSplits: [ObjectIdentifier] = []
    private var splitObservers: [NSObjectProtocol] = []

    private func watchSplits() {
        var splits: [NSSplitView] = []
        var view = superview
        while let current = view {
            if let split = current as? NSSplitView { splits.append(split) }
            view = current.superview
        }
        let ids = splits.map(ObjectIdentifier.init)
        guard ids != watchedSplits else { return }
        splitObservers.forEach(NotificationCenter.default.removeObserver)
        watchedSplits = ids
        splitObservers = splits.map { split in
            NotificationCenter.default.addObserver(
                forName: NSSplitView.didResizeSubviewsNotification,
                object: split,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self, !self.inLayout else { return }
                    self.needsLayout = true
                    // The safe area follows the split a moment later; catch the final value too.
                    DispatchQueue.main.async { self.needsLayout = true }
                }
            }
        }
    }

    deinit {
        splitObservers.forEach(NotificationCenter.default.removeObserver)
    }

    override func draggingEntered(_ sender: any NSDraggingInfo) -> NSDragOperation {
        guard FilmDrop.firstFilm(in: sender) != nil else { return [] }
        setHovering(true)
        return .copy
    }

    override func draggingUpdated(_ sender: any NSDraggingInfo) -> NSDragOperation {
        FilmDrop.firstFilm(in: sender) != nil ? .copy : []
    }

    override func draggingExited(_ sender: (any NSDraggingInfo)?) {
        setHovering(false)
    }

    override func draggingEnded(_ sender: any NSDraggingInfo) {
        setHovering(false)
    }

    override func performDragOperation(_ sender: any NSDraggingInfo) -> Bool {
        setHovering(false)
        guard let url = FilmDrop.firstFilm(in: sender) else {
            Log.drop.info("Bildfläche: kein Film im Drop")
            return false
        }
        Log.drop.info("Bildfläche: \(url.path, privacy: .public)")
        onFilm?(url)
        return true
    }

    private func setHovering(_ hovering: Bool) {
        guard self.hovering != hovering else { return }
        self.hovering = hovering
        onHover?(hovering)
    }
}

/// Small rounded note. It never takes clicks; the picture underneath does.
private final class BadgeLabel: NSTextField {
    init() {
        super.init(frame: .zero)
        isEditable = false
        isSelectable = false
        isBordered = false
        drawsBackground = false
        alignment = .center
        font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        textColor = .labelColor
        wantsLayer = true
        layer?.backgroundColor = NSColor.windowBackgroundColor.withAlphaComponent(0.85).cgColor
        layer?.cornerRadius = 8
        isHidden = true
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }

    override func hitTest(_: NSPoint) -> NSView? { nil }
}

private struct Axis {
    var low: CGFloat
    var high: CGFloat
}

struct Viewer: NSViewRepresentable {
    var zoomToFit: Bool
    var magnification: CGFloat
    var placement: Int
    var pixelWidth: Int
    var pixelHeight: Int
    var fallbackTop: CGFloat
    var fallbackLeading: CGFloat
    var fallbackBottom: CGFloat
    var fallbackTrailing: CGFloat
    var onFilm: (URL) -> Void
    var onHover: (Bool) -> Void
    var onZoom: (Bool, CGFloat) -> Void
    var badge: String?
    var onAlternate: (Bool) -> Void
    var onStage: (PictureStage) -> Void

    func makeNSView(context: Context) -> PictureStage {
        PictureStage()
    }

    func updateNSView(_ stage: PictureStage, context: Context) {
        stage.onFilm = onFilm
        stage.onHover = onHover
        stage.onZoom = onZoom
        stage.onAlternate = onAlternate
        stage.badge = badge
        let coordinator = context.coordinator
        // Only a change on the session side moves the zoom. Until a pinch reports, the session
        // still holds the old value, and every other update (frame counter, cache bar) would
        // otherwise pull the picture back.
        let zoomChanged = coordinator.zoomToFit != zoomToFit
            || coordinator.magnification.map { abs($0 - magnification) >= 0.0001 } ?? true
        coordinator.zoomToFit = zoomToFit
        coordinator.magnification = magnification
        let samePlacement = coordinator.placement == placement
        let takeZoom = zoomChanged || !samePlacement
        let sameZoom = !takeZoom
            || (stage.zoomToFit == zoomToFit && abs(stage.magnification - magnification) < 0.0001)
        let samePixels = stage.pixelWidth == pixelWidth && stage.pixelHeight == pixelHeight
        let sameInsets = stage.fallbackTop == fallbackTop
            && stage.fallbackLeading == fallbackLeading
            && stage.fallbackBottom == fallbackBottom
            && stage.fallbackTrailing == fallbackTrailing
        // The frame counter publishes on every film frame. Laying the picture out again marks it
        // dirty and the video layer drops frames to keep up with a window that is redrawing.
        if sameZoom && samePixels && sameInsets && samePlacement {
            onStage(stage)
            return
        }
        stage.fallbackTop = fallbackTop
        stage.fallbackLeading = fallbackLeading
        stage.fallbackBottom = fallbackBottom
        stage.fallbackTrailing = fallbackTrailing
        let recenter = !samePlacement
        coordinator.placement = placement
        let wasFit = stage.zoomToFit
        let oldMagnification = stage.magnification
        if takeZoom { stage.adoptZoom() }
        stage.pixelWidth = pixelWidth
        stage.pixelHeight = pixelHeight
        if takeZoom {
            stage.zoomToFit = zoomToFit
            stage.magnification = magnification
        }
        if recenter || !samePixels {
            stage.layoutPicture(recenter: true)
        } else if takeZoom && !zoomToFit && (wasFit || abs(oldMagnification - magnification) > 0.0001) {
            stage.zoomAboutSafeCenter()
        } else {
            stage.layoutPicture(recenter: false)
        }
        onStage(stage)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator {
        var placement = 0
        /// The zoom the session last handed over. Nil before the first update.
        var zoomToFit: Bool?
        var magnification: CGFloat?
    }
}
