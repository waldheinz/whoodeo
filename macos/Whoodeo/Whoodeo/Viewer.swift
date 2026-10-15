import AppKit
import SwiftUI

/// Draws one upright picture. Its frame is the picture.
final class PixelView: NSView {
    var recon: CGImage?
    var source: CGImage?
    var sharp = false
    weak var stage: PictureStage?
    var showSource = false

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }

    override func draw(_ dirty: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext, let image = shown else { return }
        ctx.interpolationQuality = sharp ? .none : .medium
        ctx.saveGState()
        ctx.translateBy(x: 0, y: bounds.height)
        ctx.scaleBy(x: 1, y: -1)
        ctx.draw(image, in: CGRect(origin: .zero, size: bounds.size))
        ctx.restoreGState()
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

    private var shown: CGImage? {
        showSource ? (source ?? recon) : recon
    }
}

/// The picture sits on the window background, which follows light and dark mode.
/// Empty space stays that plain color. Once the picture is larger than the free area,
/// dragging moves the real pixels under the sidebar and the bars.
final class PictureStage: NSView {
    let pixels = PixelView()
    var zoomToFit = true
    var magnification: CGFloat = 1
    var pixelWidth = 0
    var pixelHeight = 0
    var onZoom: ((Bool, CGFloat) -> Void)?
    var onFilm: ((URL) -> Void)?
    var onHover: ((Bool) -> Void)?
    var onOriginal: (() -> CGImage?)?

    var fallbackTop: CGFloat = 0
    var fallbackLeading: CGFloat = 0
    var fallbackBottom: CGFloat = 0
    var fallbackTrailing: CGFloat = 0

    private var pan: CGPoint?
    private var hovering = false
    private var applying = false
    private var lastReportedFit = false
    private var lastReported: CGFloat = -1
    private var lastPixelWidth = 0
    private var lastPixelHeight = 0

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { true }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        addSubview(pixels)
        pixels.stage = self
        registerForDraggedTypes(FilmDrop.pasteboardTypes)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        addSubview(pixels)
        pixels.stage = self
        registerForDraggedTypes(FilmDrop.pasteboardTypes)
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.windowBackgroundColor.setFill()
        dirtyRect.fill()
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        needsDisplay = true
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        liftSidebar()
        layoutPicture(recenter: false)
    }

    override func layout() {
        super.layout()
        liftSidebar()
        layoutPicture(recenter: false)
    }

    override func scrollWheel(with event: NSEvent) {
        if pans(event) {
            let moved = delta(of: event)
            movePicture(by: moved)
            return
        }
        guard event.momentumPhase.isEmpty else { return }
        let deltaY = event.scrollingDeltaY
        guard deltaY != 0 else { return }
        let multiplier = event.hasPreciseScrollingDeltas ? exp(deltaY * 0.008) : pow(1.15, deltaY)
        changeZoom(multiplier: multiplier, windowPoint: event.locationInWindow)
    }

    override func magnify(with event: NSEvent) {
        let multiplier = 1 + event.magnification
        guard multiplier > 0.01 else { return }
        changeZoom(multiplier: multiplier, windowPoint: event.locationInWindow)
    }

    func trackMouse(_ event: NSEvent) {
        let startWindow = event.locationInWindow
        let startOrigin = pixels.frame.origin
        let movable = panAxes()
        var panning = false
        var showing = false
        if !movable.horizontal && !movable.vertical {
            showing = true
            showOriginal(true)
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
                    showOriginal(true)
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
                    showOriginal(false)
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
        if showing { showOriginal(false) }
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
        pixels.setFrameSize(size)
        pixels.setFrameOrigin(origin)
        pixels.sharp = !zoomToFit && magnification >= 0.999
        needsDisplay = true
        pixels.needsDisplay = true
        window?.invalidateCursorRects(for: pixels)
        reportFitIfNeeded(size: size)
    }

    /// Keeps the point in the middle of the free area fixed. Used when the toolbar slider moves.
    func zoomAboutSafeCenter() {
        let safe = safeRect()
        let old = pixels.frame
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
        let origin = pixels.frame.origin
        pan = CGPoint(x: origin.x + delta.x, y: origin.y + delta.y)
        zoomToFit = false
        layoutPicture(recenter: false)
    }

    private func place(_ origin: CGPoint) {
        pan = origin
        zoomToFit = false
        layoutPicture(recenter: false)
    }

    private func changeZoom(multiplier: CGFloat, windowPoint: NSPoint) {
        guard pixelWidth > 0, pixelHeight > 0, multiplier > 0 else { return }
        let scale = screenScale
        let old = pixels.frame
        let natural = CGFloat(pixelWidth) / scale
        let visual = zoomToFit && natural > 1 ? old.width / natural : magnification
        var next = min(max(visual * multiplier, ZoomScale.minimum), ZoomScale.maximum)
        let wasHundred = !zoomToFit && abs(visual - 1) < 0.0001
        let nearHundred = abs(log2(next)) < 0.10
        let crossedHundred = (visual < 1 && next >= 1) || (visual > 1 && next <= 1)
        if !wasHundred, nearHundred {
            next = 1
        }
        if !wasHundred, nearHundred || crossedHundred {
            NSHapticFeedbackManager.defaultPerformer.perform(.alignment, performanceTime: .now)
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
        layoutPicture(recenter: false)
        onZoom?(false, next)
    }

    private func showOriginal(_ show: Bool) {
        if show, pixels.source == nil {
            pixels.source = onOriginal?()
        }
        pixels.showSource = show
        pixels.needsDisplay = true
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
        let callback = onZoom
        DispatchQueue.main.async { [weak self] in
            guard let self, self.zoomToFit else { return }
            callback?(true, factor)
        }
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
        guard let url = FilmDrop.firstFilm(in: sender) else { return false }
        onFilm?(url)
        return true
    }

    private func setHovering(_ hovering: Bool) {
        guard self.hovering != hovering else { return }
        self.hovering = hovering
        onHover?(hovering)
    }
}

private struct Axis {
    var low: CGFloat
    var high: CGFloat
}

struct Viewer: NSViewRepresentable {
    var recon: CGImage?
    var source: CGImage?
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
    var onOriginal: () -> CGImage?

    func makeNSView(context: Context) -> PictureStage {
        PictureStage()
    }

    func updateNSView(_ stage: PictureStage, context: Context) {
        stage.onFilm = onFilm
        stage.onHover = onHover
        stage.onZoom = onZoom
        stage.onOriginal = onOriginal
        stage.fallbackTop = fallbackTop
        stage.fallbackLeading = fallbackLeading
        stage.fallbackBottom = fallbackBottom
        stage.fallbackTrailing = fallbackTrailing
        stage.pixels.recon = recon
        stage.pixels.source = source
        let recenter = context.coordinator.placement != placement
        context.coordinator.placement = placement
        let wasFit = stage.zoomToFit
        let oldMagnification = stage.magnification
        let oldPixels = stage.pixelWidth == pixelWidth && stage.pixelHeight == pixelHeight
        stage.pixelWidth = pixelWidth
        stage.pixelHeight = pixelHeight
        stage.zoomToFit = zoomToFit
        stage.magnification = magnification
        if recenter || !oldPixels {
            stage.layoutPicture(recenter: true)
        } else if !zoomToFit && (wasFit || abs(oldMagnification - magnification) > 0.0001) {
            stage.zoomAboutSafeCenter()
        } else {
            stage.layoutPicture(recenter: false)
        }
        stage.pixels.needsDisplay = true
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator {
        var placement = 0
    }
}
