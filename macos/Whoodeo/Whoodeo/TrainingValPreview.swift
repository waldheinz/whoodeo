import AppKit
import SwiftUI

/// Val PNG triples for the selected run, shown under the charts in Modelle.
struct ValPreviewPane: View {
    let runName: String
    let preferredStep: Int?
    let steps: [Int]

    @State private var step: Int?
    @State private var sample = 0
    @State private var focus: ValPreview.Kind = .pred
    @State private var showSingle = false
    @State private var page: ValPreview.Page?
    @State private var zoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var followFit = true
    @State private var viewport = CGSize(width: 800, height: 360)
    @State private var imageSize: CGSize = .zero

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Validierungsbilder")
                .font(.headline)
            if steps.isEmpty {
                Text("Noch keine Validierungsbilder gespeichert — erscheinen automatisch beim nächsten Val-Schritt oder per `whoodeo-train val-preview <ckpt-id>`.")
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            } else {
                controls
                if let page {
                    sampleStrip(page)
                    zoomBar
                    stage(page)
                } else {
                    Text("Dieser Val-Schritt hat keine lesbaren Bilder.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear { sync(preferPreferred: true) }
        .onChange(of: runName) { _, _ in
            sample = 0
            showSingle = false
            sync(preferPreferred: true)
        }
        .onChange(of: steps) { _, _ in
            sync(preferPreferred: false)
        }
        .onChange(of: preferredStep) { _, _ in
            sync(preferPreferred: true)
        }
        .onChange(of: step) { _, _ in
            loadPage()
        }
        .onChange(of: sample) { _, _ in
            refreshImageSize()
            resetView()
        }
        .onChange(of: showSingle) { _, _ in
            if followFit { applyFit() }
        }
    }

    @ViewBuilder
    private var controls: some View {
        HStack(spacing: 12) {
            if steps.count > 1 {
                Picker("Schritt", selection: stepBinding) {
                    ForEach(steps.reversed(), id: \.self) { value in
                        Text("Schritt \(value)").tag(Optional(value))
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 220, alignment: .leading)
            } else if let step {
                Text("Schritt \(step)")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Picker("Ansicht", selection: $showSingle) {
                Text("Nebeneinander").tag(false)
                Text("Einzelbild").tag(true)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 280)
        }
        .labelsHidden()
    }

    private var stepBinding: Binding<Int?> {
        Binding(
            get: { step },
            set: { next in
                step = next
                sample = 0
                showSingle = false
                resetView()
            }
        )
    }

    private var zoomBar: some View {
        HStack(spacing: 8) {
            Button("Einpassen") { applyFit() }
            Button("100 %") { setZoom(1) }
            Button("200 %") { setZoom(2) }
            Button("400 %") { setZoom(4) }
            Text(zoomLabel)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(minWidth: 56, alignment: .leading)
            if showSingle {
                Picker("Bild", selection: $focus) {
                    ForEach(ValPreview.Kind.allCases, id: \.self) { item in
                        Text(item.title).tag(item)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
            }
            Spacer()
            Text("Pinch oder ⌘-Scroll zum Zoomen, Ziehen oder Scrollen zum Verschieben. Passt das Bild ganz hinein, scrollt die Seite. In der Dreieransicht laufen Zoom und Ausschnitt gleich mit.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .controlSize(.small)
        .buttonStyle(.bordered)
    }

    private var zoomLabel: String {
        "\(Int((zoom * 100).rounded())) %"
    }

    private func sampleStrip(_ page: ValPreview.Page) -> some View {
        let samples = page.meta.samples
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(samples) { row in
                    Button {
                        sample = row.index
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            previewImage(page.url(sample: row.index, kind: .pred), fit: true)
                                .frame(width: 96, height: 72)
                                .clipped()
                                .background(Color.secondary.opacity(0.08))
                                .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                                        .strokeBorder(
                                            sample == row.index ? Color.accentColor : Color.clear,
                                            lineWidth: 2
                                        )
                                }
                            Text(row.caption)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .frame(width: 120, alignment: .leading)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    @ViewBuilder
    private func stage(_ page: ValPreview.Page) -> some View {
        let kinds: [ValPreview.Kind] = showSingle ? [focus] : ValPreview.Kind.allCases
        HStack(alignment: .top, spacing: 10) {
            ForEach(kinds, id: \.self) { kind in
                VStack(alignment: .leading, spacing: 6) {
                    Text(kind.title)
                        .font(.subheadline.weight(.medium))
                    zoomPane(url: page.url(sample: sample, kind: kind), kind: kind)
                        .frame(maxWidth: .infinity)
                        .frame(height: showSingle ? 520 : 360)
                        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
            }
        }
    }

    private func zoomPane(url: URL, kind: ValPreview.Kind) -> some View {
        GeometryReader { geo in
            let size = pixelSize(url)
            ZStack {
                Color.clear
                if let image = NSImage(contentsOf: url) {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.none)
                        .antialiased(false)
                        .frame(width: size.width * zoom, height: size.height * zoom)
                        .offset(offset)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .clipped()
            .contentShape(Rectangle())
            .overlay {
                WheelCatcher(
                    onMagnify: { factor, point in
                        zoom(by: factor, around: point, in: geo.size)
                    },
                    onScroll: { delta, point, asZoom in
                        if asZoom {
                            let factor = pow(1.0015, Double(-delta.height))
                            zoom(by: factor, around: point, in: geo.size)
                            return true
                        }
                        // Only pans when the zoomed image has room in that direction; otherwise the page scrolls.
                        return pan(by: CGSize(width: delta.width, height: -delta.height), in: geo.size)
                    },
                    onDrag: { delta in
                        pan(by: delta, in: geo.size)
                    },
                    onClick: {
                        if !showSingle {
                            focus = kind
                            showSingle = true
                        }
                    }
                )
            }
            .onAppear {
                viewport = geo.size
                imageSize = size
                if followFit { applyFit() }
            }
            .onChange(of: geo.size) { _, new in
                viewport = new
                imageSize = size
                if followFit { applyFit() }
                else { clampOffset(in: new) }
            }
        }
    }

    @ViewBuilder
    private func previewImage(_ url: URL, fit: Bool) -> some View {
        if let image = NSImage(contentsOf: url) {
            let size = pixelSize(image)
            if fit {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.none)
                    .antialiased(false)
                    .scaledToFit()
            } else {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.none)
                    .antialiased(false)
                    .frame(width: size.width, height: size.height)
            }
        } else {
            Color.secondary.opacity(0.08)
                .frame(minHeight: 72)
        }
    }

    private func pixelSize(_ url: URL) -> CGSize {
        guard let image = NSImage(contentsOf: url) else { return .zero }
        return pixelSize(image)
    }

    private func pixelSize(_ image: NSImage) -> CGSize {
        if let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            return CGSize(width: cg.width, height: cg.height)
        }
        return image.size
    }

    private func fitScale(in size: CGSize = .zero) -> CGFloat {
        let view = size == .zero ? viewport : size
        let image = imageSize
        guard image.width > 0, image.height > 0, view.width > 0, view.height > 0 else { return 1 }
        return min(view.width / image.width, view.height / image.height)
    }

    private func applyFit() {
        followFit = true
        zoom = max(fitScale(), 0.05)
        offset = .zero
    }

    private func setZoom(_ value: CGFloat) {
        followFit = false
        zoom(to: value, around: CGPoint(x: viewport.width / 2, y: viewport.height / 2), in: viewport)
    }

    private func resetView() {
        followFit = true
        offset = .zero
        applyFit()
    }

    private func refreshImageSize() {
        guard let page, let url = Optional(page.url(sample: sample, kind: focus)) else { return }
        imageSize = pixelSize(url)
    }

    private func zoom(by factor: CGFloat, around point: CGPoint, in size: CGSize) {
        zoom(to: zoom * factor, around: point, in: size)
    }

    private func zoom(to target: CGFloat, around point: CGPoint, in size: CGSize) {
        let image = imageSize
        guard image.width > 0, image.height > 0 else { return }
        let minimum = min(fitScale(in: size) * 0.5, 0.25)
        let next = min(max(target, minimum), 16)
        let origin = imageOrigin(zoom: zoom, offset: offset, in: size)
        let imagePoint = CGPoint(
            x: (point.x - origin.x) / zoom,
            y: (point.y - origin.y) / zoom
        )
        let newOrigin = CGPoint(x: point.x - imagePoint.x * next, y: point.y - imagePoint.y * next)
        let centered = CGPoint(
            x: (size.width - image.width * next) / 2,
            y: (size.height - image.height * next) / 2
        )
        followFit = false
        zoom = next
        offset = CGSize(width: newOrigin.x - centered.x, height: newOrigin.y - centered.y)
        clampOffset(in: size)
    }

    /// Returns whether the image actually moved (false at the edge or when it fits).
    @discardableResult
    private func pan(by delta: CGSize, in size: CGSize) -> Bool {
        let before = offset
        offset = CGSize(width: offset.width + delta.width, height: offset.height + delta.height)
        clampOffset(in: size)
        let moved = abs(offset.width - before.width) + abs(offset.height - before.height) > 0.01
        if moved { followFit = false }
        return moved
    }

    private func imageOrigin(zoom: CGFloat, offset: CGSize, in size: CGSize) -> CGPoint {
        CGPoint(
            x: (size.width - imageSize.width * zoom) / 2 + offset.width,
            y: (size.height - imageSize.height * zoom) / 2 + offset.height
        )
    }

    private func clampOffset(in size: CGSize) {
        let scaled = CGSize(width: imageSize.width * zoom, height: imageSize.height * zoom)
        let extraW = max(0, (scaled.width - size.width) / 2)
        let extraH = max(0, (scaled.height - size.height) / 2)
        // If the image is smaller than the viewport on an axis, keep it centered on that axis.
        let maxW = scaled.width <= size.width ? 0 : extraW
        let maxH = scaled.height <= size.height ? 0 : extraH
        offset = CGSize(
            width: min(max(offset.width, -maxW), maxW),
            height: min(max(offset.height, -maxH), maxH)
        )
    }

    private func sync(preferPreferred: Bool) {
        let next: Int?
        if let step, steps.contains(step), !(preferPreferred && preferredStep != step) {
            next = step
        } else if let preferredStep, steps.contains(preferredStep) {
            next = preferredStep
        } else {
            next = steps.last
        }
        if next != step {
            sample = 0
            showSingle = false
            resetView()
        }
        step = next
        loadPage()
    }

    private func loadPage() {
        if let step {
            page = ValPreview.load(runName: runName, step: step)
            if let page, !page.meta.samples.contains(where: { $0.index == sample }) {
                sample = page.meta.samples.first?.index ?? 0
            }
            refreshImageSize()
            if followFit { applyFit() }
        } else {
            page = nil
        }
    }
}

/// Pinch and ⌘/⌥-scroll zoom the preview; plain scrolling pans a zoomed image. When there is nothing
/// to pan (image fits, or at its edge), the scroll goes on to the outer detail ScrollView, so the page
/// stays scrollable with the pointer over the pictures. One gesture (incl. momentum) keeps one target.
private struct WheelCatcher: NSViewRepresentable {
    var onMagnify: (CGFloat, CGPoint) -> Void
    /// delta, point, zoom; returns whether the preview used the event.
    var onScroll: (CGSize, CGPoint, Bool) -> Bool
    var onDrag: (CGSize) -> Void
    var onClick: () -> Void

    func makeNSView(context: Context) -> CatcherView {
        let view = CatcherView()
        view.onMagnify = onMagnify
        view.onScroll = onScroll
        view.onDrag = onDrag
        view.onClick = onClick
        return view
    }

    func updateNSView(_ view: CatcherView, context: Context) {
        view.onMagnify = onMagnify
        view.onScroll = onScroll
        view.onDrag = onDrag
        view.onClick = onClick
    }

    final class CatcherView: NSView {
        var onMagnify: ((CGFloat, CGPoint) -> Void)?
        var onScroll: ((CGSize, CGPoint, Bool) -> Bool)?
        var onDrag: ((CGSize) -> Void)?
        var onClick: (() -> Void)?
        private var lastDrag: CGPoint?
        private var dragged = false
        /// Target of the current trackpad gesture: true = page, false = preview, nil = not decided.
        private var gestureToPage: Bool?

        override var acceptsFirstResponder: Bool { true }
        override var isFlipped: Bool { true }

        override func magnify(with event: NSEvent) {
            onMagnify?(1 + event.magnification, convert(event.locationInWindow, from: nil))
        }

        override func scrollWheel(with event: NSEvent) {
            let point = convert(event.locationInWindow, from: nil)
            let zoomKeys = event.modifierFlags.contains(.command) || event.modifierFlags.contains(.option)
            let precise = event.hasPreciseScrollingDeltas
            if zoomKeys {
                let delta = precise ? event.scrollingDeltaY : event.scrollingDeltaY * 20
                _ = onScroll?(CGSize(width: 0, height: delta), point, true)
                return
            }
            let scale: CGFloat = precise ? 1 : 20
            let delta = CGSize(width: event.scrollingDeltaX * scale, height: event.scrollingDeltaY * scale)
            let starts = event.phase.contains(.began) || (event.phase.isEmpty && event.momentumPhase.isEmpty)
            if starts || gestureToPage == nil {
                let used = onScroll?(delta, point, false) ?? false
                gestureToPage = !used
                if !used { super.scrollWheel(with: event) }
            } else if gestureToPage == true {
                super.scrollWheel(with: event)
            } else {
                _ = onScroll?(delta, point, false)
            }
            if event.momentumPhase.contains(.ended) || event.phase.contains(.cancelled) {
                gestureToPage = nil
            }
        }

        override func mouseDown(with event: NSEvent) {
            lastDrag = convert(event.locationInWindow, from: nil)
            dragged = false
        }

        override func mouseDragged(with event: NSEvent) {
            let point = convert(event.locationInWindow, from: nil)
            if let lastDrag {
                let delta = CGSize(width: point.x - lastDrag.x, height: point.y - lastDrag.y)
                if abs(delta.width) + abs(delta.height) > 1 { dragged = true }
                if dragged { onDrag?(delta) }
            }
            lastDrag = point
        }

        override func mouseUp(with event: NSEvent) {
            if !dragged { onClick?() }
            lastDrag = nil
            dragged = false
        }

        override func resetCursorRects() {
            addCursorRect(bounds, cursor: .openHand)
        }
    }
}
