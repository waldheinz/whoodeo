import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var session: Session
    @State private var swiftDrop = false
    @State private var viewDrop = false
    @State private var transportInset: CGFloat = 0

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .navigationSplitViewStyle(.balanced)
        .toolbarBackgroundVisibility(.automatic, for: .windowToolbar)
        .onAppear { session.installKeys() }
        .onChange(of: session.selection) { _, _ in
            session.openSelection()
        }
        .onDrop(of: FilmDrop.contentTypes, isTargeted: $swiftDrop) { providers in
            FilmDrop.read(providers) { urls in
                Task { @MainActor in
                    _ = session.openFilms(urls)
                }
            }
            return true
        }
        .confirmationDialog(
            "„\(session.deleteName)“ in den Papierkorb legen?",
            isPresented: $session.confirmDelete,
            titleVisibility: .visible
        ) {
            Button("In den Papierkorb legen", role: .destructive) {
                session.deleteSelected()
            }
            Button("Abbrechen", role: .cancel) {}
        } message: {
            Text("Das Modell bleibt im Papierkorb, bis du es dort löschst.")
        }
        .confirmationDialog(
            "„\(session.removeName)“ aus der Liste entfernen?",
            isPresented: $session.confirmRemove,
            titleVisibility: .visible
        ) {
            Button("Entfernen", role: .destructive) {
                session.removeSelectedFilm()
            }
            Button("Abbrechen", role: .cancel) {}
        } message: {
            Text("Der Bildercache dieses Films wird gelöscht. Die Filmedatei bleibt.")
        }
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            List(selection: $session.selection) {
                ForEach(session.library) { model in
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.title)
                            Text(model.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: model.symbol)
                            .foregroundStyle(model.tint)
                    }
                    .tag(model.url)
                    .contextMenu {
                        Button("In den Papierkorb legen", role: .destructive) {
                            session.askDelete(model.url)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            filmsList
            ScaleMeter(
                rate: session.scaleRate,
                realtime: session.realtimeRate,
                remaining: session.scaleRemaining,
                work: session.exporting ? "Export" : "Cache"
            )
        }
        .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 360)
        .navigationTitle("Whoodeo")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    session.importModel()
                } label: {
                    Label("Modell importieren", systemImage: "plus")
                }
                Button {
                    session.askDelete()
                } label: {
                    Label("In den Papierkorb legen", systemImage: "trash")
                }
                .disabled(session.selection == nil)
            }
        }
    }

    private var filmsList: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Filme")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    session.askRemoveFilm()
                } label: {
                    Image(systemName: "minus")
                }
                .buttonStyle(.borderless)
                .disabled(session.filmSelection == nil)
                .help("Aus der Liste entfernen")
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 2)
            List(selection: $session.filmSelection) {
                ForEach(session.films) { film in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(film.name)
                            .lineLimit(1)
                        Text(film.subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    .tag(film.id as String?)
                    .contextMenu {
                        Button("Aus der Liste entfernen", role: .destructive) {
                            session.askRemoveFilm(film.id)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .frame(minHeight: 72)
            .onChange(of: session.filmSelection) { _, id in
                guard let id else { return }
                _ = session.showFilm(id: id)
            }
            .onDeleteCommand {
                session.askRemoveFilm()
            }
        }
    }

    private var detail: some View {
        GeometryReader { geo in
            let insets = geo.safeAreaInsets
            ZStack {
                if showsPicture {
                    Viewer(
                        recon: session.recon,
                        source: session.source,
                        zoomToFit: session.zoomToFit,
                        magnification: session.magnification,
                        placement: session.zoomPlacement,
                        pixelWidth: session.pixelWidth,
                        pixelHeight: session.pixelHeight,
                        fallbackTop: insets.top,
                        fallbackLeading: insets.leading,
                        fallbackBottom: max(insets.bottom, transportInset),
                        fallbackTrailing: insets.trailing,
                        onFilm: { _ = session.openFilms([$0]) },
                        onHover: { viewDrop = $0 },
                        onZoom: { fits, factor in
                            session.noteZoom(fits: fits, magnification: factor)
                        },
                        onOriginal: { session.originalNow() },
                        onStage: { session.attachStage($0) }
                    )
                } else if session.loadingModel || (session.busy && session.recon == nil) {
                    ProgressView(session.loadingModel ? "Modell wird geladen…" : "rechnet…")
                        .controlSize(.large)
                } else {
                    emptyState
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .ignoresSafeArea()
        .overlay {
            if swiftDrop || viewDrop {
                dropHint
            }
        }
        .navigationTitle(session.modelLabel.isEmpty ? "Whoodeo" : session.modelLabel)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    session.pickFilm()
                } label: {
                    Label("Film", systemImage: "film")
                }
                Button {
                    session.pickExport()
                } label: {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
                .disabled(!session.canExport)
                if session.exporting {
                    Button {
                        session.cancelExport()
                    } label: {
                        Label("Abbrechen", systemImage: "xmark")
                    }
                }
            }
            ToolbarItemGroup {
                ZoomSlider()
                    .disabled(session.frameCount == 0)
                Picker("Rechenwerk", selection: Binding(
                    get: { session.unit },
                    set: { session.changeUnit($0) }
                )) {
                    ForEach(Compute.allCases) { unit in
                        Text(unit.title).tag(unit)
                    }
                }
                .pickerStyle(.menu)
                .fixedSize()
            }
        }
        .safeAreaBar(edge: .bottom) {
            transportBar
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .padding(.horizontal, 16)
                .padding(.bottom, 10)
                .background {
                    GeometryReader { proxy in
                        Color.clear.preference(key: TransportInsetKey.self, value: proxy.size.height)
                    }
                }
        }
        .onPreferenceChange(TransportInsetKey.self) { transportInset = $0 }
    }

    private var showsPicture: Bool {
        session.recon != nil
    }

    private var dropHint: some View {
        ZStack {
            Color.accentColor.opacity(0.14)
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.accentColor, lineWidth: 2)
                .padding(8)
            Label("Film öffnen", systemImage: "film")
                .font(.title3)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.regularMaterial, in: Capsule())
        }
        .allowsHitTesting(false)
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label(emptyHeading, systemImage: emptySymbol)
        } description: {
            Text(session.placeholder)
        } actions: {
            if session.failure == nil && session.library.isEmpty {
                Button("Modell importieren…") { session.importModel() }
            } else if session.failure == nil && session.selection != nil && session.frameCount == 0 {
                Button("Film öffnen…") { session.pickFilm() }
            }
        }
    }

    private var emptyHeading: String {
        if session.failure != nil { return "Das hat nicht geklappt" }
        if session.library.isEmpty { return "Keine Modelle" }
        if session.selection == nil { return "Modell wählen" }
        if session.frameCount == 0 { return "Kein Film" }
        return "Whoodeo"
    }

    private var emptySymbol: String {
        if session.failure != nil { return "exclamationmark.triangle" }
        if session.library.isEmpty { return "square.stack.3d.up" }
        if session.selection == nil { return "sidebar.left" }
        return "film"
    }

    private var transportBar: some View {
        HStack(spacing: 8) {
                transport("backward.end.fill") { session.scrub(0) }
                transport("backward.frame.fill") { session.step(-1) }
                Button {
                    session.togglePlay()
                } label: {
                    Image(systemName: session.playing ? "pause.fill" : "play.fill")
                }
                .buttonStyle(.borderless)
                .disabled(session.frameCount == 0)
                transport("forward.frame.fill") { session.step(1) }
                transport("forward.end.fill") { session.scrub(max(session.frameCount - 1, 0)) }
                FrameSlider(
                    index: session.index,
                    count: session.frameCount,
                    coverage: session.coverage,
                    onPreview: { session.preview($0) },
                    onCommit: { session.endPreview() }
                )
                Text(session.frameCount == 0 ? "–" : "\(session.index + 1)/\(session.frameCount)")
                    .font(.caption.monospacedDigit())
                    .frame(width: 96, alignment: .trailing)
                Divider()
                    .frame(height: 16)
                Button("Von hier") { session.markIn() }
                    .buttonStyle(.borderless)
                    .disabled(session.frameCount == 0)
                    .help("In-Punkt (I)")
                Button("Bis hier") { session.markOut() }
                    .buttonStyle(.borderless)
                    .disabled(session.frameCount == 0)
                    .help("Out-Punkt (O)")
                Text("\(session.inPoint + 1)–\(max(session.outPoint, session.inPoint) + 1)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Divider()
                    .frame(height: 16)
                Toggle("Cache füllen", isOn: $session.fillCache)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .fixedSize()
                    .help("Der Schalter ist an, wenn der Knopf auf der rechten Seite liegt. Dann werden fehlende Bilder im Hintergrund gerechnet und gespeichert. Die Wahl bleibt nach einem Neustart erhalten.")
        }
    }

    private func transport(_ symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
        }
        .buttonStyle(.borderless)
        .disabled(session.frameCount == 0)
    }
}

/// Photos-style zoom slider. 100% is a system tick, so the mark and the thumb share one track.
private struct ZoomSlider: View {
    @EnvironmentObject private var session: Session

    var body: some View {
        let percent = Int((session.magnification * 100).rounded())
        HStack(spacing: 6) {
            Button {
                session.chooseZoom(.fit)
            } label: {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
            }
            .help("Einpassen")
            .accessibilityLabel("Einpassen")
            Text(zoomLabel(percent))
                .font(.caption.monospacedDigit())
                .foregroundStyle(session.zoomToFit ? .secondary : .primary)
                .frame(width: 64, alignment: .center)
            Slider(
                value: Binding(
                    get: { ZoomScale.position(for: session.magnification) },
                    set: { session.setSliderZoom($0) }
                ),
                in: 0...1
            ) {
                Text("Zoom")
            } ticks: {
                SliderTick(ZoomScale.hundred)
            }
            .controlSize(.small)
            .frame(width: 168)
            .help("Zoom. Der Strich ist 100 % und der Knopf rastet dort ein. Scrollrad und Trackpad zoomen und klicken beim Überqueren von 100 %. Ziehen verschiebt den Ausschnitt. Stillhalten zeigt das Original.")
        }
    }

    private func zoomLabel(_ percent: Int) -> String {
        if session.frameCount == 0 { return "–" }
        if session.zoomToFit { return "Passend" }
        return "\(percent) %"
    }
}

private struct TransportInsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// The track marks frames that are already in the archive. The thumb scrubs.
private struct FrameSlider: View {
    var index: Int
    var count: Int
    var coverage: Coverage
    var onPreview: (Int) -> Void
    var onCommit: () -> Void

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let height = geo.size.height
            ZStack {
                Capsule()
                    .fill(Color.primary.opacity(0.16))
                    .frame(height: 4)
                CoverageBar(coverage: coverage)
                    .equatable()
                Circle()
                    .fill(Color.white)
                    .shadow(color: .black.opacity(0.35), radius: 0.5, y: 0.5)
                    .frame(width: 14, height: 14)
                    .position(x: thumbX(width: width), y: height / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        onPreview(frame(at: value.location.x, width: width))
                    }
                    .onEnded { _ in
                        onCommit()
                    }
            )
        }
        .frame(height: 22)
        .frame(maxWidth: .infinity)
        .opacity(count == 0 ? 0.4 : 1)
        .allowsHitTesting(count > 0)
        .accessibilityElement()
        .accessibilityLabel("Bild")
        .accessibilityValue("\(shownNumber) von \(max(count, 1))\(coverage.contains(index) ? ", im Cache" : "")")
        .help("Farbig hinterlegt ist schon gerechnet. Ziehen zeigt vorhandene Bilder. Liegt der Regler kurz still, wird das Bild gerechnet.")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: onPreview(min(index + 1, max(count - 1, 0)))
            case .decrement: onPreview(max(index - 1, 0))
            default: break
            }
        }
    }

    private var shownNumber: Int { count == 0 ? 0 : index + 1 }

    private func thumbX(width: CGFloat) -> CGFloat {
        guard count > 1, width > 0 else { return width / 2 }
        let fraction = CGFloat(index) / CGFloat(count - 1)
        return min(max(fraction * width, 7), width - 7)
    }

    private func frame(at x: CGFloat, width: CGFloat) -> Int {
        guard count > 1, width > 0 else { return 0 }
        let fraction = min(max(x / width, 0), 1)
        return Int((fraction * CGFloat(count - 1)).rounded())
    }
}

/// The blue spans. Kept off the playhead, so moving through the film does not rebuild them.
private struct CoverageBar: View, Equatable {
    var coverage: Coverage

    var body: some View {
        Canvas { context, size in
            let path = cachedPath(width: size.width, height: size.height)
            context.fill(path, with: .color(.accentColor))
        }
        .frame(height: 4)
    }

    private func cachedPath(width: CGFloat, height: CGFloat) -> Path {
        var path = Path()
        let pixels = Int(width.rounded(.down))
        guard coverage.count > 0, pixels > 1 else { return path }
        let barY = (height - 4) / 2
        var x = 0
        while x < pixels {
            if coverage.any(from: frameIndex(x, pixels: pixels), until: frameIndex(x + 1, pixels: pixels)) {
                var end = x + 1
                while end < pixels {
                    let start = frameIndex(end, pixels: pixels)
                    let upper = frameIndex(end + 1, pixels: pixels)
                    if !coverage.any(from: start, until: upper) { break }
                    end += 1
                }
                path.addRoundedRect(
                    in: CGRect(x: CGFloat(x), y: barY, width: CGFloat(max(end - x, 1)), height: 4),
                    cornerSize: CGSize(width: 2, height: 2)
                )
                x = end
            } else {
                x += 1
            }
        }
        return path
    }

    private func frameIndex(_ pixel: Int, pixels: Int) -> Int {
        guard coverage.count > 0, pixels > 0 else { return 0 }
        return min(coverage.count, pixel * coverage.count / pixels)
    }
}
