import SwiftUI

/// One line of the sidebar. Models and films share one list; each kind keeps its own selection.
private enum SidebarItem: Hashable {
    case model(URL)
    case film(String)
}

struct ContentView: View {
    @EnvironmentObject private var session: Session
    @State private var swiftDrop = false
    @State private var viewDrop = false
    @State private var transportInset: CGFloat = 0
    @AppStorage("showPipeline") private var showPipeline = true
    @AppStorage("modelsExpanded") private var modelsExpanded = true
    @AppStorage("filmsExpanded") private var filmsExpanded = true

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .inspector(isPresented: $showPipeline) {
            // With its own toolbar item the inspector reaches up through the title bar like the
            // sidebar, and the toggle sits over it as a button of its own. The toolbar keeps the
            // item while the panel is closed, so it is the only toggle in both states.
            PipelinePanel()
                .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
                .toolbar {
                    ToolbarSpacer(.flexible)
                    ToolbarItem {
                        pipelineToggle
                    }
                }
        }
        .navigationSplitViewStyle(.balanced)
        .toolbarBackgroundVisibility(.automatic, for: .windowToolbar)
        .onAppear { session.installKeys() }
        .onChange(of: session.selection) { _, _ in
            session.openSelection()
        }
        .onDrop(of: FilmDrop.dropTypes, isTargeted: $swiftDrop) { providers in
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
            List(selection: sidebarSelection) {
                Section(isExpanded: $modelsExpanded) {
                    ForEach(session.library) { model in
                        modelRow(model)
                    }
                } header: {
                    Text("Modelle")
                }
                Section(isExpanded: $filmsExpanded) {
                    ForEach(session.films) { film in
                        filmRow(film)
                    }
                } header: {
                    Text("Filme")
                }
            }
            .listStyle(.sidebar)
            .onDeleteCommand {
                session.askRemoveFilm()
            }
            ScaleMeter(
                rate: session.scaleRate,
                realtime: session.realtimeRate,
                remaining: session.scaleRemaining,
                work: session.exporting ? "Export" : "Cache",
                note: session.exporting ? nil : session.cacheNote
            )
        }
        .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 360)
    }

    /// Shows and hides the pipeline panel. Over the panel while it is open, at the right end otherwise.
    private var pipelineToggle: some View {
        Button {
            showPipeline.toggle()
        } label: {
            Label("Pipeline", systemImage: "sidebar.right")
        }
        .help("Pipeline ein- oder ausblenden: Quelle, Upscaler, Farbe & Licht")
    }

    private func modelRow(_ model: LibraryModel) -> some View {
        Label {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.title)
                    Text(model.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if session.comparison == model.url {
                    Spacer()
                    Image(systemName: "rectangle.split.2x1")
                        .foregroundStyle(.secondary)
                        .help("Vergleichsmodell: Ein Klick ins Bild zeigt dieses Modell.")
                }
            }
        } icon: {
            Image(systemName: model.symbol)
                .foregroundStyle(model.tint)
        }
        .tag(SidebarItem.model(model.url))
        .contextMenu {
            Button(session.comparison == model.url ? "Vergleich aufheben" : "Zum Vergleich") {
                session.toggleComparison(model.url)
            }
            .disabled(model.url == session.selection)
            Button("In den Papierkorb legen", role: .destructive) {
                session.askDelete(model.url)
            }
        }
    }

    private func filmRow(_ film: OpenFilm) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(film.name)
                .lineLimit(1)
            Text(film.subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .tag(SidebarItem.film(film.id))
        .contextMenu {
            Button("Aus der Liste entfernen", role: .destructive) {
                session.askRemoveFilm(film.id)
            }
        }
    }

    /// One list, two selections: the main model and the film both stay highlighted.
    /// A click replaces one of them. An option-click on a model marks it for comparison instead,
    /// and a second one clears it.
    private var sidebarSelection: Binding<Set<SidebarItem>> {
        Binding(
            get: {
                var items: Set<SidebarItem> = []
                if let model = session.selection { items.insert(.model(model)) }
                if let film = session.filmSelection { items.insert(.film(film)) }
                return items
            },
            set: { next in
                var current: Set<SidebarItem> = []
                if let model = session.selection { current.insert(.model(model)) }
                if let film = session.filmSelection { current.insert(.film(film)) }
                let added = next.subtracting(current)
                guard let picked = added.first ?? (next.count == 1 ? next.first : nil) else { return }
                switch picked {
                case .model(let url):
                    if NSEvent.modifierFlags.contains(.option) {
                        if url != session.selection { session.toggleComparison(url) }
                    } else if url != session.selection {
                        session.selection = url
                    }
                case .film(let id):
                    session.filmSelection = id
                    _ = session.showFilm(id: id)
                }
            }
        )
    }

    /// While the comparison is asked for: which model, or what keeps it from the screen.
    private var comparisonBadge: String? {
        guard session.wantedSource == .comparison else { return nil }
        let name = session.comparisonLabel
        if let error = session.comparisonError { return "Vergleich \(name): \(error)" }
        if session.screenSource == .comparison { return "Vergleich: \(name)" }
        if session.comparisonLoading { return "Vergleich \(name): Modell lädt …" }
        return "Vergleich \(name): rechnet Bild …"
    }

    private var detail: some View {
        GeometryReader { geo in
            let insets = geo.safeAreaInsets
            ZStack {
                if showsPicture {
                    Viewer(
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
                        badge: comparisonBadge,
                        onAlternate: { session.showAlternate($0) },
                        onStage: { session.attachStage($0) }
                    )
                } else if session.loadingModel || (session.busy && !session.hasPicture) {
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
        .navigationTitle(session.filmTitle ?? "Whoodeo")
        .navigationSubtitle(session.modelLabel)
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
            // Film and export are one pill, the zoom another.
            ToolbarSpacer(.fixed)
            ToolbarItemGroup {
                ZoomSlider()
                    .disabled(session.frameCount == 0)
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
        session.hasPicture
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
            if session.failure == nil && session.selection != nil && session.frameCount == 0 {
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
