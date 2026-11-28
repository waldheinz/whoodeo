import SwiftUI

/// The time plan of the open film as a table of stretches: what the planner found, how sure it is,
/// and corrections per stretch. A click shows the stretch in the window.
struct TimePlanSheet: View {
    @EnvironmentObject private var session: Session
    @Environment(\.dismiss) private var dismiss
    @State private var selection: TimeSegment.ID?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let plan = session.timePlan {
                summary(plan)
                Table(plan.segments, selection: $selection) {
                    TableColumn("Von") { Text(Self.clock($0.from, plan.sourceFps)).monospacedDigit() }
                        .width(min: 70, ideal: 80)
                    TableColumn("Bis") { Text(Self.clock($0.to, plan.sourceFps)).monospacedDigit() }
                        .width(min: 70, ideal: 80)
                    TableColumn("Muster") { segment in
                        HStack(spacing: 6) {
                            Circle().fill(TimeStrip.color(segment)).frame(width: 8, height: 8)
                            Text(segment.kind == .blend ? "Mischbilder A" : "Original")
                        }
                    }
                    .width(min: 110, ideal: 130)
                    TableColumn("Phase") { Text($0.phase.map(String.init) ?? "–").monospacedDigit() }
                        .width(min: 40, ideal: 50)
                    TableColumn("Zuversicht") { segment in
                        Text(segment.kind == .blend ? "\(Int((segment.confidence * 100).rounded())) %" : "–").monospacedDigit()
                    }
                    .width(min: 70, ideal: 80)
                    TableColumn("Korrektur") { segment in
                        Text(correction(for: segment) ?? "")
                    }
                }
                .onChange(of: selection) { _, next in
                    guard let next, let segment = plan.segments.first(where: { $0.id == next }) else { return }
                    session.showSourceFrame(segment.from + min(3, max(segment.to - segment.from - 1, 0)))
                }
                actions(plan)
            } else {
                Text("Kein Zeitplan. Zeitstruktur auf Automatisch oder Manuell stellen.")
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("Fertig") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(minWidth: 680, minHeight: 460)
    }

    private func summary(_ plan: TimePlan) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Zeitstruktur").font(.headline)
            Text(plan.note).font(.callout)
            Text("\(plan.groups.count) Gruppen · \(Int((plan.coverage * 100).rounded())) % rekonstruiert · \(plan.jumps) Phasensprünge · Zuversicht \(Int((plan.confidence * 100).rounded())) % · Plan \(plan.hash)\(plan.accepted ? "" : " · nicht übernommen")")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
            if let warning = Self.warning(plan) {
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private func actions(_ plan: TimePlan) -> some View {
        let segment = selection.flatMap { id in plan.segments.first { $0.id == id } }
        return HStack(spacing: 8) {
            Button("Original erzwingen") {
                guard let segment else { return }
                replace(segment, with: TimeOverride(from: segment.from, to: segment.to, original: true))
            }
            .disabled(segment == nil || segment?.kind == .original)
            Button("Phase −1") { shift(segment, by: -1) }
                .disabled(segment?.phase == nil)
            Button("Phase +1") { shift(segment, by: 1) }
                .disabled(segment?.phase == nil)
            Button("Korrektur entfernen") {
                guard let segment else { return }
                session.setTimeOverrides(session.preprocess.overrides.filter { !($0.from < segment.to && $0.to > segment.from) })
            }
            .disabled(segment.map { correction(for: $0) == nil } ?? true)
            Spacer()
            Button("Alle Korrekturen entfernen") { session.setTimeOverrides([]) }
                .disabled(session.preprocess.overrides.isEmpty)
        }
        .disabled(session.exporting || session.decimateProgress != nil)
    }

    private func shift(_ segment: TimeSegment?, by delta: Int) {
        guard let segment, let phase = segment.phase else { return }
        replace(segment, with: TimeOverride(from: segment.from, to: segment.to, phase: ((phase + delta) % 6 + 6) % 6))
    }

    private func replace(_ segment: TimeSegment, with override: TimeOverride) {
        var list = session.preprocess.overrides.filter { !($0.from < segment.to && $0.to > segment.from) }
        list.append(override)
        list.sort { $0.from < $1.from }
        session.setTimeOverrides(list)
    }

    private func correction(for segment: TimeSegment) -> String? {
        let hits = session.preprocess.overrides.filter { $0.from < segment.to && $0.to > segment.from }
        guard let first = hits.first else { return nil }
        if first.original { return "Original" }
        if let phase = first.phase { return "Phase \(phase)" }
        return nil
    }

    /// One line for the panel when part of the film has no pattern the solver knows or the file is short.
    static func warning(_ plan: TimePlan?) -> String? {
        guard let plan, !plan.groups.isEmpty else { return nil }
        var parts: [String] = []
        let unknown = plan.segments.filter { $0.kind == .original }.reduce(0) { $0 + $1.to - $1.from }
        if plan.accepted, plan.usable > 0, unknown * 10 > plan.usable {
            parts.append("\(Int((Double(unknown) / Double(plan.usable) * 100).rounded())) % ohne bekanntes Muster (laufen als Original)")
        }
        let unsure = plan.segments.filter { $0.kind == .blend && $0.confidence < 0.3 }.count
        if unsure > 0 { parts.append("\(unsure) unsichere Abschnitte") }
        if plan.truncated { parts.append("Datei endet nach Bild \(plan.usable)\(plan.declared > plan.usable ? " von \(plan.declared)" : "")") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    static func clock(_ frame: Int, _ fps: Double) -> String {
        let seconds = Double(frame) / max(fps, 1)
        let whole = Int(seconds)
        return String(format: "%d:%02d:%02d.%02d", whole / 3600, whole / 60 % 60, whole % 60, Int((seconds - Double(whole)) * 100))
    }
}
