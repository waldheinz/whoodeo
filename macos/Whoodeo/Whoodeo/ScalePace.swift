import Foundation

/// Arrival times of frames the model just produced, in seconds on a monotonic clock.
/// A fast pace is averaged over about two seconds. A slow pace keeps the last few
/// intervals, so one frame every several seconds still has a rate.
struct ScalePace {
    private(set) var arrivals: [Double] = []
    private var smooth: Double = 0
    private(set) var shown: Double = 0

    var isIdle: Bool { arrivals.isEmpty && shown == 0 }

    mutating func add(_ count: Int, at time: Double) {
        guard count > 0 else { return }
        for _ in 0..<count {
            arrivals.append(time)
        }
        revise(at: time)
    }

    mutating func revise(at time: Double) {
        let gap = latestGap
        let horizon = max(2, min(120, gap * 6))
        // One lone frame has to stay until its partner arrives. Two frames define a pace,
        // and that pace is over once the next frame is three intervals late.
        let idleLimit = gap > 0 ? max(2, gap * 3) : (arrivals.count >= 2 ? 2 : 60)
        if let last = arrivals.last, time - last > idleLimit {
            arrivals.removeAll()
        } else if let last = arrivals.last {
            let stale = last - horizon
            arrivals.removeAll { $0 < stale }
        }
        if arrivals.count > 48 {
            arrivals.removeFirst(arrivals.count - 48)
        }

        guard let first = arrivals.first, let last = arrivals.last, last > first else {
            if arrivals.isEmpty {
                smooth = 0
                shown = 0
            }
            return
        }
        let instant = Double(arrivals.count - 1) / (last - first)
        if smooth == 0 {
            smooth = instant
        } else {
            smooth += (instant - smooth) * 0.2
        }
        publish()
    }

    mutating func reset() {
        arrivals.removeAll()
        smooth = 0
        shown = 0
    }

    /// Gap from the newest arrival back to the newest one that is actually earlier.
    /// Frames reported together share one timestamp and must not look like a stall.
    private var latestGap: Double {
        guard let last = arrivals.last else { return 0 }
        for earlier in arrivals.reversed() where last - earlier > 0 {
            return last - earlier
        }
        return 0
    }

    private mutating func publish() {
        let rounded = smooth < 1
            ? (smooth * 100).rounded() / 100
            : (smooth * 10).rounded() / 10
        guard rounded > 0 else {
            shown = 0
            return
        }
        // Around 10 fps, tenths would otherwise flip between 10 and 11.
        let deadband = shown >= 1 ? 0.5 : 0.02
        if shown == 0 || abs(smooth - shown) >= deadband {
            shown = rounded
        }
    }
}

enum ScaleReadout {
    static func eta(_ seconds: Double, work: String) -> String {
        let whole = max(0, Int(seconds.rounded()))
        let body: String
        if whole < 60 {
            body = "\(whole) s"
        } else if whole < 3600 {
            let minutes = whole / 60
            let rest = whole % 60
            body = minutes < 10 ? "\(minutes) min \(rest) s" : "\(minutes) min"
        } else if whole < 24 * 3600 {
            let hours = whole / 3600
            let minutes = (whole % 3600) / 60
            body = minutes == 0 ? "\(hours) h" : "\(hours) h \(minutes) min"
        } else {
            let days = whole / 86400
            let hours = (whole % 86400) / 3600
            let day = days == 1 ? "1 Tag" : "\(days) Tage"
            body = hours == 0 ? day : "\(day) \(hours) h"
        }
        return "noch \(body) · \(work)"
    }
}
