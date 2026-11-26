import os

/// Unified log. Frame numbers are public so Console and `log stream` show them.
nonisolated enum Log {
    static let fill = Logger(subsystem: "meins.whoodeo", category: "fill")
    static let play = Logger(subsystem: "meins.whoodeo", category: "play")
    static let playground = Logger(subsystem: "meins.whoodeo", category: "playground")
    static let model = Logger(subsystem: "meins.whoodeo", category: "model")
    static let drop = Logger(subsystem: "meins.whoodeo", category: "drop")
}
