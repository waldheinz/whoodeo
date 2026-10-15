import os

/// Unified log. Frame numbers are public so Console and `log stream` show them.
enum Log {
    static let fill = Logger(subsystem: "meins.whoodeo", category: "fill")
    static let play = Logger(subsystem: "meins.whoodeo", category: "play")
}
