"""The Whoodeo SQLite database used by the app, the trainer, and make-low.

The default file is ~/Library/Application Support/Whoodeo/whoodeo.sqlite
(or $WHOODEO_DB / --db). Connections use WAL mode, a 30 s busy timeout,
and foreign keys. Film paths in the tables are relative to settings.dataset_root.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from whoodeo.library import rungs_of

DEFAULT_DB = Path.home() / "Library/Application Support/Whoodeo/whoodeo.sqlite"
FNV_OFFSET = 14_695_981_039_346_656_037
FNV_PRIME = 1_099_511_628_211


def default_db_path():
    env = os.environ.get("WHOODEO_DB", "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_DB


def connect(path=None):
    """Open the database in WAL mode with foreign keys and a 30 s busy timeout."""
    db_path = Path(path) if path is not None else default_db_path()
    db_path = db_path.expanduser()
    if not db_path.is_file():
        raise SystemExit(f"no database: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def setting(conn, key):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def set_setting(conn, key, value):
    conn.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def dataset_root(conn):
    value = setting(conn, "dataset_root")
    if not value:
        raise SystemExit("no dataset_root in the database")
    return Path(value)


def title_split(name):
    """Stable 10% validation split from the title name (FNV-1a), same as WhoodeoDB."""
    hash_ = FNV_OFFSET
    for byte in name.encode("utf-8"):
        hash_ ^= byte
        hash_ = (hash_ * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return "val" if hash_ % 10 == 0 else "train"


def load_rungs(conn):
    """(name, pixel budget), largest first, from settings.rungs."""
    raw = setting(conn, "rungs")
    if not raw:
        raise SystemExit("no settings.rungs in the database")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"settings.rungs: {exc}") from exc
    if isinstance(data, list):
        data = {"rungs": data}
    return rungs_of(data, "settings.rungs")


def frame_ratio_columns(conn):
    """True when degrade_presets has frame_ratio_num / frame_ratio_den (schema v3)."""
    names = {row[1] for row in conn.execute("PRAGMA table_info(degrade_presets)")}
    return "frame_ratio_num" in names and "frame_ratio_den" in names


def load_degrade(conn, enabled_only=True):
    """Shared scale / flags / pix_fmt and an ordered {name: ffmpeg_args} map."""
    scale, flags, pix_fmt, variants, _ = load_degrade_ratios(conn, enabled_only)
    return scale, flags, pix_fmt, variants


def load_degrade_ratios(conn, enabled_only=True):
    """Like load_degrade, plus {name: (frame_ratio_num, frame_ratio_den)}.

    master_idx = (low_idx * num + den // 2) // den, and a low file has
    expected_low_frames(master_frames, num, den) frames. A database without
    the columns (schema v2) reads 1/1 for every preset.
    """
    ratio_sql = (
        "frame_ratio_num, frame_ratio_den" if frame_ratio_columns(conn)
        else "1 AS frame_ratio_num, 1 AS frame_ratio_den"
    )
    sql = (
        f"SELECT name, ffmpeg_args, scale, scale_flags, pix_fmt, enabled, {ratio_sql} "
        "FROM degrade_presets ORDER BY name"
    )
    rows = conn.execute(sql).fetchall()
    if enabled_only:
        rows = [row for row in rows if row["enabled"]]
    if not rows:
        raise SystemExit("no degrade presets in the database")
    variants = {}
    ratios = {}
    metas = []
    for row in rows:
        num, den = row["frame_ratio_num"], row["frame_ratio_den"]
        if not isinstance(num, int) or not isinstance(den, int) or num < 1 or den < 1:
            raise SystemExit(f"degrade preset {row['name']}: bad frame_ratio {num!r}/{den!r}")
        variants[row["name"]] = row["ffmpeg_args"]
        ratios[row["name"]] = (num, den)
        metas.append((row["scale"], row["scale_flags"], row["pix_fmt"]))
    unique = set(metas)
    if len(unique) != 1:
        names = ", ".join(variants)
        raise SystemExit(
            f"selected degrade presets do not share scale/flags/pix_fmt: {names}"
        )
    scale, flags, pix_fmt = metas[0]
    return scale, flags, pix_fmt, variants, ratios


def expected_low_frames(master_frames, num=1, den=1):
    """Frames a low file must have for a master of master_frames frames."""
    return (master_frames * den) // num


def master_index(low_idx, num=1, den=1):
    """Master frame that is the target of low frame low_idx."""
    return (low_idx * num + den // 2) // den


def remember_source(conn, name, origin_path=None, frames=None, now=None):
    """Insert or update a source. Split is set only for a new row."""
    if now is None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = conn.execute(
        "SELECT id, frames, origin_path FROM sources WHERE name = ?", (name,)
    ).fetchone()
    if existing is None:
        if frames is None or frames < 1:
            raise RuntimeError(f"{name}: missing frame count")
        conn.execute(
            "INSERT INTO sources(name, origin_path, frames, split, enabled, added_at) "
            "VALUES(?, ?, ?, ?, 1, ?)",
            (name, origin_path, frames, title_split(name), now),
        )
        return conn.execute("SELECT id FROM sources WHERE name = ?", (name,)).fetchone()[0]
    new_frames = existing["frames"] if frames is None or frames < 1 else frames
    new_origin = existing["origin_path"] if origin_path is None else origin_path
    conn.execute(
        "UPDATE sources SET origin_path = ?, frames = ? WHERE id = ?",
        (new_origin, new_frames, existing["id"]),
    )
    return existing["id"]


def remember_master(conn, title, rung, rel_path, width, height, frames=None, origin_path=None):
    source_id = remember_source(conn, title, origin_path=origin_path, frames=frames)
    conn.execute(
        "INSERT INTO source_masters(source_id, rung, rel_path, width, height) "
        "VALUES(?, ?, ?, ?, ?) "
        "ON CONFLICT(source_id, rung) DO UPDATE SET "
        "rel_path = excluded.rel_path, width = excluded.width, height = excluded.height",
        (source_id, rung, rel_path, width, height),
    )
    return conn.execute(
        "SELECT id FROM source_masters WHERE source_id = ? AND rung = ?",
        (source_id, rung),
    ).fetchone()[0]


def remember_variant(conn, title, rung, preset, rel_path):
    row = conn.execute(
        "SELECT m.id FROM source_masters m "
        "JOIN sources s ON s.id = m.source_id "
        "WHERE s.name = ? AND m.rung = ?",
        (title, rung),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"{title}.{rung}: not in the database")
    conn.execute(
        "INSERT INTO source_variants(master_id, preset, rel_path, enabled) "
        "VALUES(?, ?, ?, 1) "
        "ON CONFLICT(master_id, preset) DO UPDATE SET rel_path = excluded.rel_path",
        (row["id"], preset, rel_path),
    )


def list_masters(conn):
    """All masters with their source fields, for stats and encode planning."""
    return conn.execute(
        """
        SELECT s.id AS source_id, s.name, s.frames, s.split, s.origin_path, s.enabled AS source_enabled,
               m.id AS master_id, m.rung, m.rel_path, m.width, m.height
        FROM sources s
        JOIN source_masters m ON m.source_id = s.id
        ORDER BY s.name, m.rung
        """
    ).fetchall()


def listed_variants(conn, master_id):
    return [
        row["preset"]
        for row in conn.execute(
            "SELECT preset FROM source_variants WHERE master_id = ? ORDER BY preset",
            (master_id,),
        )
    ]


