"""Write training videos.

`whoodeo-make-low import` writes a source into the orig folder and records
it in the Whoodeo SQLite database (and in catalog.json beside orig/ and low/,
as a mirror). The rungs are pixel
budgets. A source larger than a rung is encoded down to it, 10-bit 4:4:4
HEVC with a short closed GOP, so training seeks stay cheap. A source that
already sits on a rung is remuxed as it is: the video bitstream stays, and
the audio is dropped. Smaller rungs are still encoded from that same
source. Nothing is upscaled. The file is <title>.<rung>.mkv. Low variants
come from the `degrade_presets` table and are written unless `--no-degrade`
is set, under the same filename. One master is finished, variants included,
before the next master starts. The missing variants of a master come
from one ffmpeg run: the master is decoded and scaled once, and each
variant encodes its own copy of those frames.

The plain command writes the missing variants of masters already in the
database. With no filenames it also encodes any rung the stored origin
file can still supply, and degrades that master before the next one.
Variant directories live under the low root. `--dry-run` prints each encode
or remux the command would start and writes nothing.

`whoodeo-make-low stats` prints one row per master size. A film is a master
that already has a low file. Pairs count finished files from the enabled
degrade presets, and frames are that master's frame count once.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from whoodeo import db
from whoodeo.library import (
    catalog_file,
    degrade_of,
    empty_catalog,
    master_filename,
    open_catalog,
    plan_masters,
    picture_size,
    same_frame_count,
    save_catalog,
    split_master_filename,
    titles_of,
)
VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
WEBM_CODECS = {"vp8", "vp9", "libvpx", "libvpx-vp9", "av1", "libaom-av1", "libsvtav1"}
MASTER_PIX_FMT = "yuv444p10le"
# No preset: x265 medium. CRF 12 is the quality knob. psy stays at x265's default.
# bframes=0 keeps encode order equal to display order. It does not repair
# timestamps: passthrough copies a duplicate source pts straight through.
X265_PARAMS = (
    "keyint=12:min-keyint=12:open-gop=0:repeat-headers=1:"
    "scenecut=40:bframes=0:aq-mode=1:rc-lookahead=12:"
    "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:"
    "log-level=error"
)
HDR_TRANSFER = {"smpte2084", "arib-std-b67"}
HDR_PRIMARIES = {"bt2020"}
HDR_MATRIX = {"bt2020nc", "bt2020c", "bt2020ncl"}
INTERLACED = {"tt", "bb", "tb", "bt"}
# Preset name → (frame_ratio_num, frame_ratio_den) from degrade_presets.
# A low file of a master with N frames has N * den // num frames, and low
# frame k trains against master frame (k * num + den // 2) // den. Presets
# not listed here are 1/1.
FRAME_RATIOS = {}


def frame_ratio(variant):
    return FRAME_RATIOS.get(variant, (1, 1))


def expected_low_frames(variant, master_frames):
    num, den = frame_ratio(variant)
    return db.expected_low_frames(master_frames, num, den)


def describe_frames(variant, master_frames):
    num, den = frame_ratio(variant)
    expected = db.expected_low_frames(master_frames, num, den)
    if (num, den) == (1, 1):
        return str(expected)
    return f"{expected} (= {master_frames} * {den} // {num})"


# ffprobe matrix name → colorspace filter `all`/`iall` value.
MATRIX_ALL = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "bt470m": "bt470m",
    "smpte240m": "smpte240m",
}


def add_db_option(parser):
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Whoodeo SQLite database (default: $WHOODEO_DB or "
             "~/Library/Application Support/Whoodeo/whoodeo.sqlite)",
    )


def open_store(args):
    """Open the database and resolve orig/, low/, and the data root."""
    conn = db.connect(args.db)
    root = db.dataset_root(conn)
    orig = Path(args.orig).expanduser() if args.orig else root / "orig"
    orig = orig.resolve() if orig.exists() else orig
    low = None
    if hasattr(args, "low"):
        low = Path(args.low).expanduser() if args.low else root / "low"
        if low.exists():
            low = low.resolve()
    return conn, root, orig, low


def load_catalog_mirror(root, create=False, dry_run=False):
    """Load catalog.json as a write-through mirror. A missing file is OK."""
    if catalog_file(root).is_file():
        return open_catalog(root, create=False)
    if create and not dry_run:
        return open_catalog(root, create=True)
    return empty_catalog()


def entry_from_db(conn, title):
    """A catalog-shaped title entry from the database, or None."""
    source = conn.execute(
        "SELECT origin_path, frames FROM sources WHERE name = ?",
        (title,),
    ).fetchone()
    if source is None:
        return None
    entry = {"origin": source["origin_path"], "frames": source["frames"], "masters": {}}
    for master in conn.execute(
        "SELECT m.id, m.rung, m.width, m.height FROM source_masters m "
        "JOIN sources s ON s.id = m.source_id WHERE s.name = ?",
        (title,),
    ):
        entry["masters"][master["rung"]] = {
            "width": master["width"],
            "height": master["height"],
            "low": db.listed_variants(conn, master["id"]),
        }
    return entry


def merge_entry_from_db(entry, conn, title):
    """Fill missing origin/masters/low names from the database."""
    found = entry_from_db(conn, title)
    if found is None:
        return entry
    if not entry.get("origin") and found.get("origin"):
        entry["origin"] = found["origin"]
    if entry.get("frames") is None:
        entry["frames"] = found.get("frames")
    masters = entry.setdefault("masters", {})
    if not isinstance(masters, dict):
        entry["masters"] = found["masters"]
        return entry
    for rung, master in found["masters"].items():
        current = masters.get(rung)
        if not isinstance(current, dict):
            masters[rung] = master
            continue
        if not isinstance(current.get("low"), list):
            current["low"] = list(master["low"])
            continue
        for name in master["low"]:
            if name not in current["low"]:
                current["low"].append(name)
    return entry



def db_remember_variant(conn, title, rung, preset, filename):
    db.remember_variant(conn, title, rung, preset, f"low/{preset}/{filename}")
    conn.commit()


def select_variants(variants, text):
    if text is None:
        selected = list(variants.items())
    else:
        names = [part.strip() for part in text.split(",") if part.strip()]
        if not names:
            raise SystemExit("--variants is empty")
        missing = [name for name in names if name not in variants]
        if missing:
            raise SystemExit("unknown variant: " + ", ".join(missing))
        selected = [(name, variants[name]) for name in names]
    for name, args in selected:
        try:
            split_variant_vf(args)
        except ValueError as exc:
            raise SystemExit(f"variant {name}: {exc}") from exc
    return selected


def codec_of(args):
    tokens = shlex.split(args)
    for flag in ("-c:v", "-codec:v"):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                return tokens[index + 1]
    return None


def split_variant_vf(args):
    """Pull an optional -vf / -filter:v out of a variant argument string.

    Returns (filter or None, remaining encoder tokens). Existing variants
    without a filter are unchanged.
    """
    tokens = shlex.split(args)
    vf = None
    rest = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("-vf", "-filter:v"):
            if index + 1 >= len(tokens):
                raise ValueError(f"{token} is missing its filter")
            if vf is not None:
                raise ValueError("only one -vf is allowed")
            vf = tokens[index + 1]
            if not vf.strip():
                raise ValueError("-vf is empty")
            if vf.count("[") != vf.count("]"):
                raise ValueError("-vf has unbalanced '[' / ']'")
            index += 2
            continue
        rest.append(token)
        index += 1
    return vf, rest


def run_probe(args):
    result = subprocess.run(
        ["ffprobe", "-v", "error", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def size_of(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text or "," not in text:
        raise RuntimeError(f"no video size: {path}")
    width, height = text.split(",", 1)
    return int(width), int(height)


def sample_aspect(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=sample_aspect_ratio",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text or text == "N/A":
        return None
    return text


def square_pixels(sar):
    if sar is None:
        return True
    if ":" not in sar:
        return False
    num, den = sar.split(":", 1)
    try:
        width, height = int(num), int(den)
    except ValueError:
        return False
    return width > 0 and height > 0 and width == height


def packets_of(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no packet count: {path}")
    return int(text)


def has_audio(path):
    text = run_probe([
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ])
    return bool(text)


def encode_variants(src, low_dir, jobs, scale, scale_flags, pix_fmt, force, strict):
    """Write several low variants of one master from a single decode.

    jobs is a list of (variant, ffmpeg_args). The master is decoded and scaled
    once, split, and each copy is encoded to its own file. A variant may
    include -vf, which runs after the shared scale. Return a dict from
    variant to its frame count, -1 when ignored, None on failure.
    """
    name = src.name
    results = {}
    todo = []
    for variant, ffmpeg_args in jobs:
        label = f"{variant}/{name}"
        if src.suffix.lower() not in VIDEO_EXTS:
            if strict:
                print(f"skip {label}: train does not read this extension", file=sys.stderr)
                results[variant] = None
            else:
                results[variant] = -1
            continue
        codec = codec_of(ffmpeg_args)
        if src.suffix.lower() == ".webm" and codec not in WEBM_CODECS:
            print(
                f"skip {label}: webm cannot hold codec {codec or '(unset)'}",
                file=sys.stderr,
            )
            results[variant] = None if strict else -1
            continue
        out = low_dir / variant / name
        if out.exists() and not force:
            try:
                kept = packets_of(out)
            except RuntimeError as exc:
                print(f"{label}: {exc}", file=sys.stderr)
                results[variant] = None
                continue
            # A preset whose frame ratio is not 1/1 may have an older 1/1 file
            # on disk (written before the ratio was set). That file is stale.
            if frame_ratio(variant) != (1, 1):
                try:
                    master_frames = packets_of(src)
                except RuntimeError as exc:
                    print(f"{label}: {exc}", file=sys.stderr)
                    results[variant] = None
                    continue
                if kept != expected_low_frames(variant, master_frames):
                    print(
                        f"stale {label}: {kept} frames, expected "
                        f"{describe_frames(variant, master_frames)}; re-encode",
                        flush=True,
                    )
                    todo.append((variant, ffmpeg_args, out))
                    continue
            print(f"keep {label}", flush=True)
            results[variant] = kept
            continue
        todo.append((variant, ffmpeg_args, out))
    if not todo:
        return results

    def fail_all(message):
        for variant, _, _ in todo:
            print(f"{variant}/{name}: {message}", file=sys.stderr)
            results[variant] = None
        return results

    sar = sample_aspect(src)
    if not square_pixels(sar):
        return fail_all(f"sample aspect ratio {sar}; stored pixels are not square")
    width, height = size_of(src)
    if width % (scale * 2) or height % (scale * 2):
        return fail_all(
            f"master is {width}x{height}. Both sides must be divisible by "
            f"{scale * 2} so the 1/{scale} frame is even."
        )
    low_w = width // scale
    low_h = height // scale

    count = len(todo)
    split_labels = []
    branches = []
    encoder_args = []
    for index, (variant, ffmpeg_args, out) in enumerate(todo):
        try:
            vf, rest = split_variant_vf(ffmpeg_args)
        except ValueError as exc:
            return fail_all(str(exc))
        encoder_args.append(rest)
        if vf:
            scoped = re.sub(r"\[([A-Za-z0-9_]+)\]", rf"[b{index}_\1]", vf)
            split_labels.append(f"[s{index}]")
            branches.append(f"[s{index}]{scoped}[v{index}]")
        else:
            split_labels.append(f"[v{index}]")
    graph = (
        f"[0:v:0]scale={low_w}:{low_h}:flags={scale_flags},"
        f"split={count}{''.join(split_labels)}"
    )
    if branches:
        graph += ";" + ";".join(branches)
    outputs = []
    temps = []
    for index, ((variant, ffmpeg_args, out), rest) in enumerate(zip(todo, encoder_args)):
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
        tmp.unlink(missing_ok=True)
        temps.append(tmp)
        print(f"encode {variant}/{name}  {width}x{height} -> {low_w}x{low_h}  {ffmpeg_args}", flush=True)
        outputs += [
            "-map", f"[v{index}]", "-an",
            *rest,
            "-pix_fmt", pix_fmt,
            "-fps_mode", "passthrough",
            str(tmp),
        ]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-filter_complex", graph,
        *outputs,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
        return fail_all("ffmpeg failed")

    try:
        in_packets = packets_of(src)
    except RuntimeError as exc:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
        return fail_all(str(exc))
    for (variant, _, out), tmp in zip(todo, temps):
        label = f"{variant}/{name}"
        try:
            out_size = size_of(tmp)
            out_packets = packets_of(tmp)
            audio = has_audio(tmp)
        except RuntimeError as exc:
            tmp.unlink(missing_ok=True)
            print(f"{label}: {exc}", file=sys.stderr)
            results[variant] = None
            continue
        problem = None
        if out_size != (low_w, low_h):
            problem = f"output is {out_size[0]}x{out_size[1]}, expected {low_w}x{low_h}"
        elif out_packets != expected_low_frames(variant, in_packets):
            problem = (
                f"{in_packets} input frames, {out_packets} output frames, "
                f"expected {describe_frames(variant, in_packets)}"
            )
        elif audio:
            problem = "output still has audio"
        if problem:
            tmp.unlink(missing_ok=True)
            print(f"{label}: {problem}", file=sys.stderr)
            results[variant] = None
            continue
        tmp.replace(out)
        print(f"wrote {out}  {low_w}x{low_h}  {out_packets} frames", flush=True)
        results[variant] = out_packets
    return results


def probe_video(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,sample_aspect_ratio,"
        "field_order,color_space,color_transfer,color_primaries,color_range,"
        "has_b_frames",
        "-of", "json",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no video stream: {path}")
    payload = json.loads(text)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream: {path}")
    return streams[0]


def stream_text(stream, key):
    value = stream.get(key)
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"N/A", "unknown", "unspecified", "none"}:
        return ""
    return text


def sar_factors(sar):
    if not sar:
        return 1, 1
    if ":" not in sar:
        return None
    num, den = sar.split(":", 1)
    try:
        num, den = int(num), int(den)
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num, den


def input_matrix(color_space):
    if not color_space:
        return "bt709"
    return MATRIX_ALL.get(color_space)


def master_filter(width, height, matrix, color_range):
    irange = "pc" if color_range in {"pc", "jpeg"} else "tv"
    fast = "1" if matrix == "bt709" else "0"
    # setpts numbers frames in display order. A source can name two pictures
    # with one timestamp; passthrough would keep that, and mpeg4 then refuses it.
    return (
        "setpts=N/(FRAME_RATE*TB),"
        f"colorspace=iall={matrix}:all=bt709:irange={irange}:range=tv:"
        f"format=yuv444p12:dither=none:fast={fast},"
        f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp:"
        "sws_dither=none,setsar=1,format=yuv444p10le,"
        "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709"
    )


def _ratio(text):
    if not text or "/" not in text:
        return None
    num, den = text.split("/", 1)
    try:
        num, den = int(num), int(den)
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num, den


def video_timing(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,time_base",
        "-of", "json",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no video timing: {path}")
    streams = json.loads(text).get("streams") or []
    if not streams:
        raise RuntimeError(f"no video timing: {path}")
    rate = _ratio(str(streams[0].get("r_frame_rate") or ""))
    base = _ratio(str(streams[0].get("time_base") or ""))
    if rate is None or base is None:
        raise RuntimeError(f"no frame rate: {path}")
    return rate, base


def packet_pts(path):
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "packet=pts",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no timestamps: {path}")
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "N/A":
            raise RuntimeError(f"missing timestamp: {path}")
        pts.append(int(line))
    if not pts:
        raise RuntimeError(f"no timestamps: {path}")
    return pts


def packet_times(path):
    """Presentation and decode timestamps for each video packet, in file order."""
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "packet=pts,dts",
        "-of", "csv=p=0",
        str(path),
    ])
    if not text:
        raise RuntimeError(f"no timestamps: {path}")
    times = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2 or parts[0] in {"", "N/A"}:
            raise RuntimeError(f"missing timestamp: {path}")
        # Matroska does not store the leading negative decode timestamps.
        # Those packets come back without a DTS; a later hole is a broken file.
        dts = None if parts[1] in {"", "N/A"} else int(parts[1])
        times.append((int(parts[0]), dts))
    if not times:
        raise RuntimeError(f"no timestamps: {path}")
    return times


def frame_tick(pts, rate, base):
    """Nearest frame index at this rate. base is the timestamp timebase."""
    rate_num, rate_den = rate
    tb_num, tb_den = base
    return (pts * tb_num * rate_num + (tb_den * rate_den) // 2) // (tb_den * rate_den)


def timestamp_problem(path):
    """None when each next packet is a later frame tick.

    mpeg4 turns the master timestamps into one tick per frame. Two packets on
    the same tick fail the encode. Packet order is display order for these
    masters because the import writes no B-frames.
    """
    rate, base = video_timing(path)
    pts = packet_pts(path)
    if any(b < a for a, b in zip(pts, pts[1:])):
        return "timestamps go backwards"
    last = None
    for value in pts:
        tick = frame_tick(value, rate, base)
        if last is not None and tick <= last:
            return "two frames share one timestamp"
        last = tick
    return None


def restamp_timestamps(src, dst):
    """Stream-copy src to dst on a constant frame-rate grid. Return an error or None.

    setts numbers packets in file order. That matches display order while
    timestamps never run backwards, which is true for a master without B-frames.
    """
    rate, base = video_timing(src)
    pts = packet_pts(src)
    if any(b < a for a, b in zip(pts, pts[1:])):
        return "timestamps go backwards"
    rate_num, rate_den = rate
    tb_num, tb_den = base
    expr = f"floor(N*{rate_den}*{tb_den}/({rate_num}*{tb_num})+0.5)"
    dst.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-i", str(src),
            "-map", "0:v:0", "-an", "-c", "copy",
            "-bsf:v", f"setts=pts={expr}:dts={expr}",
            str(dst),
        ],
        check=False,
    )
    if result.returncode != 0:
        dst.unlink(missing_ok=True)
        return "restamp failed"
    if packets_of(dst) != len(pts):
        dst.unlink(missing_ok=True)
        return "restamp changed the frame count"
    problem = timestamp_problem(dst)
    if problem:
        dst.unlink(missing_ok=True)
        return problem
    return None


def time_order_problem(times, rate, base, delay):
    """None when display ticks advance and packet DTS strictly increases.

    PTS is checked in presentation order. A source with B-frames stores a
    later picture before an earlier one, so packet PTS may step backwards
    while the display time still advances. Matroska leaves the leading
    decode timestamps empty, at most `delay` of them (the reorder delay).
    """
    last = None
    for value in sorted(item[0] for item in times):
        tick = frame_tick(value, rate, base)
        if last is not None and tick <= last:
            return "two frames share one timestamp"
        last = tick
    seen = False
    leading = 0
    prev = None
    for _pts, dts in times:
        if dts is None:
            if seen or leading >= delay:
                return "missing timestamp"
            leading += 1
            continue
        seen = True
        if prev is not None and dts <= prev:
            return "dts does not increase"
        prev = dts
    return None


def b_frame_delay(path):
    """Reorder delay. Unknown or unreadable counts as B-frames, so no restamp."""
    try:
        delay = int(probe_video(path).get("has_b_frames") or 0)
    except (TypeError, ValueError):
        return 1
    if delay < 0:
        return 1
    return delay


def settle_remux_time(path, label):
    """Accept path, or restamp it when packet order is display order.

    setts numbers packets in file order. That is wrong once B-frames have
    stored a later picture first, and it is wrong when PTS already runs
    backwards. Those files are left untouched and rejected.
    """
    rate, base = video_timing(path)
    times = packet_times(path)
    delay = b_frame_delay(path)
    err = time_order_problem(times, rate, base, delay)
    if err is None:
        return None
    pts = [item[0] for item in times]
    backwards = any(b < a for a, b in zip(pts, pts[1:]))
    # setts follows file order. Only a colliding display clock with no
    # B-frames and no backwards PTS can be renumbered that way.
    if err != "two frames share one timestamp" or delay != 0 or backwards:
        return err
    print(f"{label}: {err}; restamping timestamps", flush=True)
    stamped = path.with_name(f"{path.stem}.stamped{path.suffix}")
    restamp_error = restamp_timestamps(path, stamped)
    if restamp_error:
        stamped.unlink(missing_ok=True)
        return restamp_error
    stamped.replace(path)
    rate, base = video_timing(path)
    return time_order_problem(packet_times(path), rate, base, b_frame_delay(path))


def remux_problem(path, source, packets):
    """None when the remux kept the picture and dropped the audio."""
    stream = probe_video(path)
    if stream.get("codec_name") != source.get("codec_name"):
        return f"codec is {stream.get('codec_name')}"
    width = int(source.get("width") or 0)
    height = int(source.get("height") or 0)
    if int(stream.get("width") or 0) != width or int(stream.get("height") or 0) != height:
        return (
            f"output is {stream.get('width')}x{stream.get('height')}, "
            f"expected {width}x{height}"
        )
    pix = source.get("pix_fmt")
    if stream.get("pix_fmt") != pix:
        return f"pixel format is {stream.get('pix_fmt')}, expected {pix}"
    out_packets = packets_of(path)
    if out_packets != packets:
        return f"{packets} input frames, {out_packets} output frames"
    if has_audio(path):
        return "output still has audio"
    return None


def master_problem(path, width, height, packets):
    stream = probe_video(path)
    if stream.get("codec_name") != "hevc":
        return f"codec is {stream.get('codec_name')}"
    if int(stream.get("width") or 0) != width or int(stream.get("height") or 0) != height:
        return f"output is {stream.get('width')}x{stream.get('height')}, expected {width}x{height}"
    if stream.get("pix_fmt") != MASTER_PIX_FMT:
        return f"pixel format is {stream.get('pix_fmt')}, expected {MASTER_PIX_FMT}"
    sar = stream_text(stream, "sample_aspect_ratio") or None
    if not square_pixels(sar):
        return f"sample aspect ratio {sar}"
    tags = {
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "color_range": "tv",
    }
    for key, want in tags.items():
        got = stream_text(stream, key)
        if got != want:
            return f"{key} is {got or 'unset'}, expected {want}"
    out_packets = packets_of(path)
    if out_packets != packets:
        return f"{packets} input frames, {out_packets} output frames"
    if has_audio(path):
        return "output still has audio"
    return timestamp_problem(path)


def encode_master(src, label, out, stream, in_packets, width, height, matrix_name, out_w, out_h):
    """Encode src to a 10-bit 4:4:4 HEVC master at out_w by out_h."""
    matrix = input_matrix(matrix_name)
    if matrix is None:
        print(f"{label}: unsupported color matrix {matrix_name}", file=sys.stderr)
        return None
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    print(
        f"import {label}  {width}x{height} -> {out_w}x{out_h}  {MASTER_PIX_FMT} crf 12",
        flush=True,
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-map", "0:v:0", "-an",
        "-vf", master_filter(out_w, out_h, matrix, stream_text(stream, "color_range")),
        "-c:v", "libx265",
        "-profile:v", "main444-10",
        "-pix_fmt", MASTER_PIX_FMT,
        "-crf", "12",
        "-fps_mode", "passthrough",
        "-x265-params", X265_PARAMS,
        str(tmp),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"{label}: ffmpeg failed", file=sys.stderr)
        return None
    try:
        problem = timestamp_problem(tmp)
        if problem:
            print(f"{label}: {problem}; restamping timestamps", flush=True)
            stamped = tmp.with_name(f"{tmp.stem}.stamped{tmp.suffix}")
            restamp_error = restamp_timestamps(tmp, stamped)
            if restamp_error:
                tmp.unlink(missing_ok=True)
                stamped.unlink(missing_ok=True)
                print(f"{label}: {restamp_error}", file=sys.stderr)
                return None
            stamped.replace(tmp)
        problem = master_problem(tmp, out_w, out_h, in_packets)
    except (RuntimeError, json.JSONDecodeError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    if problem:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {problem}", file=sys.stderr)
        return None
    tmp.replace(out)
    print(f"wrote {out}  {out_w}x{out_h}  {in_packets} frames", flush=True)
    return out.name


def remux_master(src, label, out, stream, in_packets, width, height):
    """Copy the video bitstream into orig and drop the audio."""
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    print(f"import {label}  {width}x{height}  remux", flush=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-map", "0:v:0", "-an", "-c", "copy",
        str(tmp),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"{label}: ffmpeg failed", file=sys.stderr)
        return None
    try:
        problem = remux_problem(tmp, stream, in_packets)
        if problem is None:
            problem = settle_remux_time(tmp, label)
            if problem is None:
                problem = remux_problem(tmp, stream, in_packets)
    except (RuntimeError, json.JSONDecodeError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    if problem:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {problem}", file=sys.stderr)
        return None
    tmp.replace(out)
    print(f"wrote {out}  {width}x{height}  {in_packets} frames", flush=True)
    return out.name


def display_size(width, height, sar):
    """Square-pixel display size. None when the sample aspect ratio is unreadable."""
    factors = sar_factors(sar)
    if factors is None:
        return None
    num, den = factors
    return width * num / den, float(height)


def inspect_source(src):
    """Probe a source header. Return a dict, or None when the file cannot be a master.

    The packet count stays unset. Counting reads every packet, so it waits
    until a missing master is written.
    """
    label = src.name
    try:
        stream = probe_video(src)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width < 2 or height < 2:
        print(f"{label}: no video size", file=sys.stderr)
        return None
    field_order = stream_text(stream, "field_order")
    if field_order in INTERLACED:
        print(f"{label}: interlaced ({field_order})", file=sys.stderr)
        return None
    transfer = stream_text(stream, "color_transfer")
    primaries = stream_text(stream, "color_primaries")
    matrix_name = stream_text(stream, "color_space")
    if transfer in HDR_TRANSFER or primaries in HDR_PRIMARIES or matrix_name in HDR_MATRIX:
        print(f"{label}: HDR or wide-gamut source", file=sys.stderr)
        return None
    sar = stream_text(stream, "sample_aspect_ratio")
    size = display_size(width, height, sar)
    if size is None:
        print(f"{label}: bad sample aspect ratio {sar or 'unset'}", file=sys.stderr)
        return None
    return {
        "stream": stream,
        "width": width,
        "height": height,
        "matrix": matrix_name,
        "display": size,
    }


def source_packets(src, probed):
    """Packet count of this origin. The first missing master pays for the read."""
    cached = probed.get("packets")
    if cached is None:
        cached = packets_of(src)
        probed["packets"] = cached
    return cached


def title_entry(catalog, root, title):
    titles = titles_of(catalog, catalog_file(root))
    entry = titles.get(title)
    if not isinstance(entry, dict):
        entry = {"origin": None, "masters": {}}
        titles[title] = entry
    if "origin" not in entry:
        entry["origin"] = None
    if not isinstance(entry.get("masters"), dict):
        entry["masters"] = {}
    return entry


def remember_master(catalog, root, title, rung, path, frames, force, conn=None):
    entry = title_entry(catalog, root, title)
    width, height = picture_size(path)
    same_frame_count(entry, title, frames)
    previous = entry["masters"].get(rung)
    low = []
    if not force and isinstance(previous, dict) and isinstance(previous.get("low"), list):
        low = list(previous["low"])
    entry["masters"][rung] = {"width": width, "height": height, "low": low}
    save_catalog(root, catalog)
    if conn is not None:
        db.remember_master(
            conn, title, rung, f"orig/{path.name}", width, height, frames=frames,
        )
        conn.commit()


def produce_planned(src, title, planned, probed, orig_dir, force, catalog, root, dry_run=False, conn=None):
    """Write the planned masters that are missing. Return (filenames, failed).

    `dry_run` prints the master line and writes nothing. A file that is
    already present is reported as ready either way.
    """
    failed = False
    names = []
    if dry_run:
        for rung, out_w, out_h, kind in planned:
            name = master_filename(title, rung)
            out = orig_dir / name
            if out.exists() and not force:
                names.append(name)
                continue
            if kind == "encode":
                if input_matrix(probed["matrix"]) is None:
                    print(
                        f"{name}: unsupported color matrix {probed['matrix']}",
                        file=sys.stderr,
                    )
                    failed = True
                    continue
                print(
                    f"import {name}  {probed['width']}x{probed['height']} -> "
                    f"{out_w}x{out_h}  {MASTER_PIX_FMT} crf 12",
                    flush=True,
                )
            else:
                print(
                    f"import {name}  {probed['width']}x{probed['height']}  remux",
                    flush=True,
                )
            names.append(name)
        return names, failed
    entry = title_entry(catalog, root, title)
    for rung, out_w, out_h, kind in planned:
        name = master_filename(title, rung)
        out = orig_dir / name
        if out.exists() and not force:
            print(f"keep {name}", flush=True)
            if rung not in entry["masters"]:
                try:
                    frames = packets_of(out)
                except RuntimeError as exc:
                    print(f"{name}: {exc}", file=sys.stderr)
                    failed = True
                    continue
                try:
                    remember_master(catalog, root, title, rung, out, frames, False, conn=conn)
                except RuntimeError as exc:
                    print(f"{name}: {exc}", file=sys.stderr)
                    failed = True
                    continue
            names.append(name)
            continue
        try:
            frames = source_packets(src, probed)
            same_frame_count(entry, title, frames)
        except RuntimeError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            failed = True
            continue
        if kind == "encode":
            wrote = encode_master(
                src, name, out, probed["stream"], frames,
                probed["width"], probed["height"], probed["matrix"], out_w, out_h,
            )
        else:
            wrote = remux_master(
                src, name, out, probed["stream"], frames,
                probed["width"], probed["height"],
            )
        if wrote is None:
            failed = True
            continue
        try:
            remember_master(catalog, root, title, rung, out, frames, force, conn=conn)
        except RuntimeError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            failed = True
            continue
        names.append(name)
    return names, failed


def masters_to_write(orig_dir, title, planned, force):
    """Planned rungs whose master file is missing, or every rung when forcing."""
    if force:
        return [item[0] for item in planned]
    return [
        item[0] for item in planned
        if not (orig_dir / master_filename(title, item[0])).exists()
    ]


def prepare_import(src, orig_dir, rungs, catalog, root, dry_run=False, force=False, conn=None):
    """Record src as the title's origin. Return (title, planned, probed), or None.

    `dry_run` leaves the catalog untouched. The source packet count runs only
    when a master will be written, and a frame mismatch is refused then.
    """
    label = src.name
    if src.suffix.lower() not in VIDEO_EXTS:
        print(f"skip {label}: train does not read this extension", file=sys.stderr)
        return None
    if not src.is_file():
        print(f"no video: {src}", file=sys.stderr)
        return None
    if src.parent == orig_dir:
        print(f"{src} is already in {orig_dir}", file=sys.stderr)
        return None
    probed = inspect_source(src)
    if probed is None:
        return None
    planned = plan_masters(*probed["display"], rungs)
    if not planned:
        print(f"{label}: no master rung", file=sys.stderr)
        return None
    title = src.stem
    if dry_run:
        return title, planned, probed
    entry = title_entry(catalog, root, title)
    if masters_to_write(orig_dir, title, planned, force):
        try:
            frames = source_packets(src, probed)
            same_frame_count(entry, title, frames)
        except RuntimeError as exc:
            if not entry.get("masters") and entry.get("origin") is None:
                titles_of(catalog, catalog_file(root)).pop(title, None)
            print(f"{label}: {exc}", file=sys.stderr)
            return None
    entry["origin"] = str(src.resolve())
    save_catalog(root, catalog)
    if conn is not None:
        db.remember_source(conn, title, origin_path=entry["origin"], frames=entry.get("frames"))
        conn.commit()
    return title, planned, probed


def planned_geometry(item, probed):
    """Size a master would have before its file exists.

    An encode is square pixels at the planned size. A remux keeps the source.
    """
    _rung, out_w, out_h, kind = item
    if kind == "encode":
        return (out_w, out_h)
    return (
        probed["width"],
        probed["height"],
        stream_text(probed["stream"], "sample_aspect_ratio") or None,
    )


def list_low_encodes(src, low_dir, selected, scale, force, strict, geometry):
    """Print each low encode that would be started. Write nothing.

    `geometry` is None to measure an existing master, or the size a missing
    master would have. Return False when an encode would be refused.
    """
    failed = False
    measured = None
    for variant, ffmpeg_args in selected:
        name = src.name
        label = f"{variant}/{name}"
        if src.suffix.lower() not in VIDEO_EXTS:
            if strict:
                print(f"skip {label}: train does not read this extension", file=sys.stderr)
                failed = True
            continue
        codec = codec_of(ffmpeg_args)
        if src.suffix.lower() == ".webm" and codec not in WEBM_CODECS:
            print(
                f"skip {label}: webm cannot hold codec {codec or '(unset)'}",
                file=sys.stderr,
            )
            if strict:
                failed = True
            continue
        low_path = low_dir / variant / name
        if low_path.is_file() and not force:
            continue
        if measured is None:
            if geometry is None:
                width, height = size_of(src)
                sar = sample_aspect(src)
            else:
                width, height = geometry[0], geometry[1]
                sar = geometry[2] if len(geometry) > 2 else None
            measured = (width, height, sar)
        width, height, sar = measured
        if not square_pixels(sar):
            print(
                f"{label} has sample aspect ratio {sar}; stored pixels are not square",
                file=sys.stderr,
            )
            failed = True
            continue
        if width % (scale * 2) or height % (scale * 2):
            print(
                f"{label} is {width}x{height}. Both sides must be divisible by "
                f"{scale * 2} so the 1/{scale} frame is even.",
                file=sys.stderr,
            )
            failed = True
            continue
        low_w = width // scale
        low_h = height // scale
        print(
            f"encode {label}  {width}x{height} -> {low_w}x{low_h}  {ffmpeg_args}",
            flush=True,
        )
    return not failed


def degrade_master(
    src, low_dir, selected, scale, flags, pix_fmt, force, strict, catalog, root,
    dry_run=False, geometry=None, unlisted=False, conn=None,
):
    """Write the selected lows of one master and record them. Return False on failure.

    `dry_run` only prints the encode lines. `geometry` is the size of a master
    that is not on disk yet. `unlisted` allows an on-disk file the catalog
    does not name yet; a real run would record it before degrading.
    """
    parts = split_master_filename(src.name)
    if parts is None:
        print(f"{src.name}: expected <title>.<rung>.mkv", file=sys.stderr)
        return False
    title, rung = parts
    entry = titles_of(catalog, catalog_file(root)).get(title)
    masters = entry.get("masters") if isinstance(entry, dict) else None
    master = masters.get(rung) if isinstance(masters, dict) else None
    if not isinstance(master, dict):
        if not (dry_run and (geometry is not None or unlisted)):
            print(f"{src.name}: not in the catalog", file=sys.stderr)
            return False
        master = {"low": []}
    if dry_run:
        return list_low_encodes(
            src, low_dir, selected, scale, force, strict, geometry,
        )
    if not isinstance(master.get("low"), list):
        print(f"{src.name}: low variants must be a list", file=sys.stderr)
        return False
    failed = False
    jobs = []
    for variant, ffmpeg_args in selected:
        low_path = low_dir / variant / src.name
        if low_path.is_file() and not force and variant in master["low"]:
            print(f"keep {variant}/{src.name}", flush=True)
            if conn is not None:
                try:
                    db_remember_variant(conn, title, rung, variant, src.name)
                except RuntimeError as exc:
                    print(f"{variant}/{src.name}: {exc}", file=sys.stderr)
            continue
        jobs.append((variant, ffmpeg_args))
    encoded = encode_variants(
        src, low_dir, jobs, scale, flags, pix_fmt, force, strict,
    ) if jobs else {}
    for variant, _ in jobs:
        low_path = low_dir / variant / src.name
        frames = encoded.get(variant)
        if frames is None:
            failed = True
            continue
        if frames < 0:
            continue
        master_frames = entry.get("frames")
        if not isinstance(master_frames, int) or frames != expected_low_frames(variant, master_frames):
            print(
                f"{variant}/{src.name}: {frames} frames, catalog says "
                f"{master_frames} master frames, expected "
                f"{describe_frames(variant, master_frames) if isinstance(master_frames, int) else '?'}",
                file=sys.stderr,
            )
            failed = True
            continue
        try:
            low_w, low_h = picture_size(low_path)
        except RuntimeError as exc:
            print(f"{variant}/{src.name}: {exc}", file=sys.stderr)
            failed = True
            continue
        if master.get("width") != low_w * scale or master.get("height") != low_h * scale:
            print(
                f"{variant}/{src.name} is {low_w}x{low_h}, "
                f"orig is {master.get('width')}x{master.get('height')}, "
                f"expected exactly {scale}x",
                file=sys.stderr,
            )
            failed = True
            continue
        if variant not in master["low"]:
            master["low"].append(variant)
        save_catalog(root, catalog)
        if conn is not None:
            try:
                db_remember_variant(conn, title, rung, variant, src.name)
            except RuntimeError as exc:
                print(f"{variant}/{src.name}: {exc}", file=sys.stderr)
                failed = True
    return not failed


def finish_title(
    title, orig_dir, rungs, low_dir, selected,
    scale, flags, pix_fmt, force, strict, catalog, root, dry_run=False,
    conn=None,
):
    """Degrade one master before the next one is encoded.

    Masters already on disk come first, largest planned rung first. A missing
    rung is encoded from origin and degraded before the next missing rung.
    A missing origin leaves the written masters in place. `dry_run` lists
    those encodes and writes nothing. Return False when a step fails.
    """
    titles = titles_of(catalog, catalog_file(root))
    entry = titles.get(title)
    if not isinstance(entry, dict):
        entry = entry_from_db(conn, title) if conn is not None else None
        if not isinstance(entry, dict):
            return True
        titles[title] = entry
    elif conn is not None:
        merge_entry_from_db(entry, conn, title)
    failed = False
    planned = []
    src = None
    probed = None
    origin = entry.get("origin")
    if origin:
        src = Path(origin)
        if not src.is_file():
            print(f"{title}: origin is gone, masters stay", file=sys.stderr)
        else:
            probed = inspect_source(src)
            if probed is None:
                failed = True
            else:
                planned = plan_masters(*probed["display"], rungs)

    def degrade_path(path, geometry=None, unlisted=False):
        nonlocal failed
        ok = degrade_master(
            path, low_dir, selected, scale, flags, pix_fmt,
            force, strict, catalog, root,
            dry_run=dry_run, geometry=geometry, unlisted=unlisted, conn=conn,
        )
        failed = failed or not ok

    def masters_of():
        masters = entry.get("masters")
        if isinstance(masters, dict):
            return masters
        return {}

    seen = set()
    planned_rungs = {item[0] for item in planned}
    for item in planned:
        rung = item[0]
        path = orig_dir / master_filename(title, rung)
        if not path.is_file():
            continue
        unlisted = rung not in masters_of()
        if unlisted:
            names, one_failed = produce_planned(
                src, title, [item], probed, orig_dir, False, catalog, root,
                dry_run=dry_run, conn=conn,
            )
            failed = failed or one_failed
            if one_failed or path.name not in names:
                continue
        seen.add(rung)
        degrade_path(path, unlisted=dry_run and unlisted)
    for rung in list(masters_of()):
        if rung in seen or rung in planned_rungs:
            continue
        path = orig_dir / master_filename(title, rung)
        if not path.is_file():
            print(f"skip {path.name}: file missing", file=sys.stderr)
            failed = True
            continue
        seen.add(rung)
        degrade_path(path)
    for item in planned:
        rung = item[0]
        if rung in seen:
            continue
        path = orig_dir / master_filename(title, rung)
        names, one_failed = produce_planned(
            src, title, [item], probed, orig_dir, False, catalog, root,
            dry_run=dry_run, conn=conn,
        )
        failed = failed or one_failed
        ready = path.name in names and (path.is_file() or dry_run)
        if one_failed or not ready:
            if not path.is_file():
                print(f"skip {path.name}: file missing", file=sys.stderr)
                failed = True
            continue
        seen.add(rung)
        geometry = planned_geometry(item, probed) if dry_run and not path.is_file() else None
        degrade_path(path, geometry=geometry)
    return not failed


def _listed_lows(master):
    """Variant names recorded on this master, in catalog order."""
    lows = master.get("low") if isinstance(master, dict) else None
    if not isinstance(lows, list):
        return None
    names = []
    for name in lows:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or name in names
        ):
            continue
        names.append(name)
    return names


def library_stats(catalog, where, orig_dir, low_dir):
    """Ready films grouped by master size, plus masters that are not ready.

    A film is one catalog master whose file exists and that has at least one
    low file. `pairs` counts files named in the degrade list. Frames are the
    title's frame count, added once per film. Masters with no low file, and
    masters still missing a degrade variant, are listed after the table.
    """
    scale, _, _, variants = degrade_of(catalog, where)
    titles = titles_of(catalog, where)
    recipe = list(variants)
    groups = {}
    gaps = []
    for title, entry in titles.items():
        if not isinstance(entry, dict):
            gaps.append(("skip", title, "bad catalog entry"))
            continue
        frames = entry.get("frames")
        masters = entry.get("masters")
        if not isinstance(masters, dict):
            gaps.append(("skip", title, "no masters"))
            continue
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            gaps.append(("skip", title, "missing frames"))
            continue
        for rung, master in masters.items():
            name = master_filename(title, rung)
            if not isinstance(master, dict):
                gaps.append(("skip", name, "bad catalog entry"))
                continue
            width = master.get("width")
            height = master.get("height")
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                gaps.append(("skip", name, "missing width"))
                continue
            if isinstance(height, bool) or not isinstance(height, int) or height < 1:
                gaps.append(("skip", name, "missing height"))
                continue
            listed = _listed_lows(master)
            if listed is None:
                gaps.append(("skip", name, "no lows"))
                continue
            if not (orig_dir / name).is_file():
                gaps.append(("no master", name, None))
                continue
            present = [variant for variant in recipe if (low_dir / variant / name).is_file()]
            other = [
                variant for variant in listed
                if variant not in variants and (low_dir / variant / name).is_file()
            ]
            if not present and not other:
                gaps.append(("no low", name, None))
                continue
            bucket = groups.setdefault((width, height), {"films": 0, "pairs": 0, "frames": 0})
            bucket["films"] += 1
            bucket["pairs"] += len(present)
            bucket["frames"] += frames
            missing = [variant for variant in recipe if variant not in present]
            if missing:
                gaps.append(("incomplete", name, (len(present), len(recipe), missing)))
    rows = []
    for (width, height), bucket in groups.items():
        rows.append({
            "width": width,
            "height": height,
            "low_w": width // scale,
            "low_h": height // scale,
            "films": bucket["films"],
            "pairs": bucket["pairs"],
            "frames": bucket["frames"],
        })
    rows.sort(key=lambda row: (row["width"] * row["height"], row["width"]), reverse=True)
    return recipe, rows, gaps


def library_stats_from_db(conn, orig_dir, low_dir):
    """Ready films grouped by master size, plus masters that are not ready."""
    scale, _, _, variants, ratios = db.load_degrade_ratios(conn, enabled_only=True)
    FRAME_RATIOS.update(ratios)
    recipe = list(variants)
    groups = {}
    gaps = []
    for row in db.list_masters(conn):
        if not row["source_enabled"]:
            continue
        name = master_filename(row["name"], row["rung"])
        frames = row["frames"]
        width = row["width"]
        height = row["height"]
        if not isinstance(frames, int) or frames < 1:
            gaps.append(("skip", name, "missing frames"))
            continue
        if not isinstance(width, int) or width < 1 or not isinstance(height, int) or height < 1:
            gaps.append(("skip", name, "missing size"))
            continue
        listed = db.listed_variants(conn, row["master_id"])
        if not (orig_dir / name).is_file():
            gaps.append(("no master", name, None))
            continue
        present = [variant for variant in recipe if (low_dir / variant / name).is_file()]
        other = [
            variant for variant in listed
            if variant not in variants and (low_dir / variant / name).is_file()
        ]
        if not present and not other:
            gaps.append(("no low", name, None))
            continue
        bucket = groups.setdefault((width, height), {"films": 0, "pairs": 0, "frames": 0})
        bucket["films"] += 1
        bucket["pairs"] += len(present)
        bucket["frames"] += frames
        missing = [variant for variant in recipe if variant not in present]
        if missing:
            gaps.append(("incomplete", name, (len(present), len(recipe), missing)))
    rows = []
    for (width, height), bucket in groups.items():
        rows.append({
            "width": width,
            "height": height,
            "low_w": width // scale,
            "low_h": height // scale,
            "films": bucket["films"],
            "pairs": bucket["pairs"],
            "frames": bucket["frames"],
        })
    rows.sort(key=lambda row: (row["width"] * row["height"], row["width"]), reverse=True)
    return recipe, rows, gaps


def format_stats(recipe, rows, gaps):
    """The size table, then any master that is not fully degraded."""
    lines = ["degrade  " + " ".join(recipe)]
    headers = ("master", "low", "films", "pairs", "frames")
    body = [
        (
            f"{row['width']}x{row['height']}",
            f"{row['low_w']}x{row['low_h']}",
            str(row["films"]),
            str(row["pairs"]),
            f"{row['frames']:,}",
        )
        for row in rows
    ]
    widths = [len(cell) for cell in headers]
    for cells in body:
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))
    numeric = {2, 3, 4}

    def emit(cells):
        parts = []
        for index, cell in enumerate(cells):
            if index in numeric:
                parts.append(cell.rjust(widths[index]))
            else:
                parts.append(cell.ljust(widths[index]))
        lines.append("  ".join(parts).rstrip())

    emit(headers)
    for cells in body:
        emit(cells)
    for kind, name, detail in gaps:
        if kind == "incomplete":
            have, total, missing = detail
            lines.append(
                f"incomplete {name}: {have}/{total}, missing {', '.join(missing)}"
            )
        elif kind == "skip":
            lines.append(f"skip {name}: {detail}")
        else:
            lines.append(f"{kind} {name}")
    return "\n".join(lines) + "\n"


def sources_from_args(orig_dir, names):
    if not names:
        found = sorted(
            path for path in orig_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        )
        if not found:
            raise SystemExit(f"no videos in {orig_dir}")
        return found, False
    sources = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = orig_dir / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"no video: {path}")
        if path.parent != orig_dir:
            raise SystemExit(f"{path} is not in {orig_dir}")
        sources.append(path)
    return sources, True


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="encode half-resolution variants of catalog masters, "
        "finishing one master before the next. "
        "`whoodeo-make-low import` writes a source into orig first. "
        "`whoodeo-make-low stats` prints masters grouped by size",
    )
    add_db_option(parser)
    parser.add_argument("--variants", help="comma-separated names from the database degrade presets; default is all enabled presets")
    parser.add_argument("-o", "--orig", type=Path, default=None, help="master videos (default: dataset_root/orig from the database)")
    parser.add_argument("-l", "--low", type=Path, default=None, help="degraded variants (default: dataset_root/low from the database)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing low video")
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="list each encode or remux that would be written, and write nothing",
    )
    parser.add_argument(
        "videos", nargs="*",
        help="filenames in orig; default also encodes each missing rung from origin and degrades that master before the next",
    )
    return parser.parse_args(argv)


def parse_stats_args(argv):
    parser = argparse.ArgumentParser(
        prog="whoodeo-make-low stats",
        description="print catalog masters grouped by size. "
        "A film has at least one low file. Pairs are finished files from "
        "the enabled degrade presets. Frames are counted once per film.",
    )
    add_db_option(parser)
    parser.add_argument("-o", "--orig", type=Path, default=None, help="master videos (default: dataset_root/orig from the database)")
    parser.add_argument("-l", "--low", type=Path, default=None, help="degraded variants (default: dataset_root/low from the database)")
    return parser.parse_args(argv)


def stats_main(argv):
    args = parse_stats_args(argv)
    conn, root, orig_dir, low_dir = open_store(args)
    try:
        if not orig_dir.is_dir():
            raise SystemExit(f"no orig folder: {orig_dir}")
        if low_dir is None or not low_dir.is_dir():
            raise SystemExit(f"no low folder: {low_dir}")
        text = format_stats(*library_stats_from_db(conn, orig_dir, low_dir))
        sys.stdout.write(text)
    finally:
        conn.close()


def parse_import_args(argv):
    parser = argparse.ArgumentParser(
        prog="whoodeo-make-low import",
        description="write sources into orig, one file per catalog rung. "
        "A source larger than a rung is encoded down to it. A source that "
        "already sits on a rung is remuxed. Each master's low variants are "
        "written before the next master, unless --no-degrade",
    )
    add_db_option(parser)
    parser.add_argument("-o", "--orig", type=Path, default=None, help="master videos (default: dataset_root/orig from the database)")
    parser.add_argument("-l", "--low", type=Path, default=None, help="low variants, written unless --no-degrade (default: dataset_root/low from the database)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing master")
    parser.add_argument("--no-degrade", action="store_true", help="do not write the low variants of each new master")
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="list each encode or remux that would be written, and write nothing",
    )
    parser.add_argument("videos", nargs="+", help="source videos to write into orig")
    return parser.parse_args(argv)


def import_main(argv):
    args = parse_import_args(argv)
    conn, root, orig_dir, low_dir = open_store(args)
    try:
        catalog = load_catalog_mirror(root, create=True, dry_run=args.dry_run)
        if not args.dry_run:
            orig_dir.mkdir(parents=True, exist_ok=True)
        rungs = db.load_rungs(conn)
        selected = None
        scale = flags = pix_fmt = None
        if not args.no_degrade:
            if low_dir is None:
                raise SystemExit("no low folder")
            if not args.dry_run:
                low_dir.mkdir(parents=True, exist_ok=True)
            scale, flags, pix_fmt, variants, ratios = db.load_degrade_ratios(conn, enabled_only=True)
            FRAME_RATIOS.update(ratios)
            selected = select_variants(variants, None)
        failed = False
        for name in args.videos:
            src = Path(name).expanduser().resolve()
            prepared = prepare_import(
                src, orig_dir, rungs, catalog, root,
                dry_run=args.dry_run, force=args.force, conn=conn,
            )
            if prepared is None:
                failed = True
                continue
            title, planned, probed = prepared
            for item in planned:
                written, one_failed = produce_planned(
                    src, title, [item], probed, orig_dir, args.force, catalog, root,
                    dry_run=args.dry_run, conn=conn,
                )
                failed = failed or one_failed
                if args.no_degrade:
                    continue
                for master_name in written:
                    path = orig_dir / master_name
                    geometry = None
                    if args.dry_run and (args.force or not path.is_file()):
                        geometry = planned_geometry(item, probed)
                    ok = degrade_master(
                        path, low_dir, selected,
                        scale, flags, pix_fmt, args.force, True, catalog, root,
                        dry_run=args.dry_run, geometry=geometry, conn=conn,
                    )
                    failed = failed or not ok
        if failed:
            raise SystemExit(1)
    finally:
        conn.close()


def _title_names(conn, catalog, root):
    names = [row[0] for row in conn.execute(
        "SELECT name FROM sources WHERE enabled = 1 ORDER BY name"
    )]
    seen = set(names)
    for title in titles_of(catalog, catalog_file(root)):
        if title not in seen:
            names.append(title)
            seen.add(title)
    return names


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "import":
        import_main(argv[1:])
        return
    if argv and argv[0] == "stats":
        stats_main(argv[1:])
        return
    args = parse_args(argv)
    conn, root, orig_dir, low_dir = open_store(args)
    try:
        if not orig_dir.is_dir():
            raise SystemExit(f"no orig folder: {orig_dir}")
        if not args.dry_run:
            low_dir.mkdir(parents=True, exist_ok=True)
        catalog = load_catalog_mirror(root, create=False, dry_run=args.dry_run)
        scale, scale_flags, pix_fmt, variants, ratios = db.load_degrade_ratios(conn, enabled_only=False)
        FRAME_RATIOS.update(ratios)
        enabled = {
            row["name"]
            for row in conn.execute("SELECT name FROM degrade_presets WHERE enabled = 1")
        }
        if args.variants is None:
            variants = {name: args_ for name, args_ in variants.items() if name in enabled}
        selected = select_variants(variants, args.variants)
        rungs = db.load_rungs(conn)
        failed = False
        if not args.videos:
            for title in _title_names(conn, catalog, root):
                ok = finish_title(
                    title, orig_dir, rungs, low_dir, selected,
                    scale, scale_flags, pix_fmt, args.force, False, catalog, root,
                    dry_run=args.dry_run, conn=conn,
                )
                failed = failed or not ok
        else:
            sources, strict = sources_from_args(orig_dir, args.videos)
            for src in sources:
                ok = degrade_master(
                    src, low_dir, selected, scale, scale_flags, pix_fmt,
                    args.force, strict, catalog, root, dry_run=args.dry_run,
                    conn=conn,
                )
                failed = failed or not ok
        if failed:
            raise SystemExit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
