"""Write training videos.

`whoodeo-make-low import` writes a source into the orig folder and records
it in catalog.json beside orig/ and low/. The catalog's rungs are pixel
budgets. A source larger than a rung is encoded down to it, 10-bit 4:4:4
HEVC with a short closed GOP, so training seeks stay cheap. A source that
already sits on a rung is remuxed as it is: the video bitstream stays, and
the audio is dropped. Smaller rungs are still encoded from that same
source. Nothing is upscaled. The file is <title>.<rung>.mkv. Low variants
come from the catalog's degrade list and are written unless `--no-degrade`
is set, under the same filename.

The plain command writes the missing variants of masters already in the
catalog. With no filenames it also encodes any rung the catalog's origin
file can still supply. Variant directories live under the low root.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from whoodeo.catalog import EnvPath, resolve_data
from whoodeo.library import (
    catalog_file,
    degrade_of,
    master_filename,
    open_catalog,
    plan_masters,
    rungs_of,
    save_catalog,
    split_master_filename,
    titles_of,
    video_record,
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
# ffprobe matrix name → colorspace filter `all`/`iall` value.
MATRIX_ALL = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "bt470m": "bt470m",
    "smpte240m": "smpte240m",
}


def select_variants(variants, text):
    if text is None:
        return list(variants.items())
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise SystemExit("--variants is empty")
    missing = [name for name in names if name not in variants]
    if missing:
        raise SystemExit("unknown variant: " + ", ".join(missing))
    return [(name, variants[name]) for name in names]


def codec_of(args):
    tokens = shlex.split(args)
    for flag in ("-c:v", "-codec:v"):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                return tokens[index + 1]
    return None


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


def encode_one(src, low_dir, variant, ffmpeg_args, scale, scale_flags, pix_fmt, force, strict):
    """Write one low variant. Return its frame count, -1 when ignored, None on failure."""
    name = src.name
    label = f"{variant}/{name}"
    if src.suffix.lower() not in VIDEO_EXTS:
        if strict:
            print(f"skip {label}: train does not read this extension", file=sys.stderr)
            return None
        return -1
    codec = codec_of(ffmpeg_args)
    if src.suffix.lower() == ".webm" and codec not in WEBM_CODECS:
        print(
            f"skip {label}: webm cannot hold codec {codec or '(unset)'}",
            file=sys.stderr,
        )
        return None if strict else -1
    sar = sample_aspect(src)
    if not square_pixels(sar):
        print(
            f"{label} has sample aspect ratio {sar}; stored pixels are not square",
            file=sys.stderr,
        )
        return None

    out_dir = low_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    if out.exists() and not force:
        print(f"keep {label}", flush=True)
        try:
            return packets_of(out)
        except RuntimeError as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            return None

    width, height = size_of(src)
    if width % (scale * 2) or height % (scale * 2):
        print(
            f"{label} is {width}x{height}. Both sides must be divisible by "
            f"{scale * 2} so the 1/{scale} frame is even.",
            file=sys.stderr,
        )
        return None
    low_w = width // scale
    low_h = height // scale
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    print(f"encode {label}  {width}x{height} -> {low_w}x{low_h}  {ffmpeg_args}", flush=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-map", "0:v:0", "-an",
        "-vf", f"scale={low_w}:{low_h}:flags={scale_flags}",
        *shlex.split(ffmpeg_args),
        "-pix_fmt", pix_fmt,
        "-fps_mode", "passthrough",
        str(tmp),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"{label}: ffmpeg failed", file=sys.stderr)
        return None

    try:
        out_size = size_of(tmp)
        in_packets = packets_of(src)
        out_packets = packets_of(tmp)
        audio = has_audio(tmp)
    except RuntimeError as exc:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {exc}", file=sys.stderr)
        return None
    if out_size != (low_w, low_h):
        tmp.unlink(missing_ok=True)
        print(f"{label}: output is {out_size[0]}x{out_size[1]}, expected {low_w}x{low_h}", file=sys.stderr)
        return None
    if in_packets != out_packets:
        tmp.unlink(missing_ok=True)
        print(f"{label}: {in_packets} input frames, {out_packets} output frames", file=sys.stderr)
        return None
    if audio:
        tmp.unlink(missing_ok=True)
        print(f"{label}: output still has audio", file=sys.stderr)
        return None
    tmp.replace(out)
    print(f"wrote {out}  {low_w}x{low_h}  {out_packets} frames", flush=True)
    return out_packets


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
    """Probe a source. Return a dict, or None when the file cannot be a master."""
    label = src.name
    try:
        stream = probe_video(src)
        in_packets = packets_of(src)
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
        "packets": in_packets,
        "width": width,
        "height": height,
        "matrix": matrix_name,
        "display": size,
    }


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


def remember_master(catalog, root, title, rung, path, frames, force):
    entry = title_entry(catalog, root, title)
    previous = entry["masters"].get(rung)
    low = {}
    if not force and isinstance(previous, dict) and isinstance(previous.get("low"), dict):
        low = previous["low"]
    record = video_record(path, frames)
    record["low"] = low
    entry["masters"][rung] = record
    save_catalog(root, catalog)


def produce_planned(src, title, planned, probed, orig_dir, force, catalog, root):
    """Write the planned masters that are missing. Return (filenames, failed)."""
    failed = False
    names = []
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
                    remember_master(catalog, root, title, rung, out, frames, False)
                except RuntimeError as exc:
                    print(f"{name}: {exc}", file=sys.stderr)
                    failed = True
                    continue
            names.append(name)
            continue
        if kind == "encode":
            wrote = encode_master(
                src, name, out, probed["stream"], probed["packets"],
                probed["width"], probed["height"], probed["matrix"], out_w, out_h,
            )
        else:
            wrote = remux_master(
                src, name, out, probed["stream"], probed["packets"],
                probed["width"], probed["height"],
            )
        if wrote is None:
            failed = True
            continue
        try:
            remember_master(catalog, root, title, rung, out, probed["packets"], force)
        except RuntimeError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            failed = True
            continue
        names.append(name)
    return names, failed


def import_source(src, orig_dir, rungs, catalog, root, force):
    """Write every rung of src. Return (filenames, failed)."""
    label = src.name
    if src.suffix.lower() not in VIDEO_EXTS:
        print(f"skip {label}: train does not read this extension", file=sys.stderr)
        return [], True
    if not src.is_file():
        print(f"no video: {src}", file=sys.stderr)
        return [], True
    if src.parent == orig_dir:
        print(f"{src} is already in {orig_dir}", file=sys.stderr)
        return [], True
    probed = inspect_source(src)
    if probed is None:
        return [], True
    planned = plan_masters(*probed["display"], rungs)
    if not planned:
        print(f"{label}: no master rung", file=sys.stderr)
        return [], True
    title = src.stem
    entry = title_entry(catalog, root, title)
    entry["origin"] = str(src.resolve())
    save_catalog(root, catalog)
    return produce_planned(src, title, planned, probed, orig_dir, force, catalog, root)


def degrade_master(src, low_dir, selected, scale, flags, pix_fmt, force, strict, catalog, root):
    """Write the selected lows of one master and record them. Return False on failure."""
    parts = split_master_filename(src.name)
    if parts is None:
        print(f"{src.name}: expected <title>.<rung>.mkv", file=sys.stderr)
        return False
    title, rung = parts
    entry = titles_of(catalog, catalog_file(root)).get(title)
    masters = entry.get("masters") if isinstance(entry, dict) else None
    master = masters.get(rung) if isinstance(masters, dict) else None
    if not isinstance(master, dict):
        print(f"{src.name}: not in the catalog", file=sys.stderr)
        return False
    if not isinstance(master.get("low"), dict):
        master["low"] = {}
    failed = False
    for variant, ffmpeg_args in selected:
        low_path = low_dir / variant / src.name
        if low_path.is_file() and not force and variant in master["low"]:
            print(f"keep {variant}/{src.name}", flush=True)
            continue
        frames = encode_one(
            src, low_dir, variant, ffmpeg_args,
            scale, flags, pix_fmt, force, strict,
        )
        if frames is None:
            failed = True
            continue
        if frames < 0:
            continue
        try:
            record = video_record(low_path, frames)
        except RuntimeError as exc:
            print(f"{variant}/{src.name}: {exc}", file=sys.stderr)
            failed = True
            continue
        if (
            master.get("width") != record["width"] * scale
            or master.get("height") != record["height"] * scale
        ):
            print(
                f"{variant}/{src.name} is {record['width']}x{record['height']}, "
                f"orig is {master.get('width')}x{master.get('height')}, "
                f"expected exactly {scale}x",
                file=sys.stderr,
            )
            failed = True
            continue
        master["low"][variant] = record
        save_catalog(root, catalog)
    return not failed


def fill_from_origin(title, orig_dir, rungs, catalog, root):
    """Encode rungs the stored origin can still supply. A missing origin is kept."""
    entry = titles_of(catalog, catalog_file(root)).get(title)
    if not isinstance(entry, dict):
        return True
    origin = entry.get("origin")
    if not origin:
        return True
    src = Path(origin)
    if not src.is_file():
        print(f"{title}: origin is gone, masters stay", file=sys.stderr)
        return True
    probed = inspect_source(src)
    if probed is None:
        return False
    planned = plan_masters(*probed["display"], rungs)
    masters = entry.get("masters") if isinstance(entry.get("masters"), dict) else {}
    missing = []
    for item in planned:
        rung = item[0]
        path = orig_dir / master_filename(title, rung)
        if rung in masters and path.is_file():
            continue
        missing.append(item)
    if not missing:
        return True
    _names, failed = produce_planned(
        src, title, missing, probed, orig_dir, False, catalog, root,
    )
    return not failed


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
        description="encode half-resolution variants of catalog masters. "
        "`whoodeo-make-low import` writes a source into orig first",
    )
    parser.add_argument("--variants", help="comma-separated names from the catalog degrade list; default is all of them")
    parser.add_argument("-o", "--orig", type=Path, default=EnvPath("orig"), help="master videos (default: %(default)s)")
    parser.add_argument("-l", "--low", type=Path, default=EnvPath("low"), help="degraded variants (default: %(default)s)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing low video")
    parser.add_argument(
        "videos", nargs="*",
        help="filenames in orig; default fills missing rungs from origin and encodes every missing partner",
    )
    return parser.parse_args(argv)


def parse_import_args(argv):
    parser = argparse.ArgumentParser(
        prog="whoodeo-make-low import",
        description="write sources into orig, one file per catalog rung. "
        "A source larger than a rung is encoded down to it. A source that "
        "already sits on a rung is remuxed. Low variants are written unless "
        "--no-degrade",
    )
    parser.add_argument("-o", "--orig", type=Path, default=EnvPath("orig"), help="master videos (default: %(default)s)")
    parser.add_argument("-l", "--low", type=Path, default=EnvPath("low"), help="low variants, written unless --no-degrade (default: %(default)s)")
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing master")
    parser.add_argument("--no-degrade", action="store_true", help="do not write the low variants of each new master")
    parser.add_argument("videos", nargs="+", help="source videos to write into orig")
    return parser.parse_args(argv)


def import_main(argv):
    args = parse_import_args(argv)
    args.orig = resolve_data(args.orig)
    orig_dir = args.orig.expanduser().resolve()
    orig_dir.mkdir(parents=True, exist_ok=True)
    root = orig_dir.parent
    catalog = open_catalog(root, create=True)
    where = catalog_file(root)
    rungs = rungs_of(catalog, where)
    low_dir = None
    selected = None
    scale = flags = pix_fmt = None
    if not args.no_degrade:
        args.low = resolve_data(args.low)
        low_dir = args.low.expanduser().resolve()
        low_dir.mkdir(parents=True, exist_ok=True)
        scale, flags, pix_fmt, variants = degrade_of(catalog, where)
        selected = list(variants.items())
    failed = False
    for name in args.videos:
        src = Path(name).expanduser().resolve()
        written, one_failed = import_source(src, orig_dir, rungs, catalog, root, args.force)
        failed = failed or one_failed
        if args.no_degrade:
            continue
        for master_name in written:
            ok = degrade_master(
                orig_dir / master_name, low_dir, selected,
                scale, flags, pix_fmt, args.force, True, catalog, root,
            )
            failed = failed or not ok
    if failed:
        raise SystemExit(1)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "import":
        import_main(argv[1:])
        return
    args = parse_args(argv)
    args.orig = resolve_data(args.orig)
    args.low = resolve_data(args.low)
    orig_dir = args.orig.expanduser().resolve()
    low_dir = args.low.expanduser().resolve()
    if not orig_dir.is_dir():
        raise SystemExit(f"no orig folder: {orig_dir}")
    low_dir.mkdir(parents=True, exist_ok=True)
    root = orig_dir.parent
    catalog = open_catalog(root, create=False)
    where = catalog_file(root)
    scale, scale_flags, pix_fmt, variants = degrade_of(catalog, where)
    selected = select_variants(variants, args.variants)
    rungs = rungs_of(catalog, where)
    titles = titles_of(catalog, where)
    failed = False
    if not args.videos:
        for title in list(titles):
            if not fill_from_origin(title, orig_dir, rungs, catalog, root):
                failed = True
        sources = []
        for title, entry in titles.items():
            masters = entry.get("masters") if isinstance(entry, dict) else None
            if not isinstance(masters, dict):
                continue
            for rung in masters:
                path = orig_dir / master_filename(title, rung)
                if path.is_file():
                    sources.append(path)
                else:
                    print(f"skip {path.name}: file missing", file=sys.stderr)
                    failed = True
        strict = False
    else:
        sources, strict = sources_from_args(orig_dir, args.videos)
    for src in sources:
        ok = degrade_master(
            src, low_dir, selected, scale, scale_flags, pix_fmt,
            args.force, strict, catalog, root,
        )
        failed = failed or not ok
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
