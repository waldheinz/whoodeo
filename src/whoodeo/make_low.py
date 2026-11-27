"""Write training videos.

`whoodeo-make-low import` writes a source into the orig folder and records
it in the Whoodeo SQLite database. The rungs are pixel
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
import math
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from whoodeo import db
from whoodeo.library import (
    master_filename,
    plan_masters,
    picture_size,
    split_master_filename,
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


def scaled_frame_rate(master_rate, variant):
    """Output (num, den) for a non-1:1 preset, else None.

    A low file of a master with rate R and frame ratio num/den holds
    master_frames * den // num frames, so its constant frame rate is
    R * den / num. encode_variants uses this both as -r and as a trailing
    setpts so the MKV header (r_frame_rate, avg_frame_rate, DefaultDuration)
    matches the packet timestamps, instead of inheriting the master's rate
    through -fps_mode passthrough.
    """
    num, den = frame_ratio(variant)
    if (num, den) == (1, 1):
        return None
    rate_num, rate_den = master_rate
    out_num = rate_num * den
    out_den = rate_den * num
    g = math.gcd(out_num, out_den)
    return out_num // g, out_den // g


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



FFMPEG = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error"]


def run_ffmpeg(command, label=None):
    """Run ffmpeg without a status line. Report stderr only on failure."""
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        parts = [f"{label}: ffmpeg failed" if label else "ffmpeg failed"]
        err = (result.stderr or "").strip()
        out = (result.stdout or "").strip()
        if err:
            parts.append(err)
        elif out:
            parts.append(out)
        print("\n".join(parts), file=sys.stderr)
    return result


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class FilmProgress:
    """A compact progress bar over films/masters that still need work."""

    def __init__(self, total, skipped=0, stream=None):
        self.total = total
        self.skipped = skipped
        self.done = 0
        self.started = time.monotonic()
        self.stream = stream if stream is not None else sys.stderr
        self.tty = self.stream.isatty()
        self._current = ""
        self._jobs = 0
        self._drawn = False
        self._closed = False

    def start_item(self, name, jobs=0):
        if self._closed:
            return
        self._current = name
        self._jobs = jobs
        self._render()

    def advance(self):
        if self._closed:
            return
        self.done += 1
        if self.done >= self.total:
            self._current = ""
            self._jobs = 0
        self._render()

    def clear_line(self):
        if self.tty and self._drawn and not self._closed:
            self.stream.write("\r\x1b[2K")
            self.stream.flush()
            self._drawn = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.total == 0:
            if self.skipped:
                self._write(f"nothing to do ({self.skipped} already done)", newline=True)
            elif self.tty and self._drawn:
                self.stream.write("\r\x1b[2K")
                self.stream.flush()
                self._drawn = False
            return
        self._current = ""
        self._jobs = 0
        self._render()
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()
            self._drawn = False

    def _render(self, final=False):
        self._write(self._line(), newline=not self.tty)

    def _write(self, line, newline=False):
        if self.tty:
            self.stream.write("\r\x1b[2K" + line)
            if newline:
                self.stream.write("\n")
                self._drawn = False
            else:
                self._drawn = True
        else:
            self.stream.write(line + "\n")
            self._drawn = False
        self.stream.flush()

    def _line(self):
        width = max(40, shutil.get_terminal_size((80, 24)).columns)
        total = self.total
        frac = 1.0 if not total else min(max(self.done, 0) / total, 1.0)
        pct = 100 if not total else int(frac * 100)
        elapsed = time.monotonic() - self.started
        bar_w = 12
        filled = bar_w if not total else min(bar_w, int(bar_w * frac))
        bar = "[" + "#" * filled + "-" * (bar_w - filled) + "]"
        prefix = f"{bar}  {self.done}/{total}  {pct}%  {_format_duration(elapsed)} elapsed"
        if self.done >= 1 and self.done < total:
            eta = (elapsed / self.done) * (total - self.done)
            prefix += f"  {_format_duration(eta)} eta"
        if self.skipped:
            prefix += f"  {self.skipped} skipped"
        name = self._current
        jobs = ""
        if name and self._jobs:
            jobs = f"  {self._jobs} var" if self._jobs == 1 else f"  {self._jobs} vars"
        if name:
            room = width - len(prefix) - len(jobs) - 2
            if room < 4:
                jobs = ""
                room = width - len(prefix) - 2
            if room < 4:
                name = ""
            elif len(name) > room:
                name = "…" + name[-(room - 1):]
        suffix = f"  {name}{jobs}" if name else ""
        return (prefix + suffix)[:width]


class _BarAwareStderr:
    """Clear the progress bar before other stderr output, then redraw."""

    def __init__(self, wrapped, get_progress):
        self._wrapped = wrapped
        self._get_progress = get_progress

    def write(self, data):
        progress = self._get_progress()
        if progress is not None:
            progress.clear_line()
        written = self._wrapped.write(data)
        if progress is not None and not progress._closed and str(data).endswith("\n"):
            progress._render()
        return written

    def flush(self):
        return self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    def fileno(self):
        return self._wrapped.fileno()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


_ACTIVE_PROGRESS = None
_REAL_STDERR = sys.stderr


def _current_progress():
    return _ACTIVE_PROGRESS


def begin_progress(total, skipped=0):
    """Install a film progress bar on stderr. Return the bar."""
    global _ACTIVE_PROGRESS
    progress = FilmProgress(total, skipped=skipped, stream=_REAL_STDERR)
    _ACTIVE_PROGRESS = progress
    if not isinstance(sys.stderr, _BarAwareStderr):
        sys.stderr = _BarAwareStderr(_REAL_STDERR, _current_progress)
    return progress


def end_progress():
    """Close the bar and restore stderr."""
    global _ACTIVE_PROGRESS
    progress = _ACTIVE_PROGRESS
    _ACTIVE_PROGRESS = None
    if isinstance(sys.stderr, _BarAwareStderr):
        sys.stderr = _REAL_STDERR
    if progress is not None:
        progress.close()


def say(message, verbose=True):
    if verbose:
        print(message, flush=True)


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


def header_frame_count(path):
    """Frame count from the container header (nb_frames or duration * rate).

    This does not read every packet, so it is safe to use while planning work
    over a network disk. Returns None when the header has no usable count.
    """
    text = run_probe([
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,duration,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ])
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    stream = (payload.get("streams") or [{}])[0]
    nb = stream.get("nb_frames")
    if nb not in (None, "N/A", ""):
        try:
            value = int(nb)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    duration = stream.get("duration")
    if duration in (None, "N/A", ""):
        duration = (payload.get("format") or {}).get("duration")
    rate = stream.get("r_frame_rate") or stream.get("avg_frame_rate")
    try:
        duration = float(duration)
        rate = str(rate)
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if duration <= 0 or fps <= 0:
        return None
    return int(round(duration * fps))


def has_audio(path):
    text = run_probe([
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ])
    return bool(text)


def encode_variants(src, low_dir, jobs, scale, scale_flags, pix_fmt, force, strict, verbose=True):
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
                kept = header_frame_count(out)
                if kept is None:
                    kept = packets_of(out)
            except RuntimeError as exc:
                print(f"{label}: {exc}", file=sys.stderr)
                results[variant] = None
                continue
            # A preset whose frame ratio is not 1/1 may have an older 1/1 file
            # on disk (written before the ratio was set). That file is stale.
            if frame_ratio(variant) != (1, 1):
                try:
                    master_frames = header_frame_count(src)
                    if master_frames is None:
                        master_frames = packets_of(src)
                except RuntimeError as exc:
                    print(f"{label}: {exc}", file=sys.stderr)
                    results[variant] = None
                    continue
                if kept != expected_low_frames(variant, master_frames):
                    say(
                        f"stale {label}: {kept} frames, expected "
                        f"{describe_frames(variant, master_frames)}; re-encode",
                        verbose,
                    )
                    todo.append((variant, ffmpeg_args, out))
                    continue
            say(f"keep {label}", verbose)
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
    try:
        master_rate, _ = video_timing(src)
    except RuntimeError as exc:
        return fail_all(str(exc))

    split_labels = []
    branches = []
    encoder_args = []
    output_rates = []
    for index, (variant, ffmpeg_args, out) in enumerate(todo):
        try:
            vf, rest = split_variant_vf(ffmpeg_args)
        except ValueError as exc:
            return fail_all(str(exc))
        encoder_args.append(rest)
        rate = scaled_frame_rate(master_rate, variant)
        output_rates.append(rate)
        # Non-1:1 filters keep fewer frames but, with passthrough, would
        # leave the master's r_frame_rate in the MKV header. setpts plus
        # -r/-fps_mode cfr writes a constant rate that matches the new
        # spacing without dropping or duplicating a frame. 1:1 variants
        # stay on passthrough so existing timestamps are left alone.
        retimestamp = (
            f"setpts=N/({rate[0]}/{rate[1]}*TB)" if rate is not None else None
        )
        if vf or retimestamp:
            scoped = re.sub(r"\[([A-Za-z0-9_]+)\]", rf"[b{index}_\1]", vf) if vf else ""
            filt = ",".join(part for part in (scoped, retimestamp) if part)
            split_labels.append(f"[s{index}]")
            branches.append(f"[s{index}]{filt}[v{index}]")
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
    for index, ((variant, ffmpeg_args, out), rest, rate) in enumerate(
        zip(todo, encoder_args, output_rates)
    ):
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
        tmp.unlink(missing_ok=True)
        temps.append(tmp)
        say(f"encode {variant}/{name}  {width}x{height} -> {low_w}x{low_h}  {ffmpeg_args}", verbose)
        rate_args = (
            ["-r", f"{rate[0]}/{rate[1]}", "-fps_mode", "cfr"]
            if rate is not None else
            ["-fps_mode", "passthrough"]
        )
        outputs += [
            "-map", f"[v{index}]", "-an",
            *rest,
            "-pix_fmt", pix_fmt,
            *rate_args,
            str(tmp),
        ]
    command = [
        *FFMPEG,
        "-i", str(src),
        "-filter_complex", graph,
        *outputs,
    ]
    result = run_ffmpeg(command, name)
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
        say(f"wrote {out}  {low_w}x{low_h}  {out_packets} frames", verbose)
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
    result = run_ffmpeg(
        [
            *FFMPEG, "-y",
            "-i", str(src),
            "-map", "0:v:0", "-an", "-c", "copy",
            "-bsf:v", f"setts=pts={expr}:dts={expr}",
            str(dst),
        ],
        src.name,
    )
    if result.returncode != 0:
        dst.unlink(missing_ok=True)
        err = (result.stderr or "").strip()
        return f"restamp failed ({err})" if err else "restamp failed"
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


def settle_remux_time(path, label, verbose=True):
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
    say(f"{label}: {err}; restamping timestamps", verbose)
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


def encode_master(src, label, out, stream, in_packets, width, height, matrix_name, out_w, out_h, verbose=True):
    """Encode src to a 10-bit 4:4:4 HEVC master at out_w by out_h."""
    matrix = input_matrix(matrix_name)
    if matrix is None:
        print(f"{label}: unsupported color matrix {matrix_name}", file=sys.stderr)
        return None
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    say(
        f"import {label}  {width}x{height} -> {out_w}x{out_h}  {MASTER_PIX_FMT} crf 12",
        verbose,
    )
    command = [
        *FFMPEG,
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
    result = run_ffmpeg(command, label)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        return None
    try:
        problem = timestamp_problem(tmp)
        if problem:
            say(f"{label}: {problem}; restamping timestamps", verbose)
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
    say(f"wrote {out}  {out_w}x{out_h}  {in_packets} frames", verbose)
    return out.name


def remux_master(src, label, out, stream, in_packets, width, height, verbose=True):
    """Copy the video bitstream into orig and drop the audio."""
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    tmp.unlink(missing_ok=True)
    say(f"import {label}  {width}x{height}  remux", verbose)
    command = [
        *FFMPEG,
        "-i", str(src),
        "-map", "0:v:0", "-an", "-c", "copy",
        str(tmp),
    ]
    result = run_ffmpeg(command, label)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        return None
    try:
        problem = remux_problem(tmp, stream, in_packets)
        if problem is None:
            problem = settle_remux_time(tmp, label, verbose=verbose)
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
    say(f"wrote {out}  {width}x{height}  {in_packets} frames", verbose)
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

def check_frames(conn, title, frames):
    """Refuse a frame count that disagrees with the database."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise RuntimeError(f"{title}: bad frame count {frames!r}")
    row = conn.execute("SELECT frames FROM sources WHERE name = ?", (title,)).fetchone()
    if row is not None and row["frames"] is not None and row["frames"] != frames:
        raise RuntimeError(f"{title}: {frames} frames, database says {row['frames']}")


def source_record(conn, title):
    return conn.execute(
        "SELECT id, name, frames, origin_path, enabled FROM sources WHERE name = ?",
        (title,),
    ).fetchone()


def masters_for_title(conn, title):
    return conn.execute(
        """
        SELECT m.id, m.rung, m.rel_path, m.width, m.height
        FROM source_masters m
        JOIN sources s ON s.id = m.source_id
        WHERE s.name = ?
        ORDER BY m.rung
        """,
        (title,),
    ).fetchall()


def remember_master(conn, title, rung, path, frames, origin_path=None):
    width, height = picture_size(path)
    check_frames(conn, title, frames)
    db.remember_master(
        conn, title, rung, f"orig/{path.name}", width, height,
        frames=frames, origin_path=origin_path,
    )
    conn.commit()


def produce_planned(src, title, planned, probed, orig_dir, force, conn, dry_run=False, verbose=True):
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
    for rung, out_w, out_h, kind in planned:
        name = master_filename(title, rung)
        out = orig_dir / name
        if out.exists() and not force:
            say(f"keep {name}", verbose)
            row = conn.execute(
                "SELECT m.id FROM source_masters m "
                "JOIN sources s ON s.id = m.source_id "
                "WHERE s.name = ? AND m.rung = ?",
                (title, rung),
            ).fetchone()
            if row is None:
                try:
                    frames = packets_of(out)
                    remember_master(conn, title, rung, out, frames)
                except RuntimeError as exc:
                    print(f"{name}: {exc}", file=sys.stderr)
                    failed = True
                    continue
            names.append(name)
            continue
        try:
            frames = source_packets(src, probed)
            check_frames(conn, title, frames)
        except RuntimeError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            failed = True
            continue
        if kind == "encode":
            wrote = encode_master(
                src, name, out, probed["stream"], frames,
                probed["width"], probed["height"], probed["matrix"], out_w, out_h,
                verbose=verbose,
            )
        else:
            wrote = remux_master(
                src, name, out, probed["stream"], frames,
                probed["width"], probed["height"],
                verbose=verbose,
            )
        if wrote is None:
            failed = True
            continue
        try:
            origin = str(src.resolve()) if src is not None else None
            remember_master(conn, title, rung, out, frames, origin_path=origin)
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


def prepare_import(src, orig_dir, rungs, conn, dry_run=False, force=False):
    """Record src as the title's origin. Return (title, planned, probed), or None.

    `dry_run` leaves the database untouched. The source packet count runs only
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
    if masters_to_write(orig_dir, title, planned, force):
        try:
            frames = source_packets(src, probed)
            check_frames(conn, title, frames)
        except RuntimeError as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            return None
    origin = str(src.resolve())
    frames = probed.get("packets")
    if frames is None:
        existing = source_record(conn, title)
        if existing is not None:
            frames = existing["frames"]
    try:
        db.remember_source(conn, title, origin_path=origin, frames=frames)
        conn.commit()
    except RuntimeError as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None
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


def list_low_encodes(src, low_dir, selected, scale, force, strict, geometry, master_frames=None):
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
        if low_path.is_file() and not force and not low_is_stale(low_path, src, variant, master_frames=master_frames):
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


def low_is_stale(low_path, master_path, variant, master_frames=None):
    """True when an existing low file does not match the preset frame ratio.

    Uses the container header (and optional known master frame count) so a
    plan or dry-run does not have to read every packet on a network disk.
    """
    if frame_ratio(variant) == (1, 1):
        return False
    if not low_path.is_file():
        return True
    if master_frames is None and master_path is not None and Path(master_path).is_file():
        master_frames = header_frame_count(master_path)
    if not isinstance(master_frames, int) or master_frames < 1:
        return True
    kept = header_frame_count(low_path)
    if kept is None:
        return True
    return kept != expected_low_frames(variant, master_frames)


def pending_jobs(src, low_dir, selected, force, master_frames=None):
    """Variant encodes still needed for this master."""
    jobs = []
    for variant, ffmpeg_args in selected:
        out = low_dir / variant / src.name
        if force or not out.is_file() or low_is_stale(out, src, variant, master_frames=master_frames):
            jobs.append((variant, ffmpeg_args))
    return jobs


def degrade_master(
    src, low_dir, selected, scale, flags, pix_fmt, force, strict, conn,
    dry_run=False, geometry=None, unlisted=False, verbose=True,
):
    """Write the selected lows of one master and record them. Return False on failure.

    `dry_run` only prints the encode lines. `geometry` is the size of a master
    that is not on disk yet. `unlisted` allows an on-disk file the database
    does not name yet; a real run would record it before degrading.
    """
    parts = split_master_filename(src.name)
    if parts is None:
        print(f"{src.name}: expected <title>.<rung>.mkv", file=sys.stderr)
        return False
    title, rung = parts
    source = source_record(conn, title)
    master = None
    if source is not None:
        master = conn.execute(
            "SELECT id, width, height FROM source_masters "
            "WHERE source_id = ? AND rung = ?",
            (source["id"], rung),
        ).fetchone()
    if master is None:
        if not (dry_run and (geometry is not None or unlisted)):
            print(f"{src.name}: not in the database", file=sys.stderr)
            return False
    if dry_run:
        master_frames = source["frames"] if source is not None else None
        return list_low_encodes(
            src, low_dir, selected, scale, force, strict, geometry,
            master_frames=master_frames,
        )
    failed = False
    jobs = []
    for variant, ffmpeg_args in selected:
        low_path = low_dir / variant / src.name
        master_frames = source["frames"] if source is not None else None
        if low_path.is_file() and not force and not low_is_stale(low_path, src, variant, master_frames=master_frames):
            say(f"keep {variant}/{src.name}", verbose)
            try:
                db_remember_variant(conn, title, rung, variant, src.name)
            except RuntimeError as exc:
                print(f"{variant}/{src.name}: {exc}", file=sys.stderr)
            continue
        jobs.append((variant, ffmpeg_args))
    encoded = encode_variants(
        src, low_dir, jobs, scale, flags, pix_fmt, force, strict, verbose=verbose,
    ) if jobs else {}
    source = source_record(conn, title)
    if master is None and source is not None:
        master = conn.execute(
            "SELECT id, width, height FROM source_masters "
            "WHERE source_id = ? AND rung = ?",
            (source["id"], rung),
        ).fetchone()
    for variant, _ in jobs:
        low_path = low_dir / variant / src.name
        frames = encoded.get(variant)
        if frames is None:
            failed = True
            continue
        if frames < 0:
            continue
        master_frames = source["frames"] if source is not None else None
        if not isinstance(master_frames, int) or frames != expected_low_frames(variant, master_frames):
            expected = (
                describe_frames(variant, master_frames)
                if isinstance(master_frames, int) else "?"
            )
            print(
                f"{variant}/{src.name}: {frames} frames, database says "
                f"{master_frames} master frames, expected {expected}",
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
        width = master["width"] if master is not None else None
        height = master["height"] if master is not None else None
        if width != low_w * scale or height != low_h * scale:
            print(
                f"{variant}/{src.name} is {low_w}x{low_h}, "
                f"orig is {width}x{height}, "
                f"expected exactly {scale}x",
                file=sys.stderr,
            )
            failed = True
            continue
        try:
            db_remember_variant(conn, title, rung, variant, src.name)
        except RuntimeError as exc:
            print(f"{variant}/{src.name}: {exc}", file=sys.stderr)
            failed = True
    return not failed


def plan_title(conn, title, orig_dir, rungs, low_dir, selected, force):
    """Work still needed for one title.

    Returns (steps, skipped, warnings, failed, record). `steps` need encoding
    (and maybe a new master). `record` are unlisted existing masters that
    only need a database row. `warnings` are messages to print.
    """
    source = source_record(conn, title)
    if source is None:
        return [], 0, [], False, []
    warnings = []
    failed = False
    planned = []
    src = None
    probed = None
    origin = source["origin_path"]
    if origin:
        src = Path(origin)
        if not src.is_file():
            warnings.append(f"{title}: origin is gone, masters stay")
        else:
            probed = inspect_source(src)
            if probed is None:
                failed = True
            else:
                planned = plan_masters(*probed["display"], rungs)
    db_masters = {row["rung"]: row for row in masters_for_title(conn, title)}
    steps = []
    skipped = 0
    seen = set()
    planned_rungs = {item[0] for item in planned}
    record = []

    def job_count(path, assume_all=False):
        if assume_all or not path.is_file():
            return len(selected)
        return len(pending_jobs(path, low_dir, selected, force, master_frames=source["frames"]))

    for item in planned:
        rung = item[0]
        path = orig_dir / master_filename(title, rung)
        if not path.is_file():
            continue
        unlisted = rung not in db_masters
        jobs = job_count(path)
        if unlisted:
            record.append((path, item, src, probed))
        if jobs:
            steps.append({
                "name": path.name,
                "path": path,
                "jobs": jobs,
                "produce": None,
                "unlisted": unlisted,
                "item": item,
                "src": src,
                "probed": probed,
            })
        else:
            skipped += 1
        seen.add(rung)
    for rung in db_masters:
        if rung in seen or rung in planned_rungs:
            continue
        path = orig_dir / master_filename(title, rung)
        if not path.is_file():
            warnings.append(f"skip {path.name}: file missing")
            failed = True
            continue
        seen.add(rung)
        jobs = job_count(path)
        if jobs:
            steps.append({
                "name": path.name,
                "path": path,
                "jobs": jobs,
                "produce": None,
                "unlisted": False,
                "item": None,
                "src": src,
                "probed": probed,
            })
        else:
            skipped += 1
    for item in planned:
        rung = item[0]
        if rung in seen:
            continue
        path = orig_dir / master_filename(title, rung)
        if src is None or probed is None:
            warnings.append(f"skip {path.name}: file missing")
            failed = True
            continue
        steps.append({
            "name": path.name,
            "path": path,
            "jobs": job_count(path, assume_all=True),
            "produce": item,
            "unlisted": False,
            "item": item,
            "src": src,
            "probed": probed,
        })
    return steps, skipped, warnings, failed, record


def execute_title(
    conn, title, plan, orig_dir, low_dir, selected,
    scale, flags, pix_fmt, force, strict, dry_run=False, verbose=True, progress=None,
):
    """Run a plan from plan_title. Return False when a step fails."""
    steps, _skipped, warnings, failed, record = plan
    for message in warnings:
        print(message, file=sys.stderr)
    for _path, item, src, probed in record:
        if dry_run:
            continue
        _names, one_failed = produce_planned(
            src, title, [item], probed, orig_dir, False, conn,
            dry_run=False, verbose=verbose,
        )
        failed = failed or one_failed
    for step in steps:
        if progress is not None:
            progress.start_item(step["name"], step["jobs"])
        path = step["path"]
        if step["produce"] is not None:
            names, one_failed = produce_planned(
                step["src"], title, [step["produce"]], step["probed"],
                orig_dir, False, conn, dry_run=dry_run, verbose=verbose,
            )
            failed = failed or one_failed
            ready = path.name in names and (path.is_file() or dry_run)
            if one_failed or not ready:
                if not path.is_file() and not dry_run:
                    print(f"skip {path.name}: file missing", file=sys.stderr)
                    failed = True
                if progress is not None:
                    progress.advance()
                continue
        geometry = None
        if dry_run and not path.is_file() and step["item"] is not None and step["probed"] is not None:
            geometry = planned_geometry(step["item"], step["probed"])
        ok = degrade_master(
            path, low_dir, selected, scale, flags, pix_fmt,
            force, strict, conn,
            dry_run=dry_run, geometry=geometry, unlisted=step["unlisted"],
            verbose=verbose,
        )
        failed = failed or not ok
        if progress is not None:
            progress.advance()
    return not failed


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


def enabled_title_names(conn):
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sources WHERE enabled = 1 ORDER BY name"
        )
    ]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="encode half-resolution variants of database masters, "
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
        description="print database masters grouped by size. "
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
        description="write sources into orig, one file per database rung. "
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


def _import_work_items(orig_dir, low_dir, selected, prepared_list, force, no_degrade):
    """Masters that still need a write or low variants. Also a skip count."""
    items = []
    skipped = 0
    for src, title, planned, probed in prepared_list:
        for item in planned:
            path = orig_dir / master_filename(title, item[0])
            needs_write = force or not path.is_file()
            jobs = 0
            if not no_degrade and selected is not None:
                if path.is_file() and not force:
                    jobs = len(pending_jobs(path, low_dir, selected, force))
                else:
                    jobs = len(selected)
            if needs_write or jobs:
                items.append((src, title, item, probed, path, jobs, needs_write))
            else:
                skipped += 1
    return items, skipped


def import_main(argv):
    args = parse_import_args(argv)
    conn, root, orig_dir, low_dir = open_store(args)
    try:
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
        prepared_list = []
        for name in args.videos:
            src = Path(name).expanduser().resolve()
            prepared = prepare_import(
                src, orig_dir, rungs, conn,
                dry_run=args.dry_run, force=args.force,
            )
            if prepared is None:
                failed = True
                continue
            prepared_list.append((src, *prepared))
        verbose = args.dry_run
        if not args.dry_run:
            work, skipped = _import_work_items(
                orig_dir, low_dir, selected, prepared_list, args.force, args.no_degrade,
            )
            begin_progress(len(work), skipped)
        try:
            for src, title, planned, probed in prepared_list:
                for item in planned:
                    path = orig_dir / master_filename(title, item[0])
                    jobs = 0
                    if not args.no_degrade and selected is not None:
                        if path.is_file() and not args.force:
                            jobs = len(pending_jobs(path, low_dir, selected, args.force))
                        else:
                            jobs = len(selected)
                    needs = args.force or not path.is_file() or jobs
                    if not args.dry_run and needs and _ACTIVE_PROGRESS is not None:
                        _ACTIVE_PROGRESS.start_item(path.name, jobs)
                    written, one_failed = produce_planned(
                        src, title, [item], probed, orig_dir, args.force, conn,
                        dry_run=args.dry_run, verbose=verbose,
                    )
                    failed = failed or one_failed
                    if args.no_degrade:
                        if not args.dry_run and needs and _ACTIVE_PROGRESS is not None:
                            _ACTIVE_PROGRESS.advance()
                        continue
                    for master_name in written:
                        master_path = orig_dir / master_name
                        geometry = None
                        if args.dry_run and (args.force or not master_path.is_file()):
                            geometry = planned_geometry(item, probed)
                        ok = degrade_master(
                            master_path, low_dir, selected,
                            scale, flags, pix_fmt, args.force, True, conn,
                            dry_run=args.dry_run, geometry=geometry, verbose=verbose,
                        )
                        failed = failed or not ok
                    if not args.dry_run and needs and _ACTIVE_PROGRESS is not None:
                        _ACTIVE_PROGRESS.advance()
        finally:
            end_progress()
        if failed:
            raise SystemExit(1)
    finally:
        end_progress()
        conn.close()


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
        verbose = args.dry_run
        try:
            if not args.videos:
                titles = enabled_title_names(conn)
                plans = []
                total = 0
                skipped = 0
                for title in titles:
                    plan = plan_title(
                        conn, title, orig_dir, rungs, low_dir, selected, args.force,
                    )
                    plans.append((title, plan))
                    total += len(plan[0])
                    skipped += plan[1]
                if not args.dry_run:
                    begin_progress(total, skipped)
                for title, plan in plans:
                    ok = execute_title(
                        conn, title, plan, orig_dir, low_dir, selected,
                        scale, scale_flags, pix_fmt, args.force, False,
                        dry_run=args.dry_run, verbose=verbose,
                        progress=_ACTIVE_PROGRESS,
                    )
                    failed = failed or not ok
            else:
                sources, strict = sources_from_args(orig_dir, args.videos)
                if args.dry_run:
                    for src in sources:
                        ok = degrade_master(
                            src, low_dir, selected, scale, scale_flags, pix_fmt,
                            args.force, strict, conn, dry_run=True, verbose=True,
                        )
                        failed = failed or not ok
                else:
                    pending = []
                    skipped = 0
                    for src in sources:
                        jobs = pending_jobs(src, low_dir, selected, args.force)
                        if jobs:
                            pending.append((src, jobs))
                        else:
                            skipped += 1
                            ok = degrade_master(
                                src, low_dir, selected, scale, scale_flags, pix_fmt,
                                args.force, strict, conn, verbose=False,
                            )
                            failed = failed or not ok
                    begin_progress(len(pending), skipped)
                    for src, jobs in pending:
                        _ACTIVE_PROGRESS.start_item(src.name, len(jobs))
                        ok = degrade_master(
                            src, low_dir, selected, scale, scale_flags, pix_fmt,
                            args.force, strict, conn, verbose=False,
                        )
                        failed = failed or not ok
                        _ACTIVE_PROGRESS.advance()
        finally:
            end_progress()
        if failed:
            raise SystemExit(1)
    finally:
        end_progress()
        conn.close()


if __name__ == "__main__":
    main()
