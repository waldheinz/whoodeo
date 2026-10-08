"""Index of the training masters and their degraded variants.

catalog.json sits beside orig/ and low/. It holds the rung list, the degrade
recipe, and the titles. One title can have several masters, each named
<title>.<rung>.mkv, and the lows repeat that name under their variant
directory. origin is the file a later rung is encoded from. It may be null.
A missing source leaves the masters that are already written.

Import writes the document to a temp file and renames it over the old one.
Training reads it once at startup and then works from that copy. A read that
lands on the rename retries a few times. A new file is seeded from the
package degrade.yaml and the 720 and 480 rungs. After that, this file is
the list make-low encodes.
"""

import json
import os
import time
from pathlib import Path

import yaml

CATALOG = "catalog.json"
# 854*480 is the 16:9 frame at 480p. The fitter snaps each source to multiples of 4.
DEFAULT_RUNGS = (
    ("720", 1280 * 720),
    ("480", 854 * 480),
)
# A picture takes a rung's name when it already fits that pixel budget and
# fills at least this much of it. The 1320x696 films sit on the 720 rung.
# A size that only reaches the gap between two rungs keeps its own WxH name.
_ON_RUNG = 0.85


def snap4(value):
    snapped = int(round(value / 4.0)) * 4
    if snapped < 4:
        return 4
    return snapped


def fit_pixels(disp_w, disp_h, budget):
    """Square-pixel size at this budget, aspect kept, both sides a multiple of 4.

    A picture that already has fewer pixels stays at its display size. The
    multiple of 4 keeps the half-resolution low frame even.
    """
    scale = 1.0
    if disp_w * disp_h > budget:
        scale = (budget / (disp_w * disp_h)) ** 0.5
    return snap4(disp_w * scale), snap4(disp_h * scale)


def band_name(pixels, width, height, rungs):
    """Rung name for a picture that is kept as it is, or its own WxH."""
    for name, budget in rungs:
        if pixels > budget:
            continue
        if pixels >= budget * _ON_RUNG:
            return name
    return f"{width}x{height}"


def plan_masters(disp_w, disp_h, rungs):
    """Masters to write for one source.

    `rungs` is (name, pixel budget), largest first. Each budget the source
    exceeds becomes an encode at that size. The source itself is kept when
    it is not larger than the top rung, under that rung's name when it
    already sits there. Nothing is upscaled.
    """
    pixels = disp_w * disp_h
    planned = []
    for name, budget in rungs:
        if pixels > budget:
            width, height = fit_pixels(disp_w, disp_h, budget)
            planned.append((name, width, height, "encode"))
    native_w, native_h = snap4(disp_w), snap4(disp_h)
    covered = any(width == native_w and height == native_h for _, width, height, _ in planned)
    if not covered and (not rungs or pixels <= rungs[0][1]):
        name = band_name(native_w * native_h, native_w, native_h, rungs)
        planned.append((name, native_w, native_h, "remux"))
    planned.sort(key=lambda item: item[1] * item[2], reverse=True)
    return planned


def master_filename(title, rung):
    return f"{title}.{rung}.mkv"


def split_master_filename(name):
    """(title, rung) from <title>.<rung>.mkv. None when there is no rung."""
    if not name.endswith(".mkv"):
        return None
    stem = name[: -len(".mkv")]
    if "." not in stem:
        return None
    title, rung = stem.rsplit(".", 1)
    if not title or not rung or "." in rung or "/" in rung:
        return None
    return title, rung


def catalog_file(root):
    return Path(root) / CATALOG


def load_catalog(root):
    """The catalog. A missing file is an error. A torn read is retried."""
    path = catalog_file(root)
    if not path.is_file():
        raise SystemExit(f"no catalog: {path}")
    last = None
    for _ in range(8):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.05)
            continue
        if not isinstance(data, dict):
            raise SystemExit(f"{path}: expected an object")
        return data
    raise SystemExit(f"{path}: {last}")


def save_catalog(root, data):
    """Write the whole catalog and rename it into place."""
    path = catalog_file(root)
    tmp = path.with_name("catalog.json.tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def measure_clip(path):
    """Width, height, and duration. The container is closed before return.

    Duration matches the value training used to read from an open clip:
    the container duration in seconds.
    """
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        if not container.duration:
            raise RuntimeError(f"no duration: {path}")
        width = stream.codec_context.width
        height = stream.codec_context.height
        duration = container.duration / av.time_base
    finally:
        container.close()
    return width, height, duration


def _where(path):
    return str(path)


def titles_of(catalog, path):
    titles = catalog.get("titles") if isinstance(catalog, dict) else None
    if not isinstance(titles, dict):
        raise SystemExit(f"{_where(path)}: missing titles")
    return titles


def rungs_of(catalog, path):
    """(name, pixel budget), largest first."""
    rungs = catalog.get("rungs") if isinstance(catalog, dict) else None
    label = _where(path)
    if not isinstance(rungs, list) or not rungs:
        raise SystemExit(f"{label}: rungs must be a non-empty list")
    parsed = []
    seen_names = set()
    seen_pixels = set()
    for item in rungs:
        if not isinstance(item, dict):
            raise SystemExit(f"{label}: each rung needs a name and a pixel budget")
        name = item.get("name")
        pixels = item.get("pixels")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "." in name
            or "/" in name
            or "\\" in name
        ):
            raise SystemExit(f"{label}: bad rung name {name!r}")
        if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 16:
            raise SystemExit(f"{label}: rung {name} needs a pixel budget")
        if name in seen_names or pixels in seen_pixels:
            raise SystemExit(f"{label}: duplicate rung {name}")
        seen_names.add(name)
        seen_pixels.add(pixels)
        parsed.append((name, pixels))
    parsed.sort(key=lambda item: item[1], reverse=True)
    return tuple(parsed)


def degrade_of(catalog, path):
    """Scale, scale flags, pixel format, and the variant arguments."""
    degrade = catalog.get("degrade") if isinstance(catalog, dict) else None
    label = _where(path)
    if not isinstance(degrade, dict):
        raise SystemExit(f"{label}: missing degrade")
    scale = degrade.get("scale")
    flags = degrade.get("scale_flags")
    pix_fmt = degrade.get("pix_fmt")
    variants = degrade.get("variants")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 2:
        raise SystemExit(f"{label}: degrade.scale must be an integer >= 2")
    if not isinstance(flags, str) or not flags:
        raise SystemExit(f"{label}: degrade.scale_flags must be a string")
    if not isinstance(pix_fmt, str) or not pix_fmt:
        raise SystemExit(f"{label}: degrade.pix_fmt must be a string")
    if not isinstance(variants, dict) or not variants:
        raise SystemExit(f"{label}: degrade.variants must be a non-empty object")
    for name, args in variants.items():
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name:
            raise SystemExit(f"{label}: bad variant name {name!r}")
        if not isinstance(args, str) or not args.strip():
            raise SystemExit(f"{label}: variant {name} must be an ffmpeg argument string")
    return scale, flags, pix_fmt, variants


def empty_catalog():
    """A new library: 720 and 480, and the package degrade recipe."""
    path = Path(__file__).resolve().parent / "degrade.yaml"
    try:
        loaded = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no config: {path}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected an object")
    data = {
        "rungs": [{"name": name, "pixels": pixels} for name, pixels in DEFAULT_RUNGS],
        "degrade": loaded,
        "titles": {},
    }
    rungs_of(data, path)
    degrade_of(data, path)
    return data


def open_catalog(root, create=False):
    """Load the catalog and check its three sections.

    `create` writes a seeded file when none exists. Import uses that.
    The plain command does not, so a missing file stays visible.
    """
    path = catalog_file(root)
    if not path.exists():
        if not create:
            raise SystemExit(f"no catalog: {path}")
        data = empty_catalog()
        save_catalog(root, data)
        return data
    data = load_catalog(root)
    rungs_of(data, path)
    degrade_of(data, path)
    titles_of(data, path)
    return data


def video_record(path, frames):
    width, height, duration = measure_clip(path)
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "duration": duration,
    }
