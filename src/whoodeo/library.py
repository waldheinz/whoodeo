"""Master filenames, rung names, and picture sizes for the training library.

A title can have several masters, each named <title>.<rung>.mkv. Rung names
come from the pixel budgets in settings.rungs (720, 480, or a film's own
WxH when it sits between budgets). make-low and the database use these
helpers; the catalog lives in whoodeo.sqlite.
"""

from pathlib import Path

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


def picture_size(path):
    """Stored width and height. The container is closed before return."""
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
    finally:
        container.close()
    if not width or not height:
        raise RuntimeError(f"no video size: {path}")
    return width, height


def _where(path):
    return str(path)


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
