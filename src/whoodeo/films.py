"""Catalog films for training.

Pairs are a master plus the low variants named on it. A reconstruction run
can ignore the lows and read the master alone. Two spans of frames are held
out for validation. Nothing here opens a video until a clip is asked for a
frame.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import av

from whoodeo.library import load_catalog, master_filename, titles_of
from whoodeo.video import frame_tensor

SCALE = 2


def holdout_ranges(count, fraction):
    span = count * fraction / 2
    ranges = []
    for start in (0.30 * count, 0.70 * count):
        end = min(count, start + span)
        if end > start:
            ranges.append((start, end))
    return ranges


def in_ranges(index, ranges):
    return any(start <= index < end for start, end in ranges)


def frame_index(stamp, rate):
    """Display index of a presentation time at this frame rate."""
    return round(stamp * float(rate))


class Clip:
    """A video named in the catalog. Nothing stays open between reads."""

    def __init__(self, path, width, height, frames):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.frames = frames

    def frames_around(self, index, radius):
        """Frames centered on this display index.

        The seek timestamp only reaches a keyframe. The index is the frame
        rate times the presentation time, and each kept step has to be one
        frame. A gap, or a landing past the requested index, stops the read.
        """
        if index < 0 or index >= self.frames:
            raise RuntimeError(
                f"{self.path.name}: frame {index} outside 0..{self.frames - 1}"
            )
        container = av.open(str(self.path))
        try:
            stream = container.streams.video[0]
            width = stream.codec_context.width
            height = stream.codec_context.height
            if (width, height) != (self.width, self.height):
                raise RuntimeError(
                    f"{self.path.name} is {width}x{height}, "
                    f"catalog says {self.width}x{self.height}"
                )
            rate = stream.average_rate
            if not rate:
                raise RuntimeError(f"no frame rate: {self.path}")
            first = max(0, index - radius)
            last = min(self.frames - 1, index + radius)
            hint = max(0.0, (first - 1) / float(rate))
            container.seek(int(hint * av.time_base), backward=True, any_frame=False)
            found = {}
            previous = None
            for frame in container.decode(stream):
                if frame.time is None:
                    raise RuntimeError(f"no timestamp: {self.path.name}")
                current = frame_index(frame.time, rate)
                if previous is None:
                    if current > first:
                        raise RuntimeError(
                            f"{self.path.name}: seek landed on frame {current}, "
                            f"needed {first}"
                        )
                elif current != previous + 1:
                    raise RuntimeError(
                        f"{self.path.name}: frame {previous} is followed by {current}"
                    )
                previous = current
                if current < first:
                    continue
                found[current] = frame_tensor(frame)
                if current >= last:
                    break
            if first not in found or last not in found:
                raise RuntimeError(f"no frame {index} in {self.path.name}")
            return [
                found[min(max(index + delta, 0), self.frames - 1)]
                for delta in range(-radius, radius + 1)
            ]
        finally:
            container.close()


@dataclass
class Variant:
    name: str
    low: Clip


@dataclass
class Film:
    name: str
    orig: Clip
    frames: int
    holdouts: list
    variants: list = field(default_factory=list)


@dataclass
class FoundFilms:
    """Films named in the catalog."""

    films: list
    total: int
    kept: int
    problems: list


def _number(record, key, label):
    value = record.get(key) if isinstance(record, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label}: missing {key}")
    return value


def _count(record, key, label):
    value = record.get(key) if isinstance(record, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"{label}: missing {key}")
    return value


def _ordered(items):
    items.sort(key=lambda item: (item[0], -item[1]))
    return [item[2] for item in items]


def load_films(root, holdout, variants=None):
    """Pairs from catalog.json. No video is opened."""
    orig_dir = root / "orig"
    low_dir = root / "low"
    if not orig_dir.is_dir():
        raise SystemExit(f"no orig folder: {orig_dir}")
    if not low_dir.is_dir():
        raise SystemExit(f"no low folder: {low_dir}")
    catalog = load_catalog(root)
    titles = titles_of(catalog, root / "catalog.json")
    pattern = re.compile(variants) if variants is not None else None
    total = 0
    kept = 0
    problems = []
    ordered = []
    for title in titles:
        entry = titles[title]
        masters = entry.get("masters") if isinstance(entry, dict) else None
        if not isinstance(masters, dict):
            problems.append(f"skip {title}: no masters")
            continue
        frames = _count(entry, "frames", title)
        holdouts = holdout_ranges(frames, holdout)
        for rung, master in masters.items():
            name = master_filename(title, rung)
            orig_path = orig_dir / name
            if not isinstance(master, dict):
                problems.append(f"skip {name}: bad catalog entry")
                continue
            if not orig_path.is_file():
                problems.append(f"skip {name}: no master file")
                continue
            orig_w = int(_number(master, "width", name))
            orig_h = int(_number(master, "height", name))
            lows = master.get("low")
            if not isinstance(lows, list):
                problems.append(f"skip {name}: no lows")
                continue
            names = []
            for variant_name in lows:
                if (
                    not isinstance(variant_name, str)
                    or not variant_name
                    or "/" in variant_name
                    or variant_name in {".", ".."}
                ):
                    problems.append(f"skip {name}: bad variant name")
                    continue
                if variant_name not in names:
                    names.append(variant_name)
            low_w = orig_w // SCALE
            low_h = orig_h // SCALE
            variants_here = []
            for variant_name in sorted(names):
                rel = f"{variant_name}/{name}"
                low_path = low_dir / variant_name / name
                if not low_path.is_file():
                    problems.append(f"skip {rel}: file missing")
                    continue
                total += 1
                if pattern is not None and pattern.search(rel) is None:
                    continue
                if orig_w % SCALE or orig_h % SCALE:
                    raise SystemExit(
                        f"{name} is {orig_w}x{orig_h}, "
                        f"both sides must be divisible by {SCALE}"
                    )
                kept += 1
                variants_here.append(Variant(
                    name=variant_name,
                    low=Clip(low_path, low_w, low_h, frames),
                ))
            if not variants_here:
                continue
            film = Film(
                name=f"{title}.{rung}",
                orig=Clip(orig_path, orig_w, orig_h, frames),
                frames=frames,
                holdouts=holdouts,
                variants=variants_here,
            )
            ordered.append((f"{title}.mkv", orig_w * orig_h, film))
    return FoundFilms(_ordered(ordered), total, kept, problems)


def load_masters(root, holdout, crop):
    """Masters from catalog.json. A low variant is not required.

    `crop` drops a master whose side is shorter than that window. None keeps
    every master file.
    """
    orig_dir = root / "orig"
    if not orig_dir.is_dir():
        raise SystemExit(f"no orig folder: {orig_dir}")
    catalog = load_catalog(root)
    titles = titles_of(catalog, root / "catalog.json")
    total = 0
    kept = 0
    problems = []
    ordered = []
    for title in titles:
        entry = titles[title]
        masters = entry.get("masters") if isinstance(entry, dict) else None
        if not isinstance(masters, dict):
            problems.append(f"skip {title}: no masters")
            continue
        frames = _count(entry, "frames", title)
        holdouts = holdout_ranges(frames, holdout)
        for rung, master in masters.items():
            name = master_filename(title, rung)
            orig_path = orig_dir / name
            if not isinstance(master, dict):
                problems.append(f"skip {name}: bad catalog entry")
                continue
            if not orig_path.is_file():
                problems.append(f"skip {name}: no master file")
                continue
            orig_w = int(_number(master, "width", name))
            orig_h = int(_number(master, "height", name))
            total += 1
            if crop is not None and (orig_w < crop or orig_h < crop):
                problems.append(f"skip {name}: smaller than crop {crop}")
                continue
            kept += 1
            film = Film(
                name=f"{title}.{rung}",
                orig=Clip(orig_path, orig_w, orig_h, frames),
                frames=frames,
                holdouts=holdouts,
            )
            ordered.append((f"{title}.mkv", orig_w * orig_h, film))
    return FoundFilms(_ordered(ordered), total, kept, problems)


def require_films(found, root, variants):
    if found.kept:
        return
    if variants:
        raise SystemExit(
            f"no pairs match {variants!r}: {found.kept} of {found.total}"
        )
    raise SystemExit(f"no video pairs in {root / 'orig'} and {root / 'low'}")


def require_masters(found, root):
    if found.kept:
        return
    raise SystemExit(f"no masters in {root / 'orig'}")


def sample_index(rng, film, split):
    for _ in range(10000):
        index = rng.randrange(film.frames)
        inside = in_ranges(index, film.holdouts)
        if split == "val" and inside:
            return index
        if split == "train" and not inside:
            return index
    raise RuntimeError(f"could not sample a {split} frame in {film.name}")


def draw_indexes(rng, available, count):
    """count indexes into a collection of `available` items.

    The draw is without replacement when there are at least count items, and
    with replacement when there are fewer.
    """
    if count > available:
        return rng.choices(range(available), k=count)
    return rng.sample(range(available), count)


class UnreadableSample(RuntimeError):
    """A training clip could not be read. The loader draws another batch."""


def unreadable(where, index, exc):
    return UnreadableSample(f"{where} frame {index}: {exc}")
