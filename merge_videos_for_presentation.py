r"""Concatenate CSV-selected videos with pain labels below the image.

Use --video_pairs LEFT RIGHT (repeatable) with --compare_mode to specify exact
pairs. Remaining pairs are selected randomly to cover the requested classes.
All clips or pairs play in ascending numeric pain-level order; equal levels
retain their selection order. --video_pairs cannot be combined with --video_names.
Both options accept CSV sample names, filenames, or exact sample_id strings
(leading zeros are preserved). Each value must identify exactly one video.

Example:
    python merge_videos_for_presentation.py \
        --root_folder /seidenas/users/fvilli/PartA/video --num_videos 3 \
        --csv_path partA/starting_point/plottable_subjects.csv --classes 0 2 4

Requires OpenCV with an avc1 encoder and NumPy. Output is silent.
Add --compare_mode for N sequential pairs (2N distinct videos), with matching
pain levels and different subjects in each pair. Named videos specify the left
side. Each pair lasts as long as its longer clip; the shorter side holds its
final frame. The image width doubles and one shared label appears underneath.
Add --compare_sequential with --compare_mode to play left then right in each
pair, dimming the inactive panel with a 70% black overlay. The waiting right
panel shows its first frame; the finished left panel holds its final frame.
Each pair then lasts the sum of both clip durations.
Use --video_speed 0.5 for half speed, 1 for original speed (default), or 2 for
double speed. The multiplier applies to all clips; output FPS stays unchanged.
"""

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Video:
    name: str
    subject: str
    class_id: str
    path: Path
    sample_id: str = ''


def load_videos(root: Path, csv_path: Path) -> list[Video]:
    """Load available CSV videos, preferring the dataset's subject layout."""
    if not root.is_dir():
        raise ValueError(f'Video root is not a directory: {root}')
    extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.m4v'}
    index = defaultdict(list)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in extensions:
            index[path.stem].append(path)
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        header = stream.readline()
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter='\t' if '\t' in header else ',')
        required = {'sample_name', 'subject_name', 'subject_id', 'class_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must contain columns: {", ".join(sorted(required))}')
        videos = []
        seen = {}
        rows = list(reader)
        subject_names = {(row.get('subject_name') or '').strip() for row in rows}
        for line, row in enumerate(rows, 2):
            if any(not (row.get(key) or '').strip() for key in required):
                raise ValueError(f'Missing CSV values on line {line}')
            name = row['sample_name'].strip()
            stem = Path(name).stem if Path(name).suffix.lower() in extensions else name
            matches = [p for p in index.get(stem, [])
                       if len(p.relative_to(root).parts) == 1
                       or p.relative_to(root).parts[0] not in subject_names
                       or p.relative_to(root).parts[0] == row['subject_name'].strip()]
            preferred = [p for p in matches if p.parent == root / row['subject_name'].strip()]
            matches = preferred or matches
            if len(matches) > 1:
                raise ValueError(f'Ambiguous video name in CSV: {name}')
            if not matches:
                continue
            video = Video(stem, row['subject_id'].strip(), row['class_id'].strip(), matches[0],
                          sample_id=(row.get('sample_id') or '').strip())
            if video.path in seen and seen[video.path] != video:
                raise ValueError(f'Conflicting CSV metadata for {name}')
            if video.path not in seen:
                videos.append(video)
                seen[video.path] = video
    return videos


def resolve_video(videos, name):
    matches = [v for v in videos if name in (v.name, v.path.name)
               or (v.sample_id and name == v.sample_id)]
    if len(matches) != 1:
        raise ValueError(f'Video name, filename, or sample_id must resolve to one available CSV video: {name}')
    return matches[0]


def select_videos(videos, count, names, classes, seed, used_subjects=()):
    """Cover requested classes and maximize distinct subjects, honoring names."""
    if count < 1:
        raise ValueError('num_videos must be positive')
    classes = list(dict.fromkeys(classes))
    selected = []
    for name in names:
        video = resolve_video(videos, name)
        if video in selected:
            raise ValueError(f'Duplicate requested video: {name}')
        if classes and video.class_id not in classes:
            raise ValueError(f'Requested video {name} is outside the selected classes')
        selected.append(video)
    if len(selected) > count:
        raise ValueError('More video names than num_videos')
    pool = [v for v in videos if (not classes or v.class_id in classes) and v not in selected]
    if len(pool) + len(selected) < count:
        raise ValueError('Not enough available videos for the requested count/classes')
    uncovered = [c for c in classes if c not in {v.class_id for v in selected}]
    for c in uncovered:
        if not any(v.class_id == c for v in pool):
            raise ValueError(f'No available videos for class {c}')
    if len(uncovered) > count - len(selected):
        raise ValueError('Not enough remaining slots to cover every requested class')
    rng = random.Random(seed)
    rng.shuffle(pool)
    slots = uncovered + [None] * (count - len(selected) - len(uncovered))
    used_subjects = set(used_subjects) | {v.subject for v in selected}
    candidates = [[v for v in pool if c is None or v.class_id == c] for c in slots]
    subjects = [list(dict.fromkeys(v.subject for v in group if v.subject not in used_subjects))
                for group in candidates]
    # Maximum bipartite matching avoids greedy choices that unnecessarily reuse subjects.
    owners = {}

    def assign(slot, visited):
        for subject in subjects[slot]:
            if subject in visited:
                continue
            visited.add(subject)
            if subject not in owners or assign(owners[subject], visited):
                owners[subject] = slot
                return True
        return False

    for slot in range(len(slots)):
        assign(slot, set())
    assigned = {slot: next(v for v in candidates[slot] if v.subject == subject)
                for subject, slot in owners.items()}
    reserved = set(assigned.values())
    for slot, group in enumerate(candidates):
        if slot not in assigned:
            assigned[slot] = next(v for v in group if v not in reserved)
            reserved.add(assigned[slot])
    return selected + [assigned[slot] for slot in range(len(slots))]


def select_partners(videos, left, seed, used_subjects=()):
    """Match same-class partners, preferring subjects absent from the left side."""
    left_paths = {v.path for v in left}
    left_subjects = set(used_subjects) | {v.subject for v in left}
    pool = [v for v in videos if v.path not in left_paths]
    random.Random(seed).shuffle(pool)
    pool.sort(key=lambda v: v.subject in left_subjects)
    candidates = [[v for v in pool if v.class_id == a.class_id and v.subject != a.subject]
                  for a in left]

    def assign(slot, visited, groups, owners):
        for candidate in groups[slot]:
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate not in owners or assign(owners[candidate], visited, groups, owners):
                owners[candidate] = slot
                return True
        return False

    # First match unused subjects across all classes; then fill any remaining
    # slots by matching individual videos, allowing subjects to recur.
    subjects = [list(dict.fromkeys(v.subject for v in group if v.subject not in left_subjects))
                for group in candidates]
    subject_owners = {}
    for slot in range(len(left)):
        assign(slot, set(), subjects, subject_owners)
    owners = {next(v for v in candidates[slot] if v.subject == subject): slot
              for subject, slot in subject_owners.items()}
    matched = set(owners.values())
    for slot in range(len(left)):
        if slot not in matched and not assign(slot, set(), candidates, owners):
            raise ValueError(f'Cannot assign distinct partner videos to all selected left clips; '
                             f'check class {left[slot].class_id} and available subjects')
    partners = {slot: video for video, slot in owners.items()}
    return [partners[slot] for slot in range(len(left))]


def select_pairs(videos, count, named_pairs, classes, seed):
    """Preserve explicit pairs and complete the remaining class coverage."""
    if count < 1 or len(named_pairs) > count:
        raise ValueError('num_videos must be positive and at least the number of fixed pairs')
    fixed_left, fixed_right = [], []
    reserved = set()
    for names in named_pairs:
        if len(names) != 2:
            raise ValueError('Each --video_pairs argument needs exactly two names')
        a, b = [resolve_video(videos, name) for name in names]
        if a.path == b.path or a.path in reserved or b.path in reserved:
            raise ValueError('A video cannot appear more than once across fixed pairs')
        if a.class_id != b.class_id or a.subject == b.subject:
            raise ValueError('Each fixed pair needs the same class and two different subjects')
        if classes and a.class_id not in classes:
            raise ValueError(f'Fixed pair class {a.class_id} is outside the requested classes')
        reserved.update((a.path, b.path))
        fixed_left.append(a)
        fixed_right.append(b)
    uncovered = set(classes) - {v.class_id for v in fixed_left}
    if len(uncovered) > count - len(fixed_left):
        raise ValueError('Not enough remaining pair slots to cover every requested class')
    right_paths = {v.path for v in fixed_right}
    left = select_videos([v for v in videos if v.path not in right_paths], count,
                         [names[0] for names in named_pairs], classes, seed,
                         used_subjects={v.subject for v in fixed_right})
    reserved.update(v.path for v in left)
    random_right = select_partners([v for v in videos if v.path not in reserved],
                                   left[len(fixed_left):], seed,
                                   used_subjects={v.subject for v in left + fixed_right})
    return left, fixed_right + random_right


def render(videos, output: Path, partners=None, compare_sequential=False, video_speed=1.0):
    """Stream clips or side-by-side pairs with a separate shared pain footer."""
    if not math.isfinite(video_speed) or video_speed <= 0:
        raise ValueError('video_speed must be a finite positive number')
    if not videos:
        raise ValueError('No videos selected')
    if compare_sequential and partners is None:
        raise ValueError('Sequential comparison requires partners')
    if partners is not None:
        if len(partners) != len(videos):
            raise ValueError('Each left clip needs exactly one partner')
        if len({v.path for v in videos + partners}) != 2 * len(videos):
            raise ValueError('Comparison videos must not repeat')
        if any(a.class_id != b.class_id or a.subject == b.subject for a, b in zip(videos, partners)):
            raise ValueError('Each pair needs the same class and two different subjects')
    segments = list(zip(videos, partners)) if partners is not None else [(v,) for v in videos]
    segments.sort(key=lambda segment: float(segment[0].class_id))
    metadata = {}
    for video in [v for segment in segments for v in segment]:
        cap = cv2.VideoCapture(str(video.path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or frames < 1 or min(width, height) < 1:
                raise ValueError(f'Cannot read valid video metadata: {video.path}')
            metadata[video] = (fps, int(frames), width, height)
        finally:
            cap.release()
    output_fps, _, width, height = metadata[videos[0]]
    width += width % 2
    height += height % 2
    footer = max(48, round(height * 0.09))
    footer += footer % 2
    output_width = width * len(segments[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a previous presentation.
    with output.open('xb'):
        pass
    writer = None
    try:
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'avc1'),
                                 output_fps, (output_width, height + footer))
        if not writer.isOpened():
            raise ValueError('Cannot open avc1 encoder; install OpenCV with H.264 encoding support')
        for segment in segments:
            names = ' | '.join(f'{v.name} (subject {v.subject}, sample_id {v.sample_id or "N/A"})'
                               for v in segment)
            print(f'Rendering {names} | Pain level: {segment[0].class_id}', flush=True)
            canvas = np.zeros((height + footer, output_width, 3), dtype=np.uint8)
            label = f'Pain level: {segment[0].class_id}'
            if video_speed != 1:
                label += f' | Speed: {video_speed:g}x'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = min(footer / 55, max(0.1, (output_width - 16) / cv2.getTextSize(label, font, 1, 2)[0][0]))
            (tw, th), baseline = cv2.getTextSize(label, font, scale, 2)
            cv2.putText(canvas, label, ((output_width - tw) // 2, height + (footer + th - baseline) // 2),
                        font, scale, (255, 255, 255), 2, cv2.LINE_AA)
            captures = []
            try:
                for video in segment:
                    cap = cv2.VideoCapture(str(video.path))
                    captures.append(cap)
                    if not cap.isOpened():
                        raise ValueError(f'Cannot open video: {video.path}')
                clip_frames = [max(1, round(metadata[v][1] / metadata[v][0] * output_fps / video_speed))
                               for v in segment]
                output_frames = sum(clip_frames) if compare_sequential else max(clip_frames)
                source_indices = [-1] * len(segment)
                source_frames = [None] * len(segment)
                for index in range(output_frames):
                    active_side = int(index >= clip_frames[0]) if compare_sequential else None
                    canvas[:height] = 0
                    for side, (video, cap) in enumerate(zip(segment, captures)):
                        fps, frames, _, _ = metadata[video]
                        local_index = index
                        if compare_sequential and side == 1:
                            local_index = max(0, index - clip_frames[0])
                        target = min(frames - 1, int(local_index * fps / output_fps * video_speed))
                        if compare_sequential and side == 0 and active_side == 1:
                            target = frames - 1
                        while source_indices[side] < target:
                            ok, frame = cap.read()
                            if not ok:
                                raise ValueError(f'Could not decode frame {source_indices[side] + 1}: {video.path}')
                            source_indices[side] += 1
                            source_frames[side] = frame
                        frame = source_frames[side]
                        ratio = min(width / frame.shape[1], height / frame.shape[0])
                        size = (max(1, round(frame.shape[1] * ratio)), max(1, round(frame.shape[0] * ratio)))
                        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
                        if compare_sequential and side != active_side:
                            resized = cv2.convertScaleAbs(resized, alpha=0.3)
                        x, y = side * width + (width - size[0]) // 2, (height - size[1]) // 2
                        canvas[y:y + size[1], x:x + size[0]] = resized
                    writer.write(canvas)
            finally:
                for cap in captures:
                    cap.release()
        writer.release()
        writer = None
        check = cv2.VideoCapture(str(output))
        try:
            if not check.read()[0]:
                raise ValueError('Encoded output is not readable')
        finally:
            check.release()
    except BaseException:
        if writer is not None:
            writer.release()
        output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root_folder', type=Path, required=True)
    parser.add_argument('--num_videos', type=int, required=True, help='Number of clips, or pairs in compare mode')
    parser.add_argument('--csv_path', type=Path, required=True)
    named = parser.add_mutually_exclusive_group()
    named.add_argument('--video_names', nargs='+', default=[], help='CSV sample names, filenames, or sample IDs; partial lists allowed; left clips in compare mode')
    named.add_argument('--video_pairs', nargs=2, action='append', default=[], metavar=('LEFT', 'RIGHT'),
                       help='Fixed comparison pair using sample names, filenames, or sample IDs; repeat for multiple pairs; requires --compare_mode')
    parser.add_argument('--compare_mode', action='store_true', help='Show same-class clips from different subjects side by side')
    parser.add_argument('--compare_sequential', action='store_true',
                        help='Requires --compare_mode: play left then right, dimming the inactive panel')
    parser.add_argument('--classes', nargs='+', default=[], help='Allowed class IDs; include at least one video per class')
    parser.add_argument('--seed', type=int, help='Random selection seed; generated and printed if omitted')
    parser.add_argument('--dataset_name', help='Defaults to the CSV grandparent directory name')
    parser.add_argument('--output_folder', type=Path, default=Path('.'))
    parser.add_argument('--video_speed', type=float, default=1.0,
                        help='Positive playback multiplier: 0.5 half speed, 1 original, 2 double speed')
    args = parser.parse_args()
    if not math.isfinite(args.video_speed) or args.video_speed <= 0:
        parser.error('--video_speed must be a finite positive number')
    if args.compare_sequential and not args.compare_mode:
        parser.error('--compare_sequential requires --compare_mode')
    if args.video_pairs and not args.compare_mode:
        parser.error('--video_pairs requires --compare_mode')
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    dataset = args.dataset_name or args.csv_path.resolve().parent.parent.name
    safe = lambda value: re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('.-') or 'dataset'
    classes = list(dict.fromkeys(args.classes))
    output = args.output_folder / f'{safe(dataset)}_video_presentation' / (
        f'merged_video_n{args.num_videos}_c{safe("-".join(classes)) if classes else "all"}_s{seed}'
        f'{"_compare" if args.compare_mode else ""}'
        f'{"_sequential" if args.compare_sequential else ""}'
        f'{"_speed" + str(args.video_speed).removesuffix(".0") if args.video_speed != 1 else ""}.mp4')
    try:
        videos = load_videos(args.root_folder, args.csv_path)
        if args.video_pairs:
            selected, partners = select_pairs(videos, args.num_videos, args.video_pairs, classes, seed)
        else:
            selected = select_videos(videos, args.num_videos, args.video_names, classes, seed)
            partners = select_partners(videos, selected, seed) if args.compare_mode else None
        print(f'Seed: {seed}', flush=True)
        repeated = [subject for subject, count in Counter(v.subject for v in selected + (partners or [])).items() if count > 1]
        if repeated:
            print(f'Subject reuse required by selection constraints: {", ".join(repeated)}')
        render(selected, output, partners=partners, compare_sequential=args.compare_sequential,
               video_speed=args.video_speed)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Output: {output.resolve()}')


if __name__ == '__main__':
    main()
