import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

import cv2
import numpy as np

from merge_videos_for_presentation import Video, load_videos, select_videos, select_partners, select_pairs, render


class PresentationTests(unittest.TestCase):
    def test_video_speed_controls_sampling_duration_and_pair_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = []
            for side, (fps, frames) in enumerate([(10, 10), (20, 40)]):
                path = root / f'{side}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, (160, 120))
                self.assertTrue(writer.isOpened())
                for index in range(frames):
                    value = 100 + 10 * index if side == 0 else 80 + 3 * index
                    writer.write(np.full((120, 160, 3), value, np.uint8))
                writer.release()
                clips.append(Video(str(side), str(side), '2', path, sample_id='001' if side == 0 else ''))
            for speed in [0.5, 1, 2]:
                for mode, normal_frames in [('single', 10), ('compare', 20), ('sequential', 30)]:
                    with self.subTest(speed=speed, mode=mode):
                        output = root / f'{mode}_{speed}.mp4'
                        log = io.StringIO()
                        with redirect_stdout(log):
                            render([clips[0]], output, partners=None if mode == 'single' else [clips[1]],
                                   compare_sequential=mode == 'sequential', video_speed=speed)
                        names = '0 (subject 0, sample_id 001)'
                        if mode != 'single':
                            names += ' | 1 (subject 1, sample_id N/A)'
                        self.assertEqual(log.getvalue(), f'Rendering {names} | Pain level: 2\n')
                        cap = cv2.VideoCapture(str(output))
                        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
                        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), normal_frames / speed)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(2 / speed))
                        ok, frame = cap.read()
                        self.assertTrue(ok)
                        self.assertAlmostEqual(frame[60, 80].mean(), 120, delta=10)
                        if mode == 'sequential':
                            self.assertAlmostEqual(frame[60, 240].mean(), 24, delta=10)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(12 / speed))
                            ok, frame = cap.read()
                            self.assertTrue(ok)
                            self.assertAlmostEqual(frame[60, 80].mean(), 57, delta=10)
                            self.assertAlmostEqual(frame[60, 240].mean(), 92, delta=10)
                        elif mode == 'compare':
                            self.assertAlmostEqual(frame[60, 240].mean(), 92, delta=10)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(16 / speed))
                            ok, frame = cap.read()
                            self.assertTrue(ok)
                            self.assertAlmostEqual(frame[60, 80].mean(), 190, delta=10)
                        cap.release()

    def test_invalid_video_speeds_fail_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'invalid.mp4'
            for speed in [0, -1, float('nan'), float('inf'), float('-inf')]:
                with self.subTest(speed=speed):
                    with self.assertRaisesRegex(ValueError, 'video_speed'):
                        render([Video('a', 'a', '0', Path('missing.mp4'))], output, video_speed=speed)
                    self.assertFalse(output.exists())
            script = str(Path(__file__).resolve().parents[1] / 'merge_videos_for_presentation.py')
            result = subprocess.run([sys.executable, script, '--root_folder', directory,
                                     '--csv_path', 'missing.csv', '--num_videos', '1', '--video_speed', '0'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--video_speed', result.stderr)
            self.assertIn('positive', result.stderr)

    def test_render_sorts_numeric_labels_stably_in_all_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = [], []
            for i, label in enumerate(['10', '2', '2']):
                for side, clips in enumerate([left, right]):
                    path = root / f'{i}_{side}.mp4'
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), 5, (160, 120))
                    self.assertTrue(writer.isOpened())
                    value = [60, 120, 200][i] + side * 30
                    for _ in range(2):
                        writer.write(np.full((120, 160, 3), value, np.uint8))
                    writer.release()
                    clips.append(Video(path.stem, path.stem, label, path))
            for mode in ['single', 'compare', 'sequential']:
                with self.subTest(mode=mode):
                    output = root / f'{mode}.mp4'
                    render(left, output, partners=None if mode == 'single' else right,
                           compare_sequential=mode == 'sequential')
                    cap = cv2.VideoCapture(str(output))
                    segment_frames = 4 if mode == 'sequential' else 2
                    self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), segment_frames * 3)
                    for position, value in enumerate([120, 200, 60]):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, position * segment_frames)
                        ok, frame = cap.read()
                        self.assertTrue(ok)
                        self.assertAlmostEqual(float(frame[60, 80].mean()), value, delta=10)
                        if mode != 'single':
                            if mode == 'sequential':
                                cap.set(cv2.CAP_PROP_POS_FRAMES, position * segment_frames + 2)
                                ok, frame = cap.read()
                                self.assertTrue(ok)
                            self.assertAlmostEqual(float(frame[60, 240].mean()), value + 30, delta=10)
                    cap.release()

    def test_fixed_pairs_preserve_order_and_complete_missing_classes(self):
        videos = [Video('a4', 'a', '4', Path('a4.mp4')),
                  Video('b4', 'b', '4', Path('b4.mp4')),
                  Video('a2', 'a', '2', Path('a2.mp4')),
                  Video('b2', 'b', '2', Path('b2.mp4')),
                  Video('c2', 'c', '2', Path('c2.mp4')),
                  Video('d2', 'd', '2', Path('d2.mp4'))]
        for seed in range(10):
            left, right = select_pairs(videos, 2, [['a4.mp4', 'b4']], ['2', '4'], seed)
            self.assertEqual((left[0], right[0]), (videos[0], videos[1]))
            self.assertEqual({left[1].subject, right[1].subject}, {'c', 'd'})
            self.assertEqual((left[1].class_id, right[1].class_id), ('2', '2'))
            self.assertEqual(len({v.path for v in left + right}), 4)
            self.assertEqual((left, right), select_pairs(videos, 2, [['a4.mp4', 'b4']], ['2', '4'], seed))
        left, right = select_pairs(videos, 2, [['b4', 'a4'], ['d2', 'c2']], ['2', '4'], 0)
        self.assertEqual([v.name for v in left], ['b4', 'd2'])
        self.assertEqual([v.name for v in right], ['a4', 'c2'])
        for count, pairs, classes in [(1, [['a4', 'b4']], ['2', '4']),
                                      (1, [['a4', 'b4'], ['c2', 'd2']], []),
                                      (1, [['a4', 'c2']], []),
                                      (1, [['a4', 'a4']], []),
                                      (1, [['a4', 'missing']], []),
                                      (1, [['a4', 'b4']], ['2']),
                                      (2, [['a4', 'b4'], ['b4', 'a4']], [])]:
            with self.subTest(count=count, pairs=pairs, classes=classes):
                with self.assertRaises(ValueError):
                    select_pairs(videos, count, pairs, classes, 0)
        same_subject = videos + [Video('a4other', 'a', '4', Path('a4other.mp4'))]
        with self.assertRaises(ValueError):
            select_pairs(same_subject, 1, [['a4', 'a4other']], [], 0)
        same_stem = videos + [Video('a4', 'other', '4', Path('a4.avi'))]
        left, right = select_pairs(same_stem, 1, [['a4.mp4', 'b4']], [], 0)
        self.assertEqual((left, right), ([videos[0]], [videos[1]]))

    def test_fixed_pair_cli_requires_compare_and_excludes_video_names(self):
        script = str(Path(__file__).resolve().parents[1] / 'merge_videos_for_presentation.py')
        base = [sys.executable, script, '--root_folder', '.', '--csv_path', 'missing.csv',
                '--num_videos', '1', '--video_pairs', 'a', 'b']
        result = subprocess.run(base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--video_pairs requires --compare_mode', result.stderr)
        result = subprocess.run(base + ['--compare_mode', '--video_names', 'a'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('not allowed with argument', result.stderr)

    def test_sequential_compare_plays_left_then_right_and_dims_frozen_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = []
            for i, (fps, seconds, channel) in enumerate([(10, 1, 2), (20, 2, 1),
                                                         (20, 1, 0), (5, 1, 2)]):
                path = root / f'{i}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, (160, 120))
                self.assertTrue(writer.isOpened())
                for n in range(fps * seconds):
                    frame = np.zeros((120, 160, 3), np.uint8)
                    frame[:, :, channel] = 80 if n == 0 else (240 if n == fps * seconds - 1 else 160)
                    writer.write(frame)
                writer.release()
                clips.append(Video(str(i), str(i), str(i // 2), path))
            output = root / 'sequential.mp4'
            render([clips[0], clips[2]], output, partners=[clips[1], clips[3]], compare_sequential=True)
            cap = cv2.VideoCapture(str(output))
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 50)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 320)
            # Output frame: left channel/value, right channel/value.
            for position, lc, lv, rc, rv in [(0, 2, 80, 1, 24), (5, 2, 160, 1, 24),
                                            (10, 2, 72, 1, 80), (15, 2, 72, 1, 160),
                                            (29, 2, 72, 1, 160), (30, 0, 80, 2, 24),
                                            (39, 0, 160, 2, 24), (40, 0, 72, 2, 80),
                                            (49, 0, 72, 2, 240)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertAlmostEqual(int(frame[60, 80, lc]), lv, delta=10)
                self.assertAlmostEqual(int(frame[60, 240, rc]), rv, delta=10)
                self.assertGreater(frame[120:].max(), 220)
            cap.release()

    def test_sequential_requires_compare_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'invalid.mp4'
            with self.assertRaisesRegex(ValueError, 'partners'):
                render([Video('a', 'a', '0', Path('missing.mp4'))], output, compare_sequential=True)
            self.assertFalse(output.exists())
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'merge_videos_for_presentation.py'),
                                     '--root_folder', directory, '--csv_path', 'missing.csv',
                                     '--num_videos', '1', '--compare_sequential'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--compare_sequential requires --compare_mode', result.stderr)

    def test_partners_match_classes_and_prefer_unused_subjects(self):
        left = [Video('a0', 'a', '0', Path('a0.mp4')),
                Video('b2', 'b', '2', Path('b2.mp4'))]
        pool = left + [Video('c0', 'c', '0', Path('c0.mp4')),
                       Video('d0', 'd', '0', Path('d0.mp4')),
                       Video('c2', 'c', '2', Path('c2.mp4')),
                       Video('a2', 'a', '2', Path('a2.mp4'))]
        for seed in range(20):
            partners = select_partners(pool, left, seed)
            self.assertEqual([v.name for v in partners], ['d0', 'c2'])
            self.assertEqual(partners, select_partners(pool, left, seed))
        named = select_videos(pool, 2, ['a0'], ['0', '2'], 4)
        self.assertEqual(named[0], left[0])
        for a, b in zip(named, select_partners(pool, named, 4)):
            self.assertEqual(a.class_id, b.class_id)
            self.assertNotEqual(a.subject, b.subject)

    def test_partner_matching_reuses_subjects_without_repeating_videos(self):
        left = [Video('a0', 'a', '0', Path('a0.mp4')),
                Video('b0', 'b', '0', Path('b0.mp4'))]
        pool = left + [Video('a1', 'a', '0', Path('a1.mp4')),
                       Video('c0', 'c', '0', Path('c0.mp4'))]
        for seed in range(20):
            # The first left clip must take c0, leaving a1 for the second.
            partners = select_partners(pool, left, seed)
            self.assertEqual([v.name for v in partners], ['c0', 'a1'])
            self.assertEqual(len(set(left + partners)), 4)
        for pool in [left, left + [Video('a1', 'a', '0', Path('a1.mp4'))]]:
            with self.assertRaises(ValueError):
                select_partners(pool, left, 0)

    def test_compare_render_holds_shorter_clip_and_doubles_width(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = []
            # Both shorter sides are exercised, with different sizes and FPS.
            for i, (size, fps, seconds, channel) in enumerate([
                ((160, 120), 10, 1, 2), ((80, 120), 20, 2, 1),
                ((160, 120), 10, 2, 0), ((160, 80), 5, 1, 2),
            ]):
                path = root / f'{i}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, size)
                self.assertTrue(writer.isOpened())
                for n in range(fps * seconds):
                    frame = np.zeros((size[1], size[0], 3), np.uint8)
                    frame[:, :, channel] = 240 if n == fps * seconds - 1 else 100
                    writer.write(frame)
                writer.release()
                clips.append(Video(str(i), str(i), str(i // 2), path))
            output = root / 'compare.mp4'
            render([clips[0], clips[2]], output, partners=[clips[1], clips[3]])
            cap = cv2.VideoCapture(str(output))
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 320)
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 40)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
            for position, left_channel, right_channel, frozen_side in [(15, 2, 1, 80), (35, 0, 2, 240)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertGreater(frame[60, 80, left_channel], 80)
                self.assertGreater(frame[60, 240, right_channel], 80)
                self.assertGreater(frame[60, frozen_side].max(), 220)
                self.assertGreater(frame[120:].max(), 180)
                self.assertLess(frame[120:, :10].mean(), 5)
            cap.release()

    def test_selection_covers_classes_without_unnecessary_subject_reuse(self):
        videos = [Video('a0', 'a', '0', Path('a0.mp4')),
                  Video('b0', 'b', '0', Path('b0.mp4')),
                  Video('a2', 'a', '2', Path('a2.mp4')),
                  Video('c4', 'c', '4', Path('c4.mp4'))]
        for seed in range(20):
            selected = select_videos(videos, 3, [], ['0', '2', '4'], seed)
            self.assertEqual({v.subject for v in selected}, {'a', 'b', 'c'})
            self.assertEqual({v.class_id for v in selected}, {'0', '2', '4'})
            self.assertEqual(selected, select_videos(videos, 3, [], ['0', '2', '4'], seed))
        selected = select_videos(videos, 4, ['a0.mp4'], ['0', '2', '4'], 3)
        self.assertEqual(selected[0].name, 'a0')
        self.assertEqual(len({v.path for v in selected}), 4)
        self.assertEqual(len({v.subject for v in selected}), 3)
        for count, names, classes in [(2, [], ['0', '2', '4']),
                                       (3, ['a0', 'b0'], ['0', '2', '4']),
                                       (1, ['a2'], ['0']), (5, [], []),
                                       (1, ['missing'], []), (1, [], ['9']),
                                       (2, ['a0', 'a0.mp4'], []), (0, [], [])]:
            with self.subTest(count=count, names=names, classes=classes):
                with self.assertRaises(ValueError):
                    select_videos(videos, count, names, classes, 1)

    def test_selection_accepts_exact_sample_ids_and_mixed_identifiers(self):
        videos = [Video('a', 'a', '2', Path('a.mp4'), sample_id='001'),
                  Video('b', 'b', '2', Path('b.mp4'), sample_id='2'),
                  Video('c', 'c', '2', Path('c.mp4')),
                  Video('d', 'd', '2', Path('d.mp4'), sample_id='d')]
        self.assertEqual(select_videos(videos, 4, ['001', '2', 'c.mp4', 'd'], [], 0), videos)
        self.assertEqual(select_pairs(videos, 1, [['001', '2']], [], 0),
                         ([videos[0]], [videos[1]]))
        for name in ('1', '002', '', 'missing'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                select_videos(videos, 1, [name], [], 0)
        with self.assertRaisesRegex(ValueError, 'Duplicate requested video'):
            select_videos(videos, 2, ['a.mp4', '001'], [], 0)
        with self.assertRaisesRegex(ValueError, 'outside the selected classes'):
            select_videos(videos, 1, ['001'], ['4'], 0)

    def test_sample_id_collisions_are_ambiguous(self):
        video = Video('clip', 'a', '2', Path('clip.mp4'), sample_id='001')
        collisions = [Video('other', 'b', '2', Path('other.mp4'), sample_id='001'),
                      Video('001', 'b', '2', Path('other.mp4')),
                      Video('other', 'b', '2', Path('001'))]
        for other in collisions:
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, 'resolve to one'):
                select_videos([video, other], 1, ['001'], [], 0)

    def test_csv_sample_ids_preserve_text_and_allow_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'clip.mp4').touch()
            csv_path = root / 'metadata.csv'
            for delimiter in (',', '\t'):
                for values, expected in [([], ''), ([''], ''), (['  '], ''),
                                         ([' 001 '], '001'), (['0'], '0')]:
                    with self.subTest(delimiter=delimiter, values=values):
                        with csv_path.open('w', newline='') as stream:
                            writer = csv.writer(stream, delimiter=delimiter)
                            header = ['sample_name', 'subject_name', 'subject_id', 'class_id']
                            writer.writerow(header + (['sample_id'] if values else []))
                            writer.writerow(['clip', 'subject', '1', '2'] + values)
                        self.assertEqual(load_videos(root, csv_path)[0].sample_id, expected)

    def test_csv_resolution_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'subject').mkdir()
            (root / 'subject' / 'clip.mp4').touch()
            for delimiter in (',', '\t'):
                csv_path = root / 'metadata.csv'
                with csv_path.open('w', newline='') as stream:
                    writer = csv.writer(stream, delimiter=delimiter)
                    writer.writerow(['sample_name', 'subject_name', 'subject_id', 'class_id'])
                    writer.writerow(['clip', 'subject', '1', '2'])
                videos = load_videos(root, csv_path)
                self.assertEqual(videos, [Video('clip', '1', '2', root / 'subject/clip.mp4')])
            (root / 'subject/clip.mp4').rename(root / 'clip.mp4')
            self.assertEqual(load_videos(root, csv_path)[0].path, root / 'clip.mp4')
            (root / 'other').mkdir()
            (root / 'other/clip.mp4').touch()
            with self.assertRaises(ValueError):
                load_videos(root, csv_path)

    def test_render_preserves_order_duration_and_separate_footer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for i, (size, fps, color) in enumerate([
                ((320, 240), 10, (0, 0, 255)),
                ((160, 240), 20, (0, 255, 0)),
                ((320, 180), 5, (255, 0, 0)),
            ]):
                path = root / f'{i}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, size)
                self.assertTrue(writer.isOpened(), 'avc1 encoder unavailable')
                for _ in range(fps):
                    writer.write(np.full((size[1], size[0], 3), color, dtype=np.uint8))
                writer.release()
                videos.append(Video(str(i), str(i), str(i), path))
            output = root / 'merged.mp4'
            render(videos, output)
            cap = cv2.VideoCapture(str(output))
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 30)
            for position, channel in [(5, 2), (15, 1), (25, 0)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertGreater(frame.shape[0], 240)
                self.assertGreater(frame[120, 160, channel], 220)
                self.assertGreater(frame[240:].max(), 180)
                self.assertLess(frame[240:, :10].mean(), 5)
            cap.release()
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                render(videos, output)
            self.assertEqual(output.read_bytes(), before)
            broken = root / 'broken.mp4'
            with self.assertRaises(ValueError):
                render(videos + [Video('missing', 'x', '0', root / 'absent.mp4')], broken)
            self.assertFalse(broken.exists())

    def test_missing_subject_file_does_not_use_another_known_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'b').mkdir()
            (root / 'b/clip.mp4').touch()
            csv_path = root / 'metadata.csv'
            csv_path.write_text('sample_name,subject_name,subject_id,class_id\n'
                                'clip,a,1,0\nclip,b,2,4\n')
            self.assertEqual(load_videos(root, csv_path),
                             [Video('clip', '2', '4', root / 'b/clip.mp4')])


if __name__ == '__main__':
    unittest.main()
