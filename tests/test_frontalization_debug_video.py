import importlib.util
from functools import lru_cache
from pathlib import Path
import sys
import types

import cv2
import numpy as np


@lru_cache(maxsize=1)
def load_face_extractor():
    # Import the plotting code without custom.tools' unrelated multiprocessing setup.
    spec = importlib.util.spec_from_file_location(
        '_face_extractor_debug_test', Path(__file__).resolve().parents[1] / 'custom/faceExtractor.py')
    module = importlib.util.module_from_spec(spec)
    original_tools = sys.modules.get('custom.tools')
    sys.modules['custom.tools'] = types.ModuleType('custom.tools')
    try:
        spec.loader.exec_module(module)
    finally:
        if original_tools is None:
            del sys.modules['custom.tools']
        else:
            sys.modules['custom.tools'] = original_tools
    return module.FaceExtractor


def test_plot_landmarks_accepts_normalized_points_outside_frame():
    extractor = load_face_extractor().__new__(load_face_extractor())
    landmarks = np.array([[-0.1, 0.1], [0.5, 0.1], [0.1, 0.5]])

    image, _, _ = extractor.plot_landmarks_triangulation(
        np.zeros((32, 32, 3), dtype=np.uint8), landmarks)

    assert image.any()


def test_debug_video_contains_every_frame_at_source_fps(tmp_path):
    FaceExtractor = load_face_extractor()
    extractor = FaceExtractor.__new__(FaceExtractor)
    frames = [np.full((32, 32, 3), value, dtype=np.uint8) for value in (40, 180)]
    landmarks = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])

    extractor._get_list_frame = lambda *args, **kwargs: (frames, [0, 1], 7.5)
    extractor.extract_facial_landmarks = lambda *args: [landmarks, landmarks]
    extractor.compute_rigid_transform = lambda *args: (None, None)
    extractor.apply_rigid_transform = lambda rotation, translation, points: points.T
    extractor._get_frontalized_img = lambda **kwargs: kwargs['orig_frame']
    extractor.post_process_frontalized_img = lambda **kwargs: kwargs['frontalized_img']
    extractor.plot_landmarks_triangulation = lambda **kwargs: (
        np.zeros((32, 32, 3), dtype=np.uint8), (0, 0), (32, 32))

    extractor.frontalized_video(
        video_path='clip.mp4', ref_landmarks=landmarks, stabilize=False,
        plot_debug=True, plot_video=True, plot_every=30,
        plot_output_dir=str(tmp_path))

    output = tmp_path / 'clip.mp4'
    assert output.exists()
    assert list(tmp_path.glob('*.png')) == []
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert abs(capture.get(cv2.CAP_PROP_FPS) - 7.5) < 0.1
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    ok, frame = capture.read()
    assert ok and frame.shape[0] > 32 and frame.shape[1] > 32
    capture.release()

    png_dir = tmp_path / 'pngs'
    extractor.frontalized_video(
        video_path='clip.mp4', ref_landmarks=landmarks, stabilize=False,
        plot_debug=True, plot_every=30, plot_output_dir=str(png_dir))
    assert [path.name for path in png_dir.iterdir()] == ['clip_frame0.png']
