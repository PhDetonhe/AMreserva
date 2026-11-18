"""Valida apenas a integração YOLOv8/TACO, sem câmera real, rede ou banco.

Execute na raiz: python -m scripts.validate_yolov8_taco
As fotos do dataset são regressões funcionais, não avaliação independente.
"""

from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

from backend.detection.types import Detection
from backend.detection.yolov8_taco_detector import (
    COUNT_CATEGORIES, DEFAULT_CLASS_MAPPING, YoloV8TacoDetector,
)
from backend.monitoring import monitor_yolov8_taco as monitor
from backend.tracking.yolov8_presence_tracker import YoloV8PresenceTracker

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "demo" / "yolov8_taco"


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = monitor.create_detector(monitor.load_config(monitor.CONFIG_FILE))
        cls.images, cls.results = {}, {}
        for name in ("garrafa", "latinha", "papelao", "saco_de_lixo", "papel"):
            image = cv2.imdecode(np.fromfile(str(FIXTURES / f"{name}.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
            cls.images[name] = image
            cls.results[name] = cls.detector.detect(image)

    def test_real_bottle(self):
        self.assertTrue(any(d.class_name == "bottle" for d in self.results["garrafa"]))

    def test_real_can(self):
        self.assertTrue(any(d.class_name == "can" for d in self.results["latinha"]))

    def test_real_cardboard_uses_carton(self):
        self.assertTrue(any(d.class_name == "carton" for d in self.results["papelao"]))

    def test_real_garbage_bag_maps_to_plastic(self):
        self.assertTrue(any(d.class_name == "plastic" for d in self.results["saco_de_lixo"]))

    def test_real_paper(self):
        self.assertTrue(any(d.class_name == "paper" for d in self.results["papel"]))

    def test_boxes_are_in_capture_coordinates(self):
        for name, detections in self.results.items():
            height, width = self.images[name].shape[:2]
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                self.assertTrue(0 <= x1 < x2 < width)
                self.assertTrue(0 <= y1 < y2 < height)
                self.assertIn(detection.class_name, COUNT_CATEGORIES)
                self.assertGreaterEqual(detection.confidence, 0.30)

    def test_mapping_does_not_invent_a_category_for_other_waste(self):
        self.assertEqual(len(DEFAULT_CLASS_MAPPING), 37)
        for name in ("Battery", "Cigarette", "Broken glass", "Food waste", "Plastic utensils"):
            self.assertNotIn(name, self.detector.class_mapping)
        self.assertEqual(DEFAULT_CLASS_MAPPING["Garbage bag"], "plastic")
        self.assertEqual(DEFAULT_CLASS_MAPPING["Corrugated carton"], "carton")
        self.assertNotIn("card", set(DEFAULT_CLASS_MAPPING.values()))

    def test_only_selected_output_groups_are_returned(self):
        detector = YoloV8TacoDetector(allowed_classes=("can",))
        self.assertEqual(detector.detect(self.images["garrafa"]), [])
        self.assertTrue(any(d.class_name == "can" for d in detector.detect(self.images["latinha"])))

    def test_letterbox_coordinates_and_cross_class_duplicate_suppression(self):
        output = np.zeros((1, 64, 3), dtype=np.float32)
        output[0, :4, :] = np.array([[320, 320, 100], [320, 320, 100],
                                     [320, 320, 50], [180, 180, 50]], dtype=np.float32)
        output[0, 4 + 12, 0] = 0.9
        output[0, 4 + 38, 1] = 0.8  # mesma caixa, classe diferente
        output[0, 4 + 1, 2] = 0.99  # pilha sem grupo: ignorada
        detections = self.detector._decode(output, 1280, 720, (0.5, 0.5, 0, 140))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "can")
        self.assertEqual(detections[0].bbox, (320, 180, 960, 540))

    def test_rejects_invalid_input_threshold_and_mapping(self):
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((20, 20), dtype=np.uint8))
        with self.assertRaises(ValueError):
            YoloV8TacoDetector(confidence_threshold=float("nan"))
        with self.assertRaises(ValueError):
            YoloV8TacoDetector(class_mapping={"unknown class": "plastic"})
        with self.assertRaises(ValueError):
            YoloV8TacoDetector(allowed_classes=("card",))


class MonitorChecks(unittest.TestCase):
    def _simulate(self, folder, local_only, frames=3):
        config = monitor.load_config(monitor.CONFIG_FILE)
        config["counts_path"] = str(folder / "counts.json")
        config["station_document"]["path"] = str(folder / "station.json")
        clients = []
        events = []

        class FakeClient:
            def __init__(self, settings):
                self.closed = False
                self.station_checked = None
                clients.append(self)

            def ensure_station_is_available(self, station_id):
                self.station_checked = station_id

            def send_detection(self, event):
                events.append(event)

            def close(self):
                self.closed = True

        class FakeCamera:
            def __init__(self):
                self.frames = 0
                self.closed = False

            def read(self, *, copy=True):
                self.frames += 1
                if self.frames > frames:
                    raise KeyboardInterrupt
                return np.zeros((100, 640, 3), dtype=np.uint8)

            def close(self):
                self.closed = True

        class FakeDetector:
            def __init__(self):
                self.frames = 0

            def detect(self, frame):
                self.frames += 1
                y = 40  # Todos ficam parados: a contagem independe de movimento/linha.
                return [Detection(name, 0.9, (40 + i * 100, y - 5, 60 + i * 100, y + 5))
                        for i, name in enumerate(COUNT_CATEGORIES)]

        camera = FakeCamera()
        with patch.object(monitor, "create_detector", return_value=FakeDetector()), \
             patch.object(monitor, "BackendClient", FakeClient), \
             patch.object(monitor.WebcamConfig, "open_camera", return_value=object()), \
             patch.object(monitor, "LatestFrameCamera", return_value=camera), \
             patch.object(monitor, "time", SimpleNamespace(monotonic=iter([0.0, 1.0, 2.0]).__next__)), \
             redirect_stdout(io.StringIO()):
            monitor.run(config, display=False, local_only=local_only)
        self.assertTrue(camera.closed)
        counts = monitor.load_counts(folder / "counts.json", config["station_id"])
        self.assertEqual(counts, dict.fromkeys(COUNT_CATEGORIES, 1))
        document = json.loads((folder / "station.json").read_text("utf-8"))
        self.assertEqual(document["status"], "offline")
        self.assertEqual(document["detections"], 5)
        return clients, events, config

    def test_stationary_objects_are_counted_once_and_sent_with_the_five_categories(self):
        with tempfile.TemporaryDirectory() as folder:
            clients, events, config = self._simulate(Path(folder), local_only=False)
        self.assertEqual(len(events), 5)
        self.assertEqual({event["detection_type"] for event in events}, set(COUNT_CATEGORIES))
        self.assertEqual(len({event["event_id"] for event in events}), 5)
        for event in events:
            uuid.UUID(event["event_id"])
            self.assertEqual(event["station_id"], config["station_id"])
            self.assertIsNone(event["direction"])
            self.assertIsNotNone(datetime.fromisoformat(event["detected_at"]).tzinfo)
        self.assertEqual(clients[0].station_checked, config["station_id"])
        self.assertTrue(clients[0].closed)

    def test_first_frame_alone_is_enough_to_count_and_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            _, events, _ = self._simulate(Path(folder), local_only=False, frames=1)
        self.assertEqual(len(events), 5)

    def test_active_defaults_use_moderate_confidence_and_faster_sampling(self):
        config = monitor.load_config(monitor.CONFIG_FILE)
        self.assertEqual(config["yolov8_taco"]["confidence_threshold"], 0.30)
        self.assertEqual(config["detection_interval_ms"], 100)
        self.assertEqual(config["tracking"]["max_missing_frames"], 12)

    def test_local_only_never_creates_a_network_client(self):
        with tempfile.TemporaryDirectory() as folder:
            clients, events, _ = self._simulate(Path(folder), local_only=True)
        self.assertEqual(clients, [])
        self.assertEqual(events, [])

    def test_default_state_is_separate_from_existing_monitors(self):
        config = monitor.load_config(monitor.CONFIG_FILE)
        self.assertEqual(Path(config["counts_path"]), ROOT / "backend/data/yolov8_taco/counts.json")
        self.assertEqual(Path(config["station_document"]["path"]), ROOT / "backend/data/yolov8_taco/station.json")
        self.assertNotIn("detection", config)
        self.assertNotIn("ssd_mobilenet", config)

    def test_environment_and_relative_paths(self):
        with patch.dict(os.environ, {"AQUADETECTOR_API_URL": "http://localhost:8123/", "BOTTLE_COUNT_STATION_ID": "7"}):
            config = monitor.load_config(monitor.CONFIG_FILE)
        self.assertEqual(config["station_id"], 7)
        self.assertEqual(config["backend"]["base_url"], "http://localhost:8123")
        self.assertTrue(Path(config["yolov8_taco"]["model_path"]).is_file())

    def test_invalid_counts_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "counts.json"
            monitor.save_counts(path, 1, dict.fromkeys(COUNT_CATEGORIES, 0))
            with self.assertRaises(RuntimeError):
                monitor.load_counts(path, 2)
            path.write_text('{"station_id": 1, "counts": {"can": -1}}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                monitor.load_counts(path, 1)

    def test_new_entrypoint_does_not_import_the_legacy_detectors(self):
        code = ("import sys; import backend.monitoring.monitor_yolov8_taco; "
                "assert 'backend.detection.yolo_detector' not in sys.modules; "
                "assert 'backend.detection.ssd_mobilenet_detector' not in sys.modules; "
                "assert 'ultralytics' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PresenceChecks(unittest.TestCase):
    @staticmethod
    def detection(x=20, category="bottle"):
        return Detection(category, 0.8, (x, 20, x + 40, 100))

    def test_first_appearance_counts_immediately_and_motion_does_not_recount(self):
        tracker = YoloV8PresenceTracker()
        events = tracker.update([self.detection()])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].class_name, "bottle")
        self.assertEqual(tracker.update([self.detection(x=30)]), [])
        self.assertEqual(tracker.update([self.detection(x=50)]), [])

    def test_brief_detector_dropouts_do_not_create_a_new_count(self):
        tracker = YoloV8PresenceTracker(max_missing_frames=12)
        tracker.update([self.detection()])
        for _ in range(8):
            self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([self.detection()]), [])

    def test_return_after_expiry_can_count_again(self):
        tracker = YoloV8PresenceTracker(max_missing_frames=12)
        first_id = tracker.update([self.detection()])[0].track_id
        for _ in range(13):
            tracker.update([])
        events = tracker.update([self.detection()])
        self.assertEqual(len(events), 1)
        self.assertNotEqual(events[0].track_id, first_id)

    def test_class_flicker_does_not_count_the_same_box_twice(self):
        tracker = YoloV8PresenceTracker()
        tracker.update([self.detection(category="bottle")])
        self.assertEqual(tracker.update([self.detection(category="plastic")]), [])
        self.assertEqual(next(iter(tracker.objects.values())).class_name, "bottle")

    def test_two_objects_have_separate_counts(self):
        tracker = YoloV8PresenceTracker()
        events = tracker.update([self.detection(x=20), self.detection(x=300)])
        self.assertEqual(len(events), 2)
        self.assertEqual(len({event.track_id for event in events}), 2)
        self.assertEqual(tracker.update([self.detection(x=30), self.detection(x=310)]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
