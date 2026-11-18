"""Caminho independente YOLOv8/TACO: contagem imediata por objeto e publicação HTTP."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

import cv2

from backend.camera.image_processing import prepare_frame, restore_coordinates
from backend.camera.webcam_config import LatestFrameCamera, WebcamConfig
from backend.communication.backend_client import BackendClient, BackendConfig
from backend.detection.yolov8_taco_detector import COUNT_CATEGORIES, YoloV8TacoDetector
from backend.station.station_document import StationDocument
from backend.tracking.yolov8_presence_tracker import YoloV8PresenceTracker


ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "aquamonitor-yolov8-taco.json"


def load_config(path: str | Path) -> dict:
    """Lê apenas a configuração do novo caminho; não carrega SSD/YOLOv26."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("A configuração deve ser um objeto JSON.")
    backend = config["backend"]
    backend["base_url"] = os.getenv("AQUADETECTOR_API_URL", os.getenv(
        "BOTTLE_COUNT_API_URL", backend["base_url"])).rstrip("/")
    if "BOTTLE_COUNT_STATION_ID" in os.environ:
        config["station_id"] = int(os.environ["BOTTLE_COUNT_STATION_ID"])
    if type(config["station_id"]) is not int or config["station_id"] <= 0:
        raise ValueError("station_id deve ser inteiro positivo.")
    url = urlparse(backend["base_url"])
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("backend.base_url deve ser uma URL HTTP válida.")
    if type(backend["require_connection_on_startup"]) is not bool:
        raise ValueError("backend.require_connection_on_startup deve ser booleano.")
    for name, value in [("detection_interval_ms", config["detection_interval_ms"]),
                         ("backend.timeout_seconds", backend["timeout_seconds"]),
                         ("tracking.max_distance", config["tracking"]["max_distance"])]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} deve ser positivo e finito.")
    for name in ("width", "height", "fps"):
        value = config["camera"][name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"camera.{name} deve ser inteiro positivo.")
    if type(config["camera"]["device_index"]) is not int or config["camera"]["device_index"] < 0:
        raise ValueError("camera.device_index deve ser inteiro não negativo.")
    for section in ("display", "image_processing"):
        if type(config[section]["enabled"]) is not bool:
            raise ValueError(f"{section}.enabled deve ser booleano.")
    for key in ("max_width", "max_height"):
        if type(config["image_processing"][key]) is not int or config["image_processing"][key] <= 0:
            raise ValueError(f"image_processing.{key} deve ser inteiro positivo.")
    tracking = config["tracking"]
    if type(tracking["max_missing_frames"]) is not int or tracking["max_missing_frames"] < 0:
        raise ValueError("tracking.max_missing_frames deve ser inteiro não negativo.")
    for section, fields in (("station_document", ("path",)),
                            ("yolov8_taco", ("model_path", "metadata_path"))):
        for field in fields:
            config[section][field] = str((path.parent / config[section][field]).resolve())
    config["counts_path"] = str((path.parent / config["counts_path"]).resolve())
    return config


def create_detector(config: dict) -> YoloV8TacoDetector:
    settings = config["yolov8_taco"]
    return YoloV8TacoDetector(
        model_path=settings["model_path"], metadata_path=settings["metadata_path"],
        confidence_threshold=settings["confidence_threshold"], nms_threshold=settings["nms_threshold"],
        allowed_classes=settings["detection_types"], class_mapping=settings["class_mapping"],
    )


def load_counts(path: Path, station_id: int) -> dict[str, int]:
    counts = dict.fromkeys(COUNT_CATEGORIES, 0)
    if not path.exists():
        return counts
    try:
        saved = json.loads(path.read_text(encoding="utf-8-sig"))
        if saved["station_id"] != station_id:
            raise ValueError("O arquivo de contagens pertence a outra estação; configure counts_path próprio.")
        for name in counts:
            value = saved["counts"].get(name, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"Contagem inválida: {name}.")
            counts[name] = value
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f"Contagens YOLOv8/TACO inválidas em {path}: {error}") from error
    return counts


def save_counts(path: Path, station_id: int, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"station_id": station_id, "detector": "yolov8_taco", "counting_mode": "first_detection", "counts": counts,
                "detection_types": list(COUNT_CATEGORIES),
                "updated_at": datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def draw_overlay(frame, detections, counts) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(frame, f"{detection.class_name} {detection.confidence:.0%}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    labels = [f"Lixo geral: {sum(counts.values())}"]
    labels.extend(f"{name}: {counts[name]}" for name in COUNT_CATEGORIES)
    for row, text in enumerate(labels, start=1):
        cv2.putText(frame, text, (8, row * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, text, (8, row * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run(config: dict, display: bool | None = None, local_only: bool = False) -> None:
    if display is None:
        display = config["display"]["enabled"]
    station_id = config["station_id"]
    counts_path = Path(config["counts_path"])
    counts = load_counts(counts_path, station_id)
    detector = create_detector(config)
    settings = config["backend"]
    interval = config["detection_interval_ms"] / 1000
    tracking_settings = config["tracking"]
    station_settings = config["station_document"]
    station = StationDocument(Path(station_settings["path"]), station_id, station_settings)
    window = "AquaMonitor - YOLOv8 TACO"
    with ExitStack() as resources:
        client = None
        sender = None
        if not local_only:
            client = BackendClient(BackendConfig(settings["base_url"], settings["detections_path"],
                                                 settings["timeout_seconds"]))
            resources.callback(client.close)
            if settings["require_connection_on_startup"]:
                client.ensure_station_is_available(station_id)
        camera = LatestFrameCamera(WebcamConfig(**config["camera"]).open_camera())
        resources.callback(camera.close)
        if display:
            resources.callback(cv2.destroyAllWindows)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        station.update(detections=sum(counts.values()), status="online")
        resources.callback(lambda: station.update(detections=sum(counts.values()), status="offline"))
        resources.callback(lambda: save_counts(counts_path, station_id, counts))
        if not local_only:
            sender = resources.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="http-yolov8"))
        pending = deque()
        tracker = None
        frame_shape = None
        next_detection = 0.0
        detections = []
        mode = "somente local (sem envio)" if local_only else "publicação HTTP habilitada"
        print(f"Monitor YOLOv8/TACO iniciado; {mode}; intervalo mínimo {interval * 1000:g} ms.")
        print("Contagem na primeira detecção de cada objeto; Q/ESC ou Ctrl+C para sair.")
        try:
            while True:
                while pending and pending[0].done():
                    pending.popleft().result()
                frame = camera.read(copy=display)
                if frame.shape[:2] != frame_shape:
                    frame_shape = frame.shape[:2]
                    tracker = YoloV8PresenceTracker(
                        max_distance=tracking_settings["max_distance"],
                        max_missing_frames=tracking_settings["max_missing_frames"],
                    )
                    detections = []
                    next_detection = 0.0
                now = time.monotonic()
                if now >= next_detection:
                    next_detection = now + interval
                    processed = prepare_frame(frame, config["image_processing"])
                    detections = restore_coordinates(detector.detect(processed), processed.shape, frame.shape)
                    first_detections = tracker.update(detections)
                    for detected in first_detections:
                        counts[detected.class_name] += 1
                    if first_detections:
                        save_counts(counts_path, station_id, counts)
                    for detected in first_detections:
                        if client is not None:
                            if len(pending) >= 32:
                                raise RuntimeError("Backend lento: limite de 32 envios pendentes atingido.")
                            event = {"event_id": str(uuid.uuid4()), "station_id": station_id,
                                     "detection_type": detected.class_name, "confidence": detected.confidence,
                                     "track_id": detected.track_id, "direction": None,
                                     "detected_at": datetime.now(timezone.utc).isoformat()}
                            pending.append(sender.submit(client.send_detection, event))
                        print(f"Contado YOLOv8/TACO: {detected.class_name} (track #{detected.track_id})")
                if display:
                    draw_overlay(frame, detections, counts)
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            for future in pending:
                future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=os.getenv("AQUAMONITOR_YOLOV8_CONFIG", CONFIG_FILE))
    parser.add_argument("--local-only", action="store_true", help="Conta localmente, sem conectar/publicar na API.")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--display", dest="display", action="store_true", help="Exibe vídeo e contadores.")
    preview.add_argument("--no-display", dest="display", action="store_false", help="Executa sem janela.")
    parser.set_defaults(display=None)
    args = parser.parse_args()
    run(load_config(args.config), display=args.display, local_only=args.local_only)


if __name__ == "__main__":
    main()
