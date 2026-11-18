"""Adaptador independente YOLOv8/TACO em ONNX, sem importar os outros detectores."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from backend.detection.types import Detection


COUNT_CATEGORIES = ("can", "bottle", "paper", "plastic", "carton")
CLASS_GROUPS = {
    "can": ("Food Can", "Aerosol", "Drink can"),
    "bottle": ("Other plastic bottle", "Clear plastic bottle", "Glass bottle"),
    "paper": ("Paper cup", "Magazine paper", "Tissues", "Wrapping paper", "Normal paper",
              "Paper bag", "Plastified paper bag", "Paper straw"),
    "plastic": ("Plastic bottle cap", "Disposable plastic cup", "Foam cup", "Plastic lid",
                "Plastic film", "Six pack rings", "Garbage bag", "Other plastic wrapper",
                "Single-use carrier bag", "Polypropylene bag", "Crisp packet",
                "Disposable food container", "Foam food container", "Other plastic container",
                "Plastic straw", "Styrofoam piece"),
    "carton": ("Toilet tube", "Other carton", "Egg carton", "Drink carton", "Corrugated carton",
             "Meal carton", "Pizza box"),
}
DEFAULT_CLASS_MAPPING = {name: category for category, names in CLASS_GROUPS.items() for name in names}
MODEL_DIR = Path(__file__).resolve().parent / "models" / "yolov8_taco"


def _threshold(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} deve ser finito, maior que zero e no máximo 1.")
    return value


class YoloV8TacoDetector:
    """Devolve Detection com as cinco categorias solicitadas, em pixels do frame BGR.

    As classes originais são mapeadas antes do rastreamento. Uma instância por
    thread; não há fallback para SSD nem para o adaptador YOLO existente.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_DIR / "yolov8s_taco.onnx",
        metadata_path: str | Path = MODEL_DIR / "metadata.json",
        confidence_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        allowed_classes: Iterable[str] = COUNT_CATEGORIES,
        class_mapping: dict[str, str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Instale: python -m pip install -r requirements-yolov8-taco.txt") from error
        self.confidence_threshold = _threshold(confidence_threshold, "confidence_threshold")
        self.nms_threshold = _threshold(nms_threshold, "nms_threshold")
        self.allowed_classes = set(allowed_classes)
        if not self.allowed_classes or self.allowed_classes - set(COUNT_CATEGORIES):
            raise ValueError(f"detection_types deve selecionar categorias de {COUNT_CATEGORIES}.")
        self.class_mapping = dict(DEFAULT_CLASS_MAPPING if class_mapping is None else class_mapping)
        if not self.class_mapping or set(self.class_mapping.values()) - set(COUNT_CATEGORIES):
            raise ValueError("class_mapping deve mapear classes TACO para as cinco categorias.")

        model_file = Path(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
        self.class_names = dict(enumerate(metadata["names"]))
        unknown = set(self.class_mapping) - set(self.class_names.values())
        if unknown:
            raise ValueError(f"Classes inexistentes no modelo TACO: {sorted(unknown)}.")
        if not model_file.is_file():
            raise FileNotFoundError(f"Modelo YOLOv8/TACO não encontrado: {model_file.resolve()}")
        with model_file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != metadata["sha256"]:
            raise ValueError("O SHA-256 do modelo YOLOv8/TACO não corresponde a metadata.json.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_file), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        input_info = self.session.get_inputs()[0]
        if input_info.shape != [1, 3, 640, 640] or input_info.type != "tensor(float)":
            raise ValueError("Modelo incompatível: esperado float32 NCHW 1x3x640x640.")
        self.input_name = input_info.name
        onnx_names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
        if onnx_names != self.class_names:
            raise ValueError("O mapa de classes difere dos metadados ONNX.")

    @staticmethod
    def _prepare(frame: np.ndarray):
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height))
        left, top = (640 - resized_width) // 2, (640 - resized_height) // 2
        padded = cv2.copyMakeBorder(resized, top, 640 - resized_height - top,
                                    left, 640 - resized_width - left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        return tensor, (resized_width / width, resized_height / height, left, top)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or frame.size == 0):
            raise ValueError("O frame deve ser BGR uint8 HxWx3 não vazio.")
        tensor, transform = self._prepare(frame)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode(output, frame.shape[1], frame.shape[0], transform)

    def _decode(self, output, width, height, transform) -> list[Detection]:
        if output.ndim != 3 or output.shape[:2] != (1, 4 + len(self.class_names)):
            raise ValueError(f"Saída ONNX incompatível: {output.shape}.")
        rows = output[0].T
        class_ids = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), class_ids + 4]
        selected = (scores >= self.confidence_threshold) & np.isfinite(rows).all(axis=1)
        scale_x, scale_y, pad_x, pad_y = transform
        candidates = []
        for row, class_id, score in zip(rows[selected], class_ids[selected], scores[selected]):
            category = self.class_mapping.get(self.class_names[int(class_id)])
            if category not in self.allowed_classes:
                continue
            cx, cy, w, h = map(float, row[:4])
            x1 = max(0, min(width - 1, math.floor((cx - w / 2 - pad_x) / scale_x)))
            y1 = max(0, min(height - 1, math.floor((cy - h / 2 - pad_y) / scale_y)))
            x2 = max(0, min(width - 1, math.ceil((cx + w / 2 - pad_x) / scale_x)))
            y2 = max(0, min(height - 1, math.ceil((cy + h / 2 - pad_y) / scale_y)))
            if x2 > x1 and y2 > y1:
                candidates.append(Detection(category, float(score), (x1, y1, x2, y2)))
        if not candidates:
            return []
        boxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]]
                 for d in candidates]
        # Entre classes: a mesma caixa não deve gerar duas contagens de resíduos.
        kept = cv2.dnn.NMSBoxes(boxes, [d.confidence for d in candidates],
                                 self.confidence_threshold, self.nms_threshold)
        return [candidates[int(index)] for index in np.asarray(kept).reshape(-1)]
