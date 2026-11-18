"""Rastreamento exclusivo do YOLOv8/TACO: conta na primeira detecção."""

from dataclasses import dataclass
from typing import Iterable

from backend.detection.types import Detection


@dataclass
class PresenceTrack:
    track_id: int
    class_name: str
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    missing_frames: int = 0


@dataclass(frozen=True)
class FirstDetection:
    track_id: int
    class_name: str
    confidence: float


class YoloV8PresenceTracker:
    """Associa posição e caixa sem exigir linha nem uma segunda detecção.

    Um objeto soma ao criar o rastro. Oscilações de classe não geram novo
    evento: a categoria contada é a da primeira detecção. Não identifica
    permanentemente o objeto; ele pode somar novamente após o rastro expirar.
    """

    def __init__(self, max_distance: float = 90, max_missing_frames: int = 12):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.objects: dict[int, PresenceTrack] = {}
        self._next_id = 1

    @staticmethod
    def _iou(first, second) -> float:
        intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0, min(first[3], second[3]) - max(first[1], second[1]))
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, detections: Iterable[Detection]) -> list[FirstDetection]:
        pending = list(detections)
        candidates = []
        for index, detection in enumerate(pending):
            for track_id, tracked in self.objects.items():
                distance = ((tracked.centroid[0] - detection.centroid[0]) ** 2
                            + (tracked.centroid[1] - detection.centroid[1]) ** 2) ** 0.5
                overlap = self._iou(tracked.bbox, detection.bbox)
                if distance <= self.max_distance or overlap >= 0.3:
                    candidates.append((-overlap, distance, track_id, index))
        unmatched = set(self.objects)
        used = set()
        for _, _, track_id, index in sorted(candidates):
            if track_id not in unmatched or index in used:
                continue
            tracked = self.objects[track_id]
            tracked.centroid = pending[index].centroid
            tracked.bbox = pending[index].bbox
            tracked.missing_frames = 0
            unmatched.remove(track_id)
            used.add(index)
        for track_id in unmatched:
            self.objects[track_id].missing_frames += 1
        self.objects = {key: tracked for key, tracked in self.objects.items()
                        if tracked.missing_frames <= self.max_missing_frames}
        events = []
        for index, detection in enumerate(pending):
            if index not in used:
                track_id = self._next_id
                self._next_id += 1
                self.objects[track_id] = PresenceTrack(track_id, detection.class_name,
                                                       detection.centroid, detection.bbox)
                events.append(FirstDetection(track_id, detection.class_name, detection.confidence))
        return events
