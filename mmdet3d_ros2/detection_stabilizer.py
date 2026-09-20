"""Small, dependency-free multi-frame detector for socket bridge output."""

from dataclasses import dataclass
import math


def _distance(first, second):
    return math.hypot(
        first['center']['x'] - second['center']['x'],
        first['center']['y'] - second['center']['y'],
    )


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


@dataclass
class _Track:
    class_id: str
    samples: list
    last_seen: float
    missed_frames: int = 0

    @property
    def latest(self):
        return self.samples[-1][1]

    def add(self, detection, timestamp, window_size):
        self.samples.append((timestamp, detection))
        self.samples = self.samples[-window_size:]
        self.last_seen = timestamp
        self.missed_frames = 0

    def to_detection(self, recent_count):
        samples = [detection for _, detection in self.samples[-recent_count:]]
        result = dict(samples[-1])
        center = dict(result['center'])
        for axis in ('x', 'y', 'z'):
            center[axis] = _median([sample['center'][axis] for sample in samples])
        result['center'] = center
        result['score'] = sum(sample['score'] for sample in samples) / len(samples)
        result['confirmed_hits'] = len(self.samples)
        return result


class DetectionStabilizer:
    """Confirm spatially consistent detections without merging same-class objects."""

    def __init__(
        self,
        min_hits=3,
        window_size=3,
        window_seconds=2.0,
        match_distance=0.5,
        max_missed_frames=5,
    ):
        if min_hits < 1 or window_size < min_hits:
            raise ValueError('window_size must be at least min_hits')
        self.min_hits = min_hits
        self.window_size = window_size
        self.window_seconds = window_seconds
        self.match_distance = match_distance
        self.max_missed_frames = max_missed_frames
        self.tracks = []

    def update(self, detections, timestamp):
        for track in self.tracks:
            track.missed_frames += 1

        unmatched = set(range(len(detections)))
        candidates = []
        for track_index, track in enumerate(self.tracks):
            for detection_index in unmatched:
                detection = detections[detection_index]
                if detection['class_id'].lower() != track.class_id.lower():
                    continue
                distance = _distance(track.latest, detection)
                if distance <= self.match_distance:
                    candidates.append((distance, track_index, detection_index))

        for _, track_index, detection_index in sorted(candidates):
            if detection_index not in unmatched:
                continue
            track = self.tracks[track_index]
            track.add(detections[detection_index], timestamp, self.window_size)
            unmatched.remove(detection_index)

        for detection_index in unmatched:
            detection = detections[detection_index]
            self.tracks.append(_Track(
                class_id=detection['class_id'],
                samples=[(timestamp, detection)],
                last_seen=timestamp,
            ))

        self.tracks = [
            track for track in self.tracks
            if track.missed_frames <= self.max_missed_frames
            and timestamp - track.last_seen <= self.window_seconds * 2
        ]

        confirmed = []
        for track in self.tracks:
            if track.missed_frames != 0 or len(track.samples) < self.min_hits:
                continue
            recent = track.samples[-self.min_hits:]
            if recent[-1][0] - recent[0][0] > self.window_seconds:
                continue
            confirmed.append(track.to_detection(self.min_hits))
        return confirmed
