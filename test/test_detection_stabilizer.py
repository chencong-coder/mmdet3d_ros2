# Copyright 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from mmdet3d_ros2.detection_stabilizer import DetectionStabilizer


def detection(class_id, x, y, score=0.8):
    return {
        'class_id': class_id,
        'score': score,
        'center': {'x': x, 'y': y, 'z': 0.0},
    }


def test_requires_three_consistent_frames():
    stabilizer = DetectionStabilizer()
    assert stabilizer.update([detection('chair', 1.0, 2.0)], 0.0) == []
    assert stabilizer.update([detection('chair', 1.1, 2.0)], 0.2) == []
    confirmed = stabilizer.update([detection('chair', 0.9, 2.0)], 0.4)
    assert len(confirmed) == 1
    assert confirmed[0]['confirmed_hits'] == 3
    assert confirmed[0]['center']['x'] == 1.0


def test_same_class_objects_keep_separate_tracks():
    stabilizer = DetectionStabilizer()
    for timestamp in (0.0, 0.2, 0.4):
        confirmed = stabilizer.update([
            detection('chair', 1.0, 1.0),
            detection('chair', 4.0, 1.0),
        ], timestamp)
    assert len(confirmed) == 2
    assert {round(item['center']['x'], 1) for item in confirmed} == {1.0, 4.0}


def test_unstable_jump_does_not_confirm_same_track():
    stabilizer = DetectionStabilizer()
    stabilizer.update([detection('chair', 0.0, 0.0)], 0.0)
    stabilizer.update([detection('chair', 2.0, 0.0)], 0.2)
    stabilizer.update([detection('chair', 0.0, 0.0)], 0.4)
    assert stabilizer.update([detection('chair', 2.0, 0.0)], 0.6) == []
