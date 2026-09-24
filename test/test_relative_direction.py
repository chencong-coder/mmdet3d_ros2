# Copyright 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from mmdet3d_ros2.relative_direction import relative_direction


def test_splits_targets_just_left_and_right_of_center():
    assert relative_direction(2.0, 0.35) == '前方偏左'
    assert relative_direction(2.0, -0.35) == '前方偏右'


def test_keeps_a_narrow_straight_ahead_band():
    assert relative_direction(2.0, 0.05) == '正前方'


def test_invalid_coordinates_are_unknown():
    assert relative_direction(float('nan'), 0.0) == '方向未知'
