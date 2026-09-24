# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robot-relative direction labels shared by the socket bridge."""

import math


def relative_direction(x, y):
    """Describe a target using the base_link x-forward convention."""
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return '方向未知'
    if not math.isfinite(x) or not math.isfinite(y):
        return '方向未知'
    angle = math.atan2(y, x)
    center_tolerance = math.radians(5.0)
    if -center_tolerance <= angle <= center_tolerance:
        return '正前方'
    if center_tolerance < angle < math.pi / 8:
        return '前方偏左'
    if math.pi / 8 <= angle < 3 * math.pi / 8:
        return '左前方'
    if 3 * math.pi / 8 <= angle < 5 * math.pi / 8:
        return '左侧'
    if 5 * math.pi / 8 <= angle < 7 * math.pi / 8:
        return '左后方'
    if angle >= 7 * math.pi / 8 or angle < -7 * math.pi / 8:
        return '正后方'
    if -7 * math.pi / 8 <= angle < -5 * math.pi / 8:
        return '右后方'
    if -5 * math.pi / 8 <= angle < -3 * math.pi / 8:
        return '右侧'
    if -math.pi / 8 < angle < -center_tolerance:
        return '前方偏右'
    return '右前方'
