import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def generate_launch_description():
    config_file = os.environ.get(
        'MMDET3D_CONFIG_FILE',
        '/home/nvidia/mmdetection3d/configs/fcaf3d/fcaf3d_2xb8_scannet-3d-18class.py')
    checkpoint_file = os.environ.get(
        'MMDET3D_CHECKPOINT_FILE',
        '/home/nvidia/mm3d_ws/src/mmdet3d_ros2/checkpoints/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth')
    init_device = os.environ.get('MMDET3D_INIT_DEVICE', 'cuda:0')
    score_threshold = float(os.environ.get('MMDET3D_SCORE_THRESHOLD', '0.05'))
    max_input_points = int(os.environ.get('MMDET3D_MAX_INPUT_POINTS', '30000'))
    min_input_points = int(os.environ.get('MMDET3D_MIN_INPUT_POINTS', '12000'))
    target_infer_ms = float(os.environ.get('MMDET3D_TARGET_INFER_MS', '300.0'))
    use_amp = _env_bool('MMDET3D_USE_AMP', True)
    point_cloud_range = os.environ.get('MMDET3D_POINT_CLOUD_RANGE', '')

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('MMDET3D_CONFIG_FILE', config_file),
        SetEnvironmentVariable('MMDET3D_CHECKPOINT_FILE', checkpoint_file),
        SetEnvironmentVariable('MMDET3D_INIT_DEVICE', init_device),
        Node(
            package='mmdet3d_ros2',
            executable='infer_node',
            name='mmdet3d_infer_node',
            parameters=[
                {'config_file': config_file},
                {'checkpoint_file': checkpoint_file},
                {'score_threshold': score_threshold},
                {'infer_device': 'cuda:0'},
                {'init_device': init_device},
                {'allow_cpu_fallback': False},
                {'max_input_points': max_input_points},
                {'min_input_points': min_input_points},
                {'target_infer_ms': target_infer_ms},
                {'downsample_strategy': 'stride'},
                {'use_amp': use_amp},
                {'accumulate_detections': False},
                {'point_cloud_range': point_cloud_range},
                {'stale_point_cloud_timeout': 3.0},
                {'nms_interval': 0.05},
                {'point_cloud_frame': 'rslidar'},
                {'point_cloud_qos': 'best_effort'},
                {'point_cloud_topic': '/rslidar_points'}
            ]
        )
    ])
