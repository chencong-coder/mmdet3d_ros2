from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    config_file = '/home/nvidia/mmdetection3d/configs/votenet/votenet_8xb8_scannet-3d.py'
    checkpoint_file = '/home/nvidia/mm3d_ws/src/mmdet3d_ros2/checkpoints/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth'
    init_device = 'cuda:0'

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
                {'score_threshold': 0.35},
                {'infer_device': 'cuda:0'},
                {'init_device': init_device},
                {'allow_cpu_fallback': False},
                {'max_input_points': 2048},
                {'min_input_points': 2048},
                {'target_infer_ms': 300.0},
                {'downsample_strategy': 'stride'},
                {'use_amp': False},
                {'accumulate_detections': False},
                {'stale_point_cloud_timeout': 3.0},
                {'nms_interval': 0.05},
                {'point_cloud_frame': 'rslidar'},
                {'point_cloud_qos': 'best_effort'},
                {'point_cloud_topic': '/rslidar_points'}
            ]
        )
    ])
