from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    config_file = '/home/nvidia/mmdetection3d/configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py'
    checkpoint_file = '/home/nvidia/mm3d_ws/src/mmdet3d_ros2/checkpoints/groupfree3d_8x4_scannet-3d-18class-L6-O256_20210702_145347-3499eb55.pth'
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
                {'score_threshold': 0.4},
                {'infer_device': 'cuda:0'},
                {'init_device': init_device},
                {'allow_cpu_fallback': False},
                {'max_input_points': 1024},
                {'min_input_points': 512},
                {'target_infer_ms': 100.0},
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
