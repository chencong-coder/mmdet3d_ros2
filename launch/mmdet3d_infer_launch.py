from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mmdet3d_ros2',
            executable='infer_node',
            name='mmdet3d_infer_node',
            parameters=[
                {'config_file': '/home/nvidia/mmdetection3d/configs/votenet/votenet_8xb8_scannet-3d.py'},
                {'checkpoint_file': '/home/nvidia/mm3d_ws/src/mmdet3d_ros2/checkpoints/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth'},
                {'score_threshold': 0.30},
                {'infer_device': 'cpu'},
                {'nms_interval': 0.5},
                {'point_cloud_frame': 'rslidar'},
                {'point_cloud_qos': 'best_effort'},
                {'point_cloud_topic': 'rslidar_points'}
            ]
        )
    ])
