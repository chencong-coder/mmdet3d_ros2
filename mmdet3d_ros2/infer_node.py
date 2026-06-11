import sys
import types

# Mock mmengine's imports that rely on FSDP and ZeroRedundancyOptimizer
z = types.ModuleType("mmengine.optim.optimizer.zero_optimizer")
z.ZeroRedundancyOptimizer = type("ZeroRedundancyOptimizer", (), {})
sys.modules["mmengine.optim.optimizer.zero_optimizer"] = z

f = types.ModuleType("mmengine.model.wrappers.fully_sharded_distributed")
f.MMFullyShardedDataParallel = type("MMFullyShardedDataParallel", (), {})
sys.modules["mmengine.model.wrappers.fully_sharded_distributed"] = f

# Mock torch.distributed
import torch
import torch.distributed as dist

if "torch._C._distributed_c10d" not in sys.modules:
    dummy_c10d = types.ModuleType("torch._C._distributed_c10d")
    dummy_c10d.Work = list
    dummy_c10d.ProcessGroup = list
    sys.modules["torch._C._distributed_c10d"] = dummy_c10d

if getattr(dist, "distributed_c10d", None) is None:
    dummy_dist = types.ModuleType("torch.distributed.distributed_c10d")
    dummy_dist.ProcessGroup = type("ProcessGroup", (), {})
    dummy_dist.Work = type("Work", (), {})
    dist.distributed_c10d = dummy_dist
elif not hasattr(dist.distributed_c10d, "ProcessGroup"):
    dist.distributed_c10d.ProcessGroup = type("ProcessGroup", (), {})
    dist.distributed_c10d.Work = type("Work", (), {})

if not hasattr(dist, "ReduceOp"):
    class DummyOp: pass
    dist.ReduceOp = DummyOp

if not hasattr(dist, "_remote_device"):
    dist._remote_device = str
    


import time

import numpy as np
import torch
import mmcv

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, TransformStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import tf2_ros
import tf_transformations

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.models.layers import aligned_3d_nms
from visualization_msgs.msg import Marker, MarkerArray

def transform_point(trans, pt):
    # https://answers.ros.org/question/249433/tf2_ros-buffer-transform-pointstamped/
    quat = [
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w
    ]
    mat = tf_transformations.quaternion_matrix(quat)
    pt_np = [pt.x, pt.y, pt.z, 1.0]
    pt_in_map_np = np.dot(mat, pt_np)

    pt_in_map = Point()
    pt_in_map.x = pt_in_map_np[0] + trans.transform.translation.x
    pt_in_map.y = pt_in_map_np[1] + trans.transform.translation.y
    pt_in_map.z = pt_in_map_np[2] + trans.transform.translation.z

    return pt_in_map

class InferNode(Node):
    def __init__(self):
        super().__init__('infer_node')
        self.logger = self.get_logger()

        cache_time = Duration(seconds=2.0) 
        self.tf_buffer = tf2_ros.Buffer(cache_time)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('config_file', 'projects/TR3D/configs/tr3d_1xb16_sunrgbd-3d-10class.py')
        self.declare_parameter('checkpoint_file', '../checkpoints/tr3d_1xb16_sunrgbd-3d-10class.pth')
        self.declare_parameter('point_cloud_frame', 'femto_mega_color_optical_frame')
        self.declare_parameter('point_cloud_topic', '/femto_mega/depth_registered/filter_points')
        self.declare_parameter('score_threshold', 0.98)
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('nms_interval', 0.5)
        self.declare_parameter('point_cloud_qos', 'best_effort')
        # self.declare_parameter('config_file', 'configs/votenet/votenet_8xb16_sunrgbd-3d.py')
        # self.declare_parameter('checkpoint_file', '../checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth')
        # imvoxelnet
        # self.declare_parameter('config_file', 'configs/imvoxelnet/imvoxelnet_2xb4_sunrgbd-3d-10class.py')
        # self.declare_parameter('checkpoint_file', '../checkpoints/imvoxelnet_4x2_sunrgbd-3d-10class_20220809_184416-29ca7d2e.pth')

        config_file_path = self.get_parameter('config_file').get_parameter_value().string_value
        checkpoint_file_path = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        infer_device = self.get_parameter('infer_device').get_parameter_value().string_value
        self.score_thrs = self.get_parameter('score_threshold').get_parameter_value().double_value
        nms_interval = self.get_parameter('nms_interval').get_parameter_value().double_value
        self.point_cloud_frame = self.get_parameter('point_cloud_frame').get_parameter_value().string_value
        point_cloud_qos = self.get_parameter('point_cloud_qos').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value

        qos = QoSProfile(depth=5)
        if point_cloud_qos == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif point_cloud_qos == 'reliable':
            qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.logger.error('Invalid value for point_cloud_qos parameter')
            return

        self.transform_stamped = TransformStamped()
        self.det3d_array = Detection3DArray()
        self.det3d_array.header.frame_id = getattr(self, 'current_frame', 'odom')

        if 'sunrgbd' in checkpoint_file_path:
            self.dataset_type = 'sunrgbd'
            self.class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                   'night_stand', 'bookshelf', 'bathtub')
        elif 'scannet' in checkpoint_file_path:
            self.dataset_type = 'scannet'
            self.class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin')
        elif 'kitti' in checkpoint_file_path:
            self.dataset_type = 'kitti'
            self.class_names = ('Pedestrian', 'Cyclist', 'Car')
        elif 'nuscenes' in checkpoint_file_path or 'centerpoint' in checkpoint_file_path.lower():
            self.dataset_type = 'nuscenes'
            self.class_names = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
        else:
            self.dataset_type = 'unknown'
            self.logger.error('Unknown weight, path of weight should contain "sunrgbd", "scannet", "kitti" or "centerpoint"')

        self.get_logger().info('full_config_file: "%s"' % config_file_path)
        self.get_logger().info('checkpoint_file: "%s"' % checkpoint_file_path)
        if infer_device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.error(
                'Requested infer_device "%s", but torch CUDA is not available. '
                'Falling back to CPU to avoid crashing during model initialization.' %
                infer_device)
            infer_device = 'cpu'
        self.torch_device = torch.device(infer_device)
        self.model = init_model(config_file_path, checkpoint_file_path, device=infer_device)

        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self.listener_callback,
            qos)
        self.marker_pub = self.create_publisher(Detection3DArray, '/detect_bbox3d', 10)
        self.vis_pub = self.create_publisher(MarkerArray, '/detect_bbox3d_vis', 10)
        # for debug
        # self.publisher_ = self.create_publisher(PointCloud2, '/detect_bbox_infer_pcd', qos)
        # for nms and publish
        self.timer = self.create_timer(nms_interval, self.detections_callback)

        self.filtered_bboxes_nms = torch.zeros(0, 6, device=self.torch_device)
        self.filtered_bboxes_tensor = torch.zeros(0, 7, device=self.torch_device)
        self.filtered_scores = torch.zeros(0, device=self.torch_device)
        self.filtered_labels = torch.zeros(0, device=self.torch_device)


    def listener_callback(self, msg):
        self.current_frame = msg.header.frame_id
        self.current_stamp = msg.header.stamp
        # read points
        gen = pc2.read_points(msg, skip_nans=True)
        int_data = list(gen)
        
        # Determine point feature dimension based on dataset
        if self.dataset_type == 'kitti':
            infer_points = np.zeros((len(int_data), 4)) # x, y, z, intensity
        elif self.dataset_type == 'nuscenes':
            infer_points = np.zeros((len(int_data), 5)) # x, y, z, intensity, ring
        else:
            # We only provide XYZ for indoor datasets correctly here according to what MMDetection3D LoadPointsFromDict pipeline expects for SUNRGBD/ScanNet without color pipeline explicit.
            infer_points = np.zeros((len(int_data), 3)) # x, y, z
            
        # We do inference directly in the point cloud's coordinate frame
        points = np.zeros((len(int_data), 3))
        base_points = np.zeros((len(int_data), 3))
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = msg.header.stamp
        transform_stamped.header.frame_id = msg.header.frame_id
        transform_stamped.child_frame_id = msg.header.frame_id
        transform_stamped.transform.rotation.w = 1.0

        for ind, x in enumerate(int_data):
            points[ind] = [x[0], x[1], x[2]]
            pt = Point()
            pt.x, pt.y, pt.z = x[0], x[1], x[2]
            base_pt = transform_point(transform_stamped, pt)
            
            if self.dataset_type == 'kitti':
                # For KITTI/LiDAR, we expect 4 channels: x, y, z, intensity
                intensity = x[3] if len(x) > 3 else 0.0
                infer_points[ind] = [base_pt.x, base_pt.y, base_pt.z, intensity]
            elif self.dataset_type == 'nuscenes':
                # nuScenes models usually expect 5 channels: x, y, z, intensity, ring
                intensity = x[3] if len(x) > 3 else 0.0
                ring = 0.0 # Default ring to 0 if not provided
                infer_points[ind] = [base_pt.x, base_pt.y, base_pt.z, intensity, ring]
            else:
                infer_points[ind] = [base_pt.x, base_pt.y, base_pt.z]
                
            base_points[ind] = [base_pt.x, base_pt.y, base_pt.z]
        # infer_points = pc2.create_cloud_xyz32(header=msg.header, points=base_points)
        # self.publisher_.publish(infer_points)
        
        # infer_points = pc2.create_cloud_xyz32(header=msg.header, points=base_points)
        # self.publisher_.publish(infer_points)
        start_time = time.time()  # get current time
        # perform inference
        model_result, data_afterprocess = inference_detector(self.model, infer_points)
        end_time = time.time()  # get current time after inference
        # Calculate elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        self.logger.debug("Inference time: {:.2f} ms".format(elapsed_time_ms))

        bboxes = model_result.pred_instances_3d.bboxes_3d
        scores = model_result.pred_instances_3d.scores_3d
        labels = model_result.pred_instances_3d.labels_3d
        indices = torch.where(scores > self.score_thrs)
        # x_center, y_center, z_center, dx, dy, dz, yaw
        filtered_bboxes = bboxes[indices]
        filtered_scores = scores[indices]
        filtered_labels = labels[indices]
        
        if filtered_bboxes.shape[0] != 0:
            filtered_bboxes_x0 = filtered_bboxes.center[:,0]-0.5*filtered_bboxes.dims[:,0]
            filtered_bboxes_y0 = filtered_bboxes.center[:,1]-0.5*filtered_bboxes.dims[:,1]
            filtered_bboxes_z0 = filtered_bboxes.center[:,2]-0.5*filtered_bboxes.dims[:,2]
            filtered_bboxes_x1 = filtered_bboxes.center[:,0]+0.5*filtered_bboxes.dims[:,0]
            filtered_bboxes_y1 = filtered_bboxes.center[:,1]+0.5*filtered_bboxes.dims[:,1]
            filtered_bboxes_z1 = filtered_bboxes.center[:,2]+0.5*filtered_bboxes.dims[:,2]
            filtered_bboxes_nms = torch.stack((filtered_bboxes_x0, filtered_bboxes_y0,
                                               filtered_bboxes_z0, filtered_bboxes_x1,
                                               filtered_bboxes_y1, filtered_bboxes_z1), dim=1)
            self.filtered_bboxes_nms = torch.cat((self.filtered_bboxes_nms, filtered_bboxes_nms), dim=0)
            if self.filtered_bboxes_tensor.shape[1] != filtered_bboxes.tensor.shape[1]:
                self.filtered_bboxes_tensor = torch.zeros(
                    0, filtered_bboxes.tensor.shape[1], device=self.torch_device)
            self.filtered_bboxes_tensor = torch.cat((self.filtered_bboxes_tensor, filtered_bboxes.tensor), dim=0)
            self.filtered_scores = torch.cat((self.filtered_scores, filtered_scores), dim=0)
            self.filtered_labels = torch.cat((self.filtered_labels, filtered_labels), dim=0)

    def detections_callback(self):
        if self.filtered_bboxes_nms.shape[0] == 0:
            self.draw_bbox(
                self.filtered_bboxes_tensor.cpu(),
                self.filtered_labels.cpu().numpy(),
                self.filtered_scores.cpu().numpy())
            return

        pick_ind = aligned_3d_nms(
            self.filtered_bboxes_nms,
            self.filtered_scores,
            self.filtered_labels,
            0.25)
        self.logger.info("[NMS] detections {} -> {}".format(
            self.filtered_bboxes_nms.shape[0], pick_ind.shape[0]))
        self.draw_bbox(
            self.filtered_bboxes_tensor[pick_ind].cpu(),
            self.filtered_labels[pick_ind].cpu().numpy(),
            self.filtered_scores[pick_ind].cpu().numpy())

        # Clear buffers after publishing so the next timer tick reflects new frames only.
        bbox_dim = self.filtered_bboxes_tensor.shape[1]
        self.filtered_bboxes_nms = torch.zeros(0, 6, device=self.torch_device)
        self.filtered_bboxes_tensor = torch.zeros(0, bbox_dim, device=self.torch_device)
        self.filtered_scores = torch.zeros(0, device=self.torch_device)
        self.filtered_labels = torch.zeros(0, device=self.torch_device)

    
    def draw_bbox(self, bboxes, labels, scores, timestamp=None):
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = getattr(self, 'current_frame', 'odom')
        if hasattr(self, 'current_stamp'):
            det3d_array.header.stamp = self.current_stamp
        
        marker_array = MarkerArray()

        if len(bboxes) > 0:
            for ind in range(len(bboxes)):
                bbox = bboxes[ind]
                label = int(labels[ind])
                score = scores[ind]
                class_name = self.class_names[label]

                # 1. 过滤掉小车不涉及的庞大类目标，防止墙壁/树木误检导致遮挡
                if class_name in ['truck', 'construction_vehicle', 'trailer']:
                    continue
                
                # 2. 为避免低矮的行人被误检为交通锥，把 traffic_cone 的阈值单独抬高到 0.45
                cls_thr = 0.45 if class_name == 'traffic_cone' else self.score_thrs
                if score < cls_thr:
                    continue

                det3d = Detection3D()
                det3d.header.frame_id = det3d_array.header.frame_id
                if hasattr(self, 'current_stamp'):
                    det3d.header.stamp = self.current_stamp

                pose = Pose()
                pose.position.x = bbox[0].item()
                pose.position.y = bbox[1].item()
                pose.position.z = bbox[2].item()

                quat = Quaternion()
                q = tf_transformations.quaternion_from_euler(0, 0, bbox[-1].item())
                quat.x = q[0]
                quat.y = q[1]
                quat.z = q[2]
                quat.w = q[3]
                pose.orientation = quat

                dimensions = Vector3()
                dimensions.x = bbox[3].item()
                dimensions.y = bbox[4].item()
                dimensions.z = bbox[5].item()

                det3d.bbox.center = pose
                det3d.bbox.size = dimensions
                object_hypothesis = ObjectHypothesisWithPose()
                object_hypothesis.id = str(class_name)
                object_hypothesis.score = float(score.item())
                det3d.results.append(object_hypothesis)
                det3d_array.detections.append(det3d)

                # Assign colors based on category
                base_color = (0.5, 0.5, 0.5) # gray default
                if class_name in ('car', 'Car'): base_color = (0.0, 1.0, 0.0) # green
                elif class_name in ('pedestrian', 'Pedestrian'): base_color = (1.0, 0.0, 0.0) # red
                elif class_name in ('bicycle', 'Cyclist', 'motorcycle'): base_color = (1.0, 1.0, 0.0) # yellow
                elif class_name in ('bus', 'truck', 'trailer'): base_color = (0.0, 0.5, 1.0) # blue
                elif class_name in ('barrier', 'traffic_cone'): base_color = (1.0, 0.5, 0.0) # orange

                # Box marker as a wireframe LINE_LIST so RViz shows an actual 3D box.
                m = Marker()
                m.header.frame_id = det3d_array.header.frame_id
                if hasattr(self, 'current_stamp'):
                    m.header.stamp = self.current_stamp
                m.ns = "bboxes"
                m.id = ind * 2
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.pose = pose
                m.scale.x = 0.06
                m.color.r = base_color[0]
                m.color.g = base_color[1]
                m.color.b = base_color[2]
                m.color.a = 1.0
                m.lifetime = Duration(seconds=0.6).to_msg()
                hx = dimensions.x / 2.0
                hy = dimensions.y / 2.0
                hz = dimensions.z / 2.0
                corners = [
                    Point(x=float(x), y=float(y), z=float(z))
                    for x, y, z in (
                        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
                        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
                    )
                ]
                for a, b in (
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ):
                    m.points.append(corners[a])
                    m.points.append(corners[b])
                marker_array.markers.append(m)

                # Text marker (class name + score)
                t = Marker()
                t.header.frame_id = det3d_array.header.frame_id
                if hasattr(self, 'current_stamp'):
                    t.header.stamp = self.current_stamp
                t.ns = "labels"
                t.id = ind * 2 + 1
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position.x = pose.position.x
                t.pose.position.y = pose.position.y
                t.pose.position.z = pose.position.z + dimensions.z / 2.0 + 0.5
                t.pose.orientation.w = 1.0
                t.scale.z = 0.6 # text height
                t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
                t.text = f"{class_name} {score.item():.2f}"
                t.lifetime = Duration(seconds=0.6).to_msg()
                marker_array.markers.append(t)

        self.marker_pub.publish(det3d_array)
        if hasattr(self, 'vis_pub'):
            self.vis_pub.publish(marker_array)
        
def main(args=None):
    rclpy.init(args=args)
    infer_node = InferNode()
    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()
