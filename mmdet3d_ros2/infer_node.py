import sys
import types
import faulthandler
import os
import time
import traceback
import threading
from contextlib import nullcontext

faulthandler.enable(all_threads=True)


def startup_trace(message):
    print(f'[infer_node_startup] {message}', file=sys.stderr, flush=True)

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


def _point_field(msg, name):
    for field in msg.fields:
        if field.name == name:
            return field
    return None


def _selected_indices(point_count, max_points, strategy):
    if max_points <= 0 or point_count <= max_points:
        return None
    if strategy == 'random':
        return np.random.choice(point_count, max_points, replace=False)
    # Deterministic uniform sampling is much cheaper than random sampling and
    # keeps latency predictable for a 10 Hz sensor stream.
    return np.linspace(0, point_count - 1, max_points, dtype=np.int64)


def pointcloud2_to_array(msg, dataset_type, max_points=0, downsample_strategy='stride',
                         point_cloud_range=None):
    x_field = _point_field(msg, 'x')
    y_field = _point_field(msg, 'y')
    z_field = _point_field(msg, 'z')
    if x_field is None or y_field is None or z_field is None:
        raise ValueError('PointCloud2 must contain x, y and z fields')

    endian = '>' if msg.is_bigendian else '<'
    dtype = {
        'names': ['x', 'y', 'z'],
        'formats': [endian + 'f4', endian + 'f4', endian + 'f4'],
        'offsets': [x_field.offset, y_field.offset, z_field.offset],
        'itemsize': msg.point_step,
    }
    point_count = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.dtype(dtype), count=point_count)
    indices = _selected_indices(point_count, max_points, downsample_strategy)

    x = raw['x'] if indices is None else raw['x'][indices]
    y = raw['y'] if indices is None else raw['y'][indices]
    z = raw['z'] if indices is None else raw['z'][indices]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if point_cloud_range is not None:
        xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
        valid &= (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax) &
            (z >= zmin) & (z <= zmax))

    valid_count = int(valid.sum())
    if valid_count == 0:
        dims = 5 if dataset_type == 'nuscenes' else 4 if dataset_type == 'kitti' else 3
        return np.empty((0, dims), dtype=np.float32), point_count

    if dataset_type == 'kitti':
        points = np.empty((valid_count, 4), dtype=np.float32)
        points[:, 0] = x[valid]
        points[:, 1] = y[valid]
        points[:, 2] = z[valid]
        intensity_field = _point_field(msg, 'intensity')
        if intensity_field is None:
            points[:, 3] = 0.0
        else:
            intensity_dtype = {
                'names': ['intensity'],
                'formats': [endian + 'f4'],
                'offsets': [intensity_field.offset],
                'itemsize': msg.point_step,
            }
            intensity_raw = np.frombuffer(
                msg.data, dtype=np.dtype(intensity_dtype), count=point_count)
            intensity = intensity_raw['intensity'] if indices is None else intensity_raw['intensity'][indices]
            points[:, 3] = intensity[valid]
        return points, point_count

    if dataset_type == 'nuscenes':
        points = np.empty((valid_count, 5), dtype=np.float32)
        points[:, 0] = x[valid]
        points[:, 1] = y[valid]
        points[:, 2] = z[valid]
        intensity_field = _point_field(msg, 'intensity')
        if intensity_field is None:
            points[:, 3] = 0.0
        else:
            intensity_dtype = {
                'names': ['intensity'],
                'formats': [endian + 'f4'],
                'offsets': [intensity_field.offset],
                'itemsize': msg.point_step,
            }
            intensity_raw = np.frombuffer(
                msg.data, dtype=np.dtype(intensity_dtype), count=point_count)
            intensity = intensity_raw['intensity'] if indices is None else intensity_raw['intensity'][indices]
            points[:, 3] = intensity[valid]
        points[:, 4] = 0.0
        return points, point_count

    points = np.empty((valid_count, 3), dtype=np.float32)
    points[:, 0] = x[valid]
    points[:, 1] = y[valid]
    points[:, 2] = z[valid]
    return points, point_count


def _parse_point_cloud_range(value):
    if not value:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 6:
        raise ValueError('point_cloud_range must contain 6 values: xmin,ymin,zmin,xmax,ymax,zmax')
    return tuple(float(part) for part in parts)


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
    def __init__(self, preloaded_model=None, preloaded_config=None,
                 preloaded_checkpoint=None, preloaded_device=None):
        startup_trace('InferNode.__init__ begin')
        super().__init__('infer_node')
        startup_trace('Node base initialized')
        self.logger = self.get_logger()

        self.tf_buffer = None
        self.tf_listener = None
        startup_trace('TF listener skipped; inference uses point cloud frame directly')

        #self.declare_parameter('config_file', '/home/nvidia/mmdetection3d/configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py')
        #self.declare_parameter('checkpoint_file', '/home/nvidia/mm3d_ws/src/mmdet3d_ros2/checkpoints/groupfree3d_8x4_scannet-3d-18class-L6-O256_20210702_145347-3499eb55.pth')
        self.declare_parameter('point_cloud_frame', 'femto_mega_color_optical_frame')
        self.declare_parameter('point_cloud_topic', '/femto_mega/depth_registered/filter_points')
        self.declare_parameter('score_threshold', 0.98)
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('allow_cpu_fallback', False)
        self.declare_parameter('init_device', 'cpu')
        self.declare_parameter('nms_interval', 0.5)
        self.declare_parameter('point_cloud_qos', 'best_effort')
        self.declare_parameter('max_input_points', 1024)
        self.declare_parameter('min_input_points', 512)
        self.declare_parameter('target_infer_ms', 100.0)
        self.declare_parameter('downsample_strategy', 'stride')
        self.declare_parameter('use_amp', False)
        self.declare_parameter('accumulate_detections', False)
        self.declare_parameter('point_cloud_range', '')
        self.declare_parameter('stale_point_cloud_timeout', 1.0)
        startup_trace('Parameters declared')
        # votenet
        self.declare_parameter('config_file', '/home/nvidia/mmdetection3d/configs/votenet/votenet_8xb8_scannet-3d.py,')
        self.declare_parameter('checkpoint_file', '/home/nvidia/mm3d_ws/src/mmdet3d_ros2/checkpoints/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth')

        config_file_path = self.get_parameter('config_file').get_parameter_value().string_value
        checkpoint_file_path = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        infer_device = self.get_parameter('infer_device').get_parameter_value().string_value
        init_device = self.get_parameter('init_device').get_parameter_value().string_value
        allow_cpu_fallback = self.get_parameter('allow_cpu_fallback').get_parameter_value().bool_value
        self.score_thrs = self.get_parameter('score_threshold').get_parameter_value().double_value
        nms_interval = self.get_parameter('nms_interval').get_parameter_value().double_value
        self.point_cloud_frame = self.get_parameter('point_cloud_frame').get_parameter_value().string_value
        point_cloud_qos = self.get_parameter('point_cloud_qos').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value
        self.max_input_points = self.get_parameter('max_input_points').get_parameter_value().integer_value
        self.configured_max_input_points = self.max_input_points
        self.min_input_points = self.get_parameter('min_input_points').get_parameter_value().integer_value
        self.target_infer_ms = self.get_parameter('target_infer_ms').get_parameter_value().double_value
        self.downsample_strategy = self.get_parameter('downsample_strategy').get_parameter_value().string_value
        self.use_amp = self.get_parameter('use_amp').get_parameter_value().bool_value
        self.accumulate_detections = self.get_parameter('accumulate_detections').get_parameter_value().bool_value
        point_cloud_range_value = self.get_parameter('point_cloud_range').get_parameter_value().string_value
        self.point_cloud_range = _parse_point_cloud_range(point_cloud_range_value)
        self.stale_point_cloud_timeout = (
            self.get_parameter('stale_point_cloud_timeout').get_parameter_value().double_value)
        startup_trace(
            f'Parameters loaded: config={config_file_path}, checkpoint={checkpoint_file_path}, '
            f'init_device={init_device}, device={infer_device}, topic={point_cloud_topic}')

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
        startup_trace(f'torch={torch.__version__}, torch_cuda_version={torch.version.cuda}')
        if infer_device.startswith('cuda'):
            startup_trace('CUDA availability check is skipped during startup to avoid Jetson runtime crashes')
        elif infer_device == 'cpu':
            startup_trace('Using CPU by request; MMDetection3D warns some CPU paths are unsupported')
        self.torch_device = torch.device(infer_device)
        if (preloaded_model is not None and
                preloaded_config == config_file_path and
                preloaded_checkpoint == checkpoint_file_path and
                preloaded_device == init_device):
            startup_trace(f'Using preloaded init_model result from device={init_device}')
            self.model = preloaded_model
        else:
            startup_trace(f'Calling init_model on device={init_device}')
            self.model = init_model(config_file_path, checkpoint_file_path, device=init_device)
            startup_trace('init_model finished')
        if init_device != infer_device and infer_device.startswith('cuda'):
            startup_trace(f'Moving initialized model to device={infer_device}')
            self.model.to(infer_device)
            startup_trace(f'Model moved to device={infer_device}')
        else:
            infer_device = init_device
            self.torch_device = torch.device(infer_device)
        self.model.eval()

        self.filtered_bboxes_nms = torch.zeros(0, 6, device=self.torch_device)
        self.filtered_bboxes_tensor = torch.zeros(0, 7, device=self.torch_device)
        self.filtered_scores = torch.zeros(0, device=self.torch_device)
        self.filtered_labels = torch.zeros(0, device=self.torch_device)
        self.detection_lock = threading.Lock()
        self.pending_condition = threading.Condition()
        self.pending_msg = None
        self.shutdown_worker = False
        self.inference_busy = False
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.published_frames = 0
        self.result_sequence = 0
        self.last_published_result_sequence = -1
        self.log_every_n_frames = 10
        self.last_point_cloud_time = None
        self.stale_clear_published = False
        startup_trace('Detection buffers initialized')

        startup_trace(f'Creating point cloud subscription: topic={point_cloud_topic}')
        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self.listener_callback,
            qos)
        startup_trace('Point cloud subscription created')

        startup_trace('Creating Detection3DArray publisher: topic=/detect_bbox3d')
        self.marker_pub = self.create_publisher(Detection3DArray, '/detect_bbox3d', 10)
        startup_trace('Detection3DArray publisher created')

        startup_trace('Creating MarkerArray publisher: topic=/detect_bbox3d_vis')
        self.vis_pub = self.create_publisher(MarkerArray, '/detect_bbox3d_vis', 10)
        startup_trace('MarkerArray publisher created')

        # for debug
        # self.publisher_ = self.create_publisher(PointCloud2, '/detect_bbox_infer_pcd', qos)
        # for nms and publish
        startup_trace(f'Creating detections timer: interval={nms_interval}')
        self.timer = self.create_timer(nms_interval, self.detections_callback)
        startup_trace('Detections timer created')

        self.worker_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.worker_thread.start()
        startup_trace('Inference worker started')

    def listener_callback(self, msg):
        self.received_frames += 1
        frame_index = self.received_frames
        self.last_point_cloud_time = time.monotonic()
        self.stale_clear_published = False

        with self.pending_condition:
            if self.pending_msg is not None:
                self.dropped_frames += 1
            self.pending_msg = (frame_index, msg)
            self.pending_condition.notify()

    def inference_loop(self):
        while True:
            with self.pending_condition:
                while self.pending_msg is None and not self.shutdown_worker:
                    self.pending_condition.wait(timeout=0.1)
                if self.shutdown_worker:
                    return
                frame_index, msg = self.pending_msg
                self.pending_msg = None

            self.inference_busy = True
            try:
                self.process_frame(frame_index, msg)
            except Exception:
                self.logger.error('[Infer] worker failed:\n' + traceback.format_exc())
            finally:
                self.inference_busy = False

    def process_frame(self, frame_index, msg):
        current_frame = msg.header.frame_id
        current_stamp = msg.header.stamp

        try:
            convert_start = time.time()
            infer_points, original_point_count = pointcloud2_to_array(
                msg,
                self.dataset_type,
                max_points=self.max_input_points,
                downsample_strategy=self.downsample_strategy,
                point_cloud_range=self.point_cloud_range)
            convert_ms = (time.time() - convert_start) * 1000
            should_log = frame_index == 1 or frame_index % self.log_every_n_frames == 0
            if should_log:
                self.logger.info(
                    f'[PCD] frame={frame_index} frame_id={current_frame} '
                    f'raw={original_point_count} input={len(infer_points)} '
                    f'convert={convert_ms:.2f} ms dropped={self.dropped_frames}')

            if len(infer_points) == 0:
                self.logger.warn('[PCD] empty point cloud, skipping inference')
                return

            start_time = time.time()
            if should_log:
                self.logger.info(f'[Infer] start frame={frame_index} points={len(infer_points)}')
            model_result, data_afterprocess = self.run_inference(infer_points)
            if self.torch_device.type == 'cuda':
                torch.cuda.synchronize(self.torch_device)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
            total_time_ms = convert_ms + elapsed_time_ms
            self.processed_frames += 1
            self.adapt_max_input_points(total_time_ms, len(infer_points))
            if should_log:
                self.logger.info(
                    '[Infer] done frame={} infer={:.2f} ms total={:.2f} ms '
                    'target={:.1f} max_points={}'.format(
                        frame_index, elapsed_time_ms, total_time_ms,
                        self.target_infer_ms, self.max_input_points))

            bboxes = model_result.pred_instances_3d.bboxes_3d
            scores = model_result.pred_instances_3d.scores_3d
            labels = model_result.pred_instances_3d.labels_3d
            indices = torch.where(scores > self.score_thrs)
            # x_center, y_center, z_center, dx, dy, dz, yaw
            filtered_bboxes = bboxes[indices]
            filtered_scores = scores[indices]
            filtered_labels = labels[indices]
            if should_log:
                self.logger.info(
                    f'[Infer] raw={scores.shape[0]} filtered={filtered_scores.shape[0]} '
                    f'threshold={self.score_thrs:.2f}')
            
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
                with self.detection_lock:
                    self.current_frame = current_frame
                    self.current_stamp = current_stamp
                    if self.filtered_bboxes_tensor.shape[1] != filtered_bboxes.tensor.shape[1]:
                        self.filtered_bboxes_tensor = torch.zeros(
                            0, filtered_bboxes.tensor.shape[1], device=self.torch_device)
                    if self.accumulate_detections:
                        self.filtered_bboxes_nms = torch.cat(
                            (self.filtered_bboxes_nms, filtered_bboxes_nms), dim=0)
                        self.filtered_bboxes_tensor = torch.cat(
                            (self.filtered_bboxes_tensor, filtered_bboxes.tensor), dim=0)
                        self.filtered_scores = torch.cat((self.filtered_scores, filtered_scores), dim=0)
                        self.filtered_labels = torch.cat((self.filtered_labels, filtered_labels), dim=0)
                    else:
                        self.filtered_bboxes_nms = filtered_bboxes_nms
                        self.filtered_bboxes_tensor = filtered_bboxes.tensor
                        self.filtered_scores = filtered_scores
                        self.filtered_labels = filtered_labels
                    self.result_sequence += 1
            else:
                with self.detection_lock:
                    self.current_frame = current_frame
                    self.current_stamp = current_stamp
                    if not self.accumulate_detections:
                        self.clear_detection_buffers_locked()
                    self.result_sequence += 1
        except Exception:
            self.logger.error('[Infer] callback failed:\n' + traceback.format_exc())

    def run_inference(self, infer_points):
        amp_enabled = self.use_amp and self.torch_device.type == 'cuda'
        try:
            with torch.inference_mode(), self.amp_context(amp_enabled):
                return inference_detector(self.model, infer_points)
        except RuntimeError as exc:
            message = str(exc)
            if amp_enabled and 'expected scalar type Half but found Float' in message:
                self.use_amp = False
                self.logger.warn(
                    '[Infer] AMP disabled because this model/op path requires FP32 '
                    f'({message})')
                if self.torch_device.type == 'cuda':
                    torch.cuda.empty_cache()
                with torch.inference_mode():
                    return inference_detector(self.model, infer_points)
            raise

    def amp_context(self, enabled):
        if enabled:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    def adapt_max_input_points(self, elapsed_time_ms, input_points):
        if self.target_infer_ms <= 0 or self.max_input_points <= 0:
            return
        if elapsed_time_ms > self.target_infer_ms * 1.15 and self.max_input_points > self.min_input_points:
            self.max_input_points = max(self.min_input_points, int(self.max_input_points * 0.85))
            self.logger.warn(
                f'[Infer] over target ({elapsed_time_ms:.1f} ms), reducing max_input_points to '
                f'{self.max_input_points}')
        elif (elapsed_time_ms < self.target_infer_ms * 0.75 and
              input_points >= self.max_input_points and
              self.max_input_points < self.configured_max_input_points):
            self.max_input_points = min(
                self.configured_max_input_points,
                int(self.max_input_points * 1.05))

    def detections_callback(self):
        if self.is_point_cloud_stale():
            if not self.stale_clear_published:
                self.logger.warn('No recent point cloud; clearing detections')
                self.clear_detection_buffers()
                self.publish_empty_detection(clear_markers=True)
                self.stale_clear_published = True
            return

        with self.detection_lock:
            result_sequence = self.result_sequence
            filtered_bboxes_nms = self.filtered_bboxes_nms
            filtered_bboxes_tensor = self.filtered_bboxes_tensor
            filtered_scores = self.filtered_scores
            filtered_labels = self.filtered_labels
            result_frame = getattr(self, 'current_frame', 'odom')
            result_stamp = getattr(self, 'current_stamp', None)
            if self.accumulate_detections:
                self.clear_detection_buffers_locked()

        if (not self.accumulate_detections and
                result_sequence == self.last_published_result_sequence):
            return

        if filtered_bboxes_nms.shape[0] == 0:
            self.draw_bbox(
                filtered_bboxes_tensor.cpu(),
                filtered_labels.cpu().numpy(),
                filtered_scores.cpu().numpy(),
                frame_id=result_frame,
                stamp=result_stamp)
            self.last_published_result_sequence = result_sequence
            return

        if self.torch_device.type == 'cpu':
            pick_ind = torch.arange(filtered_bboxes_nms.shape[0], device=self.torch_device)
        else:
            pick_ind = aligned_3d_nms(
                filtered_bboxes_nms,
                filtered_scores,
                filtered_labels,
                0.25)
        self.published_frames += 1
        if self.published_frames == 1 or self.published_frames % self.log_every_n_frames == 0:
            self.logger.info("[NMS] detections {} -> {}".format(
                filtered_bboxes_nms.shape[0], pick_ind.shape[0]))
        self.draw_bbox(
            filtered_bboxes_tensor[pick_ind].cpu(),
            filtered_labels[pick_ind].cpu().numpy(),
            filtered_scores[pick_ind].cpu().numpy(),
            frame_id=result_frame,
            stamp=result_stamp)
        self.last_published_result_sequence = result_sequence

    def is_point_cloud_stale(self):
        if self.last_point_cloud_time is None:
            return True
        return (time.monotonic() - self.last_point_cloud_time) > self.stale_point_cloud_timeout

    def clear_detection_buffers(self):
        with self.detection_lock:
            self.clear_detection_buffers_locked()

    def clear_detection_buffers_locked(self):
        bbox_dim = self.filtered_bboxes_tensor.shape[1]
        self.filtered_bboxes_nms = torch.zeros(0, 6, device=self.torch_device)
        self.filtered_bboxes_tensor = torch.zeros(0, bbox_dim, device=self.torch_device)
        self.filtered_scores = torch.zeros(0, device=self.torch_device)
        self.filtered_labels = torch.zeros(0, device=self.torch_device)

    def publish_empty_detection(self, clear_markers=False):
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = getattr(self, 'current_frame', self.point_cloud_frame)
        if hasattr(self, 'current_stamp'):
            det3d_array.header.stamp = self.current_stamp
        self.marker_pub.publish(det3d_array)

        if clear_markers and hasattr(self, 'vis_pub'):
            marker_array = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
            self.vis_pub.publish(marker_array)

    def destroy_node(self):
        with self.pending_condition:
            self.shutdown_worker = True
            self.pending_condition.notify()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        super().destroy_node()
    
    def draw_bbox(self, bboxes, labels, scores, frame_id=None, stamp=None):
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = frame_id or getattr(self, 'current_frame', 'odom')
        if stamp is not None:
            det3d_array.header.stamp = stamp
        elif hasattr(self, 'current_stamp'):
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
                det3d.header.stamp = det3d_array.header.stamp

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
                m.header.stamp = det3d_array.header.stamp
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
                t.header.stamp = det3d_array.header.stamp
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
    preloaded_model = None
    preloaded_config = os.environ.get('MMDET3D_CONFIG_FILE')
    preloaded_checkpoint = os.environ.get('MMDET3D_CHECKPOINT_FILE')
    preloaded_device = os.environ.get('MMDET3D_INIT_DEVICE', 'cpu')
    if preloaded_config and preloaded_checkpoint:
        startup_trace(f'Preloading model before rclpy.init on device={preloaded_device}')
        preloaded_model = init_model(
            preloaded_config, preloaded_checkpoint, device=preloaded_device)
        startup_trace('Preload init_model finished')

    rclpy.init(args=args)
    infer_node = InferNode(
        preloaded_model=preloaded_model,
        preloaded_config=preloaded_config,
        preloaded_checkpoint=preloaded_checkpoint,
        preloaded_device=preloaded_device)
    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()
