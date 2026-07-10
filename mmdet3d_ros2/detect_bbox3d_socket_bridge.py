import json
import math
import socket
import threading

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray


def _yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _result_id(result):
    if hasattr(result, 'id'):
        return str(result.id)
    hypothesis = getattr(result, 'hypothesis', None)
    if hypothesis is None:
        return ''
    if hasattr(hypothesis, 'class_id'):
        return str(hypothesis.class_id)
    if hasattr(hypothesis, 'id'):
        return str(hypothesis.id)
    return ''


def _result_score(result):
    if hasattr(result, 'score'):
        return float(result.score)
    hypothesis = getattr(result, 'hypothesis', None)
    if hypothesis is not None and hasattr(hypothesis, 'score'):
        return float(hypothesis.score)
    return 0.0


def _detection_to_dict(detection):
    center = detection.bbox.center.position
    orientation = detection.bbox.center.orientation
    size = detection.bbox.size

    class_id = ''
    score = 0.0
    if detection.results:
        class_id = _result_id(detection.results[0])
        score = _result_score(detection.results[0])

    return {
        'class_id': class_id,
        'score': score,
        'center': {
            'x': float(center.x),
            'y': float(center.y),
            'z': float(center.z),
        },
        'size': {
            'x': float(size.x),
            'y': float(size.y),
            'z': float(size.z),
        },
        'orientation': {
            'x': float(orientation.x),
            'y': float(orientation.y),
            'z': float(orientation.z),
            'w': float(orientation.w),
        },
        'yaw': float(_yaw_from_quaternion(orientation)),
    }


class DetectBBox3DSocketBridge(Node):
    def __init__(self):
        super().__init__('detect_bbox3d_socket_bridge')

        self.declare_parameter('topic', '/detect_bbox3d')
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8765)

        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value

        self.socket_clients = []
        self.socket_clients_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(8)
        self.server_socket.settimeout(0.5)

        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()

        self.subscription = self.create_subscription(
            Detection3DArray,
            self.topic,
            self._detection_callback,
            10)

        self.get_logger().info(
            f'Bridging {self.topic} to tcp://{self.host}:{self.port}')

    def _accept_loop(self):
        while not self.shutdown_event.is_set():
            try:
                client, address = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return

            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.socket_clients_lock:
                self.socket_clients.append(client)
            self.get_logger().info(f'socket client connected: {address}')

    def _detection_callback(self, msg):
        payload = {
            'topic': self.topic,
            'frame_id': msg.header.frame_id,
            'stamp': {
                'sec': int(msg.header.stamp.sec),
                'nanosec': int(msg.header.stamp.nanosec),
            },
            'detections': [
                _detection_to_dict(detection)
                for detection in msg.detections
            ],
        }
        data = (json.dumps(payload, separators=(',', ':')) + '\n').encode('utf-8')
        self._send_to_clients(data)

    def _send_to_clients(self, data):
        with self.socket_clients_lock:
            clients = list(self.socket_clients)

        stale_clients = []
        for client in clients:
            try:
                client.sendall(data)
            except OSError:
                stale_clients.append(client)

        if not stale_clients:
            return

        with self.socket_clients_lock:
            for client in stale_clients:
                if client in self.socket_clients:
                    self.socket_clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass

    def destroy_node(self):
        self.shutdown_event.set()
        try:
            self.server_socket.close()
        except OSError:
            pass
        with self.socket_clients_lock:
            clients = list(self.socket_clients)
            self.socket_clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectBBox3DSocketBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
