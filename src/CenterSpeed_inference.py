#!/usr/bin/env python3
import rospy
import tf2_py as tf2
import tf2_ros
from tf2_geometry_msgs import tf2_geometry_msgs
import os
import torch
import time
from std_msgs.msg import Float32MultiArray
from frenet_converter.frenet_converter import FrenetConverter
from f110_msgs.msg import WpntArray, ObstacleArray
from f110_msgs.msg import Obstacle as ObstacleMessage
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32
from bisect import bisect_left
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import quaternion_from_euler
from dynamic_reconfigure.msg import Config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualization_msgs.msg import Marker,MarkerArray
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from models.CenterSpeed import *
from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
from pycoral.adapters.common import input_size
from pycoral.adapters import classify
from tflite_runtime.interpreter import Interpreter, load_delegate


def normalize_s(x,track_length):
        x = x % (track_length)
        if x > track_length/2:
            x -= track_length
        return x
class Obstacle :
    """
    This class implements the properties of the obstacles
    """
    current_id = 0
    def __init__(self,x,y, vx, vy ,size,theta, id=None) -> None:
        self.center_x = x
        self.center_y = y
        self.vx = vx#additional vx
        self.vy = vy#additional vy
        self.size = size
        self.id = id
        self.theta = theta

    def squaredDist(self, obstacle):
        return (self.center_x-obstacle.center_x)**2+(self.center_y-obstacle.center_y)**2
class CenterSpeed_Inference :
    """
    This class implements a ROS node for object detection with CenterSpeed.

    It subscribes to the following topics:
        - `/scan`: Publishes the lidar scans.
        - `/global_waypoints`: Publishes the global waypoints.
        - `/odom_frenet`: Publishes the car state in frenet frame.
        - `/dynamic_tracker_server/parameter_updates`: Publishes the dynamic reconfigure parameters.


    The node publishes the following topics:
        - `/detect_bound`: Publishes the detect boundaries.
        - `/raw_obstacles`: Publishes the detected obstacles.
        - `/obstacles_markers_new`: Publishes the markers of the detected obstacles.
        - `/velocity_marker`: Publishes the velocity of the detected obstacles.
        - `/obstacles_centerspeed`: Publishes the detected obstacles with estimated velocities. If traking is enabled
        - `/obstacles`: Publishes the detected obstacles. If tracking is disabled
        - `/perception/detection/latency`: Publishes the latency of the detection.
        - `/perception/obstacle_v_cartesian`: Publishes the cartesian velocities of the detected obstacles, if measuring is enabled.
        - `/perception/centerspeed_image`: Publishes the image for visualization in FOXGLOVE, if visualization is enabled.
    """
    def __init__(self) -> None:
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        print("--------USING OS_STACK---------")
        self.from_bag = False
        self.converter = None

        # track variables
        self.waypoints = None
        self.biggest_d = None
        self.smallest_d = None
        self.s_array = None
        self.d_right_array = None
        self.d_left_array = None
        self.track_length = None
        self.path_needs_update = False

        # ego car s position
        self.ego_s = None
        self.ego_yaw = None

        # --- CenterSpeed variables ---
        self.current_stamp = None
        self.tracked_obstacles = []
        self.tracked_obstacles_pointclouds = []
        self.angles = None
        self.cos_angles = None
        self.sin_angles = None

        # --- dyn params sub ---
        self.min_obs_size = 10
        self.max_obs_size = 0.5
        self.max_viewing_distance = 9
        self.boundaries_inflation = 0.3

        # --- Node properties ---
        rospy.init_node('CenterSpeedInference', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        # --- Load Parameters ---
        self.measuring = rospy.get_param("/measure", False) #mesasure the latency of the node
        self.tracking_enabled = rospy.get_param("/TinyCenterSpeed/tracking", False) #additionally to the node a tracking node is used
        self.using_raceline = rospy.get_param("/TinyCenterSpeed/using_raceline", True) #if the raceline is used for filtering
        self.visualize = rospy.get_param("/TinyCenterSpeed/visualize", False) # if the visualization is enabled
        self.rate = rospy.get_param("/TinyCenterSpeed/rate", 40) #rate of the node
        self.pixelsize = rospy.get_param("/TinyCenterSpeed/pixel_size", 0.12) #size of a pixel in meters
        self.image_size = rospy.get_param("/TinyCenterSpeed/image_size", 64) #size of the image for preprocessing
        self.feature_size = rospy.get_param("/TinyCenterSpeed/feature_size", 3) #number of features in the preprocessed image (one frame)
        self.num_opponents = rospy.get_param("/TinyCenterSpeed/num_opponents", 1) #number of peaks extracted from the heatmap
        self.boundaries_inflation = rospy.get_param("/TinyCenterSpeed/boundary_inflation", 0.3) #for boundary filtering
        self.using_coral = rospy.get_param("/TinyCenterSpeed/using_coral", False) #if the coral tpu is used
        self.dense = rospy.get_param("/TinyCenterSpeed/dense", True) #uses CenterSpeed 2.0
        self.quantize = rospy.get_param("/TinyCenterSpeed/quantize", False) #quantizes the model
        self.use_cartesian = rospy.get_param("/TinyCenterSpeed/publish_cartesian", False) #if the cartesian velocities are used
        self.publish_image = rospy.get_param("/TinyCenterSpeed/publish_foxglove", False) #publishes the image for visualization in FOXGLOVE

        # --- variables for Centerspeed ---
        self.scan1 =None #raw scans from the lidar
        self.scan2 = None #raw scans from the lidar
        self.origin_offset = (self.image_size//2) * self.pixelsize #origin is in the middle of the image
        self.frame1 = None
        self.frame2 = None
        self.opp_coordinates = [[],[]]
        self.detection_threshold = 0.2
        self.distance_threshold = self.pixelsize * self.image_size//2 + 0.5 #distance threshold for the peak detection

        # --- Publisher ---
        self.boundaries_pub = rospy.Publisher('/perception/detect_bound', Marker, queue_size=5)
        self.obstacles_marker_pub = rospy.Publisher('/perception/obstacles_markers_new', MarkerArray, queue_size=5)
        self.velocity_marker_pub = rospy.Publisher('/perception/velocity_marker', MarkerArray, queue_size=5)
        if self.publish_image:
            self.image_pub = rospy.Publisher('/perception/centerspeed_image', Image, queue_size=5)

        # --- conditional publishers ---
        if self.measuring:
            self.latency_pub = rospy.Publisher('/perception/detection/latency', Float32, queue_size=5)
            self.v_cartesian_pub = rospy.Publisher('/perception/obstacle_v_cartesian', Float32MultiArray, queue_size=5)
        if self.tracking_enabled:
            self.obstacles_msg_pub_estimated = rospy.Publisher('/perception/detection/raw_obstacles', ObstacleArray, queue_size=5)
        else:
            self.obstacles_msg_pub_estimated = rospy.Publisher('/perception/obstacles', ObstacleArray, queue_size=5)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Subscribers ---
        rospy.Subscriber('/scan', LaserScan, self.laserCb)
        rospy.Subscriber('/global_waypoints', WpntArray, self.pathCb)
        rospy.Subscriber('/car_state/odom_frenet', Odometry, self.carStateCb)
        if not self.from_bag:
            rospy.Subscriber("/dynamic_tracker_server/parameter_updates", Config, self.dyn_param_cb)

        self.converter = self.initialize_converter()
        self.wait_for_track_length()

        rospy.loginfo("[CenterSpeed]: Initialized CenterSpeed Node")

    def shutdown(self) -> None:
        """
        Shutdown the node
        """
        rospy.logwarn('CenterSpeed is shutdown')

    def wait_for_track_length(self) -> None:
        """
        Wait for the track length to be published
        """
        while(self.track_length is None and not rospy.is_shutdown()):
            rospy.sleep(0.1)
            print("waiting for track length ...")

    def initialize_converter(self) -> bool:
        """
        Initialize the FrenetConverter object
        """
        while(self.waypoints is None):
            rospy.sleep(0.1)
            print("waiting for waypoints ...")

        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.psi_array)
        rospy.loginfo("[CenterSpeed]: initialized FrenetConverter object")
        return converter

    def laserCb(self, msg: LaserScan) -> None:
        """
        Callback function for the lidar scans.
        Precomputes the cosine and sine values of the angles for preprocessing.

        Args:
            - msg: the lidar scan

        Sets:
            - angles: the angles of the lidar scan
            - cos_angles: the cosine values of the angles
            - sin_angles: the sine values of the angles
            - scan1: the first scan
            - scan2: the second scan
        """
        if self.angles is None:
            self.angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.cos_angles = np.cos(self.angles)
            self.sin_angles = np.sin(self.angles)
        if self.scan1 is None:
            self.scan1 = msg
        else:
            self.scan2 = self.scan1
            self.scan1 = msg

    def pathCb(self,data: WpntArray) -> None:
        '''
        Callback function for the global waypoints.
        Initializes the waypoints and creates the boundaries.

        Args:
            - data: the global waypoints
        '''
        # Initial calls: initialize the converter
        if self.waypoints is None:
            self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
            self.psi_array = np.array([wpnt.psi_rad for wpnt in data.wpnts])
        # Second call: create the boundaries arrays
        if (self.s_array is None or self.path_needs_update) and self.converter is not None:
            rospy.loginfo('[CenterSpeed]: received global path')
            waypoint_array = data.wpnts
            points=[]
            self.s_array = []
            self.d_right_array = []
            self.d_left_array = []
            for waypoint in waypoint_array:
                self.s_array.append(waypoint.s_m)
                self.d_right_array.append(waypoint.d_right-self.boundaries_inflation)
                self.d_left_array.append(waypoint.d_left-self.boundaries_inflation)
                resp = self.converter.get_cartesian(waypoint.s_m,-waypoint.d_right+self.boundaries_inflation)
                points.append(Point(resp[0],resp[1],0))
                resp = self.converter.get_cartesian(waypoint.s_m,waypoint.d_left-self.boundaries_inflation)
                points.append(Point(resp[0],resp[1],0))
            self.smallest_d = min(self.d_right_array+self.d_left_array)
            self.biggest_d = max(self.d_right_array+self.d_left_array)
            print("Got track length")
            self.track_length = data.wpnts[-1].s_m

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.id = 0
            marker.type = marker.SPHERE_LIST
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1
            marker.color.g = 0.
            marker.color.r = 1.
            marker.color.b = 0.
            marker.points = points

            self.boundaries_pub.publish(marker)
        self.path_needs_update = False

    def carStateCb(self, data: Odometry) -> None:
        '''
        Callback function for the car state.

        Args:
            - data: the car state

        Sets:
            - ego_s: the s-coordinate of the car
            - ego_yaw: the yaw of the car
        '''
        self.ego_s = data.pose.pose.position.x
        self.ego_yaw = Rotation.from_quat(np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])).as_euler('xyz')[2]

    def dyn_param_cb(self, params: Config) -> None:
        '''
        Callback function for the dynamic reconfigure parameters.
        (Adapted from detect.py)
        '''
        self.min_obs_size = rospy.get_param('dynamic_tracker_server/min_obs_size', 10)
        self.max_obs_size = rospy.get_param('dynamic_tracker_server/max_obs_size', 0.5)
        self.max_viewing_distance = rospy.get_param('dynamic_tracker_server/max_viewing_distance', 9)
        self.boundaries_inflation = rospy.get_param('dynamic_tracker_server/boundaries_inflation', 0.1)
        self.path_needs_update = True

        # Handle the visualization parameter
        self.visualize = rospy.get_param('dynamic_tracker_server/visualize', False)
        if not self.visualize:
            if hasattr(self, 'fig'):
                try:
                    plt.close(self.fig)
                    del self.fig
                    rospy.loginfo('[Centerspeed] Closed figure')
                except Exception as e:
                    rospy.loginfo(f'[Centerspeed] Error closing figure: {e}')

        param_list = [self.min_obs_size, self.max_obs_size, self.max_viewing_distance]
        print(f'[Centerspeed]: New dyn reconf values recieved: Min size [laser points], Max size [m], max viewing dist [m]: {param_list}')

    def clearmarkers(self) -> Marker:
        '''
        Clears the markers.
        (Adapted from detect.py)
        '''
        marker = Marker()
        marker.action = 3
        return [marker]

    def laserPointOnTrack (self, s, d) -> bool:
        '''
        Checks if a point in the frenet frame is on the track.
        (Adapted from detect.py)

        Args:
            - s: the s-coordinate of the point
            - d: the d-coordinate of the point

        Returns:
            - True if the point is on the track, False otherwise
        '''
        if normalize_s(s-self.ego_s,self.track_length)>self.max_viewing_distance:
            return False
        if abs(d) >= self.biggest_d:
            return False
        if abs(d) <= self.smallest_d:
            return True
        idx = bisect_left(self.s_array, s)
        if idx:
            idx -= 1
        if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
            return False
        return True


    def preprocess(self, scan: LaserScan) -> torch.Tensor:
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a image_sizeximage_size grid.
        Stores these grids in a new tensor.

        Args:
            - scan: the lidar scan to be preprocessed

        Returns:
            - preprocessed_scans (Tensor): the preprocessed scans in image space
        '''
        lidar_data = torch.tensor(scan.ranges, dtype=torch.float32, device=self.device)
        intensities = torch.tensor(scan.intensities, dtype=torch.float32, device=self.device)
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities

        #preprocess the lidar data
        preprocessed_scans = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32, device=self.device)
        x = lidar_data * torch.tensor(self.cos_angles, device=self.device)
        y = lidar_data * torch.tensor(self.sin_angles, device=self.device)
        x_coord = ((x + self.origin_offset) / self.pixelsize).to(torch.int)
        y_coord = ((y + self.origin_offset) / self.pixelsize).to(torch.int)
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        preprocessed_scans[0,y_coord, x_coord] = 1 #set the pixel to occupied
        preprocessed_scans[1,y_coord, x_coord] = torch.maximum(preprocessed_scans[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
        preprocessed_scans[2,y_coord, x_coord] += 1 #count the number of points in the pixel
        return preprocessed_scans

    def index_to_cartesian(self, x_img,y_img) -> tuple:
        '''
        Converts an index in image-space back to cartesian coordinates.

        Args:
            - x_img: the x-coordinate in the image space
            - y_img: the y-coordinate in the image space

        Returns:
            - Tuple: (x, y) the cartesian coordinates
        '''
        x = x_img * self.pixelsize - self.origin_offset
        y = y_img * self.pixelsize - self.origin_offset
        return x,y

    def find_k_peaks(self, image: torch.Tensor, k: int, radius: float, return_cartesian: bool = True) -> tuple:
        '''
        Finds the k highest peaks in the image, which are at least radius pixels apart.

        Args:
            - image (torch.Tensor): the image to be analyzed
            - k (int): the number of peaks to be found
            - radius (float): the minimum distance between the peaks
            - return_cartesian (bool): whether to return the peaks in cartesian coordinates or image coordinates

        Returns:
            - x (np.Array): the x-coordinates of the peaks
            - y (np.Array): the y-coordinates of the peaks
        '''
        radius = radius//self.pixelsize
        image = image.clone()
        peaks_found = 0
        x = []
        y = []

        for i in range(k):
            # Find the maximum value in the image
            max_val, max_idx = torch.max(image.view(-1), 0)
            if max_val < self.detection_threshold:
                break
            max_coords = np.unravel_index(max_idx, image.shape)
            x.append(max_coords[0])
            y.append(max_coords[1])
            peaks_found += 1
            if i == k-1:#don't zero the last peak, not necessary
                break
            # Set the surrounding area to zero, square with side length 2*radius: more efficient than circular mask
            top = max(0, int((max_coords[0] - radius)))
            bottom = min(image.shape[0], int((max_coords[0] + radius)))
            left = max(0, int((max_coords[1] - radius)))
            right = min(image.shape[1], int((max_coords[1] + radius)))
            image[top:bottom, left:right] = 0
        if return_cartesian:
            y = np.array(y)
            x = np.array(x)
            return self.index_to_cartesian(y, x)

        if peaks_found == 0: #if no peak was found
            return None, None

        return np.array(y), np.array(x)

    def print_centerspeed_parameters(self) -> None:
        """
        Prints a summary of the CenterSpeed parameters in a tabular format.
        """
        parameters = [
            ["Parameter", "Value"],
            ["Image Size", self.image_size],
            ["Pixel Size (m)", self.pixelsize],
            ["Detection Threshold", self.detection_threshold],
            ["Device", self.device.type],
            ["Origin Offset (m)", self.origin_offset],
            ["Feature Size", self.feature_size],
            ["Number of Opponents", self.num_opponents],
            ["Boundaries Inflation", self.boundaries_inflation],
            ["Quantize", self.quantize],
            ["Using Coral", self.using_coral],
            ["Dense", self.dense],
            ["Publish Image", self.publish_image],
            ["Use Cartesian", self.use_cartesian],
        ]

        print("\n[CenterSpeed Parameters]:")
        for param, value in parameters:
            print(f"{param:25}: {value}")

    def load_model(self, hardware: str = 'NUC', version: int = 2) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hardware == 'NUC':
            base_path = os.path.dirname(os.path.abspath(__file__))

            if version == 1: #CenterSpeed 1.0
                self.model_path = os.path.join(base_path, 'trained_models/CenterSpeed64v2.pt')
                self.net = CenterSpeedModular(image_size=self.image_size)

            if version == 2: #CenterSpeed 2.0
                self.model_path = os.path.join(base_path, 'trained_models/CenterSpeedDense.pt')
                self.net = CenterSpeedDenseResidual(image_size=self.image_size)
            else:
                self.net = CenterSpeedModular(image_size=self.image_size)
            self.net.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            if self.quantize:
                rospy.loginfo('[CenterSpeed]: Quantizing model...')
                self.net = torch.quantization.quantize_dynamic(self.net, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
            self.net.eval()
            self.net.to(self.device)
        elif hardware == 'Coral':
            rospy.loginfo('[CenterSpeed]: Using Coral TPU')
            print("Available Edge TPUs:", list_edge_tpus())
            coral_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models/tinycs_float32_edgetpu.tflite')
            rospy.loginfo('[CenterSpeed]: Loading Coral model from %s', coral_path)
            try:
                self.interpreter = make_interpreter(coral_path)
            except ValueError as e:
                rospy.logerr(f"Error loading Edge TPU delegate: {e}")

            self.interpreter.allocate_tensors()
            rospy.loginfo('[CenterSpeed]: Coral model loaded and initialized')
        else:
            rospy.logerr('[CenterSpeed]: Invalid hardware specified')

        self.print_centerspeed_parameters()

    def inference(self, preprocess_data, hardware = 'NUC', version = 2):
        '''
        Takes the preprocessed data and performs inference.

        Args:
            - preprocess_data: the preprocessed data
            - hardware: the hardware to be used for the inference (NUC or Coral)
                output = self.net(preprocess_data)
            - version: the version of the model to be used (1 or 2: Dense)

            - output_hm: the heatmap prediction
            - output_data: the speed and yaw prediction (if version 1)
        '''
        if hardware == 'NUC' and version == 1:
            with torch.no_grad():
                preprocess_data = preprocess_data.unsqueeze(0).to(self.device)
                output_hm, output_data = self.net(preprocess_data)
                return output_hm, output_data
        elif hardware == 'NUC' and version == 2:
            with torch.no_grad():
                preprocess_data = preprocess_data.unsqueeze(0).to(self.device)
                output = self.net(preprocess_data)
                return output
        elif hardware == 'Coral':
            input_details = self.interpreter.get_input_details()
            self.interpreter.set_tensor(input_details[0]['index'], preprocess_data.unsqueeze(0).permute(0, 2,3,1).numpy())
            self.interpreter.invoke()
            ouptut_details = self.interpreter.get_output_details()
            output_hm = self.interpreter.get_tensor(ouptut_details[0]['index'])
            return torch.tensor(output_hm)
        else:
            rospy.logerr('[CenterSpeed]: Invalid hardware specified')

    def publishObstaclesMessageEstimation(self) -> None:
        '''
        Publishes the detected obstacles as a message.
        Already processes the obstacles and estimates their velocity.
        Published to the topic `/perception/obstacles_centerspeed` or `/perception/obstacles` if tracking is disabled.
        '''
        obstacles_array_message = ObstacleArray()
        obstacles_array_message.header.stamp = self.current_stamp
        obstacles_array_message.header.frame_id = "map"
        x_center = []
        y_center = []
        remove_list = []

        for obstacle in self.tracked_obstacles:
            x_center.append(obstacle.center_x)
            y_center.append(obstacle.center_y)

        s_points, d_points = self.converter.get_frenet(np.array(x_center), np.array(y_center))
        sorted_d_indices = [i for i, _ in sorted(enumerate(d_points), key=lambda x: abs(x[1]))] #sort the d-values ascending

        for num_obs, index in enumerate(sorted_d_indices):
            obstacle = self.tracked_obstacles[index]
            if self.using_raceline:
                s = s_points[index]
                d = d_points[index]
                if not self.laserPointOnTrack(s,d):
                    remove_list.append(obstacle)
                    continue
                if self.use_cartesian: #if cartesian velocities are used and published
                    s = x_center[index]
                    d = y_center[index]
            else:#If no raceline is used, only for testing
                s = x_center[index]
                d = y_center[index]

            obsMsg = ObstacleMessage()
            obsMsg.id = obstacle.id
            obsMsg.s_start = s-obstacle.size/2
            obsMsg.s_end = s+obstacle.size/2
            obsMsg.d_left = d+obstacle.size/2
            obsMsg.d_right = d-obstacle.size/2
            obsMsg.s_center = s
            obsMsg.d_center = d
            obsMsg.size = obstacle.size
            yaw_diff = (obstacle.theta)%(2*np.pi)
            if yaw_diff > np.pi:
                yaw_diff -= 2*np.pi
            yaw_diff = -yaw_diff
            ROT = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff)],
                            [np.sin(yaw_diff), np.cos(yaw_diff)]])
            speed = np.array([obstacle.vx, obstacle.vy])
            vx, vy = np.dot(ROT, speed)#the velocity in the opponents frame

            ###Publish cartesian velocities in global frame###
            if self.measuring: #for catkin test also cartesian velocities are used
                yaw_diff = (self.ego_yaw+ obstacle.theta)%(2*np.pi)
                if yaw_diff > np.pi:
                    yaw_diff -= 2*np.pi
                ROT = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff)],
                                [np.sin(yaw_diff), np.cos(yaw_diff)]])
                speed = np.array([obstacle.vx, obstacle.vy])
                vx_glob, vy_glob = np.dot(ROT, speed)
                self.v_cartesian_pub.publish(Float32MultiArray(data=[vx_glob, vy_glob]))
            ##################################################

            obstacle_orientation_global = (obstacle.theta + self.ego_yaw) % (2 * np.pi)
            if obstacle_orientation_global > np.pi:
                obstacle_orientation_global -= 2*np.pi

            if not self.use_cartesian:
                vs, vd = self.converter.get_frenet_velocities(vx, vy, obstacle_orientation_global, s) #convert cartesian velocities to frenet velocities
            else:
                vs = vx
                vd = vy

            obsMsg.vs = vs
            obsMsg.vd = vd
            obsMsg.is_static = vs < 0.1 and vd < 0.1
            obstacles_array_message.obstacles.append(obsMsg)

        if len(remove_list) > 0:
            for obs in remove_list:
                if obs in self.tracked_obstacles:
                    self.tracked_obstacles.remove(obs)

        self.obstacles_msg_pub_estimated.publish(obstacles_array_message)

    def publishObstaclesMarkers(self) -> None:
        '''
        Publishes the detected obstacles as markers.
        Published to the topic `/perception/obstacles_markers_new`.
        '''
        markers_array = []
        for i ,obs in  enumerate(self.tracked_obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.current_stamp
            marker.id = i
            marker.type = marker.SPHERE
            marker.scale.x = obs.size*1.5
            marker.scale.y = obs.size
            marker.scale.z = obs.size
            marker.color.a = 0.5
            marker.color.g = 1.
            marker.color.r = 0.
            marker.color.b = i
            marker.pose.position.x = obs.center_x
            marker.pose.position.y = obs.center_y
            orientation = (self.ego_yaw + obs.theta) % (2 * np.pi)
            if orientation > np.pi:
                orientation -= 2*np.pi
            q = quaternion_from_euler(0, 0, orientation)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            markers_array.append(marker)

        self.obstacles_marker_pub.publish(self.clearmarkers())
        self.obstacles_marker_pub.publish(markers_array)

    def publish_image_from_tensor(self, tensor) -> None:
        '''
        Publishes the image to the topic `/perception/centerspeed_image`.
        '''
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = tensor.cpu().numpy()

        # Handle grayscale vs color images
        if len(tensor.shape) == 3:  # (C, H, W)
            tensor = np.transpose(tensor, (1, 2, 0))  # Convert to (H, W, C)
        elif len(tensor.shape) != 2:
            raise ValueError("Tensor must have shape (C, H, W) or (H, W).")
        if tensor.dtype != np.uint8:
            tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)

        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(tensor, encoding="mono8")
        self.image_pub.publish(ros_image)

    def plot_velocity(self) -> None:
        '''
        Plots the velocity of the detected obstacles as an arrow.
        Published to the topic `/perception/velocity_marker`.
        '''
        marker_array = MarkerArray()
        for obstacle in self.tracked_obstacles:
            yaw_diff = (self.ego_yaw)%(2*np.pi)
            if yaw_diff > np.pi:
                yaw_diff -= 2*np.pi
            ROT = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff)],
                            [np.sin(yaw_diff), np.cos(yaw_diff)]])
            speed = np.array([obstacle.vx, obstacle.vy])
            vx, vy = np.dot(ROT, speed)

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "velocity"
            marker.id = obstacle.id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.center_x
            marker.pose.position.y = obstacle.center_y
            marker.pose.position.z = 0
            orientation = quaternion_from_euler(0, 0, np.arctan2(vy, vx))
            marker.pose.orientation.x = orientation[0]
            marker.pose.orientation.y = orientation[1]
            marker.pose.orientation.z = orientation[2]
            marker.pose.orientation.w = orientation[3]
            marker.scale.x = np.sqrt(vx**2 + vy**2)  # Length of the arrow
            marker.scale.y = 0.1  # Width of the arrow
            marker.scale.z = 0.1  # Height of the arrow
            marker.color.a = 0.5  # Opacity
            marker.color.r = 0.0  # Red
            marker.color.g = 0.0  # Green
            marker.color.b = 1.0  # Blue
            marker_array.markers.append(marker)
        self.velocity_marker_pub.publish(marker_array)

    def lidar_to_global(self, x, y) -> tuple:
        '''
        Transforms the coordinates from the lidara to the global frame.

        Args:
            - x: the x-coordinate in the lidar frame
            - y: the y-coordinate in the lidar frame

        Returns: (tuple)
            - x: the x-coordinate in the global frame
            - y: the y-coordinate in the global frame
        '''
        point_in_lidar_frame = PointStamped()
        point_in_lidar_frame.header.frame_id = 'laser'
        point_in_lidar_frame.header.stamp = rospy.Time.now()
        point_in_lidar_frame.point = Point(x, y, 0)
        while not rospy.is_shutdown():
            try:
                t_bl = self.tf_buffer.lookup_transform(target_frame='base_link', source_frame='laser', time=rospy.Time(0))
                break
            except tf2.LookupException:
                rospy.loginfo("Waiting for transform from 'laser' to 'base_link'...")
                rospy.sleep(1)  # Wait for a second before trying again

        t_bl = self.tf_buffer.lookup_transform(target_frame='base_link', source_frame='laser', time=rospy.Time(0))
        t_global = self.tf_buffer.lookup_transform(target_frame='map', source_frame='base_link', time=rospy.Time(0))
        point_in_base_link = tf2_geometry_msgs.do_transform_point(point_in_lidar_frame, t_bl)
        point_in_global_frame = tf2_geometry_msgs.do_transform_point(point_in_base_link, t_global)
        return point_in_global_frame.point.x, point_in_global_frame.point.y

    def vis(self, i, output, data) -> None:
        '''
        Visualizes the input and output of the model in an live plot.
        Shows from left to right: the input, the output, the output with the detected peaks.

        Args:
            - i: the input image
            - output: the output heatmap
        '''

        if not hasattr(self, 'fig'):
            self.fig, self.axs = plt.subplots(1, 3, figsize=(10, 10))
            plt.ion()
            plt.show()

        # Clear previous plots
        for ax in self.axs:
            ax.clear()

        x,y = self.find_k_peaks(output, self.num_opponents, 0.8, return_cartesian=False)
        x_vel = x[0]#only one peak for speed and yaw
        y_vel = y[0]#only one peak for speed and yaw

        # Plot Input
        self.axs[0].imshow(i, origin='lower', cmap='plasma')
        self.axs[0].set_title('Input')
        self.axs[0].quiver(x_vel, y_vel, data[0], -data[1], color='red', scale=10)

        # Plot output
        self.axs[1].imshow(output, origin='lower', cmap='plasma')
        self.axs[1].set_title('Output')

        # Plot predicted peak on output
        self.axs[2].imshow(output, origin='lower', cmap='plasma')
        self.axs[2].scatter(x,y, color='red', s=10)
        self.axs[2].quiver(x_vel, y_vel, data[0], -data[1], color='red', scale=10)
        yaw_degrees = np.rad2deg(data[2])
        rectangle = patches.Rectangle((x_vel-5, y_vel-2.5), 10, 5, angle=yaw_degrees, fill=False, color='r')
        # Add the rectangle to the plot
        self.axs[2].add_patch(rectangle)
        self.axs[2].set_title('Modified Output')

        plt.draw()
        plt.pause(0.001)

    def main_v1(self):
        """
        Main function for CenterSpeed 1.0.
        Differs from main_v2 in the way the output is processed.
        """
        preprocessed_scans = torch.cat([self.frame1, self.frame2], dim=0)
        output_hm, output_data = self.inference(preprocessed_scans, version=1)
        output_data = output_data.squeeze(0)
        print(output_hm.shape)
        print(torch.max(output_hm))

        if not self.using_coral:
            output_hm = output_hm.squeeze(0).squeeze(0)
        else:
            output_hm = output_hm.squeeze(0).squeeze(-1)
        x,y = self.find_k_peaks(output_hm, self.num_opponents, 0.8)
        if x is not None:
            for x,y in zip(x,y):
                if np.sqrt(x**2 + y**2) > self.distance_threshold:#filter out points that are too far away
                    continue
                x_g, y_g = self.lidar_to_global(x,y)#transform the point to the global frame
                self.tracked_obstacles.append(Obstacle(x=x_g, y=y_g,vx=output_data[0], vy=-output_data[1], size=0.5, theta=output_data[2], id=0))#uses fixed size and id for now
        self.publishObstaclesMessageEstimation()#Directly give estimations without tracking.py
        self.publishObstaclesMarkers()#publish the markers
        self.tracked_obstacles.clear()#clear the obstacles for the next iteration
        if self.visualize:
            self.vis(preprocessed_scans[0], output_hm, output_data)

    def main_v2(self):
        """
        Main function for CenterSpeed 2.0.
        Differs from main_v1 in the way the output is processed.
        """
        preprocessed_scans = torch.cat([self.frame1, self.frame2], dim=0)
        plt.imshow(preprocessed_scans[0])
        plt.show()
        output = self.inference(preprocessed_scans)
        if not self.using_coral:
            output = output.squeeze(0).squeeze(0)
        else:
            output= output.squeeze(0)
            output = torch.permute(output, (2,0,1))
        plt.imshow(output[0])
        plt.show()
        y_ind,x_ind = self.find_k_peaks(output[0], self.num_opponents, 0.5, return_cartesian=False)
        if x_ind is not None:
            x,y = self.index_to_cartesian(y_ind, x_ind)
            for i, cords in enumerate(zip(x,y)):
                x,y = cords
                if np.sqrt(x**2 + y**2) > self.distance_threshold:#filter out points that are too far away
                    continue
                x_g, y_g = self.lidar_to_global(x,y)#transform the point to the global frame
                vx = output[1, int(x_ind[i]), int(y_ind[i])]
                vy = -output[2, int(x_ind[i]), int(y_ind[i])]
                theta = output[3, int(x_ind[i]), int(y_ind[i])]
                self.tracked_obstacles.append(Obstacle(x=x_g, y=y_g,vx=vx, vy=vy, size=0.5, theta=theta, id=0))#uses fixed size and id for now
        self.publishObstaclesMessageEstimation()#Directly give estimations without tracking.py
        self.publishObstaclesMarkers()#publish the markers
        self.tracked_obstacles.clear()#clear the obstacles for the next iteration

        # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        # ax[0].imshow(preprocessed_scans[0], origin='lower', cmap='plasma')
        # ax[1].imshow(output[0], origin='lower', cmap='plasma')
        # plt.show()
        if self.publish_image:
            self.publish_image_from_tensor(output[0])

    def main(self):
        '''
        Main function of the node.
        Multiplexes between the different versions of CenterSpeed (1.0 / 2.0).
        '''

        rate = rospy.Rate(self.rate)
        rospy.loginfo('[CenterSpeed]: Waiting for global wpnts')
        rospy.wait_for_message('/global_waypoints', WpntArray)
        rospy.loginfo('[CenterSpeed]: Waiting for subsequent scans')
        rospy.sleep(0.5)
        version = 2 if self.dense else 1
        self.load_model(version=version)
        self.grid_y, self.grid_x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))#for the peak detection, precompute the grid
        initial = True

        with torch.no_grad():
            while not rospy.is_shutdown():
                if self.measuring:#for measuring the latency
                    start_time = time.perf_counter()
                if initial:#initialize both frames
                    self.frame1 = self.preprocess(self.scan1)
                    self.frame2 = self.preprocess(self.scan2)
                    initial = False
                else:#shift the frames
                    self.frame1 = self.frame2
                    self.frame2 = self.preprocess(self.scan1)
                self.current_stamp = rospy.Time.now()
                if self.dense or self.using_coral:
                    self.main_v2()
                else:
                    self.main_v1()
                if self.measuring:
                    end_time = time.perf_counter()
                    latency = end_time - start_time
                    self.latency_pub.publish(latency)#publish the latency in ms
                rate.sleep()

if __name__ == '__main__':
    detect = CenterSpeed_Inference()
    detect.main()
