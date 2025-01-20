#!/usr/bin/env python3
import rospy
import tf
import tf2_ros
from tf2_geometry_msgs import tf2_geometry_msgs
import torch
import time
from frenet_converter.frenet_converter import FrenetConverter
from f110_msgs.msg import WpntArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32, Float32MultiArray
from bisect import bisect_left
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import quaternion_from_euler
from dynamic_reconfigure.msg import Config
import matplotlib.pyplot as plt
from models.CenterSpeed import *
from visualization_msgs.msg import Marker,MarkerArray
from scipy.spatial.transform import Rotation
import tensorrt as trt
##profiling
# import cProfile
# import pstats
from f110_msgs.msg import ObstacleArray
from f110_msgs.msg import Obstacle as ObstacleMessage

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
        self.vx = vx
        self.vy = vy
        self.size = size
        self.id = id
        self.theta = theta

    def squaredDist(self, obstacle):
        return (self.center_x-obstacle.center_x)**2+(self.center_y-obstacle.center_y)**2


class CenterSpeed_Inference :
    """
    This class implements a ROS node that detects obstacles on the track using ML-Methods.

    It subscribes to the following topics:
        - `/scan`: Publishes the lidar scans.
        - `/global_waypoints`: Publishes the global waypoints.
        - `/odom_frenet`: Publishes the car state in frenet frame.


    The node publishes the following topics:
        - `/breakpoints_markers`: Publishes the breakpoint markers of the obstacles.
        - `/detect_bound`: Publishes the detect boundaries.
        - `/raw_obstacles`: Publishes the detected obstacles.
        - `/obstacles_markers_new`: Publishes the markers of the detected obstacles.
        - `/obstacles`: Publishes the estimated obstacles.
        - `/detection/latency`: Publishes the latency of the detection.
    """
    def __init__(self) -> None:
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        self.from_bag = False
        self.converter = None

        # --- Node properties ---
        rospy.init_node('StaticDynamic', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        self.measuring = rospy.get_param("/measure", False)

        # --- Subscribers ---
        rospy.Subscriber('/scan', LaserScan, self.laserCb)
        rospy.Subscriber('/global_waypoints', WpntArray, self.pathCb)
        rospy.Subscriber('/car_state/odom_frenet', Odometry, self.carStateCb)
        if not self.from_bag:
            rospy.Subscriber("/dynamic_tracker_server/parameter_updates", Config, self.dyn_param_cb)

        # --- Publisher ---
        self.breakpoints_markers_pub = rospy.Publisher('/perception/breakpoints_markers', MarkerArray, queue_size=5)
        self.boundaries_pub = rospy.Publisher('/perception/detect_bound', Marker, queue_size=5)
        self.obstacles_msg_pub = rospy.Publisher('/perception/detection/raw_obstacles', ObstacleArray, queue_size=5)
        self.obstacles_msg_pub_estimated = rospy.Publisher('/perception/obstacles', ObstacleArray, queue_size=5)
        self.obstacles_marker_pub = rospy.Publisher('/perception/obstacles_markers_new', MarkerArray, queue_size=5)
        if self.measuring:
            self.latency_pub = rospy.Publisher('/perception/detection/latency', Float32, queue_size=5)
        
        self.marker_pub = rospy.Publisher('/visualization_marker_opp', Marker, queue_size=5)
        # --- Tunable params ---
       
        # --- dyn params sub ---
        self.min_obs_size = 10
        self.max_obs_size = 0.5
        self.max_viewing_distance = 9
        self.boundaries_inflation = 0.1

        # --- variables for Centerspeed ---
        self.pixelsize = 0.1#size of a pixel in meters
        self.image_size = 64 #size of the image for preprocessing
        self.feature_size = 3 #number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize #origin is in the middle of the image
        self.rate = 40
        self.frame1 = None
        self.frame2 = None
        self.num_opponents = 1
        self.opp_coordinates = np.zeros((self.num_opponents, 2))

        # track variables
        self.waypoints = None
        self.biggest_d = None
        self.smallest_d = None
        self.s_array = None
        self.d_right_array = None
        self.d_left_array = None
        self.track_length =None

        # ego car s position
        self.car_s = 0

        # raw scans from the lidar
        self.scan1 =None
        self.scan2 = None

        self.current_stamp = None
        self.tracked_obstacles = []
        self.tracked_obstacles_pointclouds = []
        self.angles = None
        self.cos_angles = None
        self.sin_angles = None

        self.tf_listener = tf.TransformListener()
        self.path_needs_update = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        #self.processed = True

        while(self.waypoints is None):
            rospy.sleep(0.1)
            print("waiting for waypoints ...")

        self.converter = self.initialize_converter()

    def shutdown(self):
        '''
        Shutdown the node
        '''
        rospy.logwarn('Detect is shutdown')

    def initialize_converter(self) -> bool:
        """
        Initialize the FrenetConverter object
        """
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.psi_array)
        rospy.loginfo("[Perception Detect] initialized FrenetConverter object")

        return converter

    def laserCb(self,msg):
        '''
        Callback for the lidar scan
        '''
        if self.angles is None:
            self.angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.cos_angles = np.cos(self.angles)
            self.sin_angles = np.sin(self.angles)
        if self.scan1 is None:
            self.scan1 = msg
        else:
            self.scan2 = self.scan1
            self.scan1 = msg

    def pathCb(self,data):
        '''
        Callback for the global waypoints
        '''
        # Initial calls: initialize the converter
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.psi_array = np.array([wpnt.psi_rad for wpnt in data.wpnts])
        self.track_length = data.wpnts[-1].s_m

        # Second call: create the boundaries arrays
        if (self.s_array is None or self.path_needs_update) and self.converter is not None:
            rospy.loginfo('received global path')
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
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.id = 0
            marker.type = marker.SPHERE_LIST
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1
            marker.color.g = 0.
            marker.color.r = 1.
            marker.color.b = 0.
            marker.points = points

            self.boundaries_pub.publish(marker)
        self.path_needs_update = False

    def carStateCb(self,data):
        '''
        Callback for the car state
        '''
        self.ego_s = data.pose.pose.position.x
        self.ego_yaw = Rotation.from_quat(np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])).as_euler('xyz')[2]

    def dyn_param_cb(self, params: Config):
        '''
        Callback for the dynamic reconfigure parameters
        '''
        self.min_obs_size = rospy.get_param('dynamic_tracker_server/min_obs_size', 10)
        self.max_obs_size = rospy.get_param('dynamic_tracker_server/max_obs_size', 0.5)
        self.max_viewing_distance = rospy.get_param('dynamic_tracker_server/max_viewing_distance', 9)
        self.boundaries_inflation = rospy.get_param('dynamic_tracker_server/boundaries_inflation', 0.1)

        self.path_needs_update = True
        param_list = [self.min_obs_size, self.max_obs_size, self.max_viewing_distance]
        print(f'[DETECT] New dyn reconf values recieved: Min size [laser points], Max size [m], max viewing dist [m]: {param_list}')
        
    def clearmarkers(self):
        '''
        Clears the markers
        '''
        marker = Marker()
        marker.action = 3
        return [marker]
    
    def laserPointOnTrack (self, s, d):
        '''
        Checks if a point in the frenet frame is on the track.
        (Adapted from detect.py)
        
        Args:
            - s: the s-coordinate of the point
            - d: the d-coordinate of the point
        
        Returns:
            - True if the point is on the track, False otherwise
        '''
        if normalize_s(s-self.car_s,self.track_length)>self.max_viewing_distance:
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

    
    def preprocess(self, scan: LaserScan):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
        Stores these grids in a new tensor.
        '''
        lidar_data = np.array(scan.ranges, dtype=np.float16)
        intensities = np.array(scan.intensities, dtype=np.float16) 
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())      

        #preprocess the lidar data
        preprocessed_scans = np.zeros((1,self.feature_size, self.image_size, self.image_size), dtype=np.float16)
        x = lidar_data * self.cos_angles
        y = lidar_data * self.sin_angles
        x_coord = ((x + self.origin_offset) / self.pixelsize).astype(int)
        y_coord = ((y + self.origin_offset) / self.pixelsize).astype(int)
       
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        preprocessed_scans[:,0,y_coord, x_coord] = 1 #set the pixel to occupied
        #TODO: Does this work? I think it does
        preprocessed_scans[:,1,y_coord, x_coord] = np.maximum(preprocessed_scans[:,1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
        preprocessed_scans[:,2,y_coord, x_coord] += 1 #count the number of points in the pixel
        return preprocessed_scans
    
    def index_to_cartesian(self, x_img,y_img):
        '''
        Converts the index of the imagespace back to cartesian coordinates.

        Args:
            - x_img: the x-coordinate in the image space
            - y_img: the y-coordinate in the image space
        '''
        x = x_img * self.pixelsize - self.origin_offset
        y = y_img * self.pixelsize - self.origin_offset
        return x,y
        
    def find_k_peaks(self, image, k, radius, return_cartesian = True):
        '''Finds the k highest peaks in the image, which are at least radius pixels apart.

        Args:
            - image (torch.Tensor): the image to be analyzed
            - k (int): the number of peaks to be found
            - radius (float): the minimum distance between the peaks
            - return_cartesian (bool): whether to return the peaks in cartesian coordinates or image coordinates

        Returns:
            - x (torch.Tensor): the x-coordinates of the peaks
            - y (torch.Tensor): the y-coordinates of the peaks
        '''
        radius = radius/self.pixelsize
        image = image.clone()

        for i in range(k):
            # Find the maximum value in the image
            max_idx = np.argmax(image.reshape(-1))
            max_coords = np.unravel_index(max_idx, image.shape)
            self.opp_coordinates[i] = max_coords
            if i == k-1:
                break
            # Set the surrounding area to zero
            top = max(0, max_coords[0] - radius)
            bottom = min(image.shape[0], max_coords[0] + radius)
            left = max(0, max_coords[1] - radius)
            right = min(image.shape[1], max_coords[1] + radius)
            image[top:bottom, left:right] = 0

        if return_cartesian:
            return self.index_to_cartesian(self.opp_coordinates[:,1], self.opp_coordinates[:,0])               

        return self.opp_coordinates[:,1], self.opp_coordinates[:,0]
    
    def load_model(self):
        """
        Load the model from the specified path.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess_device = torch.device('cpu')
        rospy.loginfo("[CenterSpeed]: Using Device for inference:  %s",self.device)
        rospy.loginfo("[CenterSpeed]: Using Device for preprocessing:  %s",self.preprocess_device)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open('trained_models/CenterSpeed64.engine', 'rb') as f:
            selialized_engine = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(selialized_engine)

        #TODO: optimize model here, add config
        self.context = self.engine.create_execution_context()

        self.input = torch.empty(1,6,64,64, device='cuda', dtype=torch.float32)
        self.output_hm = torch.empty(1, 1, 64, 64, device='cuda', dtype=torch.float32)
        self.output_data = torch.empty(1, 3, device='cuda', dtype=torch.float32)

        self.context.set_input_shape('input.1', (1, 6, 64, 64))
        self.context.set_tensor_address("input.1", self.input.data_ptr())
        self.context.set_tensor_address("35", self.output_data.data_ptr())
        self.context.set_tensor_address("40", self.output_hm.data_ptr())

        rospy.loginfo('[CenterSpeed]: TensorRT Model loaded!')


    def inference(self, preprocess_data):
        """
        This function takes the preprocessed data and returns the model output.
        """
        self.input = torch.tensor(preprocess_data, device=self.device, dtype=torch.float32)
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        output_hm = self.output_hm.cpu()
        output_data = self.output_data.cpu()
        return output_hm, output_data

    def preallocate(self):
        '''
        Preallocates memory on the GPU for the lidar data, intensities and preprocessed scans.
        '''
        self.lidar_data = torch.zeros((self.len_lidar,), dtype=torch.float16, device=self.device)
        self.intensities = torch.zeros((self.len_lidar,), dtype=torch.float16, device=self.device)
        self.preprocessed_scans = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float16, device=self.device)
        rospy.loginfo('[CenterSpeed]: Memory preallocated on GPU')

    def publishObstaclesMessageEstimation (self):
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
            else:#If no raceline is used, only for testing TODO: remove
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
            obsMsg.is_opponent = True
            #print(f"Obstacle {obstacle.id}: Velocity in lidar frame: vx={obstacle.vx}, vy={obstacle.vy}")
            yaw_diff = (obstacle.theta)%(2*np.pi)
            if yaw_diff > np.pi:
                yaw_diff -= 2*np.pi
            yaw_diff = -yaw_diff
            ROT = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff)],
                            [np.sin(yaw_diff), np.cos(yaw_diff)]])
            speed = np.array([obstacle.vx, obstacle.vy])
            vx, vy = np.dot(ROT, speed)#the velocity in the opponents frame
            #print(f"Obstacle {obstacle.id}: Velocity in obstacle frame: vx={vx}, vy={vy}, s: {s}, d: {d}")

            ###Publish cartesian velocities in global frame###
            if self.measuring:#for catkin test also cartesian velocities are used!
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
            vs, vd = self.converter.get_frenet_velocities(vx, vy, obstacle_orientation_global, s)#convert cartesian velocities to frenet velocities
            #print(f"VS: {vs}, VD: {vd}")

            obsMsg.vs = vs 
            obsMsg.vd = vd 
            obsMsg.is_static = vs < 0.1 and vd < 0.1
            obstacles_array_message.obstacles.append(obsMsg)

        if len(remove_list) > 0:
            for obs in remove_list:
                if obs in self.tracked_obstacles:
                    self.tracked_obstacles.remove(obs)

        self.obstacles_msg_pub_estimated.publish(obstacles_array_message)


    def publishObstaclesMarkers(self):
        markers_array = []
        for obs in self.tracked_obstacles:
            if True: 
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.current_stamp
                marker.id = obs.id
                marker.type = marker.CUBE
                marker.scale.x = obs.size
                marker.scale.y = obs.size*1.5
                marker.scale.z = obs.size
                marker.color.a = 0.5
                marker.color.g = 1.
                marker.color.r = 0.
                marker.color.b = 1.
                marker.pose.position.x = obs.center_x
                marker.pose.position.y = obs.center_y
                q = quaternion_from_euler(0, 0, obs.theta)
                marker.pose.orientation.x = q[0]
                marker.pose.orientation.y = q[1]
                marker.pose.orientation.z = q[2]
                marker.pose.orientation.w = q[3]
                markers_array.append(marker)
        self.obstacles_marker_pub.publish(self.clearmarkers())
        self.obstacles_marker_pub.publish(markers_array)
        Obstacle.current_id = 0

    def lidar_to_global(self, x, y):
        '''
        Transforms the detected obstacle to the global frame.
        ''' 
        point_in_lidar_frame = PointStamped()
        point_in_lidar_frame.header.frame_id = 'laser'
        point_in_lidar_frame.header.stamp = rospy.Time.now()
        point_in_lidar_frame.point = Point(x, y, 0)

        t_bl = self.tf_buffer.lookup_transform(target_frame='base_link', source_frame='laser', time=rospy.Time(0))
        t_global = self.tf_buffer.lookup_transform(target_frame='map', source_frame='base_link', time=rospy.Time(0))

        point_in_base_link = tf2_geometry_msgs.do_transform_point(point_in_lidar_frame, t_bl)

        point_in_global_frame = tf2_geometry_msgs.do_transform_point(point_in_base_link, t_global)

        return point_in_global_frame.point.x, point_in_global_frame.point.y

    def vis(self, i, output):
            i_vis = np.array(i, dtype=np.float32)
            output_vis = np.array(output, dtype=np.float32)
            if not hasattr(self, 'fig'):
                self.fig, self.axs = plt.subplots(1, 3, figsize=(10, 10))
                plt.ion()
                plt.show()

            # Clear previous plots
            for ax in self.axs:
                ax.clear()

            # Plot Input
            self.axs[0].imshow(i_vis, origin='lower', cmap='plasma')
            self.axs[0].set_title('Input')

            # Plot output
            self.axs[1].imshow(output_vis, origin='lower', cmap='plasma')
            self.axs[1].set_title('Output')

            # Plot predicted peak on output
            x,y = self.find_k_peaks(output, 3, 20, return_cartesian=False)
            self.axs[2].imshow(output_vis, origin='lower', cmap='plasma')
            self.axs[2].scatter(x,y, color='red', s=10)
            self.axs[2].set_title('Modified Output')

            plt.draw()
            plt.pause(0.001)
            
    def main (self):
        rate = rospy.Rate(self.rate)
        self.load_model()
        self.grid_y, self.grid_x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size))#for the peak detection
        
        rospy.loginfo('[CenterSpeed]: Waiting for global wpnts')
        rospy.wait_for_message('/global_waypoints', WpntArray)
        self.converter = self.initialize_converter()
        rospy.loginfo('[CenterSpeed]: Waiting for subsequent scans')
        rospy.sleep(1)
        
        #latencies = []
        initial = True
        #profiler = cProfile.Profile()
        #profiler.enable()
        rospy.loginfo('[CenterSpeed]: Ready')
        self.measuring = True
        while not rospy.is_shutdown():
            if self.measuring:
                start_time = time.perf_counter()
            if initial:
                #self.processed = False
                self.frame1 = self.preprocess(self.scan1)
                self.frame2 = self.preprocess(self.scan2)
                initial = False
            else:
                #self.processed = False
                self.frame1 = self.frame2
                self.frame2 = self.preprocess(self.scan1)

            self.current_stamp = rospy.Time.now()
            preprocessed_scans = torch.cat([torch.tensor(self.frame1),torch.tensor(self.frame2)], dim=1)
            output_hm, output_data = self.inference(preprocessed_scans.clone())
            self.processed = True
            output_data = output_data.squeeze(0)
            output_hm = output_hm.squeeze(0).squeeze(0)
            x,y = self.find_k_peaks(output_hm, self.num_opponents, 8)
            #add the detected obstacle to the list of tracked obstacles
            if x is not None:
                for x,y in zip(x,y):
                    x_g, y_g = self.lidar_to_global(x,y)
                    self.tracked_obstacles.append(Obstacle(x=x_g, y=y_g,vx=output_data[0], vy=output_data[1], size=0.5, theta=output_data[2], id=0))
            self.publishObstaclesMessageEstimation()#Directly give estimations without tracking.py
            self.publishObstaclesMarkers()
            self.tracked_obstacles.clear()
            if self.measuring:
                end_time = time.perf_counter()
                latency = end_time - start_time
                self.latency_pub.publish(latency)
            rate.sleep()
        #profiler.disable()
        #stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
        #stats.print_stats(200)
        
if __name__ == '__main__':
    detect = CenterSpeed_Inference()
    detect.main()
