#!/usr/bin/env python3
import rosbag
import rospy
import numpy as np
import csv
import matplotlib.pyplot as plt
from frenet_converter import frenet_converter
import tf2_py as tf2
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point
import os




class RosbagDataset:
    def __init__(self, bag_file_path, outputfile, start,  end):
        self.bag_file_path = bag_file_path
        self.topics = ["/scan", "/opp/car_state/odom", "/opp/car_state/odom_frenet", "/tf", "/tf_static", "/car_state/odom"]
        self.topic_map = {'scan': 0, 'opp': 1, 'opp_frenet': 2, 'transform':3}
        self.all_messages = [[] for _ in range(6)]
        self.dataset = []
        self.dataset_csv = []
        self.initial_time = None
        self.time_offset = 0
        self.opp_time_offset = None
        self.opp_car_frenet = []
        self.output_file = outputfile
        self.test_dataset = False
        self.test_messages = [[] for _ in range(2)]
        self.start = start if start is not None else 0
        self.end = end if end is not None else 1

    def process_bag(self):
        '''
        Reads messages from a specified topics in the ROS bag file. (adapted from JFR-Plotting)
        '''
        rospy.loginfo("[Dataset Generator]: Processing bag")
        bag_obj = rosbag.Bag(self.bag_file_path)
        self.raceline = None
        for topic, msg, t in bag_obj.read_messages(topics=['/global_waypoints']):
            self.raceline = msg.wpnts
            self.track_length = msg.wpnts[-1].s_m
            break


        raceline_x = np.array([p.x_m for p in self.raceline])
        raceline_y = np.array([p.y_m for p in self.raceline])

        #Frenet converter
        self.fc = frenet_converter.FrenetConverter(raceline_x, raceline_y)

        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in self.raceline])


        # Second call: create the boundaries arrays

        # Second call: create the boundaries arrays

        #needed for corrections of data: fuse with detect.py
        rospy.loginfo('received global path')
        waypoint_array = self.raceline
        points=[]
        self.s_array = []
        self.d_right_array = []
        self.d_left_array = []
        for waypoint in waypoint_array:
            self.s_array.append(waypoint.s_m)
            self.d_right_array.append(waypoint.d_right-self.boundaries_inflation)
            self.d_left_array.append(waypoint.d_left-self.boundaries_inflation)
            resp = self.fc.get_cartesian(waypoint.s_m,-waypoint.d_right+self.boundaries_inflation)
            points.append(Point(resp[0],resp[1],0))
            resp = self.fc.get_cartesian(waypoint.s_m,waypoint.d_left-self.boundaries_inflation)
            points.append(Point(resp[0],resp[1],0))
        self.smallest_d = min(self.d_right_array+self.d_left_array)
        self.biggest_d = max(self.d_right_array+self.d_left_array)


        laser_scan_sync = None
        car_state_pose_sync = None
        initial_time = None
        time_offset = 0
        opp_time_offset = None
        ego_time_offset = None
        i =0
        boundary = None

        lasertimes_arr = np.arange(self.start,self.end, 0.025)#Used to controll the size of the dataset, 0.025  for the 40Hz lidar
        all_laser_point = []
        lasertimes = (t for t in lasertimes_arr)
        lasertime = next(lasertimes)
        laser_ranges = []
        laser_intensities = []

        car_positions = []
        car_speeds = []
        car_yaws = []
        car_times = []

        opp_car_positions = []
        opp_car_positions_lidar = []
        opp_car_yaws = []
        opp_car_times = []
        opp_car_frenet = []
        opp_car_frenet_sync = []


        tf_buffer = tf2.BufferCore(rospy.Duration(1e9))

        for topic, msg, t in bag_obj.read_messages(topics=self.topics):
            if initial_time is None:
                if ego_time_offset is None:
                    if topic == "/tf" or topic == "/tf_static":
                        ego_time_offset = t.to_sec() - msg.transforms[-1].header.stamp.to_sec()
                    else:
                        ego_time_offset = t.to_sec() - msg.header.stamp.to_sec()
                initial_time = t

            if topic == self.topics[0]:
                laser_scan_sync = msg

            if topic == '/car_state/odom':
                car_state_pose_sync = msg

            if topic == self.topics[1]:
                opp_car_state_pose_sync = msg

            if topic == self.topics[2]:
                if opp_time_offset is None:
                    opp_time_offset = initial_time.to_sec() - msg.header.stamp.to_sec()
                s = msg.pose.pose.position.x
                d = msg.pose.pose.position.y
                vs = msg.twist.twist.linear.x
                vd = msg.twist.twist.linear.y
                stamp = msg.header.stamp.to_sec() + opp_time_offset
                #TODO want velocity in cartesian coordinates#############################
                #vs, vd = self.fc.get_cartesian(vs, vd)
                ####################################
                opp_car_frenet.append(np.array([s, d, vs, vd, stamp]))


            if "tf" in topic:
                for msg_tf in msg.transforms:
                    if topic == "/tf_static":
                        tf_buffer.set_transform_static(msg_tf,'default_authority')
                    else:
                        tf_buffer.set_transform(msg_tf,'default_authority')

            # get relative time in milliseconds
            cur_t = (t - initial_time).to_sec() * 1000 - time_offset * 1000
            if cur_t/1000>=lasertime:
                car_pos = np.array([car_state_pose_sync.pose.pose.position.x, car_state_pose_sync.pose.pose.position.y])
                car_positions.append(car_pos)
                car_speed = np.array([car_state_pose_sync.twist.twist.linear.x, car_state_pose_sync.twist.twist.linear.y])
                car_speeds.append(car_speed)
                car_rot = Rotation.from_quat(np.array([car_state_pose_sync.pose.pose.orientation.x, car_state_pose_sync.pose.pose.orientation.y, car_state_pose_sync.pose.pose.orientation.z, car_state_pose_sync.pose.pose.orientation.w]))
                car_yaw = car_rot.as_euler('xyz')[2]
                car_yaws.append(car_yaw)
                car_times.append(car_state_pose_sync.header.stamp.to_sec() + ego_time_offset)

                #opponent stuff
                opp_car_pos = np.array([0,0])

                transform_pos = tf_buffer.lookup_transform_core("laser","map",rospy.Time(0))# want the opponents position in the lidar frame

                T_pos = np.array([transform_pos.transform.translation.x, transform_pos.transform.translation.y, transform_pos.transform.translation.z])
                R_pos = Rotation.from_quat(np.array([transform_pos.transform.rotation.x, transform_pos.transform.rotation.y, transform_pos.transform.rotation.z, transform_pos.transform.rotation.w]))
                opp_car_pos_lidar = R_pos.apply(np.vstack((opp_car_pos[0], opp_car_pos[1], np.zeros_like(opp_car_pos[0]))).T) + T_pos
                opp_car_positions.append(opp_car_pos)
                opp_car_positions_lidar.append(opp_car_pos_lidar)
                opp_car_rot = Rotation.from_quat(np.array([1, 0, 1, 0]))
                #-------------Transform Yaw into Ego Frame and Speed calculations-------
                opp_car_yaw = (opp_car_rot.as_euler('xyz')[2] - car_yaw) % (2* np.pi)
                if opp_car_yaw > np.pi:
                    opp_car_yaw -= 2*np.pi
                opp_car_yaws.append(opp_car_yaw)
                ROT = np.array([
                    [np.cos(-opp_car_yaw), - np.sin(-opp_car_yaw)],
                    [np.sin(-opp_car_yaw), np.cos(-opp_car_yaw)]])
                opp_velocity_bl = np.array([0, 0])
                opp_velocity_behind = np.dot(ROT, opp_velocity_bl)
                opp_car_frenet_sync.append(np.array([0,0,opp_velocity_behind[0], opp_velocity_behind[1], 0]))
                #-------------------------------------------------
                opp_car_times.append(0)

                # convert laser scans
                ranges = np.array(laser_scan_sync.ranges)
                angles = np.linspace(laser_scan_sync.angle_min, laser_scan_sync.angle_max, len(ranges))
                self.increment = laser_scan_sync.angle_increment
                x_laser = ranges * np.cos(angles)
                y_laser = ranges * np.sin(angles)
                R = Rotation.from_rotvec([0, 0, 0])
                T = np.array([0, 0, 0])
                x_map = R.apply(np.vstack((x_laser, y_laser, np.zeros_like(x_laser))).T) + T
                x_map = x_map.T

                all_laser_point.append(x_map)
                laser_ranges.append(laser_scan_sync.ranges)
                laser_intensities.append(laser_scan_sync.intensities)



                try:
                    lasertime = next(lasertimes)
                except StopIteration:
                    print(f"Done, we have processed all the lasertimes: {lasertimes_arr} [s]")
                    break
        #Store all necessary data for further processing
        self.all_messages[0] = all_laser_point#laser_intensities
        self.all_messages[1] = opp_car_positions_lidar
        self.all_messages[2] = opp_car_frenet_sync
        self.all_messages[3] = laser_ranges
        self.all_messages[4] = laser_intensities
        self.all_messages[5] = opp_car_yaws

        if self.test_dataset:
            self.test_messages[0] = car_yaws
            self.test_messages[1] = car_speeds
        print(np.array(all_laser_point[0][0], dtype=float))
        rospy.loginfo("[Dataset Generator]: Size of raw dataset: %s", [len(self.all_messages[i]) for i in range(4)])

    def dataset_creator(self):#unused now
        '''
        Creates a dictionary of the necessary data for easier csv-processing.
        '''
        self.headers = ['lidar','lidarx', 'lidary', 'x', 'y', 'vs', 'vd']
        self.dataset_csv = []
        size = 0

        for i in range(len(self.all_messages[0])):
            #TODO Do i want the ranges of the lidar in the set or the baselink coordinates?
            #Extract the data and write it in more useful format
            lidar = self.all_messages[3][i]
            lx = self.all_messages[self.topic_map['scan']][i][0]
            ly = self.all_messages[self.topic_map['scan']][i][1]
            x = self.all_messages[self.topic_map['opp']][i][0][0]
            y = self.all_messages[self.topic_map['opp']][i][0][1]
            vs = self.all_messages[self.topic_map['opp_frenet']][i][2]
            vd = self.all_messages[self.topic_map['opp_frenet']][i][3]
            self.dataset_csv.append({'lidar': lidar, 'lidarx': lx, 'lidary': ly,'x': x, 'y': y, 'vs': vs, 'vd': vd})
            size += 1
        rospy.loginfo("[Dataset Generator]: Dataset Created! Size: %s", size)


    def visualize_raw(self):
        '''
        Visualizes the raw data extracted from the rosbag.
        '''
        plt.figure(figsize=(10, 10))
        for i in range(len(self.all_messages[0])):

            print("point    ",self.all_messages[1][i])
            plt.scatter(self.all_messages[0][i][0], self.all_messages[0][i][1], s=0.1)
            print("point    ",self.all_messages[1][i][0])
            plt.scatter(self.all_messages[1][i][0][0], self.all_messages[1][i][0][1])
            plt.scatter(0,0)
            plt.axis('equal')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('2D LiDAR Scan Visualization')
            plt.show()

    def visualize_compact(self):
        '''
        Visualizes the data after the conversion into the dictionary.
        Used to verify the conversion.
        '''
        plt.figure(figsize=(10, 10))
        for i in range(len(self.all_messages[0])):

            rospy.loginfo("[Dataset Generator]: Currently showing data from time: %s s. This is the %s. entry in the dataset", 0.0125*i, i)
            plt.scatter(self.dataset_csv[i]['lidarx'], self.dataset_csv[i]['lidary'], s=0.1)
            plt.scatter(self.dataset_csv[i]['x'], self.dataset_csv[i]['y'])
            plt.scatter(0,0)
            plt.axis('equal')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('2D LiDAR Scan Visualization')
            plt.show()

    def write_csv(self):
        '''
        Writes the dataset to a csv file.
        '''
        with open(self.output_file, mode='w', newline='') as file:
            # Create a DictWriter object, specifying the file and the fieldnames
            writer = csv.DictWriter(file, fieldnames=self.headers)

            # Write the header row to the CSV file
            writer.writeheader()

            # Write the data rows to the CSV file
            for row in self.dataset_csv:
                writer.writerow(row)
        rospy.loginfo("[Dataset Generator]: Dataset written to: %s", self.output_file)

    def get_bag_info(self):
        '''
        Prints topic information from the current bag being analyzed.
        '''
        bag = rosbag.Bag(self.bag_file_path)
        topics = bag.get_type_and_topic_info()[1].keys()
        types = []
        for val in bag.get_type_and_topic_info()[1].values():
            types.append(val[0])

        for topic in topics:
            rospy.loginfo("[Dataset Generator] Topic: %s", topic)

if __name__ == '__main__':
    rospy.init_node('Bag_to_dataset', anonymous=True)
    bag_file_path = '/home/f1tenth/rosbags/perception/combined_var_egoopp_v0.bag'
    output_csv_path = "/home/f1tenth/catkin_ws/src/race_stack/perception/dataset/out.csv"  #specify the path to the Rosbag to be analyzed
    bag_reader = RosbagDataset(bag_file_path, output_csv_path)
    bag_reader.get_bag_info()
    bag_reader.process_bag()
    bag_reader.dataset_creator()
    #Uncomment to visualize the dataset!
    #bag_reader.visualize_compact()
    #uncomment to write to csv-file!
    bag_reader.write_csv()


    #     return xyz    #     return xyz