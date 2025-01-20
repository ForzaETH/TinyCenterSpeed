#!/usr/bin/env python3

import rospy
from bisect import bisect_left
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import os
import argparse

#import RosbagDataset class
from parse_bag import RosbagDataset

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
    def __init__(self,x,y,size,theta) -> None:

        self.center_x = x
        self.center_y = y
        self.size = size
        self.id = None
        self.theta = theta

    def squaredDist(self, obstacle):
        return (self.center_x-obstacle.center_x)**2+(self.center_y-obstacle.center_y)**2


class Detect(RosbagDataset):
    """
    This class extends the RosbagDataset class. It adds functionality to process the lidar-scans offline
    to provide additional information about GT-positions of the enemy in the given rosbag.
    """
    def __init__(self, bag_file, output, start, end, correct, speedy, only_gt) -> None:
        """
        Initialize the Detect class
        """
        super().__init__(bag_file, output, start, end)
        self.from_bag = False
        self.converter = None


        # --- Node properties ---
        rospy.init_node('StaticDynamic', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        # --- Tunable params ---
        #TODO: what are these parameters?
        #Not used because algo is offline:self.rate = 1#rospy.get_param("/detect/rate")
        self.lambda_angle = 10#rospy.get_param("/detect/lambda")*math.pi/180
        self.sigma = 0.03#rospy.get_param("/detect/sigma")
        self.min_2_points_dist = 0.1#rospy.get_param("/detect/min_2_points_dist")

        # --- dyn params sub ---
        self.min_obs_size = 10
        self.max_obs_size = 0.5
        self.max_viewing_distance = 9
        self.boundaries_inflation = 0.1

        # --- variables ---

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
        self.scans =None

        self.current_stamp = None
        self.tracked_obstacles = []
        self.tracked_obstacles_pointclouds = []
        self.path_needs_update = False

        self.final = False
        self.changed = 0
        self.correct = correct if correct is not None else True #True if you want to correct the positions, False if you want to use the transmitted positions
        self.speed_mode = speedy if speedy is not None else True #True if you want to correct the positions faster
        self.to_be_inspected = 0
        self.only_gt = only_gt if only_gt is not None else False #True if you want to use only the GT-positions

    def shutdown(self):
        rospy.logwarn('Detect is shutdown')

    def laserPointOnTrack (self, s, d):
        # this old version could not detect obstacles on the side/behind
        # if normalize_s(s-self.car_s,self.track_length)<self.max_dist or normalize_s(s-self.car_s,self.track_length)>self.max_viewing_distance:
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

    def scans2ObsPointCloud(self, laser_points_in_bl_frame):
        """
        Converts the lidar scans to a 2D PointCloud (adapded from detect.py)
        """

        # --- initialisation of some sutility parameters ---
        l = self.lambda_angle
        d_phi = self.increment #self.scans.angle_increment
        sigma = self.sigma
        xyz_map = laser_points_in_bl_frame

        cloudPoints_list = []
        for i in range(xyz_map.shape[1]):
            pt = (xyz_map[0,i], xyz_map[1,i])
            cloudPoints_list.append(pt)
        # --------------------------------------------------
        # segment the cloud point into smaller point clouds
        # that represent potential object using the adaptive
        # method
        # --------------------------------------------------

        objects_pointcloud_list = [[[cloudPoints_list[0][0],cloudPoints_list[0][1]]]]
        for idx, point in enumerate(cloudPoints_list):
            if (idx == 0):
                continue
            dist = math.sqrt(point[0]**2 + point[1]**2)
            d_max = (dist * math.sin(d_phi)/math.sin(l-d_phi)+3*sigma) / 2
            if (math.dist([cloudPoints_list[idx-1][0],cloudPoints_list[idx-1][1]],
            [point[0],point[1]])>d_max):
                objects_pointcloud_list.append([[point[0],point[1]]])
            else:
                objects_pointcloud_list[-1].append([point[0],point[1]])

        # ------------------------------------------------
        # removing point clouds that are too small or too
        # big or that have their center point not on the
        # track
        # ------------------------------------------------
        x_points = []
        y_points = []
        for obs in objects_pointcloud_list:
            x_points.append(obs[int(len(obs)/2)][0])
            y_points.append(obs[int(len(obs)/2)][1])
        s_points, d_points = self.fc.get_frenet(np.array(x_points), np.array(y_points))


        remove_array=[]
        for idx, object in enumerate(objects_pointcloud_list):
            # if len(object)<self.min_obs_size or len(object)>self.max_obs_size:
            if len(object) < self.min_obs_size:
                remove_array.append(object)
                continue
            if not (self.laserPointOnTrack(s_points[idx], d_points[idx])):
                remove_array.append(object)
                continue

        for object in remove_array:
            objects_pointcloud_list.remove(object)

        return objects_pointcloud_list

    def obsPointClouds2obsArray (self,objects_pointcloud_list):
        '''
        Turns the obstacle pointcloud into the obstacles array (adapted from detect.py)
        '''
        current_obstacle_array =[]
        min_dist = self.min_2_points_dist
        for obstacle in objects_pointcloud_list:

            # --- fit a rectangle to the data points ---
            theta = np.linspace(0,np.pi/2-np.pi/180,90)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            distance1 = np.dot(obstacle,[cos_theta,sin_theta])
            distance2 = np.dot(obstacle,[-sin_theta,cos_theta])
            D10 = -distance1 + np.amax(distance1,axis=0)
            D11 = distance1 - np.amin(distance1,axis=0)
            D20 = -distance2 + np.amax(distance2,axis=0)
            D21 = distance2 - np.amin(distance2,axis=0)
            min_array = np.argmin([np.linalg.norm(D10,axis=0),np.linalg.norm(D11,axis=0)],axis=0)
            D10 = np.transpose(D10)
            D11 = np.transpose(D11)
            D10[min_array==1]=D11[min_array==1]
            D10 = np.transpose(D10)
            min_array = np.argmin([np.linalg.norm(D20,axis=0),np.linalg.norm(D21,axis=0)],axis=0)
            D20 = np.transpose(D20)
            D21 = np.transpose(D21)
            D20[min_array==1]=D21[min_array==1]
            D20 = np.transpose(D20)
            D = np.minimum(D10,D20)
            D[D<min_dist]=min_dist

            # --------------------------------------------
            # extract the center of the obstacle assuming
            # that it is actually a square obstacle
            # --------------------------------------------

            theta_opt = np.argmax(np.sum(np.reciprocal(D),axis=0))*np.pi/180
            distances1 = np.dot(obstacle,[np.cos(theta_opt),np.sin(theta_opt)])
            distances2 = np.dot(obstacle,[-np.sin(theta_opt),np.cos(theta_opt)])
            max_dist1 = np.max(distances1)
            min_dist1 = np.min(distances1)
            max_dist2 = np.max(distances2)
            min_dist2 = np.min(distances2)

            # corners are detected in a anti_clockwise manner
            corner1 = None
            corner2 = None
            if(np.var(distances2)>np.var(distances1)): # the obstacle has more detection in the verticle direction
                if (np.linalg.norm(-distances1+max_dist1)<np.linalg.norm(distances1-min_dist1)):
                    #the detections are nearer to the right edge
                    # lower_right_corner
                    corner1 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*min_dist2])
                    # upper_right_corner
                    corner2 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*max_dist2])
                else :
                    #the detections are nearer to the left edge
                    # upper_left_corner
                    corner1 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*max_dist2])
                    # lower_left_corner
                    corner2 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*min_dist2])
            else: # the obstacle has more detection in the horizontal direction
                if (np.linalg.norm(-distances2+max_dist2)<np.linalg.norm(distances2-min_dist2)):
                    #the detections are nearer to the top edge
                    # upper_right_corner
                    corner1 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*max_dist2])
                    # upper_left_corner
                    corner2 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*max_dist2])
                else :
                    #the detections are nearer to the bottom edge
                    # lower_left_corner
                    corner1 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*min_dist2])
                    # lower_right_corner
                    corner2 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*min_dist2])
            # vector that goes from corner1 to corner2
            colVec = np.array([corner2[0]-corner1[0],corner2[1]-corner1[1]])
            # orthogonal vector to the one that goes from corner1 to corner2
            orthVec = np.array([-colVec[1],colVec[0]])
            # center position
            center = corner1 + 0.5*colVec + 0.5*orthVec

            current_obstacle_array.append(Obstacle(center[0],center[1],np.linalg.norm(colVec),theta_opt))


        return current_obstacle_array

    def checkObstacles(self, current_obstacles, current_obstacles_corresponding_pointclouds):
        '''
        Double-checks the obstacles (adapted from detect.py)
        '''
        remove_list = []
        remove_list_pointclouds = []
        self.tracked_obstacles.clear()
        self.tracked_obstacles_pointclouds.clear()
        for obs, pointcloud in zip(current_obstacles, current_obstacles_corresponding_pointclouds):
            if(obs.size > self.max_obs_size):
                remove_list.append(obs)
                remove_list_pointclouds.append(pointcloud)

        for obs in remove_list:
            current_obstacles.remove(obs)
        for pointcloud in remove_list_pointclouds:
            current_obstacles_corresponding_pointclouds.remove(pointcloud)

        for idx, curr_obs in enumerate(current_obstacles):
            curr_obs.id = idx
            self.tracked_obstacles.append(curr_obs)

        for idx, curr_obs_pointcloud in enumerate(current_obstacles_corresponding_pointclouds):
            self.tracked_obstacles_pointclouds.append(curr_obs_pointcloud)


    def positions(self):
        '''
        For each lidar scan the detection algorithm predicts a position for the opponent.
        Always the closest detected point to the GT-position from the rosbag is chosen. If they
        diverge too much the interactive plot is launched for manual inspection.
        '''
        self.detected_positions = []
        self.corrected_positions = []
        for i in range(len(self.all_messages[0])):
            object_pointcloud_list = self.scans2ObsPointCloud(self.all_messages[0][i])
            self.current_obstacles = self.obsPointClouds2obsArray(object_pointcloud_list)
            self.checkObstacles(self.current_obstacles, object_pointcloud_list)
            gt_pos = self.all_messages[1][i][0]
            min_dist = float('inf')
            min_candidate = None
            for candidate in self.current_obstacles:
                dist = np.sqrt((gt_pos[0]-candidate.center_x)**2 + (gt_pos[1]-candidate.center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_candidate = candidate
            self.detected_positions.append([min_candidate.center_x, min_candidate.center_y] if min_candidate is not None else [0,0])#Take the closest one to GT-pos from bag
            if self.only_gt:
                self.corrected_positions.append([gt_pos[0], gt_pos[1]])
                print("Only GT-Positions are used!")
                continue
            if min_dist > 0.6:
                if gt_pos[0] < 0:#opponent car is behind
                    self.corrected_positions.append([gt_pos[0], gt_pos[1]])
                    continue
                self.to_be_inspected += 1
                if self.correct:
                    self.launch_corrector(i)
            else:
                self.corrected_positions.append(self.detected_positions[-1])

        print("Number of weird points: ", self.to_be_inspected)


    def launch_corrector(self, i):
        '''
        Launches the interactive plot for manual inspection

        Args:
            - i: the index of the datapoint to be inspected
        '''
        print("Correcting Datapoint: ", i, " of ", len(self.all_messages[0]))
        self.i = i
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        axchoose_gt = self.fig.add_axes([0.4, 0.01, 0.1, 0.075])
        axchoose_det = self.fig.add_axes([0.6, 0.01, 0.1, 0.075])
        axcoose_next = self.fig.add_axes([0.8, 0.01, 0.1, 0.075])
        axchoose_discard = self.fig.add_axes([0.2, 0.01, 0.1, 0.075])
        axchoose_save = self.fig.add_axes([0.0, 0.01, 0.1, 0.075])
        btn_choose_gt = Button(axchoose_gt, 'GT')
        btn_choose_det = Button(axchoose_det, 'DET')
        btn_choose_next = Button(axcoose_next, 'Next')
        btn_discard = Button(axchoose_discard, 'Discard')
        btn_save = Button(axchoose_save, 'Save')
        btn_choose_gt.on_clicked(self.choose_gt_position)
        btn_choose_det.on_clicked(self.choose_det_position)
        btn_choose_next.on_clicked(self.next_scan)
        btn_discard.on_clicked(self.discard)
        btn_save.on_clicked(self.save)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.plot_scan(i)

    def on_click(self,event):
        '''
        Callback function for clicking inside the plot.

        Args:
            -event: captures what happens during the click
        '''
        if not self.final:
            if event.inaxes is self.ax:
                self.clicked_point = [event.xdata, event.ydata]
                self.corrected_positions.append([event.xdata, event.ydata]) #corrected position is clicked point now
                self.changed += 1
                self.update_plot()
                rospy.loginfo("Set point manually at: %s", self.clicked_point)
                if self.speed_mode:
                    plt.pause(0.3)
                    self.next_scan(1)
        else:
            pass

    def choose_gt_position(self, event):
        '''
        Callback function for GT button on interactive plot.
        '''
        if not self.final:
            rospy.loginfo("[Dataset Generator]: Use GT-Position!")
            self.corrected_positions.append([self.all_messages[1][self.i][0][0],self.all_messages[1][self.i][0][1]]) #corrected position is GT now
            self.changed += 1
            self.next_scan(event)
        else:
            pass

    def choose_det_position(self,event):
        '''
        Callback function for DET button on interactive plot.
        '''
        if not self.final:
            rospy.loginfo("[Dataset Generator]: Use Det-Position!")
            self.corrected_positions.append([self.detected_positions[self.i][0], self.detected_positions[self.i][1]]) #corrected position is Det now
            self.changed+= 1
            self.next_scan(event)
        else:
            pass

    def next_scan(self, event):
        '''
        Callback function for next button on interactive plot.
        '''
        if self.final:
            self.i += 1
            if self.i >= len(self.all_messages[0]):
                rospy.loginfo("[Dataset Generator]: End of scans reached.")
                plt.close(self.fig)
                self.i = len(self.all_messages[0]) - 1  # Stay at the last scan
            else:
                self.plot_scan(self.i)
        else:
            plt.close(self.fig)

    def discard(self, event):
        '''
        Callback function for discard button on interactive plot.
        '''
        if not self.final:
            rospy.loginfo("[Dataset Generator]: Discard this datapoint!")
            self.corrected_positions.append([None, None])#only so that inidces add up
            self.next_scan(event)

    def save(self, event):
        '''
        Callback function for save button on interactive plot.
        '''
        if not self.final:
            rospy.loginfo("[Dataset Generator]: Dataset saved!")
            self.dataset_creator_corrected()
            self.write_csv()


    def update_plot(self):
        '''
        Updates the plot after manually placing a GT-position for the opponent.
        '''
        if self.clicked_point:
            self.ax.scatter(self.clicked_point[0], self.clicked_point[1], color='magenta', zorder=5)
            self.ax.text(self.clicked_point[0], self.clicked_point[1], 'Manual', zorder=5)
            self.fig.canvas.draw_idle()


    def plot_scan(self, i):
        '''
        Plots the lidar scan.

        Args:
            - i: the index of the datapoint to be inspected
        '''
        scans = self.all_messages[0][i]
        gt_pos = self.all_messages[1][i][0]
        det_pos = self.detected_positions[i]
        if self.final:
            discarded = False
            correct = self.corrected_positions[i]
            rospy.loginfo("[Dataset Generator]: Showing Entry %s of %s", i, len(self.corrected_positions))
            if correct[0] is None:
                rospy.loginfo("[Dataset Generator]: This datapoint was discarded!")
                discarded = True



        # Clear the axes for fresh plot
        self.ax.clear()

        # Plotting elements
        self.ax.scatter(scans[0], scans[1], s=0.1, label='Scans')
        self.ax.scatter(gt_pos[0], gt_pos[1], color='green', label='GT-Pos')
        self.ax.text(gt_pos[0], gt_pos[1],'GT-Pos')
        self.ax.scatter(*det_pos, color='red', label='Det-Pos')
        self.ax.text(*det_pos,'Det-Pos')
        self.ax.scatter(0, 0, color='blue', label='Ego-Pos')
        self.ax.text(0, 0,'Ego-Pos')  # Ego position
        if self.final:
            if not discarded:
                self.ax.scatter(correct[0], correct[1], label='corrected')
                self.ax.text(correct[0], correct[1], 'Corrected')
            else:
                self.ax.text(1, 1, 'Discarded this Entry!', fontsize=30, color='red')



        # Adjusting view, focusing on GT-position
        dx = dy = 2
        self.ax.set_xlim(gt_pos[0] - dx, gt_pos[0] + dx)
        self.ax.set_ylim(gt_pos[1] - dy, gt_pos[1] + dy)
        # self.ax.set_xlabel('X coordinate')
        # self.ax.set_ylabel('Y coordinate')
        self.ax.set_title('Dataset Builder: 2D LiDAR Scan ')

        # Adding legend
        self.ax.legend()
        plt.show()


    def visualize_corrected(self):
        '''
        Visualizes the raw data extracted from the rosbag.
        '''
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        axchoose_gt = self.fig.add_axes([0.4, 0.01, 0.1, 0.075])
        axchoose_det = self.fig.add_axes([0.6, 0.01, 0.1, 0.075])
        axcoose_next = self.fig.add_axes([0.8, 0.01, 0.1, 0.075])
        btn_choose_gt = Button(axchoose_gt, 'GT')
        btn_choose_det = Button(axchoose_det, 'DET')
        btn_choose_next = Button(axcoose_next, 'Next')
        btn_choose_gt.on_clicked(self.choose_gt_position)
        btn_choose_det.on_clicked(self.choose_det_position)
        btn_choose_next.on_clicked(self.next_scan)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.i = 0
        self.plot_scan(self.i)

    def plot_final(self):
        '''
        Plots the final dataset as well as some infos.
        '''
        rospy.loginfo("[Dataset Generator]: This is the final data being visualized!")
        # print("Original Data: ", len(self.all_messages[0]))
        # print("Detected Data: ", len(self.detected_positions))
        # print("Corrected Data: ", len(self.corrected_positions))
        # print("Number of corrected Datapoints: ", self.changed)
        self.final = True
        self.visualize_corrected()

    def dataset_creator_corrected(self):
        '''
        Creates a dictionary of the necessary data for easier csv-processing.
        '''
        if self.test_dataset:
            self.headers = ['lidar', 'intensities', 'x', 'y', 'vs', 'vd', 'yaw', 'ego-vx', 'ego-vy' ,'ego-yaw']
        else:
            self.headers = ['lidar', 'intensities', 'x', 'y', 'vs', 'vd', 'yaw']

        self.dataset_csv = []
        size = 0
        discarded = 0
        rospy.loginfo("[Dataset Generator]: Creating Dataset!")
        rospy.loginfo("[Dataset Generator]: Number of all points: %s", len(self.all_messages[0]))
        rospy.loginfo("[Dataset Generator]: Number of points in dataset: %s", len(self.corrected_positions))
        for i in range(len(self.corrected_positions)):
            #Extract the data and write it in more useful format
            lidar = self.all_messages[3][i]
            lidar_intensities = self.all_messages[4][i]
            #lx = list(self.all_messages[self.topic_map['scan']][i][0])
            #ly = list(self.all_messages[self.topic_map['scan']][i][1])
            yaw = self.all_messages[5][i]
            x = self.corrected_positions[i][0]
            y = self.corrected_positions[i][1]
            if x is None:
                discarded += 1
                continue
            vs = self.all_messages[self.topic_map['opp_frenet']][i][2]
            vd = self.all_messages[self.topic_map['opp_frenet']][i][3]

            if self.test_dataset:
                ego_yaw = self.test_messages[0][i]
                ego_vx = self.test_messages[1][i][0]
                ego_vy = self.test_messages[1][i][1]
                self.dataset_csv.append({'lidar': lidar, 'intensities': lidar_intensities,'x': x, 'y': y, 'vs': vs, 'vd': vd, 'yaw': yaw, 'ego-vx': ego_vx, 'ego-vy':ego_vy, 'ego-yaw': ego_yaw})
            else:
                self.dataset_csv.append({'lidar': lidar, 'intensities': lidar_intensities,'x': x, 'y': y, 'vs': vs, 'vd': vd, 'yaw': yaw})
            size += 1
        rospy.loginfo("[Dataset Generator]: Dataset Created! Size: %s   %s Entries were discarded!", size, discarded)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, help="Path to the bag file")
    parser.add_argument("--output", type=str, help="Path to the output csv file")
    parser.add_argument("--start", type=int, help="Start time in the rosbag")
    parser.add_argument("--end", type=int, help="End time in the rosbag")
    parser.add_argument("--correct", type=bool, help="Correct the positions")
    parser.add_argument("--speedy", type=bool, help="Speedy mode")
    parser.add_argument("--only_gt", type=bool, help="Use only GT-positions")
    args = parser.parse_args()

    if args.bag is None:
        print("[Dataset Creator]: Please provide the path to the bag file!")
        exit()
    if args.output is None:
        output_csv_path = args.bag.split('.')[0] + ".csv"
    else:
        output_csv_path = args.output
    if not os.path.exists(args.bag):
        print("[Dataset Creator]: Bag file does not exist!")
        exit()
    if not os.path.exists(os.getcwd() + '/dataset_output'):
        os.makedirs(os.getcwd() +'/dataset_output')
    output_csv_path = os.getcwd() + '/dataset_output/' + output_csv_path
    if os.path.exists(output_csv_path):
        print("[Dataset Creator]: Output file already exists! Please delete or rename it!")
        exit()
    corrector = Detect(args.bag, output_csv_path, args.start, args.end, args.correct, args.speedy, args.only_gt)
    corrector.process_bag()
    corrector.positions()
    corrector.dataset_creator_corrected()
    corrector.write_csv()
    corrector.plot_final()
