#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import conv2d
import csv
import ast
import random


class LidarDataset(Dataset):
    def __init__(self, dataset_path):
        '''
        Initialize the dataset object.
        Gets labels, lidar-data and the rest of the data from the provided csv file
        '''
        self.lidar_data = []
        self.lidarx = [] #lidar data in BL-frame
        self.lidary = [] #lidar data in BL-frame
        self.data = []
        self.intenisies = []
        self.time_dependent  = False
        self.time_steps = 5
        self.use_heatmaps = False
        self.pixelsize = 0.015#size of a pixel in meters
        self.image_size = 256 #size of the image for preprocessing
        self.feature_size = 3 #number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5 #standard deviation of the gaussian peaks
        with open(dataset_path, "r") as dataset_file:
            csv_reader = csv.DictReader(dataset_file)
            
            for row in csv_reader:
                self.lidar_data.append(ast.literal_eval(row['lidar']))# because the lidar array is saved as a string in the csv file
                self.intenisies.append(ast.literal_eval(row['intensities']))
                self.data.append([row[column] for column in list(row.keys())[2:]]) # skip the base link frame lidar scans for now, only raw lidar data?

        self.labels = list(row.keys())
        self.intenisies = torch.tensor(np.array(self.intenisies, dtype=float), dtype=torch.float32)
        self.lidar_data = torch.tensor(np.array(self.lidar_data, dtype=float), dtype=torch.float32)#hacky casting because data is a string originally
        self.data = torch.tensor(np.array(self.data, dtype=float), dtype=torch.float32)#------------^

        self.normintens = (self.intenisies - self.intenisies.min()) / (self.intenisies.max() - self.intenisies.min())
    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        if self.time_dependent:#if the dataset is time dependent, the length is reduced by the number of time steps
            return len(self.data) - self.time_steps
        else:
            return len(self.data)
    
    def __getitem__(self, index):
        '''
        Gets an item from the dataset

        Args:
            index: the index, which should be returned.
        
        '''
        if torch.is_tensor(index):
            index = index.tolist()

        if self.time_dependent:
            if index + self.time_steps > len(self.data):
                raise IndexError("Index out of range")
            lidar = self.lidar_data[index:index+self.time_steps]
            intensity = self.intenisies[index:index+self.time_steps]
            data = self.data[index:index+self.time_steps]
        elif self.use_heatmaps:
            input = self.input_data[index]
            gt = self.heatmaps[index]
            data = self.data[index]
            return input, gt, data
        else:
            lidar = self.lidar_data[index]
            intensity = self.intenisies[index]
            data = self.data[index]

        return lidar, intensity, data
    
    def add_data(self, csv_file):
        '''
        Adds data to the dataset

        Args:
            - csv_file: the csv file to be added
        '''
        new_lidar_data = []
        new_intensities = []
        new_data = []

        with open(csv_file, "r") as dataset_file:
            csv_reader = csv.DictReader(dataset_file)
            
            for row in csv_reader:
                assert self.labels == list(row.keys()), "The labels of the new csv file do not match the labels of the dataset"

                new_lidar_data.append(ast.literal_eval(row['lidar']))# because the lidar array is saved as a string in the csv file
                new_intensities.append(ast.literal_eval(row['intensities']))
                new_data.append([row[column] for column in list(row.keys())[2:]])
            new_intensities = torch.tensor(np.array(self.intenisies, dtype=float), dtype=torch.float32)#hacky casting because data is a string originally
            new_lidar_data = torch.tensor(np.array(self.lidar_data, dtype=float), dtype=torch.float32)#hacky casting because data is a string originally
            new_data = torch.tensor(np.array(self.data, dtype=float), dtype=torch.float32)#------------^
            new_normintens = (new_intensities - new_intensities.min()) / (new_intensities.max() - new_intensities.min()) 

            self.lidar_data = torch.cat((self.lidar_data, new_lidar_data), 0)
            self.intenisies = torch.cat((self.intenisies, new_intensities), 0)
            self.data = torch.cat((self.data, new_data), 0)
            self.normintens = torch.cat((self.normintens, new_normintens), 0)

    def time_dependent(self, time_steps):
        '''
        Transforms the datapoints into time dependent data. 
        This is done to create a time series dataset.

        Args:
            - time_steps: the number of time steps to be considered for one datapoint
        '''
        concat_data = torch.tensor([0 for _ in range(len(self.data)-time_steps)])
        concat_lidar_data = torch.tensor([0 for _ in range(len(self.lidar_data)-time_steps)])
        concat_intenisies = torch.tensor([0 for _ in range(len(self.intenisies)-time_steps)])
        concat_normintens = torch.tensor([0 for _ in range(len(self.normintens)- time_steps)])
        for i in range(len(self.data) - time_steps):
            concat_data[i] = torch.cat([self.data[i+j] for j in range(time_steps)], 0)
            concat_lidar_data[i] = torch.cat([self.lidar_data[i+j] for j in range(time_steps)], 0)
            concat_intenisies[i] = torch.cat([self.intenisies[i+j] for j in range(time_steps)], 0)
            concat_normintens[i] = torch.cat([self.normintens[i+j] for j in range(time_steps)], 0)

        self.data = concat_data
        self.lidar_data = concat_lidar_data
        self.intenisies = concat_intenisies
        self.normintens = concat_normintens



    def vis(self, index, point=[0,0]):
        '''
        Plots the lidar scan.
        
        Args:
            - index: the index of the datapoint to be inspected
            - point: a point to be highlighted, prediction
        '''
        if not self.time_dependent:
            x = self.lidar_data[index] * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
            y = self.lidar_data[index] * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
            plt.scatter(x, y, s=0.1, label='Scans', alpha=self.normintens[index])
            plt.scatter(self.data[index][0], self.data[index][1], color='green', label='GT-Pos')
            plt.text(self.data[index][0], self.data[index][1],'GT-Pos')

            label = 'Point' if point != [0,0] else 'Ego-Pos'
            plt.scatter(point[0], point[1], color='red', label=label)
            plt.text(point[0], point[1],label)  # Point position

            #connect the points:
            label = 'Error: '+ str(self.error(index, point)) if point != [0,0] else 'Distance to Opponent: ' + str(self.error(index, point))
            plt.plot([self.data[index][0], point[0]], [self.data[index][1], point[1]], 'red', linewidth=0.5, alpha=0.5, label=label)
            # Adjusting view, focusing on GT-position
            dx = dy = 2
            plt.xlim(self.data[index][0] - dx, self.data[index][0] + dx)
            plt.ylim(self.data[index][1] - dy,self.data[index][1] + dy)
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('2D LiDAR Scan Visualization')

            # Adding legend
            plt.legend()
            plt.show()
            return
        else:# if the dataset is time dependent multiple scans are plotted corresponding to the time steps
            for i in range(self.time_steps):
                x = self.lidar_data[index+i] * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
                y = self.lidar_data[index+i] * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
                plt.scatter(x, y, s=0.1, label='Scans', alpha=self.normintens[index+ i])
                plt.scatter(self.data[index+i][0], self.data[index+i][1], color='green', label='GT-Pos')
                plt.text(self.data[index+i][0], self.data[index+i][1],'GT-Pos')
                if i == self.time_steps -1:
                    plt.scatter(point[0], point[1], color='red', label='Point')
                    plt.text(point[0], point[1],'Point')
                dx = dy = 2
                plt.xlim(self.data[index][0] - dx, self.data[index][0] + dx)
                plt.ylim(self.data[index][1] - dy,self.data[index][1] + dy)
                plt.xlabel('X coordinate')
                plt.ylabel('Y coordinate')
                plt.title('2D LiDAR Scan Visualization')
                plt.legend()
                plt.show()

    def error(self, index, point):
        '''
        Calculates the error between the ground truth position and the given point.

        Args:
            - index: the index of the datapoint to be inspected
            - point: the point to be compared with the ground truth position
        '''
        gt_pos = self.data[index,:2]
        error = np.sqrt((gt_pos[0] - point[0])**2 + (gt_pos[1] - point[1])**2)
        return error.item()
    

    def augment(self):
        '''
        Augments the dataset by flipping the lidar scans and the ground truth positions
        '''
        #TODO Do I keep the normal data or both flipped and normal data?
        if self.time_dependent:
            raise ValueError("Cannot augment time dependent data")

        # Create lists to hold the augmented data
        lidar_data_aug = []
        data_aug = []
        intensities_aug = []
        normintens_aug = []

        for index in range(len(self.data)):
            if bool(random.getrandbits(1)):  # flip the data with a 50% chance
                lidar_data_aug.append(torch.flip(self.lidar_data[index], [0]).unsqueeze(0))
                flipped_data = self.data[index].clone()
                flipped_data[1] = -flipped_data[1]  # flip the y-coordinate
                flipped_data[3] = -flipped_data[3]  # flip the y-coordinate velocity
                flipped_data[4] = -flipped_data[4]  # flip the orientation
                data_aug.append(flipped_data.unsqueeze(0))
                intensities_aug.append(torch.flip(self.intenisies[index], [0]).unsqueeze(0))
                normintens_aug.append(torch.flip(self.normintens[index], [0]).unsqueeze(0))

        # Convert lists to tensors and concatenate with original data
        self.lidar_data = torch.cat((self.lidar_data, torch.cat(lidar_data_aug, 0)), 0)
        self.data = torch.cat((self.data, torch.cat(data_aug, 0)), 0)
        self.intenisies = torch.cat((self.intenisies, torch.cat(intensities_aug, 0)), 0)
        self.normintens = torch.cat((self.normintens, torch.cat(normintens_aug, 0)), 0)

        print("Augmented! New length: ", len(self.data))

    def preprocess(self):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
        Stores these grids in a new tensor.
        '''

        self.use_heatmaps = True#use heatmaps for training after preprocessing

        self.input_data = torch.zeros((len(self.lidar_data), self.image_size, self.image_size, self.feature_size), dtype=torch.float32)
        for i, scan in enumerate(self.lidar_data):
            x = scan * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
            y = scan * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
            x_coord = ((x + self.origin_offset) / self.pixelsize)
            y_coord = ((y + self.origin_offset) / self.pixelsize)
            x_coord = x_coord.to(torch.int)
            y_coord = y_coord.to(torch.int)
            valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
            x_coord = x_coord[valid_indices]
            y_coord = y_coord[valid_indices]
            self.input_data[i,y_coord, x_coord, 0] = 1 #set the pixel to occupied
            self.input_data[i,y_coord, x_coord, 1] = torch.maximum(self.input_data[i, y_coord,x_coord,1], self.normintens[i, valid_indices])#store the maximum intensity value in the pixel
            self.input_data[i,y_coord, x_coord, 2] +=1 #count the number of points in the pixel
        print("Preprocessed data!")    
        
    def vis_preprocessed(self, index):
        '''
        Visualizes the preprocessed data. 

        Args:
            - index: the index of the preprocessed data to be inspected
        '''
        # Create a figure with a 4x1 grid of subplots
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])

        axs = plt.subplot(gs[0,:])
        # Use the first subplot for the larger plot
        x = self.lidar_data[index] * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        y = self.lidar_data[index] * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        axs.scatter(x, y, s=0.1, label='Scans', alpha=self.normintens[index])
        axs.scatter(self.data[index][0], self.data[index][1], color='green', label='GT-Pos')
        axs.text(self.data[index][0], self.data[index][1],'GT-Pos')

       
        # Adjusting view, focusing on GT-position
        dx = dy = 2
        axs.set_xlim(self.data[index][0] - dx, self.data[index][0] + dx)
        axs.set_ylim(self.data[index][1] - dy,self.data[index][1] + dy)
        axs.set_xlabel('X coordinate')
        axs.set_ylabel('Y coordinate')
        axs.set_title('2D LiDAR Scan Visualization')

        axs.legend()
        names = ['Occupancy', 'Intensity', 'Point count']
        for i in range(3):
            axs = plt.subplot(gs[1, i])
            axs.imshow(self.input_data[index, :, :, i], origin='lower')
            axs.set_title(names[i])

        plt.show()

    def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
        '''
        2D Gaussian function.

        Args:
            x: x-coordinate
            y: y-coordinate
            x0: x-coordinate of the peak
            y0: y-coordinate of the peak
            sx: standard deviation in x
            sy: standard deviation in y
            A: amplitude'''
        return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

    def heatmap(self):
        '''
        Creates a heatmap of the GT-data as gaussian peaks.
        Set the standard deviation of the peaks to self.sx and self.sy.
        '''
        self.use_heatmaps = True#use heatmaps for training after preprocessing

        self.heatmaps = torch.zeros((len(self.data), self.image_size, self.image_size))
        for i,gt in enumerate(self.data):
            x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
            x0 = int((gt[0] + self.origin_offset) / self.pixelsize)
            y0 = int((gt[1] + self.origin_offset) / self.pixelsize)
            heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
            self.heatmaps[i] = torch.tensor(heatmap, dtype=torch.float32)
        print("Heatmaps created!")

    def vis_heatmap(self, index):
        '''
        Visualizes the heatmap and corresponding lidar scan in cartesian coordinates as well as image space.

        Args:
            - index: the index of the heatmap to be inspected
        '''
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.5,1])

        axs = plt.subplot(gs[0,:])
        # Use the first subplot for the larger plot
        x = self.lidar_data[index] * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        y = self.lidar_data[index] * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        axs.scatter(x, y, s=0.1, label='Scans', alpha=self.normintens[index])
        axs.scatter(self.data[index][0], self.data[index][1], color='green', label='GT-Pos')
        axs.text(self.data[index][0], self.data[index][1],'GT-Pos')

       
        # Adjusting view, focusing on GT-position
        dx = dy = 2
        axs.set_xlim(self.data[index][0] - dx, self.data[index][0] + dx)
        axs.set_ylim(self.data[index][1] - dy,self.data[index][1] + dy)
        axs.set_xlabel('X coordinate')
        axs.set_ylabel('Y coordinate')
        axs.set_title('2D LiDAR Scan Visualization')

        axs.legend()
        axs = plt.subplot(gs[1,0])
        axs.imshow(self.input_data[index, :, :, 0], origin='lower')
        axs.set_title('Occupancy')
        axs = plt.subplot(gs[1,1])
        axs.imshow(self.heatmaps[index], origin='lower')
        axs.set_title('Heatmap')
        axs = plt.subplot(gs[1,2])
        axs.imshow(self.input_data[index, :, :, 0]+ self.heatmaps[index], origin='lower')
        axs.set_title('Combined')

        plt.show()

    def get_peak(self, index):
        '''
        Returns the peak of the heatmap for the given index.
        Only returns one value.

        Args:
            - index: the index of the heatmap to be inspected
        '''
        x,y = np.unravel_index(self.heatmaps[index].argmax(), self.heatmaps[index].shape)
        print("Extracted: ", self.index_to_cartesian(y,x))
        print("GT: ", self.data[index][:2])
        return x,y
    
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
    


    
class LidarDatasetOD(Dataset):
    class LidarDataset(Dataset):
        def __init__(self, dataset_path):
            self.dataset_path = dataset_path
            self.rows = self._get_rows()

        def _get_rows(self):
            with open(self.dataset_path, "r") as dataset_file:
                return list(csv.DictReader(dataset_file))

        def __getitem__(self, index):
            row = self.rows[index]
            lidar_data = torch.tensor(ast.literal_eval(row['lidar']), dtype=torch.float32)
            intensities = torch.tensor(ast.literal_eval(row['intensities']), dtype=torch.float32)
            data = torch.tensor([row[column] for column in list(row.keys())[2:]], dtype=torch.float32)
            return lidar_data, intensities, data

        def __len__(self):
            return len(self.rows)

if __name__ == '__main__':

    set = LidarDataset(dataset_path="/home/f1tenth/catkin_ws/src/race_stack/perception/dataset/output/test.csv")

    #lidar,intensities,data = set[0]
    #print(data[0].dtype, data[0])
    #print(lidar[0].dtype, lidar[0])
    #print(len(lidar))
    #print(len(intensities))
    #print(len(data))
    #print(len(set))
   # print(set.labels)
    #print(data)
    #set.add_data(csv_file="/home/f1tenth/catkin_ws/src/race_stack/perception/dataset/output/test.csv")
    #print(len(set))
    #lidar, intensities, data = set[154]
    #print(data)

    set.time_dependent = False
    set.time_steps = 2
    print(len(set))
    #set.augment()
    print(len(set))
    lidar, intensities, data = set[0]
    print(data)
    

   # set.vis(0)
    set.preprocess()
    set.vis_preprocessed(20)
    set.heatmap()
    set.vis_heatmap(20)
    set.get_peak(20)

    