import torch
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CenterSpeedDataset(Dataset):
    '''
    Dataset class for the CenterSpeed dataset.
    '''
    def __init__(self, dataset_path, transform=None, dense=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.use_heatmaps = True
        self.dense = dense
        self.consider_free_paths = True
        self.pixelsize = 0.08#size of a pixel in meters
        self.image_size = 64 #size of the image for preprocessing
        self.feature_size = 3 #number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5 #standard deviation of the gaussian peaks
        self.len = None
        self.seq_len = 2 #number of frames in a sequence
        self.number_of_sets = None
        self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        self.setup()

    def setup(self):
        '''
        Sets up the dataset by reading the files and determining the number of rows in each file.
        '''
        self.file_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        self.len = 0
        self.file_indices = []
        num_rows_per_file = []
        for file_path in self.file_paths:
                num_rows = sum(1 for row in open(file_path))- 1 -(self.seq_len-1) #subtract 2 because of the header and the last row
                self.file_indices.append((self.len, self.len+num_rows))
                self.len += num_rows
                num_rows_per_file.append(num_rows)
        for path in self.file_paths:
            print("Reading the following files: ", path)
            print("Number of entries: ", num_rows_per_file[self.file_paths.index(path)])

        print("Number of rows: ", self.len)
        print("File indices: ", self.file_indices)

    def change_pixel_size(self, pixelsize):
        '''
        Changes the pixel size and the origin offset accordingly.

        Args:
            pixelsize (int): New pixel size in meters.
        '''
        self.pixelsize = pixelsize
        self.origin_offset = (self.image_size//2) * self.pixelsize
        print("Pixel size changed to: ", self.pixelsize)
        print("Origin offset changed to: ", self.origin_offset)

    def change_image_size(self, image_size):
        '''
        Changes the image size and the origin offset accordingly.

        Args:
            image_size (int): New image size in pixels.
        '''
        self.image_size = int(image_size)
        self.origin_offset = (self.image_size//2) * self.pixelsize
        print("Image size changed to: ", self.image_size)
        print("Origin offset changed to: ", self.origin_offset)

    def __getitem__(self, index):
        '''
        Returns the preprocessed data and the ground truth data for a given index.

        Args:
            index: Index of the data to be returned.

        Returns:
            input_data: Preprocessed data in the form of a tensor of size (3, 64, 64).
            heatmap: Ground truth heatmap in the form of a tensor of size (64, 64).
            data: Ground truth data in the form of a tensor of size (5).
            free: Boolean indicating whether the path is free or not.
            '''
        free = False
        # Determine which file the data should come from
        file_index = next(i for i, (start, end) in enumerate(self.file_indices) if start <= index < end)
        row_index = index - self.file_indices[file_index][0]

        seq_data = []
        df = pd.read_csv(self.file_paths[file_index], skiprows=row_index, nrows=self.seq_len, header=None, names=['lidar','intensities','x','y','vx','vy','yaw'])
        if len(df) == 0:
            raise IndexError
        for i in range(self.seq_len):
            df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
            df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
            row = df.iloc[i]
            self.lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
            intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
            try:
                self.intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
            except:
                print("Intensities: ", intensities)
                print("Row: ", row)
                print("Index: ", index)
            seq_data.append(self.preprocess(self.lidar_data, self.intensities))
            if i == self.seq_len - 1:
                data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
                self.data_for_plot = data.numpy().copy()
                heatmap = self.heatmap(data)
        input_data = torch.stack([item for item in seq_data]).view(self.seq_len*3,self.image_size,self.image_size)

        if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:
            free = True

        if self.transform:
            input_data, heatmap, data = self.transform((input_data, heatmap, data))

        if self.dense:
            print(f'Using dense features with data: {data}')
            dense_features = self.populate_dense_features(data=data)
            return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5), dense_features, free

        return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5), free


    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        if self.len is not None:
            return self.len
        else:
            with open(self.dataset_path, 'r') as f:
                self.len = sum(1 for row in csv.reader(f))
                return self.len

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

    def populate_dense_features(self, data) -> torch.Tensor:
        '''
        Populates a tensor with dense speed and orientation values.

        Args:
            x: x-coordinate of the peak
            y: y-coordinate of the peak
            values: List of values to be populated in the tensor.
        '''
        tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
        x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        print(f'Data length: {len(data)}')
        for i in range(3):
            tensor[:,:,i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i+2])

            if self.consider_free_paths:
                if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
                    tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
                    print(f'Car Behind, setting zero')
                    return tensor


        return tensor



    def heatmap(self, data):
        '''
        Creates a heatmap from the ground truth data.
        '''
        self.heatmaps = torch.zeros(self.image_size, self.image_size)
        x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
        if self.consider_free_paths:
            if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
                heatmap = np.zeros((self.image_size, self.image_size))
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        return heatmap

    def preprocess(self, lidar_data, intensities):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into an image.
        Creates 3 feature maps: one for the occupancy, one for the intensity and one for the number of points in a pixel.

        Args:
            lidar_data: Lidar data in the form of a tensor of size (n).
            intensities: Intensity data in the form of a tensor of size (n).

        Returns:
            input_data: Preprocessed data in the form of a tensor of size (3, 64, 64).

        '''
        self.use_heatmaps = True#use heatmaps for training after preprocessing
        #preprocess the lidar data
        input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
        x = lidar_data * self.cos
        y = lidar_data * self.sin
        x_coord = ((x + self.origin_offset) / self.pixelsize)
        y_coord = ((y + self.origin_offset) / self.pixelsize)
        x_coord = x_coord.to(torch.int)
        y_coord = y_coord.to(torch.int)
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
        input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
        input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

        return input_data

    def cartesian_to_pixel(self, x, y):
        '''
        Converts cartesian coordinates to pixel coordinates.
        '''
        pixel_x = int(x / self.pixelsize + self.image_size / 2)
        pixel_y = int(y / self.pixelsize + self.image_size / 2)
        return pixel_x, pixel_y


    def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
        '''
        Visualizes the data for a given index.

        Args:
            index: Index of the data to be visualized.
            show_preprocessed: Boolean indicating whether to show the preprocessed data.
            show_gt: Boolean indicating whether to show the ground truth data.
            show_raw: Boolean indicating whether to show the raw data.
        '''
        config = [show_preprocessed, show_gt, show_raw]
        plot_rows = 0
        for c in config:
            if c:
                plot_rows += 1
        if plot_rows == 0:
            print("No plots selected!")
            return

        fig, axs = plt.subplots(plot_rows+1, 3, figsize=(10, 15))
        input, gt, data, free = self.__getitem__(index)
        if self.transform is not None:
            transform_names = ', '.join([t.__class__.__name__ for t in self.transform.transforms])
        else:
            transform_names = 'None'

        axs[0,0].axis('off')
        axs[0,1].axis('off')
        axs[0,2].axis('off')
        axs[0,0].set_title('Dataset Info')
        axs[0,0].text(0, 0.3, f'Length of dataset: {self.len}\
                                \nPath: {self.dataset_path}\
                                \nTransforms: {transform_names}\
                                \n\nIndex: {index}\
                                \nFree track: {free}\
                                \nPixel size: {self.pixelsize}\
                                \nImage size: {self.image_size}\
                                \nGaussian radius: {self.sx}', fontsize=10)

        plot_nr = 1
        if show_preprocessed:
            axs[plot_nr, 0].set_title('Occupancy')
            axs[plot_nr, 0].imshow(input[0], cmap='plasma')
            axs[plot_nr, 1].imshow(input[1], cmap='plasma')
            axs[plot_nr, 1].set_title('Intensity')
            axs[plot_nr, 2].imshow(input[2], cmap='plasma')
            axs[plot_nr, 2].set_title('Density')
            for i in range(3):
                axs[plot_nr, i].axis('off')
                axs[plot_nr,i].scatter(self.image_size//2,self.image_size//2, label='Ego Position', color='g')
                x,y = self.cartesian_to_pixel(data[0],data[1])
                axs[plot_nr,i].scatter(x,y, label='GT Position', color='r')
                axs[plot_nr,i].quiver(x,y ,data[2],data[3], label='GT Velocity', color='r')
                yaw_degrees = np.rad2deg(data[4])
                rectangle = patches.Rectangle((x-2, y-4), 8, 4, angle=yaw_degrees, fill=False, color='r')
                axs[plot_nr, i].add_patch(rectangle)
            axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plot_nr += 1

        if show_gt:
            axs[plot_nr, 0].imshow(gt, cmap='plasma')
            axs[plot_nr, 0].set_title('GT heatmap')
            axs[plot_nr, 1].imshow(np.clip(input[0]+gt, 0, 1), cmap='plasma')
            axs[plot_nr, 1].set_title('Occupancy + GT heatmap')
            axs[plot_nr, 0].axis('off')
            axs[plot_nr, 1].axis('off')
            axs[plot_nr, 2].axis('off')
            plot_nr += 1

        if show_raw:
            axs[plot_nr, 0].plot(self.lidar_data)
            axs[plot_nr, 0].set_title('Raw lidar ranges')
            axs[plot_nr, 1].plot(self.intensities)
            axs[plot_nr, 1].set_title('Raw lidar intensities')
            x = self.lidar_data * self.cos
            y = self.lidar_data * self.sin
            axs[plot_nr,2].scatter(x, y, s=0.1, label='Scans', alpha=self.intensities)
            axs[plot_nr,2].scatter(self.data_for_plot[0], self.data_for_plot[1], color='r', label='GT-Pos')
            axs[plot_nr,2].text(self.data_for_plot[0], self.data_for_plot[1],'GT-Pos')
            # Adjusting view, focusing on GT-position
            dx = dy = 2
            axs[plot_nr,2].set_xlim(self.data_for_plot[0] - dx, self.data_for_plot[0] + dx)
            axs[plot_nr,2].set_ylim(self.data_for_plot[1] - dy,self.data_for_plot[1] + dy)
            axs[plot_nr,2].set_xlabel('X coordinate')
            axs[plot_nr,2].set_ylabel('Y coordinate')
            axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axs[plot_nr, 2].set_title('Raw lidar data')





class RandomRotation:
    '''
    Randomly rotates the input data and the ground truth data.
    '''
    def __init__(self, angle=45, image_size=64, feature_size=6):
        self.angle = angle
        self.image_size = image_size
        self.feature_size = feature_size

    def __call__(self, sample):
        input, heatmap, data = sample
        angle = random.uniform(-self.angle, self.angle)
        angle_rad = -math.radians(angle)
        #print("THis was rotated by: ", angle)
        input_rotated = []
        for i in range(input.shape[0]):
            input_pil = transforms.functional.to_pil_image(input[i])
            input_rotated_tensor = transforms.ToTensor()(transforms.functional.rotate(input_pil, angle))
            input_rotated.append(input_rotated_tensor)
        input = torch.stack(input_rotated, dim = 1)


        heatmap_image = transforms.functional.to_pil_image(heatmap)
        rotated_hm_image = transforms.functional.rotate(heatmap_image, angle)
        heatmap = transforms.ToTensor()(rotated_hm_image)

        rotation_matrix = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
                                        [np.sin(angle_rad), np.cos(angle_rad)]])

        # Apply the rotation
        data[0:2] = torch.matmul(rotation_matrix, data[0:2])
        data[2:4] = torch.matmul(rotation_matrix.T, data[2:4])
        data[4] = (data[4] - math.radians(angle))% (2*math.pi)
        if data[4] > math.pi:
            data[4] -= 2*math.pi

        return input.view(self.feature_size,self.image_size,self.image_size), heatmap.view(self.image_size,self.image_size), data.view(5)

class RandomFlip:
    '''
    Randomly flips the input data and the ground truth data.
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        input, heatmap, data = sample
        if random.random() < self.p:
            #print("This was flipped")
            input = torch.flip(input, [1])
            heatmap = torch.flip(heatmap, [0])
            data[1] = -data[1]
            data[3] = -data[3]
            data[4] = -data[4]
        return input, heatmap, data


################OLD IMPLEMENTATIONS####################

class LidarDatasetOD(Dataset):
    '''V1, Not used anymore'''
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.use_heatmaps = True
        self.pixelsize = 0.025#size of a pixel in meters, was 0.015 i think this bigger makes more sense for FOV
        self.image_size = 256 #size of the image for preprocessing
        self.feature_size = 3 #number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5 #standard deviation of the gaussian peaks
        self.len = None


    def __getitem__(self, index):
        df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=1, header=None, names=['lidar','intensities','x','y','vx','vy','yaw'])
        if len(df) == 0:
            raise IndexError
        df.loc[0, 'lidar'] = df.loc[0, 'lidar'].replace('(', '').replace(')', '')
        df.loc[0, 'intensities'] = df.loc[0, 'intensities'].replace('(', '').replace(')', '')
        row = df.iloc[0]
        lidar_data = torch.tensor(np.fromstring(df.loc[0, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
        intensities = torch.tensor(np.fromstring(df.loc[0, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
        #print(len(intensities))
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
        data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)

        return self.preprocess(lidar_data, intensities, data)

    def __len__(self):
        if self.len is not None:
            return self.len
        else:
            with open(self.dataset_path, 'r') as f:
                self.len = sum(1 for row in csv.reader(f))
                return self.len

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


    def preprocess(self, lidar_data, intensities, data):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
        Stores these grids in a new tensor.
        Completely vectorized, efficient asf!
        Does it make sense to put the origin in the middle of the grid?
        Maybe it is better to put it in the bottom left corner? Or closer to the corner?
        '''

        self.use_heatmaps = True#use heatmaps for training after preprocessing
        #preprocess the lidar data
        input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
        x = lidar_data * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        y = lidar_data * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        x_coord = ((x + self.origin_offset) / self.pixelsize)
        y_coord = ((y + self.origin_offset) / self.pixelsize)
        x_coord = x_coord.to(torch.int)
        y_coord = y_coord.to(torch.int)
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
        input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
        input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

        #preprocess the gt's into heatmaps

        self.heatmaps = torch.zeros(self.image_size, self.image_size)
        x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))

        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
        #FIXME: i think this is the wrong place to do this because sometimes we can also see the opponent behind us!
        if data[0] < 0:#the other car is behind us, no peak in the heatmap
            heatmap = np.zeros((self.image_size, self.image_size))
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        return input_data, heatmap, data


class LidarDatasetSeqOD(Dataset):
    '''
    V2, not used anymore
    '''
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.use_heatmaps = True
        self.consider_free_paths = True
        self.pixelsize = 0.025#size of a pixel in meters, was 0.015 i think this bigger makes more sense for FOV
        self.image_size = 256 #size of the image for preprocessing
        self.feature_size = 3 #number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5 #standard deviation of the gaussian peaks
        self.len = None
        self.seq_len = 2 #number of frames in a sequence
        self.number_of_sets = None
        self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
        self.setup()

    def setup(self):
       df = pd.read_csv(self.dataset_path, header=None, names=['setid','lidar','intensities','x','y','vx','vy','yaw'])
       self.number_of_sets = df.max()['setid']
       print("Number of sets", self.number_of_sets)
       self.len = len(df) - 1 - self.number_of_sets
       print("Length of Dataset", self.len)
       print("Dataset Setup!")

    def change_pixel_size(self, pixelsize):
        self.pixelsize = pixelsize
        self.origin_offset = (self.image_size//2) * self.pixelsize
        print("Pixel size changed to: ", self.pixelsize)
        print("Origin offset changed to: ", self.origin_offset)


    def __getitem__(self, index):
        seq_data = []
        df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=self.seq_len, header=None, names=['setid','lidar','intensities','x','y','vx','vy','yaw'])
        if len(df) == 0:
            raise IndexError
        if df.iloc[0]['setid'] != df.iloc[-1]['setid']:
            return self.__getitem__(index+1)
        for i in range(self.seq_len):
            df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
            df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
            row = df.iloc[i]
            lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
            intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
            seq_data.append(self.preprocess(lidar_data, intensities))
            if i == self.seq_len - 1:
                data = torch.tensor(row[3:].values.astype(float), dtype=torch.float32)
                heatmap = self.heatmap(data)
        input_data = torch.stack([item for item in seq_data]).view(self.seq_len*3,self.image_size,self.image_size)

        if self.transform:
            input_data, heatmap, data = self.transform((input_data, heatmap, data))

        return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5)


    def __len__(self):
        if self.len is not None:
            return self.len
        else:
            with open(self.dataset_path, 'r') as f:
                self.len = sum(1 for row in csv.reader(f))
                return self.len

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

    def heatmap(self, data):
        #preprocess the gt's into heatmaps

        self.heatmaps = torch.zeros(self.image_size, self.image_size)
        x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
        if self.consider_free_paths:
            if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
                heatmap = np.zeros((self.image_size, self.image_size))
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        return heatmap

    def preprocess(self, lidar_data, intensities):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
        Stores these grids in a new tensor.
        Completely vectorized, efficient asf!
        Does it make sense to put the origin in the middle of the grid?
        Maybe it is better to put it in the bottom left corner? Or closer to the corner?
        '''

        self.use_heatmaps = True#use heatmaps for training after preprocessing
        #preprocess the lidar data
        input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
        x = lidar_data * self.cos
        y = lidar_data * self.sin
        x_coord = ((x + self.origin_offset) / self.pixelsize)
        y_coord = ((y + self.origin_offset) / self.pixelsize)
        x_coord = x_coord.to(torch.int).long()
        y_coord = y_coord.to(torch.int).long()
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
        input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
        input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

        return input_data

