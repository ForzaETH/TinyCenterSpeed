import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterSpeedModular(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, size_linear_layer = 64, p_dropout=0.3, image_size=256):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=channel_two, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  #nn.LeakyReLU(),
                                  nn.Flatten(),
                                  #nn.BatchNorm1d(*2),
                                  #nn.Dropout1d(p=p_dropout),
                                    nn.Linear((image_size//4)**2, size_linear_layer),
                                   nn.LeakyReLU(),
                                   nn.Linear(size_linear_layer, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        return x,y


class CenterSpeedDense(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.bn4(self.deconv2(x))
        return x

class CenterSpeedDensesigmoid(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.bn4(self.deconv2(x))
        channels = []
        for i in range(4):
            if i == 0:
                channels.append(self.sigmoid(x[:,i:i+1,:,:]))
            else:
                channels.append(x[:,i:i+1,:,:])

        output = torch.cat(channels, dim=1)

        return output

class CenterSpeedDensev2(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)
        self.sigmoid = nn.Sigmoid()
        self.velocity_cap = 10.0
        self.theta_cap = 3.14159 # 180 degrees


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.deconv2(x)

        heatmap, vx, vy, theta = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]

        # Apply Sigmoid to heatmap channels, Tanh to velocity channels
        heatmap = torch.sigmoid(heatmap)  # Constrain heatmap values between 0 and 1
        vx = torch.tanh(vx) * self.velocity_cap   # Allow vx to take on both positive and negative values
        vy = torch.tanh(vy) * self.velocity_cap   # Allow vy to take on both positive and negative values
        theta = torch.tanh(theta) * self.theta_cap # Allow theta to take on both positive and negative values

        # Recombine outputs
        output = torch.stack((heatmap, vx, vy, theta), dim=1)

        return output

class CenterSpeedDenseResidual(nn.Module):
    '''
        Main model for CenterSpeed
        '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)


    def forward(self, x):
        res = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(res)))
        x = F.leaky_relu(self.bn3(self.deconv1(x))) + res
        x = self.bn4(self.deconv2(x))
        return x



############Old models for testing purposes ################

class HourglassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.conv3(x)
        return x

class HourglassModelDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.conv3(x)
        return x

class HourglassModelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.conv3(x)

class HourglassModelDeepExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.dropout(x)
        x = self.conv3(x)
        self.sigmoid(x)
        return x

class HourglassModelDeepExp2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(129*129, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        y = self.head(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x,y


class CenterSpeed(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(64*64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y
class CenterSpeed_NOPAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=0, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(63*63, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        print(x.shape)
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        print(x.shape)
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        print(x.shape)
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y

class CenterSpeed_NODECONV(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(64*64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.conv3(self.upsample1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv4(self.upsample2(x)))
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y

class CenterSpeed_hm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        fm = x
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        return x , fm


class CenterSpeed_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                    nn.Flatten(),
                                    nn.Linear(64*64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        y = self.head(x)
        return y


class CenterSpeedModular2(nn.Module):
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, size_linear_layer = 64, p_dropout=0.3, image_size=256):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=1, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=1, kernel_size=4, stride=1, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=channel_two, out_channels=1, kernel_size=3, stride=1, padding=1),
                                #nn.LeakyReLU(),
                                nn.Flatten(),
                                #nn.BatchNorm1d(*2),
                                #nn.Dropout1d(p=p_dropout),
                                    nn.Linear((image_size//4)**2, size_linear_layer),
                                nn.LeakyReLU(),
                                nn.Linear(size_linear_layer, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y