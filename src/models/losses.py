import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

def CenterSpeedLoss(heatmap,data, gt_heatmap, gt_data, is_free):
    '''
    This loss function neglects the speed and orientation losses, if the opponent is not visible.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
    '''
    batch_size = data.size()[0]
    try:
        assert(heatmap.shape == gt_heatmap.shape)
        assert(gt_data.shape[1]==5)
    except AssertionError as e:
        print("HEATMAP SHAPE: ", heatmap.shape)
        print("GT-HEATMAP SHAPE: ", gt_heatmap.shape)
        print("GT-DATA SHAPE: ", data.shape)
        raise e
    # Weight generation function
    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    
    #print(loss)
    heatmap_loss = heatmap_loss.sum()
    data_loss = torch.zeros((batch_size))
    for i in range(batch_size):
        if is_free[i] == True:
            data_loss[i] = 0
        else:
            data_loss[i] = ((data[i] - gt_data[i,2:])**2).sum()
    
    data_loss = data_loss.sum()
    print(f"Data Loss: {data_loss/batch_size}, Heatmap Loss: {heatmap_loss/batch_size}")
    #wandb.log({"Data Loss": data_loss/batch_size, "Heatmap Loss": heatmap_loss/batch_size})

    return (heatmap_loss + data_loss) / batch_size

########### OLD LOSS FUNCTIONS ############

def msle_loss(y_pred, y_true):#penalizing underestimates more than overestimates.
    return torch.mean((torch.log(y_pred + 1) - torch.log(y_true + 1))**2)

def heatmap_loss(pred, gt):
    # Weight generation function
    F = gt**2
    #print(F)
    # Weighted MSE loss
    loss = (F + 1) * (pred - gt)**2
    #print(loss)
    loss = loss.sum()
    return loss

def CenterSpeedLoss2(heatmap,data, gt_heatmap, gt_data):
    assert(heatmap.shape == gt_heatmap.shape)
    #assert(data.shape == gt_data.shape)
    # Weight generation function
    print(data.shape)
    print(gt_data.shape)
    F = gt_heatmap**2
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    #print(loss)
    heatmap_loss = heatmap_loss.sum()
    batch_size = data.size()[0]
    data_loss = torch.zeros((batch_size))
    print(data_loss.shape)
    for i in range(batch_size):
        if gt_data[i,0] < 0:
            data_loss[i] = 0
        else:
            # Otherwise, compute the loss as before
            for j in range(3):
                data_loss[i] += (data[i,j] - gt_data[i,j+2])**2
    print(data_loss)
    data_loss = data_loss.sum()
    data_loss = 0
    return (heatmap_loss + data_loss) / batch_size

def CenterSpeedLoss3(heatmap,data, gt_heatmap, gt_data):
    '''
    Loss function for the CenterSpeed Model. This version intrinically first weighs the heatmap more and as that converges
    starts to consider the dataloss as soon as the loss becomes low enough.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
    '''
    batch_size = data.size()[0]
    try:
        assert(heatmap.shape == gt_heatmap.shape)
        assert(gt_data.shape == data.shape)
    except AssertionError as e:
        print("HEATMAP SHAPE: ", heatmap.shape)
        print("GT-HEATMAP SHAPE: ", gt_heatmap.shape)
        print("DATA SHAPE: ", data.shape)
        print("GT-DATA SHAPE: ", gt_data.shape)
        raise e
    # Weight generation function

    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    
    #print(loss)
    heatmap_loss = heatmap_loss.sum()

    data_loss = (data - gt_data)**2
    data_loss = data_loss.sum()

    return (heatmap_loss + data_loss) / batch_size

def CenterSpeedLossFree(heatmap,data, gt_heatmap, gt_data):
    '''
    This loss function neglects the speed and orientation losses, if the opponent is not visible.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
    '''
    batch_size = data.size()[0]
    try:
        assert(heatmap.shape == gt_heatmap.shape)
        assert(gt_data.shape[1]==5)
    except AssertionError as e:
        print("HEATMAP SHAPE: ", heatmap.shape)
        print("GT-HEATMAP SHAPE: ", gt_heatmap.shape)
        print("GT-DATA SHAPE: ", data.shape)
        raise e
    # Weight generation function
    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    
    #print(loss)
    heatmap_loss = heatmap_loss.sum()
    data_loss = torch.zeros((batch_size))
    for i in range(batch_size):
        if gt_data[i,0] < 0 or torch.sqrt(gt_data[i,0]**2 + gt_data[i,1]**2) > 3:
            data_loss[i] = 0
        else:
            data_loss[i] = ((data[i] - gt_data[i,2:])**2).sum()
    
    data_loss = data_loss.sum()
    print(f"Data Loss: {data_loss/batch_size}, Heatmap Loss: {heatmap_loss/batch_size}")
    wandb.log({"Data Loss": data_loss/batch_size, "Heatmap Loss": heatmap_loss/batch_size})

    return (heatmap_loss + data_loss) / batch_size


def CenterNetLoss(heatmap, data, gt_heatmap, gt_data, alpha=2, beta=4, lambda_data=1):
    '''
    Loss function for the CenterNet model. Adapted from the original implementation.
    Currently not used.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
        - alpha: alpha parameter for the focal loss
        - beta: beta parameter for the focal loss
        - lambda_data: parameter for the data loss
    '''
    batch_size = data.size()[0]
    assert(heatmap.shape == gt_heatmap.shape)
    assert(data.shape == gt_data.shape)
    # Weight generation function
    pos_inds = gt_heatmap.eq(1)
    neg_inds = gt_heatmap.lt(1)

    neg_weights = torch.pow(1 - gt_heatmap[neg_inds], beta)

    loss = 0
    pos_pred = heatmap[pos_inds]
    neg_pred = heatmap[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, alpha)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, beta) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    data_loss = (data - gt_data)**2
    data_loss = data_loss.sum() / batch_size

    return loss + lambda_data * data_loss

def index_to_cartesian( x_img,y_img):
    '''
    Converts the index of the imagespace back to cartesian coordinates.

    Args:
        - x_img: the x-coordinate in the image space
        - y_img: the y-coordinate in the image space
    '''
    pixelsize = 0.025
    origin_offset = (256//2) * pixelsize
    x = x_img * pixelsize - origin_offset
    y = y_img * pixelsize - origin_offset
    return x,y

def get_peak(heatmap, gt, threshold=0.2, print_values = False):
    '''
    Returns the peak of the heatmap for the given index.
    Only returns one value.

    Args:
        - index: the index of the heatmap to be inspected
    '''
    x, y = np.unravel_index(heatmap.argmax(), heatmap.shape)
    xcart, ycart = index_to_cartesian(x, y)
    return ycart,xcart

def CenterSpeedLoss_Peaks(heatmap,data, gt_heatmap, gt_data):
    batch_size = data.size()[0]
    assert(heatmap.shape == gt_heatmap.shape)
    # Weight generation function

    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    #print(loss)
    heatmap_loss = heatmap_loss.mean()

    data_loss = (data - gt_data[:,2:])**2
    data_loss = data_loss.mean()

    x = []
    y = []
    for i in range(batch_size):
        x_i,y_i = get_peak(heatmap[i].squeeze(0), gt_heatmap[i].squeeze(0), print_values=True)
        x.append(x_i)
        y.append(y_i)
        # print("Predicted:  ",x_i,y_i)
        # print("Ground Truth:", gt_data[i,0],gt_data[i,1])
    x = torch.tensor(x)
    y = torch.tensor(y)
    pos_loss = (x - gt_data[:,0])**2 + (y - gt_data[:,1])**2
    pos_loss = pos_loss.mean()
   
    return (heatmap_loss + data_loss + pos_loss) / batch_size

def CenterSpeedLoss_mean(heatmap,data, gt_heatmap, gt_data):
    '''
    CURRENTLY NOT USED
    Loss function for the CenterNet model. This version equally looks at the heatmap and the data loss.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
    '''
    batch_size = data.size()[0]
    assert(heatmap.shape == gt_heatmap.shape)
    assert(data.shape == gt_data.shape)
    # Weight generation function

    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    #print(loss)
    heatmap_loss = heatmap_loss.mean()

    data_loss = (data - gt_data)**2
    data_loss = data_loss.mean()

    return ((heatmap_loss + data_loss) / batch_size)*100 #scaling the loss for stability

def CenterSpeedLoss_hm(heatmap,data, gt_heatmap, gt_data):
    '''
    CURRENTLY NOT USED
    Loss function for the CenterSpeed model. This version isolates the heatmap loss.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
    '''
    batch_size = heatmap.size()[0]
    assert(heatmap.shape == gt_heatmap.shape)
    # Weight generation function

    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    #print(loss)
    heatmap_loss = heatmap_loss.sum()

    return heatmap_loss  / batch_size

def CenterSpeedLoss_head(heatmap,data, gt_heatmap, gt_data):
    '''
    CURRENTLY NOT USED
    Loss function for the CenterSpeed model. This version isolates the data loss.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
    '''
    batch_size = data.size()[0]
    assert(data.shape == gt_data.shape)
    # Weight generation function
    data_loss = (data - gt_data)**2
    data_loss = data_loss.sum()

    return data_loss / batch_size

def CenterSpeedLoss_Adaptive(heatmap,data, gt_heatmap, gt_data, fac_hm, fac_data):
    '''
    CURRENTLY NOT USED
    Loss function for the CenterSpeed model. This version allows for adaptive weighting of the heatmap and data loss.
    ATTENTION: the decrease of the loss also automatically adapts the weights.

    Args:
        - heatmap: the predicted heatmap
        - data: the predicted data
        - gt_heatmap: the ground truth heatmap
        - gt_data: the ground truth data
        - fac_hm: the weight for the heatmap loss
        - fac_data: the weight for the data loss
    '''
    batch_size = data.size()[0]
    assert(heatmap.shape == gt_heatmap.shape)
    assert(data.shape == gt_data.shape)
    # Weight generation function

    F = gt_heatmap #unit weight
    #print(F)
    # Weighted MSE loss
    heatmap_loss = (F + 1) * (heatmap - gt_heatmap)**2
    
    #print(loss)
    heatmap_loss = heatmap_loss.sum()

    data_loss = (data - gt_data)**2
    data_loss = data_loss.sum()

    data_loss = fac_data * data_loss / batch_size
    heatmap_loss = fac_hm * heatmap_loss / batch_size

    print("[Loss Info] Heatmap Loss: ",heatmap_loss.item(), "  Weight: ",fac_hm)
    print("[Loss Info] Data Loss: ",data_loss.item(), "  Weight: ",fac_data)

    return heatmap_loss + data_loss