import numpy as np
import scipy.misc
import scipy.io as sio
import h5py
import os
import glob
import PIL
import pywt
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import skimage.measure as measure

def c_psnr(im1, im2):
    '''
    Compute PSNR
    :param im1: input image 1 ndarray ranging [0,1]
    :param im2: input image 2 ndarray ranging [0,1]
    :return: psnr=-10*log(mse(im1,im2))
    '''
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_psnr(im1, im2)

def c_ssim(im1, im2):
    '''
    Compute PSNR
    :param im1: input image 1 ndarray ranging [0,1]
    :param im2: input image 2 ndarray ranging [0,1]
    :return: psnr=-10*log(mse(im1,im2))
    '''
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)

    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_ssim(im1, im2, win_size=11, data_range=1, gaussian_weights=True)

def batch_PSNR(im_true, im_fake, data_range):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clone().resize_(N, C*H*W)
    Ifake = im_fake.clone().resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)

def PSNR_GPU(im_true, im_fake):
    data_range = 1
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone().resize_(C*H*W)
    Ifake = im_fake.clone().resize_(C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=0, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)

def batch_SAM_GPU(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clone().resize_(N, C, H*W)
    Ifake = im_fake.clone().resize_(N, C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=1).resize_(N, H*W)
    denom1 = torch.pow(Itrue,2).sum(dim=1).sqrt_().resize_(N, H*W)
    denom2 = torch.pow(Ifake,2).sum(dim=1).sqrt_().resize_(N, H*W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(N, H*W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (N*H*W)
    return sam

def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone().resize_(C, H*W)
    Ifake = im_fake.clone().resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0).resize_(H*W)
    denom1 = torch.pow(Itrue,2).sum(dim=0).sqrt_().resize_(H*W)
    denom2 = torch.pow(Ifake,2).sum(dim=0).sqrt_().resize_(H*W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(H*W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (H*W)
    return sam

def batch_SAM_CPU(im_true, im_fake):
    I_true = im_true.data.cpu().numpy()
    I_fake = im_fake.data.cpu().numpy()
    N = I_true.shape[0]
    C = I_true.shape[1]
    H = I_true.shape[2]
    W = I_true.shape[3]
    batch_sam = 0
    for i in range(N):
        true = I_true[i,:,:,:].reshape(C, H*W)
        fake = I_fake[i,:,:,:].reshape(C, H*W)
        nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
        denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
        denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
        sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
        sam = sam/np.pi*180
        # ignore pixels that have zero norm
        idx = (np.isfinite(sam))
        batch_sam += np.sum(sam[idx])/np.sum(idx)
        if np.sum(~idx) != 0:
            print("waring: some values were ignored when computing SAM")
    return batch_sam/N

def SAM_CPU(im_true, im_fake):
    I_true = im_true.data.cpu().numpy()
    I_fake = im_fake.data.cpu().numpy()
    N = I_true.shape[0]
    C = I_true.shape[1]
    H = I_true.shape[2]
    W = I_true.shape[3]
    batch_sam = 0
    for i in range(N):
        true = I_true[i,:,:,:].reshape(C, H*W)
        fake = I_fake[i,:,:,:].reshape(C, H*W)
        nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
        denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
        denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
        sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
        sam = sam/np.pi*180
        # ignore pixels that have zero norm
        idx = (np.isfinite(sam))
        batch_sam += np.sum(sam[idx])/np.sum(idx)
        if np.sum(~idx) != 0:
            print("waring: some values were ignored when computing SAM")
    return batch_sam/N

def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label

# Convert an image into patches
def Im2Patch(img,img2, win, stride=1, istrain=False): # Based on code written by Shuhang Gu (cssgu@comp.polyu.edu.hk)
    k = 0
    endw = img.shape[0]
    endh = img.shape[1]
    if endw<win or endh<win:
        return None,None
    patch = img[0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    # TotalPatNum = (img.shape[0]-win+1)*(img.shape[1]-win+1)
    Y = np.zeros([win*win,TotalPatNum])
    Y2 = np.zeros([win*win,TotalPatNum])
    for i in range(win):
        for j in range(win):
            patch = img[i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[k,:] = np.array(patch[:]).reshape(TotalPatNum)
            patch2 = img2[i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y2[k, :] = np.array(patch2[:]).reshape(TotalPatNum)
            k = k + 1
    if istrain:
        return Y.reshape([win, win, TotalPatNum]), Y2.reshape([win, win, TotalPatNum])
    else:
        return Y.transpose().reshape([TotalPatNum,win,win,1]), Y2.transpose().reshape([TotalPatNum,win,win,1])