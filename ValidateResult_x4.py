# -*- coding: utf-8 -*-
import argparse
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import warnings
import os
import scipy.io as sio
import scipy.misc
from utils import SAM_GPU, PSNR_GPU
import torch.nn.functional as func
from scipy.misc import imresize
from skimage.transform import resize

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PyTorch SANet Test")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--model", default=30, type=int, help="model path")
parser.add_argument("--model_s", default=1, type=int, help="model path")
# parser.add_argument("--model", default="model/model_ISSR_epoch_15.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument('--dataset', default='./generate_training_dataset/testset/', type=str, help='path to general model')

model_name = 'HSI_SR_GDRRN_Harvard_up4_saml_1e1_g2'
set_name = 'Harvard'
filep = './results/results_' + model_name + '/'

def denoising(path, imgName, model):

    mat = scipy.io.loadmat(path + "/" + imgName + ".mat")
    label_ = mat['GT'].astype(np.float32)[:,0:512,0:512]
    hsi_ = mat['H'].astype(np.float32)
    H = np.zeros((label_.shape[0], label_.shape[1]*2, label_.shape[2]*2), dtype=np.float32)
    for i in range(hsi_.shape[0]):
        H[i, :, :] = imresize(hsi_[i, :, :], (label_.shape[1]*2, label_.shape[2]*2), 'bicubic', mode='F')
    H = H[:,0:512,0:512]

    # H = mat['H'].astype(np.float32)[:,0:512,0:512]
    # M = mat['M'].astype(np.float32)

    # sio.savemat('tmp.mat', {'H_': hsi_[:,0:128,0:128], 'H': H, 'GT': label_})

    H_input = Variable(torch.from_numpy(H).float().contiguous(), volatile=True).view(1, H.shape[0], H.shape[1], H.shape[2])
    # M_input = Variable(torch.from_numpy(M).float().contiguous(), volatile=True).view(1, M.shape[0], M.shape[1], M.shape[2])
    # M = np.transpose(M, [1,2,0])
    if cuda:
        model = model.cuda()
        H_input = H_input.cuda()
        # M_input = M_input.cuda()
    # model = model.cpu()
    start_time = time.time()
    # out = model(H_input, M_input)
    out = model(H_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()

    result = out.data[0].numpy().astype(np.float32)

    result[result < 0] = 0
    result[result > 1.] = 1.

    psnr_predicted = PSNR_GPU(torch.from_numpy(result), torch.from_numpy(label_))
    sam_predicted = SAM_GPU(torch.from_numpy(result), torch.from_numpy(label_))
    # ssim_predicted = c_ssim(result, label_)
    # im_h.save("./result/" + imgName + "_" + mName +".bmp")

    return result, psnr_predicted, sam_predicted


opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

psnr_summary = []
sam_summary = []

for index in range(opt.model_s, opt.model + 1):
# for index in range(1, 30):
    modelName = "model_"+model_name+"/model_epoch_" + str(index) + ".pth"
    model = torch.load(modelName)["model"]
    model.eval()

    fpath = opt.dataset
    path = fpath + str(opt.scale) + '/' + set_name  # file folder
    files = os.listdir(path)
    PSNRs = []
    SAMs = []

    for file in files:
        if not os.path.isdir(file):
            imageName = os.path.splitext(file)[0]
            # if imageName != 'imgf2': continue
            # print("==========   SANet on " + imageName + " ==============")
            recon, psnr, sam = denoising(path, imageName, model)
            PSNRs.append(psnr)
            SAMs.append(sam)
            # SSIMs_SANet.append(ssim)
            # scipy.misc.imsave(filepath+'/'+imageName+'.png', recon)

            # !!!!! please be fine if you see the numerical results are wrong. Someone may debug the measurements' code. !!!!!
            # print("   "+imageName+": PSNR={}   SAM={}".format(PSNRs[PSNRs.__len__()-1],SAMs[SAMs.__len__()-1]))
            if index >= 1:
                filepath = filep + set_name + '/'+ str(opt.scale) + '/' + str(index)
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                sio.savemat(filepath+'/'+imageName+'_recon.mat',{'result': recon})

    # print("========== model {} Average results ==============".format(index))
    # !!!!! please be fine if you see the numerical results are wrong. Someone may debug the measurements' code. !!!!!
    print("model {}   PSNR={}   SAM={}".format(index, np.mean(PSNRs), np.mean(SAMs)))
    psnr_summary.append(np.array(PSNRs))
    sam_summary.append(np.array(SAMs))
sio.savemat(filep + set_name + '_'+ str(opt.scale) +'_resArray.mat', {'psnr': np.array(psnr_summary),
                                                                      'sam': np.array(sam_summary)})
