import os
import argparse
import scipy.io as sio
import numpy as np
import time
import random
import torch
import glob
from scipy.misc import imresize
import h5py
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torch.nn as nn
from torch.autograd import Variable
from models_SRConv4 import HSISRCONV4
from torch.utils.data import DataLoader
import torch.utils.data as data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description="PyTorch HSISRCONV4")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
# parser.add_argument("--resume", default="model/model_ISSR_epoch_80.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.005, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
# parser.add_argument('--dataset', default='data/fusion_trainset.mat', type=str, help='path to general model')
# parser.add_argument('--dataset', default='../HyperDatasets/trainset_of_CNNbasedFusion/train_data/fusion_trainset_CAVE_x32.mat', type=str, help='path to general model')
parser.add_argument('--dataset', default='./generate_training_dataset/generate_trainset_of_GDRRN/train_data/fusion_trainset_Harvard_x4_32/', type=str, help='path to general model')

method_name = 'HSI_SR_HSISRCONV4_Harvard_up4_nsam_g1'
sigma = 25

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    # print("===> Loading datasets")
    #train_set = DatasetFromMat(opt.dataset, sigma)
    # train_set = DatasetFromMat7_3(opt.dataset)

    print("===> Building model")
    model = HSISRCONV4(input_chnl_hsi=31, group=1)
    # criterion = nn.MSELoss()
    # criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = dataparallel(model, 1)  # set the number of parallel GPUs
        # criterion = criterion.cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    # optimizer = optim.SGD([
    #     {'params': model.parameters()}
    # ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam([
        {'params': model.parameters()}
        ], lr=opt.lr, weight_decay=opt.weight_decay)

    print("===> Training")
    lossAarry = np.zeros(opt.nEpochs)
    losspath = 'losses/'
    if not os.path.exists(losspath):
        os.makedirs(losspath)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        start_time = time.time()
        print("===> Loading datasets ")
        train_set = DatasetFromHDF5_bicubic(opt.dataset)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=False)
        lossAarry[epoch - 1] = lossAarry[epoch - 1] + train(training_data_loader, optimizer, model, epoch)
        print("===> Epoch[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, lossAarry[epoch - 1],time.time() - start_time))
        save_checkpoint(model, epoch)

    sio.savemat(losspath + method_name+'_lossArray.mat', {'lossArray': lossAarry})

def train(training_data_loader, optimizer, model, epoch):
    lr = adjust_learning_rate(epoch - 1, opt.step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, low_lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    start_time = time.time()

    model.train()
    lossValue = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        hsi, label = Variable(batch[0]), Variable(batch[2], requires_grad=False)
        if opt.cuda:
            hsi = hsi.cuda()
            label = label.cuda()
        res = model(hsi)

        # loss = criterion(res, label)
        sam_lamd = 0
        mse_lamd = 1
        lossfunc = myloss_spe(hsi.data.shape[0], lamd=sam_lamd, mse_lamd=mse_lamd)
        loss = lossfunc.forward(res, label)

        # loss = criterion(res, label)/(input.data.shape[0]*2)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data.item()
        if (iteration+1)%50 == 0:
            elapsed_time = time.time() - start_time
            # save_checkpoint(model, iteration)
            print("===> Epoch[{}]: iteration[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, iteration+1,
                                            # criterion(lres + hres, target).data[0], loss_low.data[0], 0, elapsed_time))
                                            loss.data.item(), elapsed_time))

    elapsed_time = time.time() - start_time
    lossValue = lossValue / (iteration + 1)
    # print("===> Epoch[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, lossValue, elapsed_time))
    return lossValue

class myloss_spe(nn.Module):
    def __init__(self, N, lamd = 1e-1, mse_lamd=1):
        super(myloss_spe, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        return

    def forward(self, res, label):
        mse = func.mse_loss(res, label, size_average=False)
        # mse = func.l1_loss(res, label, size_average=False)
        loss = mse / (self.N * 2)
        esp = 1e-12
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0
        sam_sum = torch.sum(sam) / (self.N * H * W)
        total_loss = self.mse_lamd*loss + self.lamd * sam_sum
        return total_loss

def adjust_learning_rate(epoch, step):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    # if epoch < step:
    #     lr = opt.lr #* (0.1 ** (epoch // opt.step))#0.2
    # elif epoch < 3 * step:
    #     lr = opt.lr * 0.1 #* (0.1 ** (epoch // opt.step))#0.2
    # elif epoch < 5 * step:
    #     lr = opt.lr * 0.01  # * (0.1 ** (epoch // opt.step))#0.2
    # else:
    #     lr = opt.lr * 0.001
    lr = opt.lr  * (0.1 ** (epoch // opt.step))#0.2
    return lr

class DatasetFromHDF5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHDF5, self).__init__()
        self.file_path = file_path
        data = h5py.File(os.path.join(self.file_path,'gt.h5'), 'r')
        self.keys = list(data.keys())
        random.shuffle(self.keys)
        data.close()

    def __getitem__(self, index):
        hdf_gt = h5py.File(os.path.join(self.file_path,'gt.h5'), 'r')
        key = str(self.keys[index])
        hdf_c = h5py.File(os.path.join(self.file_path, 'c.h5'), 'r')
        hdf_hsi = h5py.File(os.path.join(self.file_path, 'hsi_t.h5'), 'r')
        # test patch pair
        # hsi_ = np.array(hdf_hsi[key])
        # c_ = np.array(hdf_c[key])
        # gt_ = np.array(hdf_gt[key])
        # sio.savemat('tmp.mat', {'hsi': hsi_, 'c': c_, 'gt': gt_})
        gt = torch.from_numpy(np.array(hdf_gt[key], dtype=np.float32))
        c = torch.from_numpy(np.array(hdf_c[key], dtype=np.float32))
        hsi = torch.from_numpy(np.array(hdf_hsi[key], dtype=np.float32))

        hdf_gt.close()
        hdf_c.close()
        hdf_hsi.close()
        return hsi, c, gt

    def __len__(self):
        return len(self.keys)

class DatasetFromHDF5_bicubic(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHDF5_bicubic, self).__init__()
        self.file_path = file_path
        data = h5py.File(os.path.join(self.file_path,'gt.h5'), 'r')
        self.keys = list(data.keys())
        random.shuffle(self.keys)
        data.close()

    def __getitem__(self, index):
        hdf_gt = h5py.File(os.path.join(self.file_path,'gt.h5'), 'r')
        key = str(self.keys[index])
        hdf_c = h5py.File(os.path.join(self.file_path, 'c.h5'), 'r')
        hdf_hsi = h5py.File(os.path.join(self.file_path, 'hsi.h5'), 'r')
        # test patch pair
        # hsi_ = np.array(hdf_hsi[key])
        # c_ = np.array(hdf_c[key])
        # gt_ = np.array(hdf_gt[key])
        # sio.savemat('tmp.mat', {'hsi': hsi_, 'c': c_, 'gt': gt_})
        gt = torch.from_numpy(np.array(hdf_gt[key], dtype=np.float32))
        c = torch.from_numpy(np.array(hdf_c[key], dtype=np.float32))
        # hsi = torch.from_numpy(np.array(hdf_hsi[key], dtype=np.float32))
        hsi_ = np.array(hdf_hsi[key], dtype=np.float32)
        hsi_t = np.zeros(gt.shape, dtype=np.float32)
        for i in range(hsi_.shape[0]):
            hsi_t[i,:,:] = imresize(hsi_[i,:,:], (gt.shape[1], gt.shape[2]), 'bicubic', mode='F')
        # hsi1 = np.transpose(np.array(hdf_hsi[key], dtype=np.float32), [1,2,0])
        # hsi_ = np.transpose(resize(hsi1, (gt.shape[1], gt.shape[2])), [2,0,1]) # spline interpolation
        hsi = torch.from_numpy(hsi_t.astype(np.float32))
        hdf_gt.close()
        hdf_c.close()
        hdf_hsi.close()
        return hsi, c, gt

    def __len__(self):
        return len(self.keys)

def save_checkpoint(model, epoch):
    fold = "model_"+method_name+"/"
    model_out_path = fold + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(fold):
        os.makedirs(fold)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model

if __name__ == "__main__":
    main()
    exit(0)
