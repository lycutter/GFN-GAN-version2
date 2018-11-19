# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import torch.optim as optim
import argparse
from torch.autograd import Variable
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataSet
from networks.GFN_4x import Net
import random
import re
from torchvision import transforms

from data.data_loader import CreateDataLoader
from networks.Discriminator import Discriminator
# from Discriminator_ import Discriminator
from networks.Generator import Generator


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--resumeD", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, Default: 4")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--gated", type=bool, default=False, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
# parser.add_argument('--dataset', required=True, help='Path of the training dataset(.h5)')


# add lately
parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--loadSizeX', type=int, default=640, help='scale images to this size')
parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 10
aphal = 1
lambda1 = 1
lambda2 = 0.001
lambda3 = 0.1

training_settings=[
    {'nEpochs': 25, 'lr': 1e-4, 'step':  7, 'lr_decay': 1, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 60, 'lr': 1e-4, 'step': 30, 'lr_decay': 1, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 55, 'lr': 5e-5, 'step': 25, 'lr_decay': 1, 'lambda_db':   0, 'gated': True}
]



def mkdir_steptraing():
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    step1_folder, step2_folder, step3_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(step3_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        print("===> Step training models store in models/1 & /2 & /3.")

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])



def adjust_learning_rate(epoch):
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        print("learning_rate:",lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def checkpoint(epoch):
    model_out_path = "models/GFN_epoch_{}.pkl".format(epoch)
    model_out_path_D = "models/GFN_D_epoch_{}.pkl".format(epoch)
    torch.save(model, model_out_path)
    torch.save(netD, model_out_path_D)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, netD, criterion, optimizer, epoch):
    epoch_loss = 0
    train_gen = train_gen.load_data() ###############
    for iteration, batch in enumerate(train_gen):

        LR_Blur = batch['A']
        LR_Deblur = batch['C']
        HR = batch['B']

        LR_Blur = LR_Blur.to(device)
        LR_Deblur = LR_Deblur.to(device)
        HR = HR.to(device)

        sr = model(LR_Blur)

        # calculate loss_D
        fake_feature_map, fake_sr = netD(sr)
        real_feature_map, real_sr = netD(HR)

        d_loss_real = torch.mean(real_sr)
        d_loss_fake = torch.mean(fake_sr)

        mse_d_loss = criterion(sr, HR)

        feature_loss = 0
        for layer in range(len(fake_feature_map)):
            v1 = Variable(real_feature_map[layer])
            v2 = Variable(fake_feature_map[layer])
            feature_loss += criterion(v1, v2)

        # Compute gradient penalty of HR and sr
        alpha = torch.rand(HR.size(0), 1, 1, 1).cuda().expand_as(HR)
        interpolated = Variable(alpha * HR.data + (1 - alpha) * sr.data, requires_grad=True)
        _, disc_interpolates = netD(interpolated)

        grad = torch.autograd.grad(outputs=disc_interpolates,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)



        # Backward + Optimize
        gradient_penalty = LAMBDA * d_loss_gp

        relu = torch.nn.ReLU(0.2)
        loss_d_mse = relu(mse_d_loss + mse_d_loss + aphal)

        loss_D = d_loss_fake - d_loss_real + gradient_penalty + lambda3 * loss_d_mse

        optimizer_D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # for p in netD.parameters():
        #     p.data.clamp_(-0.01, 0.01)


        # calculate loss_G
        fake_feature_map, G_GAN = netD(sr)
        real_feature_map, _ = netD(HR)

        feature_loss = 0
        for layer in range(len(fake_feature_map)):
            v1 = Variable(real_feature_map[layer])
            v2 = Variable(fake_feature_map[layer])
            feature_loss += criterion(v1, v2)

        loss_G_GAN = - torch.mean(G_GAN)
        loss = criterion(sr, HR)
        Loss_G = loss + lambda2 * loss_G_GAN + feature_loss * lambda1
        epoch_loss += Loss_G
        optimizer.zero_grad()
        Loss_G.backward()
        optimizer.step()


        if iteration % 50 == 0:
            print("===> Epoch[{}]: G_GAN:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, gredient_penalty:{:.4f}, d_real_loss:{:.4f}, d_fake_loss:{:.4f}"
                  .format(epoch, loss_G_GAN.cpu(), Loss_G.cpu(), loss_D.cpu(), gradient_penalty.cpu(), d_loss_real.cpu(), d_loss_fake.cpu()))


            sr_save = transforms.ToPILImage()(sr.cpu()[0])
            sr_save.save('./pictureShow/sr_save.png')
            hr_save = transforms.ToPILImage()(HR.cpu()[0])
            hr_save.save('./pictureShow/hr_save.png')
            blur_lr_save = transforms.ToPILImage()(LR_Blur.cpu()[0])
            blur_lr_save.save('./pictureShow/blur_lr_save.png')


    print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))

opt = parser.parse_args()
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)



train_dir = opt.dataset
train_sets = [x for x in sorted(os.listdir(train_dir)) if is_hdf5_file(x)]
print("===> Loading model and criterion")



if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model = torch.load(opt.resume)
        model.load_state_dict(model.state_dict())
        netD = torch.load(opt.resumeD)
        netD.load_state_dict(netD.state_dict())
        # opt.start_epoch = which_trainingstep_epoch(opt.resume)

else:
    model = Generator()
    netD = Discriminator()
    mkdir_steptraing()

model = model.to(device)
netD = netD.to(device)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer_D = optim.Adam(netD.parameters(), lr = 0.0002)
print()



# opt.start_training_step = 2
# for i in range(opt.start_training_step, 4):
#     opt.nEpochs   = training_settings[i-1]['nEpochs']
#     opt.lr        = training_settings[i-1]['lr']
#     opt.step      = training_settings[i-1]['step']
#     opt.lr_decay  = training_settings[i-1]['lr_decay']
#     opt.lambda_db = training_settings[i-1]['lambda_db']
#     opt.gated     = training_settings[i-1]['gated']
#     print(opt)

opt.start_epoch = 2
opt.nEpochs = 1000
# trainloader = CreateDataLoader(opt)
for epoch in range(opt.start_epoch, opt.nEpochs+1):
    for j  in range(len(train_sets)):
        print("Training folder is {}-------------------------------".format(train_sets[j]))
        train_set = DataSet(join(train_dir, train_sets[j]))
        trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=0)
        train(trainloader, model, netD, criterion, optimizer, epoch)
        if epoch % 5 == 0:
            checkpoint(epoch)

