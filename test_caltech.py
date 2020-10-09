# Python
import time
import os
import random
import numpy as np
import networkx as nx
import torch.nn.functional as F
# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10,Caltech101
# from torchvision.datasets import CIFAR100, CIFAR10
# from my_caltechdataset import *

# Utils
import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from config import *
from data.sampler import SubsetSequentialSampler
import copy
import pickle

import arguments
from utils import *
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import sparse
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from gcn import GCN
from vat import VATLoss

#1.seed 2.device 3.recifar 4.backends 5.mask
# Seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) #cpu
torch.cuda.manual_seed_all(seed)  #并行gpu
torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
torch.backends.cudnn.benchmark = True   #训练集变化不大
alpha = 0.3
beta = 0.033
accuracies = []

##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])
cifar10_test  = CIFAR10('./cifar10', train=False, download=True, transform=test_transform)
cifar10_train = CIFAR10('./cifar10', train=True, download=True, transform=train_transform)

class CIFAR10_re(Dataset):
    def __init__(self, path):
        self.cifar10 = CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=train_transform)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)

cifar10_unlabeled   = CIFAR10_re('./cifar10')

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


##
# Train Utils
iters = 0



def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    # models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        # print(inputs.shape, labels.shape)
        iters += 1
#         print(inputs.shape)
        optimizers['backbone'].zero_grad()
        # optimizers['module'].zero_grad()
        models['backbone'].cuda()
        scores, features, for_gcn = models['backbone'](inputs)
        target_loss = criterion(scores, torch.max(labels, 1)[1].cuda())

        # if epoch > epoch_loss:
        #     # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
        #     features[0] = features[0].detach()
        #     features[1] = features[1].detach()
        #     features[2] = features[2].detach()
        #     features[3] = features[3].detach()
        # pred_loss = models['module'](features)
        # pred_loss = pred_loss.view(pred_loss.size(0))

        # m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        # m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        # loss            = m_backbone_loss + WEIGHT * m_module_loss

        target_loss.backward()
        optimizers['backbone'].step()
        # optimizers['module'].step()

        # Visualize
        # if (iters % 100 == 0) and (vis != None) and (plot_data != None):
        #     plot_data['X'].append(iters)
        #     plot_data['Y'].append([
        #         m_backbone_loss.item(),
        #         m_module_loss.item(),
        #         loss.item()
        #     ])
        #     vis.line(
        #         X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
        #         Y=np.array(plot_data['Y']),
        #         opts={
        #             'title': 'Loss over Time',
        #             'legend': plot_data['legend'],
        #             'xlabel': 'Iterations',
        #             'ylabel': 'Loss',
        #             'width': 1200,
        #             'height': 390,
        #         },
        #         win=1
        #     )



def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data, cycle, accuracies):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
#     for epoch in range(num_epochs):
    schedulers['backbone'].step()
    # schedulers['module'].step()

    train_epoch(models, criterion, optimizers, dataloaders, 1, epoch_loss, vis, plot_data)
#     print(accuracies, epoch)

    # Save a checkpoint
    # acc = test(models, dataloaders, 'val')
    # if best_acc < acc:
    #     best_acc = acc
    #     best_model = copy.deepcopy(models['backbone'])
    # # test_acc = test(models, dataloaders, 'test')
    # print(acc, 'val_acc', best_acc, 'best_acc', 'accs', accuracies, 'epoch', epoch, 'cycle' ,cycle)



    # if False and epoch % 5 == 4:
    #     acc = test(models, dataloaders, 'test')
    #     if best_acc < acc:
    #         best_acc = acc
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'state_dict_backbone': models['backbone'].state_dict(),
    #             'state_dict_module': models['module'].state_dict()
    #         },
    #         '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
    #     print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

    torch.save(models['backbone'].state_dict(), "model_" + str(cycle) + ".pt")
    print('>> Finished.')




def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    # models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()



RGB_MEAN = [0.5429, 0.5263, 0.4994]
RGB_STD = [0.2422, 0.2392, 0.2406]

train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(256, (.8, 1)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(RGB_MEAN, RGB_STD),
                            ])

train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(256, (.8, 1)),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomRotation(degrees=15),
                            # transforms.ColorJitter(),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Normalize(RGB_MEAN, RGB_STD),
                            ])

# train_transform = transforms.Compose([
#     T.RandomHorizontalFlip(),
#     T.RandomCrop(size=32, padding=4),
#     T.ToTensor(),
#     T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#     ])

# train_transform = transforms.Compose(
#     [     transforms.Resize((224, 224)),
#      transforms.ToTensor(),
#      transforms.Lambda(lambda x: x.repeat(3,1,1)),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225])])
cifar10_train = Caltech101('./caltech', download=False, transform=train_transform)
# cifar10_train = Caltech101('./caltech', download=False, transform=None)
print(cifar10_train)


##
# Main
# if __name__ == '__main__':
# args = arguments.get_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# main(args)
# vis = visdom.Visdom(server='http://localhost', port=9000)
vis = None
plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

for trial in range(TRIALS):
    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.

    all_indices = set(np.arange(NUM_TRAIN))
    val_indices = random.sample(all_indices, 0)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), ADDENDUM)
    f = open("./init_indice.pkl", 'wb')
    pickle.dump(initial_indices, f)
    # indices = all_indices
    # random.shuffle(indices)
    # labeled_set = indices[:ADDENDUM]

    # unlabeled_set = indices[ADDENDUM:]

    current_indices = list(initial_indices)

    train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                              sampler=SubsetRandomSampler(initial_indices), 
                              pin_memory=True,drop_last=False)
    test_loader  = DataLoader(cifar10_train, batch_size=BATCH)
    val_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                              sampler=SubsetRandomSampler(val_indices), 
                              pin_memory=True)
    dataloaders  = {'train': train_loader, 'test': test_loader, 'val': val_loader}

    # Model
    # resnet18    = resnet.ResNet18(num_classes=10).cuda()
    # # loss_module = lossnet.LossNet().cuda()
    # models      = {'backbone': resnet18}
    # torch.backends.cudnn.benchmark = False

    # Active learning cycles

#         for cycle in range(CYCLES):
    # Loss, criterion and scheduler (re)initialization

    # Randomly sample 10000 unlabeled data points
    # random.shuffle(unlabeled_set)
    unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                  sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
                                  pin_memory=True)

    resnet18    = resnet.ResNet18(num_classes=102).cuda()
    # loss_module = lossnet.LossNet().cuda()
    models      = {'backbone': resnet18}
    criterion      = nn.CrossEntropyLoss()
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                            momentum=MOMENTUM, weight_decay=WDECAY)
    # optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
    #                         momentum=MOMENTUM, weight_decay=WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
    # sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}

    # Training and test
    train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data, 0, accuracies)
    acc = test(models, dataloaders, mode='test')
    accuracies.append(acc)
    print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(current_indices), acc))


    #semi
