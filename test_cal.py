import time
import os
import random
import numpy as np
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision
# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
from torchvision.transforms import transforms
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10, Caltech101,Cityscapes

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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from gcn import GCN
from vat import VATLoss
from imutils import paths
import cv2
import pretrainedmodels

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED']=str(SEED)
    # torch.backends.cudnn.benchmark = False # as all the inputs are not of same size
SEED=42
seed_everything(SEED=SEED)

RGB_MEAN = [0.5429, 0.5263, 0.4994]
RGB_STD = [0.2422, 0.2392, 0.2406]

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

epochs = 100

image_paths = list(paths.list_images('./caltech/101_ObjectCategories'))

data = []
labels = []
label_names = []
for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    if label == 'BACKGROUND_Google':
        continue

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data.append(image)
    label_names.append(label)
    labels.append(label)
data = np.array(data)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(len(lb.classes_))

(X, x_val , Y, y_val) = train_test_split(data, labels, 
                                                    test_size=0.2,  
                                                    stratify=labels,
                                                    random_state=42)

(x_train, x_test, y_train, y_test) = train_test_split(X, Y, 
                                                    test_size=0.25, 
                                                    random_state=42)

print(f"x_train examples: {x_train.shape}\nx_test examples: {x_test.shape}\nx_val examples: {x_val.shape}")

# define transforms
# train_transform = transforms.Compose(
#     [transforms.ToPILImage(),
# 	 transforms.Resize((224, 224)),
#     #  transforms.RandomRotation((-30, 30)),
#     #  transforms.RandomHorizontalFlip(p=0.5),
#     #  transforms.RandomVerticalFlip(p=0.5),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225])])

train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(256, (.8, 1)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(RGB_MEAN, RGB_STD),
                            ])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
	 transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
 
train_data = ImageDataset(x_train, y_train, train_transform)
val_data = ImageDataset(x_val, y_val, val_transform)
test_data = ImageDataset(x_test, y_test, val_transform)
 
# dataloaders
trainloader = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=16)
valloader = DataLoader(val_data, batch_size=16, shuffle=True,num_workers=16)
testloader = DataLoader(test_data, batch_size=16, shuffle=False,num_workers=16)

def imshow(img):
    plt.figure(figsize=(15, 12))
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

train_transform = transforms.Compose(
    [transforms.ToPILImage(),
	 transforms.Resize((224, 224)),
#      transforms.RandomCrop(224),
#      transforms.RandomHorizontalFlip(),
    #  transforms.RandomRotation((-30, 30)),
    #  transforms.RandomHorizontalFlip(p=0.5),
    #  transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

# train_transform = transforms.Compose([
#                             transforms.ToPILImage(),
#                             transforms.RandomResizedCrop(256, (.8, 1)),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.RandomRotation(degrees=15),
#                             transforms.ColorJitter(),
#                             transforms.CenterCrop(224),
#                             transforms.ToTensor(),
#                             transforms.Normalize(RGB_MEAN, RGB_STD),
#                             ])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
	 transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
 
train_data = ImageDataset(x_train, y_train, train_transform)
val_data = ImageDataset(x_val, y_val, val_transform)
test_data = ImageDataset(x_test, y_test, val_transform)
 
# dataloaders
trainloader = DataLoader(train_data, batch_size=128, shuffle=True,num_workers=16)
valloader = DataLoader(val_data, batch_size=128, shuffle=True,num_workers=16)
testloader = DataLoader(test_data, batch_size=128, shuffle=False,num_workers=16)

def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    # models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        # print(inputs.shape, labels.shape)
#         iters += 1

        optimizers['backbone'].zero_grad()
        # optimizers['module'].zero_grad()

#         scores, features, for_gcn = models['backbone'](inputs)
        scores = models['backbone'](inputs)
#         print(scores.shape)
#         print(torch.max(labels, 1)[1])
        target_loss = criterion(scores, torch.max(labels, 1)[1])

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

#
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
            labels = torch.max(labels, 1)[1]
#             scores, _, _ = models['backbone'](inputs)
            scores = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data, cycle, accuracies):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)
        print(accuracies, epoch)

        # Save a checkpoint
        acc = test(models, dataloaders, 'test')
        print(acc)
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
# args = arguments.get_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
accuracies = []
# main(args)
# vis = visdom.Visdom(server='http://localhost', port=9000)
vis = None
plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

for trial in range(1):
    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.

#     all_indices = set(np.arange(NUM_TRAIN))
#     val_indices = random.sample(all_indices, 0)
#     all_indices = np.setdiff1d(list(all_indices), val_indices)

#     initial_indices = random.sample(list(all_indices), ADDENDUM)
#     f = open("./init_indice.pkl", 'wb')
#     pickle.dump(initial_indices, f)
    # indices = all_indices
    # random.shuffle(indices)
    # labeled_set = indices[:ADDENDUM]

    # unlabeled_set = indices[ADDENDUM:]

#     current_indices = list(initial_indices)

    train_loader = trainloader
    test_loader  = testloader
    val_loader = valloader
    dataloaders  = {'train': train_loader, 'test': test_loader, 'val': val_loader}

    # Model
    # resnet18    = resnet.ResNet18(num_classes=10).cuda()
    # # loss_module = lossnet.LossNet().cuda()
    # models      = {'backbone': resnet18}
    # torch.backends.cudnn.benchmark = False

    # Active learning cycles

    for cycle in range(1):
        # Loss, criterion and scheduler (re)initialization

        # Randomly sample 10000 unlabeled data points
        # random.shuffle(unlabeled_set)
#         unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
#         # Create unlabeled dataloader for the unlabeled subset
#         unlabeled_loader = DataLoader(train_data, batch_size=BATCH, 
#                                       sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
#                                       pin_memory=True)

        resnet18    = resnet.ResNet18()
        resnet18 = nn.DataParallel(resnet18)
        resnet18.cuda()
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
        train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data, 1, accuracies)
        acc = test(models, dataloaders, mode='test')
        accuracies.append(acc)
        print(acc)