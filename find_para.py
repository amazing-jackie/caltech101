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
from torchvision.datasets import CIFAR100, CIFAR10

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

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
random.seed("CVPR21")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
# alpha = 0.1
# beta = 0.033
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
                                        transform=test_transform)

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
vis = None
plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

trial = 0

all_indices = set(np.arange(NUM_TRAIN))
val_indices = random.sample(all_indices, 0)
all_indices = np.setdiff1d(list(all_indices), val_indices)

file2 = open("./init_indice.pkl", 'rb')
initial_indices = pickle.load(file2)

# initial_indices = random.sample(list(all_indices), ADDENDUM)
# f = open("./init_indice.pkl", 'rb')
# pickle.dump(initial_indices, f)
# indices = all_indices
# random.shuffle(indices)
# labeled_set = indices[:ADDENDUM]

# unlabeled_set = indices[ADDENDUM:]

# current_indices = list(initial_indices)

train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                          sampler=SubsetRandomSampler(initial_indices), 
                          pin_memory=True)
test_loader  = DataLoader(cifar10_test, batch_size=BATCH)
val_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                          sampler=SubsetRandomSampler(val_indices), 
                          pin_memory=True)
dataloaders  = {'train': train_loader, 'test': test_loader, 'val': val_loader}

# Model
# resnet18    = resnet.ResNet18(num_classes=10).cuda()
# # loss_module = lossnet.LossNet().cuda()
# models      = {'backbone': resnet18}
torch.backends.cudnn.benchmark = False

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

        scores, features, for_gcn = models['backbone'](inputs)
        target_loss = criterion(scores, labels)


        target_loss.backward()
        optimizers['backbone'].step()
        # optimizers['module'].step()



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

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data, cycle, accuracies):
#     print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)
#         print(accuracies, epoch)

file = open("./graph0.pkl", 'rb')
graph = pickle.load(file)
file1 = open("./vector0.pkl", 'rb')
data_graph = pickle.load(file1)
file2 = open("./label0.pkl", 'rb')
labels = pickle.load(file2)
file2 = open("./cur_indice0.pkl", 'rb')
current_indices = pickle.load(file2)

labels = torch.LongTensor(labels)

pos_train = len(current_indices)
mask = np.zeros(data_graph.shape[0])
mask[np.arange(pos_train)] = 1
mask = np.array(mask, dtype=np.bool)
train_mask = mask
# print(train_mask.shape)
mask = np.zeros(data_graph.shape[0])
# mask[np.arange(pos_train, pos_train+5000)] = 1
mask = np.array(mask, dtype=np.bool)
val_mask = mask
mask = np.zeros(data_graph.shape[0])
mask[np.arange(50000, 60000)] = 1
mask = np.array(mask, dtype=np.bool)
test_mask = mask
features = torch.FloatTensor(data_graph)

if hasattr(torch, 'BoolTensor'):
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
else:
    train_mask = torch.ByteTensor(train_mask)
    val_mask = torch.ByteTensor(val_mask)
    test_mask = torch.ByteTensor(test_mask)

in_feats = features.shape[1]
n_classes = 10
graph = nx.from_scipy_sparse_matrix(graph)
n_edges = graph.number_of_edges()
print("""----Data statistics------'
#Edges %d
#Classes %d
#Train samples %d
#Val samples %d
#Test samples %d""" %
    (n_edges, n_classes,
        train_mask.int().sum().item(),
        val_mask.int().sum().item(),
        test_mask.int().sum().item()))

cuda = True
# torch.cuda.set_device(args.gpu)
features = features.cuda()
labels = labels.cuda()
train_mask = train_mask.cuda()
val_mask = val_mask.cuda()
test_mask = test_mask.cuda()    


# graph preprocess and calculate normalization factor
g = graph
# add self loop
if True:
    g.remove_edges_from(nx.selfloop_edges(g))
    g.add_edges_from(zip(g.nodes(), g.nodes()))
g = DGLGraph(g)
n_edges = g.number_of_edges()
# normalization
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
if cuda:
    norm = norm.cuda()
g.ndata['norm'] = norm.unsqueeze(1)

# create GCN model
model = GCN(g,
            in_feats,
            50,
            n_classes,
            1,
            F.relu,
            0.1)

model.load_state_dict(torch.load("model_gcn" + str(0) + ".pt"))

if cuda:
    model.cuda()
# loss_fcn = torch.nn.CrossEntropyLoss()
# vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)



print()
acc = evaluate(model, features, labels, test_mask)
print("Test accuracy {:.2%}".format(acc))
# accuracies.append(acc)

mask = np.ones(data_graph.shape[0])
mask[np.arange(pos_train)] = 0
# mask[np.arange(pos_train, pos_train+5000)] = 0
mask[np.arange(50000, 60000)] = 0
mask = np.array(mask, dtype=np.bool)
mask = torch.BoolTensor(mask)
mask = mask.cuda()
model.eval()

cycle = 0 

a = [0.1, 0.08]
b = [10, 8 , 5 ,3, 1, 0.8 , 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01]
for ai in a:
    for bi in b:
        ac_list=[]
        for l in range(5):
            
            file2 = open("./all_un_indice0.pkl", 'rb')
            all_un_indice = pickle.load(file2)
            file2 = open("./cur_indice0.pkl", 'rb')
            current_indices = pickle.load(file2)
            vat_loss = VATLoss(xi=ai, eps=bi, ip=1) #0.05
            lds, lds_each = vat_loss(model, features)
            lds_each = lds_each[mask]
            lds_each = lds_each.view(-1)
            _, querry_indices = torch.topk(lds_each, int(3000))
            querry_indices = querry_indices.cpu()
            querry_pool_indices = np.asarray(all_un_indice)[querry_indices]


            excout = model(features)
            excout = F.softmax(excout, dim=1)
            excout = excout.cpu()
            excout = excout.detach().numpy()
            entropy_list = -excout * np.log(excout)
            entropy_list = np.add.reduce(entropy_list, axis=1)
            entropy_list = torch.FloatTensor(entropy_list)
            entropy_list = entropy_list[mask]
            entropy_list = entropy_list[querry_indices]
            _, querry_indices_en = torch.topk(entropy_list, int(1000))
            sampled_indices = querry_pool_indices[querry_indices_en]

            current_indices = list(current_indices) + list(sampled_indices)
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
                                      sampler=SubsetRandomSampler(current_indices), 
                                      pin_memory=True)
            resnet18    = resnet.ResNet18(num_classes=10).cuda()
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
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data, cycle, accuracies)
            acc = test(models, dataloaders, mode='test')
            accuracies.append(acc)
            # print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(current_indices), acc))
            ac_list.append(acc)
            
        print(ac_list, ai, bi)