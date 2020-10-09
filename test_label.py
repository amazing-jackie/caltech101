'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

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
# Seed
random.seed("Inyoung Cho")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

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
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

##
# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    # models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        # print(inputs.shape, labels.shape)
        iters += 1

        optimizers['backbone'].zero_grad()
        # optimizers['module'].zero_grad()

        # class LabelSmoothingLoss(nn.Module):
        # def __init__(self, classes, smoothing=0.0, dim=-1):

        scores, features, for_gcn = models['backbone'](inputs)
        # smooth_label = smooth_one_hot(labels, 10)
        # target_loss = criterion(scores, labels)

        # outputs = model(data)
        loss_criterion = LabelSmoothingLoss(classes = 10 , smoothing=0.1)
        # smooth_label = smooth_one_hot(true_labels= labels, classes=10)

        target_loss = loss_criterion(scores, labels)
        # loss = (outputs, smooth_label)

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

            scores, _, _ = models['backbone'](inputs)
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

        # Save a checkpoint
        # acc = test(models, dataloaders, 'val')
        # if best_acc < acc:
        #     best_acc = acc
        #     best_model = copy.deepcopy(models['backbone'])
        # test_acc = test(models, dataloaders, 'test')
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

    # torch.save(best_model.state_dict(), "model_" + str(cycle) + ".pt")
    print('>> Finished.')

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


##
# Main
if __name__ == '__main__':
    args = arguments.get_args()
    # main(args)
    # vis = visdom.Visdom(server='http://localhost', port=9000)
    vis = None
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.

        all_indices = set(np.arange(NUM_TRAIN))
        val_indices = random.sample(all_indices, 5000)
        all_indices = np.setdiff1d(list(all_indices), val_indices)

        initial_indices = random.sample(list(all_indices), ADDENDUM)
        # indices = all_indices
        # random.shuffle(indices)
        # labeled_set = indices[:ADDENDUM]

        # unlabeled_set = indices[ADDENDUM:]

        current_indices = list(initial_indices)
        
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

        # Active learning cycles
        accuracies = []
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization

            # Randomly sample 10000 unlabeled data points
            # random.shuffle(unlabeled_set)
            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
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
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(current_indices), acc))


            #semi
            best_model = models['backbone']
            # best_model.load_state_dict(torch.load("model_" + str(cycle) + ".pt"))
            best_model.eval()
            x = None
            y_l = None
            for img, label in dataloaders['train']:
                img = img.cuda()
                label = label.cuda()
                with torch.no_grad():
                    _,_, feature_last = best_model(img)
                    if y_l is not None:
                        y_l = torch.cat((y_l, label),0)
                        # print(y_l.shape)
                    else:
                        y_l = label

                    if x is not None:
                        x = torch.cat((x,feature_last),0)
                        # print(x.shape)
                    else:
                        x=feature_last

            for img, label in val_loader:
                img = img.cuda()
                label = label.cuda()
                with torch.no_grad():
                    _,_,feature_last = best_model(img)
                    if y_l is not None:
                        y_l = torch.cat((y_l, label),0)
                        # print(y_l.shape)
                    else:
                        y_l = label

                    if x is not None:
                        x = torch.cat((x,feature_last),0)
                        # print(x.shape)
                    else:
                        x=feature_last
            all_un_indice = []

            for img, label, indice_un in unlabeled_loader:
                img = img.cuda()
                label = label.cuda()
                with torch.no_grad():
                    _,_,feature_last = best_model(img)
                    if y_l is not None:
                        y_l = torch.cat((y_l, label),0)
                        # print(y_l.shape)
                    else:
                        y_l = label

                    if x is not None:
                        x = torch.cat((x,feature_last),0)
                        # print(x.shape)
                    else:
                        x=feature_last
                all_un_indice.extend(indice_un)

            for imgs, labels in test_loader:
                imgs = imgs.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    _,_,feature_test = best_model(imgs)
                    if y_l is not None:
                        y_l = torch.cat((y_l, labels), 0)
                        # print(y_l.shape)
                    else:
                        y_l = labels

                    if x is not None:
                        x = torch.cat((x,feature_test),0)
                        # print(x.shape)
                    else:
                        x=feature_test
            print(x.shape , y_l.shape)
            x = x.cpu()
            y_l = y_l.cpu()
            f = open("./vector.pkl", 'wb')
            pickle.dump(x.numpy(), f)

            f = open("./label.pkl", 'wb')
            pickle.dump(y_l.numpy(), f)
            k = 201
            file_vector = open("./vector.pkl", 'rb')
            data_vector = pickle.load(file_vector)
            # m = distance.cdist(data, data, 'euclidean')
            m = metrics.pairwise.euclidean_distances(data_vector, data_vector)
            idx = np.argpartition(m, k, axis=1)
            idx = idx[:, :k]
            mask = np.zeros(shape=(data_vector.shape[0], data_vector.shape[0]), dtype=bool)   # undo
            I = np.arange(data_vector.shape[0])[:, np.newaxis]
            mask[I, idx] = True
            mask = np.logical_or(mask, mask.transpose())
            mask = mask.astype(float)
            for i in range(data_vector.shape[0]):
                mask[i][i]=0.
            # mask = mask - np.identity(data.shape[0])
            r = sparse.csr_matrix(mask)
            f = open("./graph_ec_50.pkl", 'wb')
            pickle.dump(r, f)

            file = open("./graph_ec_50.pkl", 'rb')
            graph = pickle.load(file)
            file1 = open("./vector.pkl", 'rb')
            data_graph = pickle.load(file1)
            file2 = open("./label.pkl", 'rb')
            labels = pickle.load(file2)

            labels = torch.LongTensor(labels)

            pos_train = len(current_indices)
            mask = np.zeros(data_graph.shape[0])
            mask[np.arange(pos_train)] = 1
            mask = np.array(mask, dtype=np.bool)
            train_mask = mask
            # print(train_mask.shape)
            mask = np.zeros(data_graph.shape[0])
            mask[np.arange(pos_train, pos_train+5000)] = 1
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

            if args.gpu  < 0:
                cuda = False
            else:
                cuda = True
                torch.cuda.set_device(args.gpu)
                features = features.cuda()
                labels = labels.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()    


            # graph preprocess and calculate normalization factor
            g = graph
            # add self loop
            if args.self_loop:
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
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        F.relu,
                        args.dropout)

            if cuda:
                model.cuda()
            loss_fcn = torch.nn.CrossEntropyLoss()
            vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)

            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

            # initialize graph
            dur = []
            for epoch in range(args.n_epochs):
                model.train()
                if epoch >= 3:
                    t0 = time.time() 

                logits = model(features)
                loss = loss_fcn(logits[train_mask], labels[train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                acc = evaluate(model, features, labels, val_mask)
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                    "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                    acc, n_edges / np.mean(dur) / 1000))

            print()
            acc = evaluate(model, features, labels, test_mask)
            print("Test accuracy {:.2%}".format(acc))
            accuracies.append(acc)

            mask = np.ones(data_graph.shape[0])
            mask[np.arange(pos_train)] = 0
            mask[np.arange(pos_train, pos_train+5000)] = 0
            mask[np.arange(50000, 60000)] = 0
            mask = np.array(mask, dtype=np.bool)
            mask = torch.BoolTensor(mask)
            mask = mask.cuda()
            model.eval()
            lds, lds_each = vat_loss(model, features)
            print(lds.shape, lds_each.shape)
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

            f = open("./sam" + str(cycle) + ".pkl", 'wb')
            pickle.dump(sampled_indices, f)

            # _, querry_indices = torch.topk(entropy_list, int(1000))
            # querry_indices = querry_indices.cpu()
            # querry_pool_indices = np.asarray(all_un_indice)[querry_indices]
            # sampled_indices = querry_pool_indices

            # random_indice = random.sample(set(np.arange(len(all_un_indice))), SUBSET)
            # lds_each = lds_each[np.asarray(random_indice)]
            # all_un_indice = np.asarray(all_un_indice)[np.asarray(random_indice)]
            # _, querry_indices = torch.topk(lds_each, int(1000))
            # querry_indices = querry_indices.cpu()
            # querry_pool_indices = np.asarray(all_un_indice)[querry_indices]
            # sampled_indices = querry_pool_indices


            current_indices = list(current_indices) + list(sampled_indices)
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(current_indices), 
                                              pin_memory=True)


            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # # Randomly sample 10000 unlabeled data points
            # random.shuffle(unlabeled_set)
            # subset = unlabeled_set[:SUBSET]

            # # Create unlabeled dataloader for the unlabeled subset
            # unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
            #                               sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
            #                               pin_memory=True)

            # Measure uncertainty of each data points in the subset
            # uncertainty = get_uncertainty(models, unlabeled_loader)

            # # Index in ascending order
            # arg = np.argsort(uncertainty)
            
            # # Update the labeled dataset and the unlabeled dataset, respectively
            # labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            # unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # # Create a new dataloader for the updated labeled dataset
            # dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
            #                                   sampler=SubsetRandomSampler(labeled_set), 
            #                                   pin_memory=True)
        
        # Save a checkpoint

        print(accuracies)
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                    # 'state_dict_module': models['module'].state_dict()
                },
                './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))