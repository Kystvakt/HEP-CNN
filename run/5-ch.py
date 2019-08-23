import h5py
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

#: parser
def get_parser():
    parser = argparse.ArgumentParser(
        description='Run SUSY RPV training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--train-events', action='store', type=int,
                        default=412416, help='Number of events to train on.')
    parser.add_argument('--test-events', action='store', type=int,
                        default=137471, help='Number of events to test on.')
    parser.add_argument('--batch-size', action='store', type=int, default=256,
                        help='batch size per update')
    parser.add_argument('--lr', action='store', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--ls', action='store', type=bool, default=False,
                        help='log scaling')
    parser.add_argument('--patch-size', action='store', type=int, default=0,
                        help='circular padding size')
    parser.add_argument('--base-value', action='store', type=float,
                        default=1e-6, help='non-zero base value for log')
    parser.add_argument('train_data', action='store', type=str,
                        help='path to HDF5 file to train on')
    parser.add_argument('val_data', action='store', type=str,
                        help='path to HDF5 file to validate on')
    # parser.add_argument('test_data', action='store', type=str,
    #                     help='path to HDF5 file to validate on')
    parser.add_argument('model', action='store', type=str,
                        help='one of: "CNN", "3ch-CNN"')
    return parser

if torch.cuda.is_available(): print("\nGPU Acceleration Available")
else: print("\nGPU Acceleration Unavailable")

parser = get_parser()
args = parser.parse_args()

#: load data
class H5Dataset(Dataset):
    def __init__(self, filePath, evtnum):
        super(H5Dataset, self).__init__()
        h5File = h5py.File(filePath)
        evtnum = evtnum
        im = h5File['all_events']['hist'][:evtnum]
        im = np.expand_dims(im, 1)
        if args.model == 'multi':
            layer_em = h5File['all_events']['histEM'][:evtnum]
            layer_em = np.expand_dims(layer_em, 1)

            layer_track = h5File['all_events']['histtrack'][:evtnum]
            layer_track = np.expand_dims(layer_track, 1)

            layer_em = layer_em / layer_em.max()
            layer_em_ls = np.log10(layer_em + args.base_value) - np.log10(args.base_value)

            layter_track = layer_track / layer_track.max()
            layer_track_ls = np.log10(layer_track + args.base_value) - np.log10(args.base_value)

            im_ls = np.log10(im + args.base_value) - np.log10(args.base_value)
            im = np.concatenate((im, im_ls), axis=1)
            im = np.concatenate(
                (np.concatenate((im, layer_em), axis=1),
                 layer_em_ls), axis=1
            )
            im = np.concatenate((im, layer_track), axis=1)

        if args.patch_size:
            imp = np.empty((im.shape[0],                    # N
                            im.shape[1],                    # C
                            im.shape[2],                    # H
                            im.shape[3] + args.patch_size)) # W
            for idx in range(im.shape[0]):
                patch = im[idx][:, :, :args.patch_size]
                img = np.concatenate((im[idx], patch), axis=2)
                imp[idx] = img
            im = imp

        self.data = im
        lb = np.expand_dims(h5File['all_events']['y'][:args.train_events], 1)
        self.label = torch.Tensor(lb)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]), self.label[idx])
    def __len__(self):
        return self.data.shape[0]
if torch.cuda.is_available: kwargs = {'num_workers': 1, 'pin_memory': True}

trainDataset = H5Dataset(args.train_data, args.train_events)
trainLoader  = DataLoader(trainDataset, batch_size=args.batch_size)

valDataset = H5Dataset(args.val_data, args.test_events)
valLoader  = DataLoader(valDataset, batch_size=args.batch_size)

# testDataset = H5Dataset(args.test_data, args.test_events)
# testLoader  = DataLoader(testDataset, batch_size=args.batch_size)

#: model and train
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=(3, 3), stride=1, padding=(0, 2),
                      padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=(0, 2),
                      padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(0, 2),
                      padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),

            # image dimension < kernel size
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=(0, 2),
            # padding_mode='circular'),
            # nn.ReLU(),
            )
        self.fc = nn.Sequential(
            nn.Linear(256*2*4, 512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
            )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(1, 256*2*4)
        x = self.fc(x)
        return x

model = CNN()
if torch.cuda.is_available(): model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCELoss()

def train(lossList):
    model.train()
    for batch_idx, (data, label) in enumerate(trainLoader):
        if torch.cuda.is_available():
            data, label = data.float().cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {}\t[{}/{}\t({:.2f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(trainDataset),
            100. * batch_idx / len(trainLoader), loss.item()
        ))
    lossList.append(loss.item())

def evaluation(accList):
    model.eval()
    acc = 0
    for batch_idx, (data, label) in enumerate(valLoader):
        if torch.cuda.is_available():
            data, label = data.float().cuda(), label.cuda()
        prediction = model(data.float().to('cuda')).cpu().detach().numpy()
        acc += accuracy_score(label.cpu().detach().numpy(),
                              np.where(prediction > 0.5, 1, 0)
                             )
    acc /= (batch_idx + 1)
    print(acc)
    accList.append(acc)

#: train
from sklearn.metrics import roc_curve, accuracy_score, auc
lossList, accList = [], []
for epoch in tqdm(range(1, args.epochs + 1)):
    train(lossList)
    evaluation(accList)

print(lossList)
print(accList)

#: evaluate
from sklearn.metrics import roc_curve, accuracy_score, auc

def evaluate():
    model.eval()

    acc = 0
    for idx, (data, label) in enumerate(testLoader):
        y_pred = model(data.float().to('cuda')).cpu().detach().numpy()
        acc += accuracy_score(label, np.where(y_pred > 0.5, 1, 0))
        #print("Accuracy:", accuracy_score(label,np.where(y_pred > 0.5, 1, 0)))

    acc /= (idx + 1)
    print(acc)
    fpr, tpr, thr = roc_curve(label.numpy().reshape(-1),
                              y_pred.reshape(-1,))
    roc_auc = auc(fpr, tpr)

    plt.plot(tpr, 1 - fpr, label='AUC = %03f' % roc_auc)
    plt.legend()
    plt.show()

evaluate()
