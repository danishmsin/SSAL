from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.return_dataset import return_dataset_rot, return_dataset
from utils.loss import entropy, adentropy
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--multi', type=float, default=0.01, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--target', type=str, default='real',
                    help='source domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

args = parser.parse_args()
print('Dataset %s Target %s Network %s' % (args.dataset, args.target, args.net))
target_loader = return_dataset_rot(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/' % (args.dataset)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir, 'net_%s_%s' %(args.net, args.target))

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

F1 = nn.Linear(inc,4)

lr = args.lr
print("learning rate: ",lr)
G.cuda()
F1.cuda()

im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.cuda()
gt_labels_t = gt_labels_t.cuda()

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(G.parameters(), momentum=0.9, lr=0.001,weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=0.01, momentum=0.9,weight_decay=0.0005, nesterov=True)
    criterion = nn.CrossEntropyLoss().cuda()
    best_acc = 0
    counter = 0 
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, G, F1, target_loader, optimizer_g, optimizer_f, criterion)
        loss_train, acc_train = test(target_loader)
        G.train()
        F1.train()
        if acc_train >= best_acc:
            best_acc = acc_train
            counter = 0
        else:
            counter += 1
        if args.early:
            if counter > args.patience:
                break
        print('best acc  %f' % (best_acc))
        print('record %s' % record_file)
        with open(record_file, 'a') as f:
            f.write('epoch %d best %f  \n' % (epoch, best_acc))
        if args.save_check:
            print('saving model')
            torch.save(G.state_dict(), os.path.join(args.checkpath, "G_iter_model_{}_epoch_{}.pth.tar".format(args.target, epoch)))
            torch.save(F1.state_dict(), os.path.join(args.checkpath, "F1_iter_model_{}_epoch_{}.pth.tar".format(args.target, epoch)))

def train_epoch(epoch, args, G, F1, data_loader, optimizer_g, optimizer_f, criterion):
    optimizer_g.zero_grad()
    optimizer_f.zero_grad()
    for batch_idx, data_t in enumerate(data_loader):
        im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
        output = G(im_data_t)
        out1 = F1(output)
        loss = criterion(out1, gt_labels_t)
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(im_data_t), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = 4
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\Test set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()
