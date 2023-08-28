# Copyright (c) 2023-present
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=512, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--mixupBatch', default=1, type=int,
                    help='number of points to do assignment mixup with')
parser.add_argument('--manifold_mixup', action='store_true',
                    help='Randomly choose layer on which to do matching')
parser.add_argument('--test_noise', default=0.0, type=float,
                    help='Std dev of Gaussian noise added to test data')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])


class AddNoise(object):
    def __call__(self, sample):
        return sample + args.test_noise * torch.randn(list(sample.size()))


trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8, drop_last=True)

trainloaderAlpha = torch.utils.data.DataLoader(trainset,
                                               batch_size=64*20,
                                               shuffle=True, num_workers=8, drop_last=True)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
    AddNoise(),
])

testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
if not os.path.isdir('results/CIFAR'):
    os.mkdir('results/CIFAR')
logname = ('results/CIFAR/log_' + 'ResNet' + '_' + args.name + '_' + str(args.seed) +
           '_mix' + str(args.mixupBatch) + '_alpha' + str(args.alpha) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


def get_k_alpha(x, alpha, mixupBatch=1, use_cuda=True):
    # Chooses an alpha_k for k > 1 such that the squared match distances correspond to
    # k = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    # chunk off blocks of size mixupBatch
    # for each chunk, do closest assigments (hungarian)
    xx = x.view(batch_size, -1)
    indexA = index
    lam2 = 0.5
    cost1 = (torch.norm((1 - lam2) * x + (lam2 - 1) * x[indexA, :]) ** 2).cpu()
    cost = cost1
    vals = np.random.beta(alpha, alpha, 1000)
    exp_1 = np.mean((np.minimum(vals, 1 - vals)) ** 2)

    if mixupBatch > 1:
        cost = 1000. * np.ones((batch_size, batch_size))
        outs = net(x)
        layer = 0
        hidLay = outs[layer]

        if layer != 0:
            hidLay2 = hidLay.detach()
        else:
            hidLay2 = hidLay

        hidLayX = hidLay2.view(batch_size, -1)
        index = index.cpu().numpy()
        for i in range(int(np.ceil(batch_size/mixupBatch))):
            hidCPU = (hidLayX).cpu().detach().numpy()
            cost[i*mixupBatch:(i+1)*mixupBatch, i*mixupBatch:(i+1)*mixupBatch] = distance_matrix(
                    hidCPU[i*mixupBatch:(i+1)*mixupBatch, :], 
                    hidCPU[index[i*mixupBatch:(i+1)*mixupBatch], :])
        row_ind, indMatches = linear_sum_assignment(cost)
        indexA = index[indMatches]

        lam2 = 0.5
        cost = (torch.norm((1 - lam2) * x + (lam2 - 1) * x[indexA, :]) ** 2).cpu()

        # Adjust alpha via ratio of expectations
        # trial and error (sort of)
        vals = np.random.beta(alpha, alpha, 1000)
        exp_1 = np.mean((np.minimum(vals, 1 - vals)) ** 2)

        alpha_k = alpha
        exp_k = exp_1

        while exp_k < exp_1*np.sqrt(cost1/cost) and alpha_k <= 100:
            alpha_k *= 1.025
            vals = np.random.beta(alpha_k, alpha_k, 1000)
            exp_k = np.mean((np.minimum(vals, 1 - vals)) ** 2)

    else:
        alpha_k = alpha
        exp_k = exp_1
    print('alpha_k = ' + str(alpha_k))
    return (exp_k*cost)/batch_size, alpha_k


def mixup_data(x, y, alpha=1.0, mixupBatch=1,  use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    # chunk off blocks of size mixupBatch
    # for each chunk, do closest assigments (hungarian)
    xx = x.view(batch_size, -1)
    if mixupBatch == 1:
        indexA = index
        layer = 0
        if args.manifold_mixup:
            outs = net(x)
            maxLay = len(outs)
            layer = np.random.randint(0, 2)
            hidLay = outs[layer]
    else:
        cost = 1000. * np.ones((batch_size, batch_size))
        outs = net(x)
        if args.manifold_mixup:
            layer = np.random.randint(0, 2)
        else:
            layer = 0
        hidLay = outs[layer]
        if layer != 0:
            hidLay2 = hidLay.detach()
        else:
            hidLay2 = hidLay
        hidLayX = hidLay2.view(batch_size, -1)

        if use_cuda:
            hidLayX = hidLayX.cpu().numpy()

        for i in range(int(np.ceil(batch_size/mixupBatch))):
            cost[i*mixupBatch:(i+1)*mixupBatch, i*mixupBatch:(i+1)*mixupBatch] = distance_matrix(
                hidLayX[i*mixupBatch:(i+1)*mixupBatch, :], 
                hidLayX[index[i*mixupBatch:(i+1)*mixupBatch], :])
        row_ind, indMatches = linear_sum_assignment(cost)
        if mixupBatch > 1:
            indexA = index[indMatches]
        else:
            indexA = index

    if args.manifold_mixup:
        if mixupBatch == 1:
            mixed_x = lam * hidLay + (1 - lam) * hidLay[indexA, :]
        else:
            mixed = lam * hidLayX + (1 - lam) * hidLayX[indexA, :]
            med = torch.matmul(hidLayX, torch.transpose(mixed, 0, 1))
            med = torch.transpose(med.real, 0, 1)
            med = med.float()
            mixed_x = torch.tensordot(med.cuda(), hidLay, 1)
        cost = 0
    else:
        layer = 0
        mixed_x = lam * x + (1 - lam) * x[indexA, :]
        if lam < .5:
            lam2 = 1 - lam
        else:
            lam2 = lam
        cost = torch.norm((1 - lam2) * x + (lam2 - 1) * x[indexA, :])**2
    y_a, y_b = y, y[indexA]
    return mixed_x, y_a, y_b, lam, layer, cost


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_alpha():
    for batch_ids, (inputs, targets) in enumerate(trainloaderAlpha):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        cost, alpha_k = get_k_alpha(
            inputs, args.alpha, args.mixupBatch, use_cuda=True)
        break
    return cost, alpha_k


def train(epoch, alpha_k):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam, layer, cost = mixup_data(inputs, targets,
                                                                    alpha_k, args.mixupBatch, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = (net(inputs, start_lay=layer))['out']
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        if train_loss == 0:
            train_loss = 0.0001
        else:
            train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total, cost/batch_idx)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = (net(inputs))['out']
        loss = criterion(outputs, targets)

        test_loss += loss.data  # [0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


sqrdist, alpha_k = get_alpha()
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc', 'match cost', 'alphaK', 'sqrdist'])

for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer, epoch)
    train_loss, reg_loss, train_acc, cost = train(epoch, args.alpha)
    test_loss, test_acc = test(epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc.numpy(), cost, alpha_k, sqrdist])
