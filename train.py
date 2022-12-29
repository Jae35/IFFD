import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as pil_image
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from datetime import datetime
from dataset import get_imagefolder_train_loader, get_imagefoler_test_loader
from torch.optim.lr_scheduler import _LRScheduler
from cw_layer import XceptionTransfer,XceptionBN,ResNet18BN,ResNet18Transfer,Densenet121, DensenetTransfer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_face_path(data, number, index):
    num_row = math.sqrt(number)
    size = 256 / num_row
    h = index // num_row
    w = index % num_row
    data_path = torch.zeros(data.shape)
    data_path[:, :, int(w*size):int((w+1)*size), int(h*size):int((h+1)*size)] = \
        data[:, :, int(w*size):int((w+1)*size), int(h*size):int((h+1)*size)]
    return data_path


def train(model, train_data, path_number, use_gpu):
    model.train()
    correct = 0

    for batch_index, (images, labels) in enumerate(train_data):
        if epoch <= args.warm_epoch:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        if use_gpu:
            model = model.cuda()
            labels = labels.cuda()
            images = images.cuda()

        if (batch_index+1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                # update the gradient matrix G
                for path_index in range(path_number):
                    model.change_mode(path_index)
                    X = get_face_path(images, path_number, path_index)
                    x_var = Variable(X).cuda()
                    model(x_var)
                model.update_rotation_matrix()
                # change to ordinary mode
                model.change_mode(-1)
            model.train()

        optimizer.zero_grad()
        outputs = model(images)

        # classification loss
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        print('Training Epoch: {epoch} [{trained}/{total}] Accuracy: {acc} Loss: {loss} LR: {LR}'.format(
            epoch=epoch,
            acc=float(preds.eq(labels).sum())/len(labels),
            trained=batch_index * args.batch_size + len(images),
            total=len(train_data.dataset),
            loss=loss.item(),
            LR=optimizer.param_groups[0]['lr'])
        )
    return


def Test(model, test_data, use_gpu):
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for (images, labels) in test_data:
            images = Variable(images)
            labels = Variable(labels)

            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
                model = model.cuda()

            outputs = model(images)
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            total_loss / len(test_data.dataset),
            float(correct) / len(test_data.dataset)
        ))
    return total_loss / len(test_data.dataset), float(correct) / len(test_data.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action="store_true", default=False, help='disables CUDA training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--mi', default=[5, 10, 15], help='MILESTONES')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight-decay')
    parser.add_argument('--checkpoints-path', type=str, default='./checkpoints/', help='CHECKPOINTS_PATH')
    parser.add_argument('--warm-epoch', type=int, default=1, help='warm-epoch')
    parser.add_argument('--save-epoch', type=int, default=10, help='save-epoch')
    parser.add_argument('--epoch', type=int, default=41, help='epoch')
    parser.add_argument('--batch-size', type=int, default=32, help='TRAINING_BATCH_SIZE')
    parser.add_argument('--num-workers', type=int, default=4, help='NUM_WORKERS')
    parser.add_argument('--path-num', type=int, default=100, help='PATH_NUMBER')
    parser.add_argument('--train-dir', type=str, default='/home/huayingying/Documents/Data/xai_deepfake/datasets/faceforensics++/train',
                        help='train-dir')
    parser.add_argument('--val-dir', type=str, default='/home/huayingying/Documents/Data/xai_deepfake/datasets/faceforensics++/val',
                        help='test-dir')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    new_model = XceptionTransfer(whitened_layers=[14], momentum=0.9)
    #new_model =Densenet121()

    train_loader = get_imagefolder_train_loader(args.train_dir, args.batch_size, args.num_workers)
    print('get train loader done')
    val_loader = get_imagefoler_test_loader(args.val_dir, args.batch_size, args.num_workers)
    print('get val loader done')

    checkpoints_path = os.path.join(args.checkpoints_path, datetime.now().isoformat())
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    # save test accuracy
    save_path = checkpoints_path + '/result.txt'
    res_file = open(save_path, 'a')
    checkpoints_path = os.path.join(checkpoints_path, '{model}-{epoch}-{type}.pth')

    optimizer = optim.SGD(new_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.mi, gamma=args.gamma)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm_epoch)

    best_acc = 0.0
    for epoch in range(1, args.epoch):
        if epoch > args.warm_epoch:
            train_scheduler.step(epoch)
        train(new_model, train_loader, args.path_num, args.cuda)
        loss, acc = Test(new_model, val_loader, args.cuda)
        res_file.write('\tepoch: {epoch}; Val loss: {l}; Val accuracy: {accuracy}\n'.format(epoch=epoch, l=loss,
                                                                                              accuracy=acc))
        if best_acc < acc:
            torch.save(new_model.state_dict(), checkpoints_path.format(model='xception', epoch=epoch, type='best'))
            best_acc = acc
            continue
        if not epoch % args.save_epoch:
            torch.save(new_model.state_dict(), checkpoints_path.format(model='xception', epoch=epoch, type='regular'))
    res_file.close()

