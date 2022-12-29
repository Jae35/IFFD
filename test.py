import math
from torchvision import transforms
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from cw_layer import XceptionTransfer, XceptionBN, ResNet18BN, ResNet18Transfer, Densenet121, DensenetTransfer
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import torch.nn.functional as F
from matplotlib import cm
from  sklearn.metrics import roc_auc_score as AUC
from dataset import get_imagefoler_test_loader
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import os

def normalization_data(data):
    max_index = np.unravel_index(np.argmax(data), data.shape)
    max_data = data[max_index]
    min_index = np.unravel_index(np.argmin(data), data.shape)
    min_data = data[min_index]
    data_normalize = (data - min_data) / (max_data-min_data)
    return data_normalize

def preprocess_image(image, cuda):
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    preprocessed_image = preprocess(Image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def accuracy(model, test_data, use_gpu):
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

def auc_score(model, test_data, use_gpu):
    model.eval()
    i = 0
    with torch.no_grad():
        for (images, labels) in test_data:
            images = Variable(images)
            labels = Variable(labels)

            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
                model = model.cuda()

            outputs = model(images)
            _, preds = outputs.max(1)
            i += 1
            if i == 1:
                frame_labels = labels
                frame_preds = preds
                continue
            frame_labels = torch.cat([frame_labels, labels],dim=0)
            frame_preds = torch.cat([frame_preds, preds],dim=0)

    frame_auc = AUC(frame_labels, frame_preds)
    print("AUC:", frame_auc)
    return frame_auc
	
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j, batch) in enumerate(dtloader):
            x, y_true = batch
            y_pred = model.forward(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))

    return y_true_all.detach(), y_pred_all.detach()

    y_true_all, y_pred_all = Eval(model, dataloader)

    y_true_all, y_pred_all = np.array(
        y_true_all.cpu()), np.array(y_pred_all.cpu())

    fprs, tprs, ths = roc_curve(
        y_true_all, y_pred_all, pos_label=1, drop_intermediate=False)

    # acc = accuracy_score(y_true_all, np.where(y_pred_all >= 0.5, 1, 0))*100.
    # print("acc:", acc)
    #
    # ind = 0
    # for fpr in fprs:
    #     if fpr > 1e-2:
    #         break
    #     ind += 1
    # TPR_2 = tprs[ind-1]
    # print("TPR_2:", TPR_2)

    # ind = 0
    # for fpr in fprs:
    #     if fpr > 1e-3:
    #         break
    #     ind += 1
    # TPR_3 = tprs[ind-1]
    #
    # ind = 0
    # for fpr in fprs:
    #     if fpr > 1e-4:
    #         break
    #     ind += 1
    # TPR_4 = tprs[ind-1]
    au = auc(fprs, tprs)
    print("auc:", au)
    # ap = average_precision_score(y_true_all, y_pred_all)
    # print("AP:", ap)
    return au
    import pandas as pd
    import seaborn as sns
    image = preprocess_image(image, False)
    model_vis = nn.Sequential(*list(model.model.children())[:-1])
    feature = model_vis(image).squeeze(0)
    m = int(feature.shape[0])
    df = np.zeros([m, m])
    for i in range(m):
        feature1 = feature[i, :, :]
        feature1 = feature1.squeeze(0).data.cpu().numpy()
        feature1 = feature1.reshape(feature1.size, order='C')
        for j in range(m):
            feature2 = feature[j, :, :]
            feature2 = feature2.squeeze(0).data.cpu().numpy()
            feature2 = feature2.reshape(feature2.size, order='C')
            df[i, j] = np.corrcoef(feature1, feature2)[0, 1]
    dff = pd.DataFrame(df)

    plt.subplots(figsize=(12, 12))
    sns.heatmap(dff, annot=False, vmax=1, vmin=0)
    # plt.savefig("/home/huayingying/Documents/results/corr_after.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    y1 = np.zeros((1, 2025))
    for img_path in os.listdir('./datasets/test_img/real/'):
        image = cv2.imread('./datasets/test_img/real/' + img_path)

        # saliency map of xceptiontransfer
        height, width, _ = image.shape
        image_tensor = preprocess_image(image, False)
        hook_a = None

        submodule_dict = dict(model.named_modules())
        target_layer = submodule_dict['model.bn4']
        hook1 = target_layer.register_forward_hook(_hook_a)

        scores = model(image_tensor)
        print(torch.softmax(scores, dim=-1))
        hook1.remove()
        class_idx = scores.squeeze(0).argmax().item()
        weights = model.model.last_linear.weight.data[class_idx, :]

        feature = hook_a.squeeze(0)
        x = weights.view(*weights.shape, 1, 1) * (F.adaptive_avg_pool2d(feature, (1, 1)))
        data_f = x.reshape(1, 2048)
        data_f = F.relu(data_f)
        data_f.sub_(data_f.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        data_f.div_(data_f.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        data_f = data_f[:, :2025].reshape(45, 45)

        # saliency map
        cam = (weights.view(*weights.shape, 1, 1) * hook_a.squeeze(0)).sum(0)
        cam = F.relu(cam)
        cam.sub_(cam.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cam.div_(cam.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        # cam[cam < 0.8] = 0
        cam = cam.data.cpu().numpy()
        heatmap = to_pil_image(cam, mode='F')
        overlay = heatmap.resize((45, 45), resample=Image.BICUBIC)
        # plt.figure(figsize=(20, 5))
        y1 += (np.array(overlay) ** 1).reshape(1, 2025).squeeze(0)
    x1 = range(0, 2025)
    y1 = (y1 / 1000)
    print(np.sum(y1 > 0.5) / 2025, np.sum(y1 > 0.6) / 2025, np.sqrt(np.var(y1)))

if __name__ == '__main__':
    model = XceptionTransfer([14], 0.9)
    model.load_state_dict(torch.load('./checkpoints/xceptiontransfer/xception-40-best.pth'))
    model.eval()

    test_loader = get_imagefoler_test_loader('./datasets/celeb-df-v2/test', 32, 4)
    auc_score(model, test_loader, False)
    accuracy(model, test_loader, True)
