import cv2
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_imagefoler_test_loader
from cw_layer import XceptionBN, XceptionTransfer
import torch.nn
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image

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
    preprocess = transforms.Compose([transforms.Resize((400, 400)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    preprocessed_image = preprocess(Image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def show_all_channel(model, image, k):
    model_vis = nn.Sequential(*list(model.model.children())[0:k])
    feature = model_vis(image).squeeze(0)
    print(feature.shape)
    for i in range(60):
        x = feature[i, :, :]
        x = x.squeeze(0)
        x = x.data.cpu().numpy()
        x = 1.0 / (1 + np.exp(-1 * x))
        plt.subplot(10, 6, i+1)
        plt.imshow(x)
        plt.axis("off")
    plt.show()

def show_one_channel(model, image, layers):
    for k in range(1, layers):
        model_vis = nn.Sequential(*list(model.model.children())[0:k])
        feature = model_vis(image).squeeze(0)
        x = feature[0, :, :]
        x = x.squeeze(0)
        x = x.data.cpu().numpy()
        #x = 1.0 / (1 + np.exp(-1 * x))
        plt.subplot(4, 6, k)
        plt.imshow(x)
        plt.axis("off")
    plt.show()

def get_logits(model, image):
    Classification = nn.Sequential(*list(model.model.children())[21:22])
    func = nn.Softmax(dim=1)
    for i in range(21):
        model_vis = nn.Sequential(*list(model.model.children())[0:i])
        feature = model_vis(image)
        model_additional = nn.Sequential(*list(model.model.children())[i:21])
        relu = nn.ReLU(inplace=True)
        noise = torch.randn(size=feature.shape)
        x = relu(model_additional(feature+noise))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size()[0], -1)
        print(i, func(Classification(x)))

def get_test_acc(test_dir, model):
    correct = 0
    test_loader = get_imagefoler_test_loader(test_dir, 128, 4)
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            model = model.cuda()

            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
        print(float(correct) / len(test_loader.dataset))

def show_saliency_map(model, img_path, k):
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    img = preprocess_image(image, cuda=False)
    feature = nn.Sequential(*list(model.model.children())[0:k])(img).squeeze(0)
    x1 = feature[1, :, :]
    x1 = x1.data.cpu().numpy()
    x1 = 1.0 / (1 + np.exp(-1 * x1))
    saliency_img = to_pil_image(x1, mode='F')
    heatmap = normalization_data(np.asarray(saliency_img))
    heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255 * heatmap), (width, height)), cv2.COLORMAP_HOT)
    return heatmap


model = XceptionBN()
#model.load_state_dict(torch.load('/home/huayingying/Documents/Data/trained-model/DF/xception/xception-1-best.pth'))
model.eval()
img_path = '/home/huayingying/Documents/tip_img/234_2.png'

heatmap = show_saliency_map(model, img_path, 7)
cv2.imshow('', heatmap)
cv2.waitKey(0)







