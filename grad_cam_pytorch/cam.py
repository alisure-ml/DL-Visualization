import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torchvision import datasets, transforms, models


def get_cam(feature_conv, weight_softmax, class_id_list):
    size_up_sample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for class_id in class_id_list:
        cam = weight_softmax[class_id].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_up_sample))
    return output_cam


model = models.resnet18(pretrained=True)
model.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
    pass

model._modules.get("layer4").register_forward_hook(hook_feature)

params = list(model.parameters())
_weight_softmax = np.squeeze(params[-2].data.numpy())

img_pil = Image.open('./input_images/cat_dog.png')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
img_tensor = preprocess(img_pil)

img_variable = torch.Tensor(img_tensor.unsqueeze(0))
logit = model(img_variable)
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)

cam_list = get_cam(features_blobs[0], _weight_softmax, idx[probs > 0.1].tolist())

print()
