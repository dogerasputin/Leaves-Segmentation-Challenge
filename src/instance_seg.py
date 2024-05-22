'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-22 23:44:45
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-23 00:23:12
FilePath: \Leaves-Segmentation-Challenge\src\instance_seg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import utils
import torchvision
import matplotlib.pyplot as plt
import argparse

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models import ResNet101_Weights, ResNet152_Weights, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN

from engine import train_one_epoch, evaluate

# set up argparse

parser = argparse.ArgumentParser(description='Instance Segmentation')
parser.add_argument('--backbone', default='resnet18', type=str, help='backbone model: resnet18, resnet34, resnet50, resnet101, resnet152')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')

args = parser.parse_args()

class LeavesDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, path="../data/A1"):
        self.path = path
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = [x for x in os.listdir(path) if x.endswith("_rgb.png")]
        self.masks = [x for x in os.listdir(path) if x.endswith("_label.png")]
        self.cnt = 1

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.path + "/" + self.imgs[idx]
        mask_path = self.path + "/" + self.masks[idx]
        img = read_image(img_path,ImageReadMode.RGB)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        # check boxes area is not zero
        boxes = boxes[area > 0]
        labels = labels[area > 0]
        masks = masks[area > 0]
        area = area[area > 0]
        iscrowd = np.zeros((len(boxes),), dtype=np.int64)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.cnt == 0:
            print(img.shape)
            self.cnt += 1
            
        return img, target

    def __len__(self):
        return len(self.imgs)
    
def get_model_instance_segmentation(num_classes, backbone_name='resnet18'):
    # load an instance segmentation model pre-trained on COCO
    # sample 1
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn() 
    # sample 2
    backbone_weights = {
        'resnet18': ResNet18_Weights.DEFAULT,
        'resnet34': ResNet34_Weights.DEFAULT,
        'resnet50': ResNet50_Weights.DEFAULT,
        'resnet101': ResNet101_Weights.DEFAULT,
        'resnet152': ResNet152_Weights.DEFAULT
    }
    backbone = resnet_fpn_backbone(
        backbone_name,
        weights=backbone_weights[backbone_name],
        trainable_layers=4
    )
    # sample 3 (not working)
    # backbone = BackboneWithFPN(
    #     backbone=MyBackbone(),
    #     out_channels=MyBackbone().out_channels,
    # )
    model = MaskRCNN(backbone, num_classes=num_classes)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and Leaf
num_classes = 2
# use our dataset and defined transformations
dataset = LeavesDataset(get_transform(train=True))
dataset_test = LeavesDataset(get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-10])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

# define training and validation data loaders
data_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes, args.backbone)

# move model to the right device
model.to(device)

print(model)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
print(f"Parameters size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024} MB")
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = args.epochs

loss_hist = []

for epoch in range(num_epochs):
    print(f"===== Epoch {epoch} =====")
    # train for one epoch, printing every 10 iterations
    _,loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    loss_hist.append(loss)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("Training Done!")

# plot loss history
plt.plot(loss_hist)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.savefig("../plot/loss.png")

image = read_image("../data/A1/plant002_rgb.png", ImageReadMode.RGB)
labels = read_image("../data/A1/plant002_label.png")
labels_unique = torch.unique(labels)
eval_transform = get_transform(train=False)

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]


image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

masks = (pred["masks"] > 0.7).squeeze(1)
image = draw_segmentation_masks(image, masks[:labels_unique.shape[0]-3])


plt.figure(figsize=(10, 10))
plt.imshow(image.permute(1, 2, 0))
plt.savefig("../plot/instance_seg.png")