import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image

def load_image(img_dir_path):
    img_list = []
    for img_name in tqdm(os.listdir(img_dir_path)):
        if img_name.endswith('_rgb.png'):
            img_path = os.path.join(img_dir_path, img_name)
            img = Image.open(img_path)
            img = img.convert('RGB')
            img_array = np.array(img)
            img_list.append(img_array)
    return img_list

def load_label(label_dir_path):
    label_list = []
    for label_name in tqdm(os.listdir(label_dir_path)):
        if label_name.endswith('_label.png'):
            label_path = os.path.join(label_dir_path, label_name)
            label = Image.open(label_path)
            label = label.convert('L')
            label_array = np.array(label)
            # element > 0 is 1
            label_array[label_array > 0] = 1
            # float32
            label_array = label_array.astype(np.float32)
            label_list.append(label_array)
    return label_list

# load data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.data)
    
def get_data_loader(data, label, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    dataset = MyDataset(data, label, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def get_data_loader_from_path(data_path, label_path, batch_size):
    data = load_image(data_path)
    label = load_label(label_path)
    return get_data_loader(data, label, batch_size)

# model
from self_model import Transformer_VAE

def get_model():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=32, pretrained=False)
    # model = Transformer_VAE()
    return model

# train
from warnings import filterwarnings
from matplotlib import pyplot as plt
filterwarnings('ignore')

## set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

## set hyperparameters
batch_size = 32
lr = 1e-2
epochs = 18

## load data
data_path = '../data/A1/'
label_path = '../data/A1/'
data_loader = get_data_loader_from_path(data_path, label_path, batch_size)

## load criterion
criterion = nn.L1Loss().to(device)

## load model
model = get_model().to(device)

## set optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)

## load scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

## train
model.train()
losses = []
for epoch in range(epochs):
    temp_loss = 0
    for img, label in data_loader:
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        temp_loss += loss.item()
    scheduler.step()
    losses.append(temp_loss)
    print(f'epoch: {epoch+1}, loss: {temp_loss}, lr: {scheduler.get_last_lr()[0]}', end='\r')#

## plot

### loss
plt.plot(losses)
plt.title('loss')
plt.savefig('../plot/loss.png')

# get output
model.eval()
img, label = next(iter(data_loader))
img = img.to(device)
label = label.to(device)
pred = model(img)

# plot pred

pred = pred[0].cpu().detach().numpy()
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
pred = pred[0]
print(pred.shape)
plt.figure(figsize=(10, 10))
plt.imshow(pred,cmap='gray')
plt.axis('off')
plt.savefig('../plot/pred.png')

# plot label

label = label[0].cpu().detach().numpy()
label = label[0]
print(label.shape)
plt.figure(figsize=(10, 10))
plt.imshow(label,cmap='gray')
plt.axis('off')
plt.savefig('../plot/label.png')

## convert to onnx
dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_path = '../model/model.onnx'
torch.onnx.export(model, dummy_input, onnx_path)
print('onnx model saved at', onnx_path)