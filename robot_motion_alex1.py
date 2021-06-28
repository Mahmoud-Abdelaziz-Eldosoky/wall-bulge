
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
#from .utils import load_state_dict_from_url
from typing import Any
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from skimage import io, transform
from PIL import Image

plt.ion()   # interactive mode

######################################################################
# Load Data
# Data augmentation and normalization for training

im_path_train = 'datasets/dataset3_before/train/image_train/'
im_path_test = 'datasets/dataset3_before/test/image_test/'
# ImageFolder assumes that the folder name is the class label
im_path_infer = 'datasets/dataset3_before/test/'

# path for saving and loading the model parametrs
model_param_path = 'scraper_param.pth'

csv_train = 'csv/d3_train.csv'
csv_test = 'csv/d3_test.csv'


__all__ = ['AlexNet', 'alexnet']


model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000, num_classes1: int = 1000, num_classes2: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes1),
        )

        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.features(x)
        feature = self.avgpool(feature)
        feature = torch.flatten(feature, 1)
        out1 = self.classifier(feature)
        out2 = self.classifier1(feature)
        out3 = self.classifier2(feature)
        return out1,out2,out3


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



# ToTensor convert numpy image to pytorch image
# swap color axis because
# numpy image: H x W x C
# torch image: C X H X W

def my_crop(image):
# the below 4 cropping values must be adjusted with each experiment
   return TF.crop(image,150,310,520,720)
        

# default transfrom apply on image only
data_transform = transforms.Compose([transforms.Lambda(my_crop), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# transforms.RandomHorizontalFlip(),

      

# read csv file, note it automatically ignore the first row (header)
angle_data = pd.read_csv(csv_train)

#read first sample data, automatically excludes the first row (Header)
img_name = angle_data.iloc[0, 0]
img_label = angle_data.iloc[0, 1:]
img_label = np.asarray(img_label)
img_label = img_label.astype('float')#.reshape(-1, 2)

print('image name: {}'.format(img_name))
print('label: {}'.format(img_label))
print(img_label.shape)

# a function to display images

def show_image_theta(image_name, image_label):
    """Show image with label"""
    plt.imshow(image_name)
    #plt.text(0,0, image_label)
    plt.title('Parameters = {}'.format(image_label))
    plt.pause(0.001)      # pause a bit so that plots are updated



class AngleRegressDataset(Dataset):
    """Andle Regress dataset."""

    def __init__(self, csv_file, root_dir, transform = None, custom_transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.custom_transform = custom_transform

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images_labels.iloc[idx, 0])
# with using ImageFolder, it automatically converts gray scale image to RGB
# here as we do not use the defalut ImageFolder we need to convert gray scale image to RGB
        image = Image.open(img_name).convert('RGB')     #image = io.imread(img_name)

        # label = [x1, x2, x3]  
        label = self.images_labels.iloc[idx,1:]
        label = np.asarray(label)
        label = label.astype('long')
        #label = np.array([label])
        
        #sample = {'image': image, 'label': label}
        
        #if self.custom_transform:
            #sample = self.custom_transform(sample)

# if you skip the below line, image transformation will not applied to image
# as transformtion updates object sample not variable image   
        #image = sample['image']
        #label = sample['label']

# the below transformation applies only to image varible 
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': label}
        return sample


train_dataset = AngleRegressDataset(csv_file = csv_train, root_dir = im_path_train)
transformed_train_dataset = AngleRegressDataset(csv_file = csv_train, root_dir = im_path_train, transform = data_transform)

test_dataset = AngleRegressDataset(csv_file = csv_test, root_dir = im_path_test)
transformed_test_dataset = AngleRegressDataset(csv_file = csv_test, root_dir = im_path_test, transform = data_transform)

infer_dataset = datasets.ImageFolder(im_path_infer)
transformed_infer_dataset = datasets.ImageFolder(im_path_infer, transform = data_transform)


# Display 1 sample images before image processing with annotations

fig = plt.figure(1)

for i in range(len(train_dataset)):
    sample = train_dataset[i]
  
    print(i, sample['image'].size, sample['label'].shape)

    #ax = plt.subplot(1, 2, i + 1)
    plt.tight_layout()
    #ax.set_title('Sample #{}'.format(i))
    #ax.axis('off')
    show_image_theta(sample['image'], sample['label'])
           
    if i == 0:
        plt.show()
        break


# Display 1 sample images after image processing with annotations

fig = plt.figure(2)

for i in range(len(transformed_train_dataset)):
    sample = transformed_train_dataset[i]
  
    print(i, sample['image'].shape, sample['label'].shape)

    #ax = plt.subplot(1, 2, i + 1)
    plt.tight_layout()
    #ax.set_title('Sample #{}'.format(i))
    #ax.axis('off')
    #In PyTorch, images are represented as [channels, height, width] so need to converted to [height, width, channels]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    sample['image'] = sample['image'].numpy().transpose((1, 2, 0)) 
    sample['image'] = std * sample['image'] + mean
    sample['image'] = np.clip(sample['image'], 0, 1)   
    show_image_theta(sample['image'], sample['label'])

    if i == 0:
        plt.show()
        break

# Turn on shuffluing so test samples are selected
angle_train_dataloader = torch.utils.data.DataLoader(dataset=transformed_train_dataset, batch_size=5, shuffle=True)

# Turn off shuffling so test samples are selected in order
angle_val_dataloader = torch.utils.data.DataLoader(dataset=transformed_test_dataset, batch_size=5, shuffle=False)

scraper_infer_dataloader = torch.utils.data.DataLoader(dataset=transformed_infer_dataset, batch_size=5, shuffle=False)

dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(test_dataset)
dataset_sizes['infer'] = len(infer_dataset)


print("training set size =", dataset_sizes['train'] )
print("testing set size =", dataset_sizes['val'])

dataloaders = {}
dataloaders['train'] = angle_train_dataloader
dataloaders['val'] = angle_val_dataloader
dataloaders['infer'] = scraper_infer_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device",device)

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)  # pause a bit so that plots are updated


# Display 8 training samples after applying transformations

nump_images=4
images_so_far = 0
plt.figure(3)
plt.suptitle('Training Samples')

for i, sample_batched in enumerate(dataloaders['train']):
    if images_so_far == nump_images:
        break
    inputs = sample_batched['image']
    labels = sample_batched['label']   
    labels = labels.long()    # convert labels from float to int
   
    inputs = inputs.to(device)
    labels = labels.to(device)

    for j in range(inputs.size()[0]):
       images_so_far += 1
       ax = plt.subplot(nump_images, 1, images_so_far)
       ax.axis('off')
       ax.set_title('Parameters = {}'.format(labels[j].cpu().numpy()))
       imshow(inputs.cpu().data[j])
       if images_so_far == nump_images:
           break


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion1, criterion2,criterion3, optimizer, pretrained_dict, num_epochs):
    since = time.time()
   
    model_dict = model.state_dict()

# copy similar parts only of pretrained model to current model  
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}



# Load pretrained model parameters to current model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

 
    best_model_wts = copy.deepcopy(model.state_dict())

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_val_acc1 = 0.0
    best_train_acc1 = 0.0
    best_val_acc2 = 0.0
    best_train_acc2 = 0.0
    best_val_acc3 = 0.0
    best_train_acc3 = 0.0

    epoch_acc  = 0.0
    epoch_acc1 = 0.0
    epoch_acc2 = 0.0
    epoch_acc3 = 0.0
    train_loss = []
    train_loss1 = [] 
    train_loss2 = []
    train_loss3 = []
    train_acc  = []
    train_acc1 = []
    train_acc2 = []
    train_acc3 = []
    val_loss = []
    val_loss1 = []
    val_loss2 = []
    val_loss3 = []
    val_acc  = []
    val_acc1 = []
    val_acc2 = []
    val_acc3 = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        ccc = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_corrects = 0.0
            running_corrects1 = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0

            # Iterate over data.

            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                #print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())             
                inputs = sample_batched['image']
                labels = sample_batched['label']
                labels = labels.long()
                labels_1 = labels[:,0]
                labels_2 = labels[:,1]
                labels_3 = labels[:,2]
                #print("label_1 shape ",labels_1.shape)
                inputs = inputs.to(device)
                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)
                labels_3 = labels_3.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1,outputs2,outputs3 = model(inputs)  
                    #print("outputs1 shape ",outputs1.shape)
                    #print("label1",labels_1)                    
                    loss1 = criterion1(outputs1, labels_1)
                    loss2 = criterion2(outputs2, labels_2)
                    loss3 = criterion3(outputs3, labels_3)
                    loss = loss1 + loss2 + loss3

                    _, preds1 = torch.max(outputs1, 1) 
                    _, preds2 = torch.max(outputs2, 1) 
                    _, preds3 = torch.max(outputs3, 1)
                    
                    corr1 = (preds1 == labels_1) # shape = batch size * 1
                    corr2 = (preds2 == labels_2) # shape = batch size * 1
                    corr3 = (preds3 == labels_3) # shape = batch size * 1
                    corr  = torch.stack((corr1, corr2, corr3), dim=1) # shape = batch size * 3
                    corrects = corr.all(axis=1)
                    #print("pred1",preds1)
                    #print("corr1",corr1)
                    #print("corrects",corr)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
# the MSE output is a veraged over the batch size so to compensate multiply by the batch size
# inputs.size(0) = batch size
                running_loss += loss.item() * inputs.size(0)
                running_loss1 += loss1.item() * inputs.size(0)
                running_loss2 += loss2.item() * inputs.size(0)
                running_loss3 += loss3.item() * inputs.size(0)
                
                running_corrects  += torch.sum(corrects)

                running_corrects1 += torch.sum(preds1 == labels_1.data)
                running_corrects2 += torch.sum(preds2 == labels_2.data)
                running_corrects3 += torch.sum(preds3 == labels_3.data)
                

                if phase == 'val':
                    ccc = ccc +1
                    #print("outputs ",outputs)
                    #print("labels ",labels)
                    #print("counter = ",ccc)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_loss2 = running_loss2 / dataset_sizes[phase]
            epoch_loss3 = running_loss3 / dataset_sizes[phase]

            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            epoch_acc1 = running_corrects1.double() / dataset_sizes[phase]
            epoch_acc2 = running_corrects2.double() / dataset_sizes[phase]
            epoch_acc3 = running_corrects3.double() / dataset_sizes[phase]

            print('{} Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f} Acc1: {:.4f} Acc2:{:.4f} Acc3:{:.4f} Acc: {:.4f}'.format(phase, epoch_loss1,epoch_loss2, epoch_loss3, epoch_acc1,epoch_acc2,epoch_acc3,epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                            
            if phase == 'val' and epoch_acc1 > best_val_acc1:
                best_val_acc1 = epoch_acc1

            if phase == 'val' and epoch_acc2 > best_val_acc2:
                best_val_acc2 = epoch_acc2

            if phase == 'val' and epoch_acc3 > best_val_acc3:
                best_val_acc3 = epoch_acc3

            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
                                
            if phase == 'train' and epoch_acc1 > best_train_acc1:
                best_train_acc1 = epoch_acc1

            if phase == 'train' and epoch_acc2 > best_train_acc2:
                best_train_acc2 = epoch_acc2

            if phase == 'train' and epoch_acc3 > best_train_acc3:
                best_train_acc3 = epoch_acc3
           
            if phase == 'train':
                train_loss.append(epoch_loss)
                #train_loss1.append(epoch_loss1)
                #train_loss2.append(epoch_loss2)
                #train_loss3.append(epoch_loss3)
                train_acc.append(epoch_acc.cpu().detach().numpy())
                train_acc1.append(epoch_acc1.cpu().detach().numpy())
                train_acc2.append(epoch_acc2.cpu().detach().numpy())
                train_acc3.append(epoch_acc3.cpu().detach().numpy())

            if phase == 'val':
                val_loss.append(epoch_loss)
                #val_loss1.append(epoch_loss1)
                #val_loss2.append(epoch_loss2)
                #val_loss3.append(epoch_loss3)
                val_acc.append(epoch_acc.cpu().detach().numpy())
                val_acc1.append(epoch_acc1.cpu().detach().numpy())
                val_acc2.append(epoch_acc2.cpu().detach().numpy())
                val_acc3.append(epoch_acc3.cpu().detach().numpy())
    ''' 
    plt.figure(4)
    #plt.suptitle('Vgg16')
    plt.plot(range(1, len(train_loss)+1), train_loss, 'b', label = "Training loss")
    plt.plot(range(1, len(val_loss)+1), val_loss,'r', label = "Testing loss")
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    plt.legend()
    '''

    plt.figure(5)
    plt.suptitle('Vertical Distance')
    plt.plot(range(1, len(train_acc1)+1), train_acc1, 'b', label = "Training accuracy")
    plt.plot(range(1, len(val_acc1)+1), val_acc1,'r', label = "Testing accuracy")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,1.1)
    plt.legend()
    
    plt.figure(6)
    plt.suptitle('Tilt Angle')
    plt.plot(range(1, len(train_acc2)+1), train_acc2, 'b', label = "Training accuracy2")
    plt.plot(range(1, len(val_acc2)+1), val_acc2,'r', label = "Testing accuracy2")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,1.1)
    plt.legend()
    
    plt.figure(7)
    plt.suptitle('Scraper Stiffness')
    plt.plot(range(1, len(train_acc3)+1), train_acc3, 'b', label = "Training accuracy3")
    plt.plot(range(1, len(val_acc3)+1), val_acc3,'r', label = "Testing accuracy3")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,1.1)
    plt.legend()
    
    plt.figure(8)
    plt.suptitle('Net Accuracy')
    plt.plot(range(1, len(train_acc)+1), train_acc, 'b', label = "Training accuracy")
    plt.plot(range(1, len(val_acc)+1), val_acc,'r', label = "Testing accuracy")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,1.1)
    plt.legend()
   
 
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Test Acc: {:.4f}'.format(best_val_acc))    
    print('Best Test Acc1: {:.4f}'.format(best_val_acc1))
    print('Best Test Acc2: {:.4f}'.format(best_val_acc2))
    print('Best Test Acc3: {:.4f}'.format(best_val_acc3))
    print('Best Train Acc: {:.4f}'.format(best_train_acc))    
    print('Best Train Acc1: {:.4f}'.format(best_train_acc1))
    print('Best Train Acc2: {:.4f}'.format(best_train_acc2))
    print('Best Train Acc3: {:.4f}'.format(best_train_acc3))
    print()    

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(best_model_wts), model_param_path)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def visualize_model(model, num_images=6):

    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure(9)
    #plt.suptitle('Vgg16')

    with torch.no_grad():
        for i, sample_batched in enumerate(dataloaders['val']):
            inputs = sample_batched['image']
            labels = sample_batched['label']
            labels = labels.long() # convert labels from float to int
            labels_1 = labels[:,0]
            labels_2 = labels[:,1]
            labels_3 = labels[:,2]
            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_3 = labels_3.to(device)

            #print("label shape ",labels[:,4].shape)
            inputs = inputs.to(device)
            output1,output2,output3 = model(inputs)

            _, pred1 = torch.max(output1, 1) 
            _, pred2 = torch.max(output2, 1) 
            _, pred3 = torch.max(output3, 1)           

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('true: {} {} {}\n pred: {}{} {}'.format(labels_1[j].cpu().numpy(),labels_2[j].cpu().numpy(), labels_3[j].cpu().numpy(),pred1[j].cpu().numpy(),pred2[j].cpu().numpy(),pred3[j].cpu().numpy()))

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Evaluating the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def evaluate_model(model):
    was_training = model.training
    model.eval()
    images_so_far = 0
    dataFile = open('results/results_endpoints.txt', 'w')

    with torch.no_grad():
        for i, sample_batched in enumerate(dataloaders['val']):
            inputs = sample_batched['image']
            labels = sample_batched['label']

            labels_end = labels[:,:4].float()
            labels_class = labels[:,4].long()
            #print("label shape ",labels[:,4].shape)
            inputs = inputs.to(device)
            labels_end = labels_end.to(device)
            labels_class = labels_class.to(device)

            outputs1,outputs2 = model(inputs)
            _, preds = torch.max(outputs2, 1)

            correct1 = abs(outputs1 - labels_end.data)
# AND logic the result for all of x1,y1,x2,y2, a sample is assumed to be correctly predicted
# if all of the 4 indices are correctly predicted 
            correct1 = correct1.all(axis=1)

            correct2 = (preds == labels_class.data)
 
            for j in range(inputs.size()[0]):
                images_so_far += 1

                dataFile.write('Batch: {}  sample: {} End_Pred: {} Label_Pred: {}\n'.format(i+1, images_so_far, correct1[j], correct2[j]))
                #dataFile.write('true: {:.2f}'.format(l) for l in labels[j].cpu().numpy())
                dataFile.write('Rs = {:.2f}  Re = {:.2f} Ravg = {:.2f}\n'.format(distance_start_point_eva[j], distance_end_point_eva[j], average_distance_eva[j]))
                dataFile.write('end_true: {} class_true: {} \nend_pred: {} class_pred: {}\n'.format(labels_end[j].cpu().numpy(),labels_class[j].cpu().numpy(), outputs1[j].cpu().numpy(),preds[j].cpu().numpy()))
        dataFile.close()
        model.train(mode=was_training)

######################################################################

def infer_model(model):

    sample_counter = 0

    running_corrects = 0.0
    running_corrects1 = 0.0
    running_corrects2 = 0.0
    running_corrects3 = 0.0

    acc  = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    
    prediction = torch.LongTensor([]) # initialize int tensor 
    prediction = prediction.to(device)
    
    was_training = model.training
    model.load_state_dict(torch.load(model_param_path))
    model.eval()
    i = 0
    with torch.no_grad():
        for inputs, _ in (dataloaders['infer']):
            #inputs = sample_batched['image']
            '''
            labels = sample_batched['label']
            labels = labels.long() # convert labels from float to int
            labels_1 = labels[:,0]
            labels_2 = labels[:,1]
            labels_3 = labels[:,2]
            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_3 = labels_3.to(device)
            '''

            inputs = inputs.to(device)

            output1,output2,output3 = model(inputs)

            _, pred1 = torch.max(output1, 1) 
            _, pred2 = torch.max(output2, 1) 
            _, pred3 = torch.max(output3, 1)
            
            pred = torch.stack((pred1, pred2, pred3), dim=1) # shape = batch size * 3
            prediction = torch.cat((prediction,pred),dim=0) # shape = test dataset size * 3
            
            '''
            corr1 = (pred1 == labels_1) # shape = batch size * 1
            corr2 = (pred2 == labels_2) # shape = batch size * 1
            corr3 = (pred3 == labels_3) # shape = batch size * 1
            corr  = torch.stack((corr1, corr2, corr3), dim=1) # shape = batch size * 3
            corrects = corr.all(axis=1)         

            running_corrects  += torch.sum(corrects)
            running_corrects1 += torch.sum(pred1 == labels_1.data)
            running_corrects2 += torch.sum(pred2 == labels_2.data)
            running_corrects3 += torch.sum(pred3 == labels_3.data)
            '''
               
            for j in range(inputs.size()[0]):
                sample_counter += 1

                print('Batch: {} sample: {}'.format(i+1, sample_counter))
                #print('Label1: {} Label2: {} Label3: {}'.format(labels_1[j],labels_2[j],labels_3[j]))
                print('Pred1:  {} Pred2:  {} Pred3:  {}'.format(pred1[j],pred2[j],pred3[j]))
                
            i = i + 1

        #acc  = running_corrects.double() / dataset_sizes['val']
        #acc1 = running_corrects1.double() / dataset_sizes['val']
        #acc2 = running_corrects2.double() / dataset_sizes['val']
        #acc3 = running_corrects3.double() / dataset_sizes['val']

        print() 
        prediction = prediction.cpu().detach().numpy()      
        print('Model Predictions', prediction)              
        print()
        #print('Acc1: {:.4f} Acc2:{:.4f} Acc3:{:.4f} Acc: {:.4f}'.format(acc1,acc2,acc3,acc))
        
        model.train(mode=was_training)
######################################################################


# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

#model_ft = models.vgg16(pretrained=True)
pretrained_model = models.alexnet(pretrained=True)
model_ft = alexnet(pretrained=False)
#model_ft = models.resnet50(pretrained=True)


#copy pretrained model parameters
pretrained_dict = pretrained_model.state_dict()

print(model_ft)


'''
#freeze some layers
for i in range (13):
    for param in model_ft.features[i].parameters():
        param.requires_grad = False
'''

'''
#freeze all convolutional layers parameters
for param in model_ft.features.parameters():
      param.requires_grad = False        
'''     

'''
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft.fc = nn.Sequential(
      nn.Linear(num_ftrs, 4),
      nn.Sigmoid()
)
'''


# Both AlexNet and Vgg16 are modified by the same code as below
#model_ft.features = nn.Sequential(*[model_ft.features[i] for i in range (24)])


model_ft.classifier[6] = nn.Linear(4096, 3)
model_ft.classifier1[6] = nn.Linear(4096, 3)
model_ft.classifier2[6] = nn.Linear(4096, 2)

#model_ft.classifier = nn.Sequential(*list(model_ft.classifier) + [nn.Sigmoid()])
#model_ft.classifier1 = nn.Sequential(*list(model_ft.classifier1) + [nn.Sigmoid()])
#model_ft.classifier2 = nn.Sequential(*list(model_ft.classifier2) + [nn.Sigmoid()])



'''
# add ReLU + linear + Sigmoid to original Alexnet
model_ft.classifier[6] = nn.Linear(4096, 2048)
model_ft.classifier = nn.Sequential(*list(model_ft.classifier) + [nn.ReLU(True)])
model_ft.classifier = nn.Sequential(*list(model_ft.classifier) + [nn.Linear(2048, 1)])
model_ft.classifier = nn.Sequential(*list(model_ft.classifier) + [nn.Sigmoid()])
'''

'''
# instead we can override the original pytorch AlexNet classifier

model_ft.classifier = nn.Sequential(
    nn.Dropout(0.5, False),
    nn.Linear(9216, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5, False),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Linear(4096, 2048),
    nn.ReLU(True),
    nn.Linear(2048, 1),
    nn.Sigmoid()
)
'''

print(model_ft)

model_ft = model_ft.to(device)

# loss function
# The target values are out of bounds. the loss "nn.CrossEntropyLoss" expects a torch.LongTensor with values in the range [0, nb_classes-1].

#criterion1 = nn.MSELoss()  # lr = 0.01
criterion1 = nn.CrossEntropyLoss()  
criterion2 = nn.CrossEntropyLoss()
criterion3 = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.
#

model_trained = train_model(model_ft, criterion1, criterion2, criterion3, optimizer_ft, pretrained_dict, num_epochs= 30)

######################################################################

#evaluate_model(model_trained)	# this function takes the trained model
visualize_model(model_trained)	# this function takes the trained model
infer_model(model_ft)		# this function takes the untrained model

plt.ioff()
plt.show()

