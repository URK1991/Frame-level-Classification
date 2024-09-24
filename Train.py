from __future__ import print_function, division
import torch.nn.functional as nnf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

def get_transforms(split):
    print('getting data transforms')
    data_transforms = {
    'Train': [
        transforms.RandomResizedCrop(size=(224, 224),scale=(0.9, 1.0),ratio=(9 / 10, 10 / 9)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(23)], p=0.8),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
        transforms.ToTensor()
        ],
    'Test': [
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ]
    }
    return transforms.Compose(data_transforms[split])

def device_avail():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    return device

def train_model():
    
    torch.manual_seed(0)
    data_dir = '/projectnb/ivcgroup/ukhan3/GAN_Data'
    device = device_avail()
    dataloaders = {};
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),get_transforms(x)) for x in ['Train', 'Test']}
    dataloaders['Train'] = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=4,
                                             shuffle=True, num_workers=3)

    dataloaders['Test'] = torch.utils.data.DataLoader(image_datasets['Test'], batch_size=4,
                                             shuffle=False, num_workers=3)
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}
    model = torchvision.models.resnet18(pretrained = False)
    num_ftrs = model.fc.in_features
    
  #  new_lin = nn.Sequential(
  #      nn.Dropout(0.25),
  #      nn.Linear(num_ftrs, 64),
  #      nn.Dropout(0.2),
  #      nn.Linear(64, 4)
  #  ) 
    new_lin = nn.Linear(num_ftrs, 4)
    
    model.fc = new_lin 
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    train_dir = '/projectnb/ivcgroup/ukhan3/GAN_Data/Train'

    class_weights = []
    
    for root, subdir, files in os.walk(train_dir):
        if len(files)>0:
            class_weights.append(1/len(files))

    criterion = nn.CrossEntropyLoss().to(device)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.000001)

    best_acc = 0.0
    # Early stopping
    the_last_loss = 100
    low_loss = the_last_loss
    patience = 5
    trigger_times = 0
    flag = 0
    num_epochs = 70
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)
                   
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                    

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'Train':
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)  
            # deep copy the model
            
            if phase == 'Test' and epoch_loss <= the_last_loss:
                trigger_times = 0
                if epoch_loss < low_loss:
                    low_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
            if phase == 'Test' and epoch_loss > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!')
                    flag = 1
                     
            if phase == 'Test':
                the_last_loss = epoch_loss
   
        if flag == 1:
            break
        

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_losses, train_losses
    
if __name__ == "__main__":
    print('VAE')
    tr_model, tstloss, trloss = train_model()
    torch.save(tr_model, '/projectnb/ivcgroup/ukhan3/output/' + 'InterGAN_Res18.pth')
