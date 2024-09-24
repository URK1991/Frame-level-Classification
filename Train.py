import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from models import get_model
from data_transforms import get_transforms

def device_avail():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    return device

def train_model():
    torch.manual_seed(0)
    data_dir = ''
    device = device_avail()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), get_transforms(x)) for x in ['Train', 'Test']}
    dataloaders = {
        'Train': DataLoader(image_datasets['Train'], batch_size=4, shuffle=True, num_workers=3),
        'Test': DataLoader(image_datasets['Test'], batch_size=4, shuffle=False, num_workers=3)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}
    
    model = get_model(num_classes=4).to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.000001)

    best_acc = 0.0
    the_last_loss = 100
    low_loss = the_last_loss
    patience = 5 #For early stopping : if the loss is not reduced from the lowest loss encountered for 5 consecutive epochs, training will stop 
    trigger_times = 0
    flag = 0
    num_epochs = 70
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['Train', 'Test']:
            model.train() if phase == 'Train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)

            if phase == 'Test' and epoch_loss < low_loss:
                trigger_times = 0
                if epoch_loss < low_loss:
                    low_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
            if phase == 'Test' and epoch_loss > low_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)
                if trigger_times = patience:
                    print('Early stopping!')
                    flag = 1
                     
            if phase == 'Test':
                the_last_loss = epoch_loss
   
        if flag == 1:
            break
        
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, test_losses, train_losses
