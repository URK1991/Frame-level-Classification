import torchvision.transforms as transforms

def get_transforms(split):
    print('Getting data transforms')
    data_transforms = {
        'Train': transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(9 / 10, 10 / 9)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(23)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
            transforms.ToTensor()
        ]),
        'Test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }
    return data_transforms[split]
