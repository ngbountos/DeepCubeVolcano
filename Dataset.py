import cv2 as cv
import numpy as np
import torch
import os
import random
import albumentations as A
import torchvision
# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset='../Data/Dataset', setting='train', sim=True, original=False):
        self.path = dataset
        self.classes = os.listdir(self.path)
        self.interferograms = []
        self.interferograms_normal = []
        self.interferograms_deformation = []
        self.sim = sim
        self.original = original
        for data_class in self.classes:
            images = os.listdir(self.path + '/' + data_class)
            for image in images:
                image_dict = {'path': self.path + '/' + data_class + '/' + image, 'label': data_class}
                self.interferograms.append(image_dict)
                if int(data_class) == 0:
                    self.interferograms_normal.append(image_dict)
                else:
                    self.interferograms_deformation.append(image_dict)

        self.num_examples = len(self.interferograms)
        self.set = setting

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        if self.set == 'train' and self.sim == False:
            choice = random.randint(0, 10)

            if choice % 2 != 0:
                choice_normal = random.randint(0, len(self.interferograms_normal) - 1)
                image_data = self.interferograms_normal[choice_normal]
            else:
                choice_deform = random.randint(0, len(self.interferograms_deformation) - 1)
                image_data = self.interferograms_deformation[choice_deform]
        else:
            image_data = self.interferograms[index]
        image_file = image_data['path']
        image_label = image_data['label']
        image = cv.imread(image_file)
        zero = np.zeros_like(image)
        if image is None:
            print(image_file)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        original = image
        original = original[:224, :224, :]
        zero[:, :, 0] = gray
        zero[:, :, 1] = gray
        zero[:, :, 2] = gray
        image = zero
        image = image[:224, :224, :]

        if self.set == 'train':
            transform = A.Compose([
                A.augmentations.transforms.HorizontalFlip(p=0.8),
                A.augmentations.transforms.VerticalFlip(p=0.8),
                A.augmentations.transforms.ElasticTransform(p=0.8),
                A.augmentations.transforms.Cutout(p=0.8),
                A.augmentations.transforms.MultiplicativeNoise(p=0.8),
                A.augmentations.transforms.GaussianBlur(p=0.8),
                A.augmentations.transforms.GaussNoise(p=0.8)
            ])
            transformed = transform(image=image)
            augmented = transformed['image']

            sim = self.sim
            if sim == True:
                transform2 = A.Compose([
                    A.augmentations.transforms.HorizontalFlip(p=0.8),
                    A.augmentations.transforms.VerticalFlip(p=0.8),
                    A.augmentations.transforms.ElasticTransform(p=0.8),
                    A.augmentations.transforms.Cutout(p=0.8),
                    A.augmentations.transforms.MultiplicativeNoise(p=0.8),
                    A.augmentations.transforms.GaussianBlur(p=0.8),
                    A.augmentations.transforms.GaussNoise(p=0.8)
                ])
                transformed2 = transform2(image=image)
                image = transformed2['image']
                flag = True

        else:

            augmented = None

            flag = False

        image = torch.from_numpy(image).float().permute(2, 0, 1)
        original = torch.from_numpy(original).float().permute(2, 0, 1)

        image = torchvision.transforms.Normalize((127.0710, 127.0710, 127.0710), (71.4902, 71.4902, 71.4902))(image)
        if augmented is None:
            augmented = torch.tensor(int(image_label))
        else:
            augmented = torch.from_numpy(augmented).float().permute(2, 0, 1)
            # merged
            augmented = torchvision.transforms.Normalize((127.0710, 127.0710, 127.0710), (71.4902, 71.4902, 71.4902))(
                augmented)

        if image.shape[1] < 224 or image.shape[2] < 224:
            print(image_file)
        if self.original:
            return (image, augmented, original), int(image_label), image_file
        return (image, augmented), int(image_label)

