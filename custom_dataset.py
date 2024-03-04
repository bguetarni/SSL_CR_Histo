import os, glob, random
import numpy as np
import pandas
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from albumentations import Compose, Rotate, CenterCrop, RandomScale, Resize, RandomCrop
from models.randaugment import RandAugment

class DatasetBreastPathQ_Supervised_train:

    def __init__(self, datalist, image_size, transform=None, maxsize=100000):

        """
        BreastPathQ dataset: supervised fine-tuning on downstream task
        """

        self.datalist = datalist
        self.image_size = image_size
        self.transform = transform

        if len(self.datalist) > maxsize:
            self.datalist = random.sample(self.datalist, maxsize)

        # Resize images
        self.transform1 = Compose([Resize(image_size, image_size, interpolation=2)])  # 256

        # Data augmentations
        self.transform4 = Compose([Rotate(limit=(-90, 90), interpolation=2), CenterCrop(image_size, image_size)])
        self.transform5 = Compose(
            [Rotate(limit=(-90, 90), interpolation=2), RandomScale(scale_limit=(0.8, 1.2), interpolation=2),
             Resize(image_size + 20, image_size + 20, interpolation=2),
             RandomCrop(image_size, image_size)])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img_path, label = self.datalist[index]
        
        img = Image.open(img_path).convert('RGB').reduce(2)

        if self.transform:

            # Convert PIL image to numpy array
            img = np.array(img)

            # First image
            img1 = self.transform1(image=img)
            img1 = Image.fromarray(img1['image'])
            img1 = np.array(img1)

            Aug1_img1 = self.transform4(image=img1)
            Aug2_img1 = self.transform5(image=img1)

            # Convert numpy array to PIL Image
            img1 = Image.fromarray(img1)
            Aug1_img1 = Image.fromarray(Aug1_img1['image'])
            Aug2_img1 = Image.fromarray(Aug2_img1['image'])

            # Convert to numpy array
            img1 = np.array(img1)
            Aug1_img1 = np.array(Aug1_img1)
            Aug2_img1 = np.array(Aug2_img1)

            # Stack along specified dimension
            img = np.stack((img1, Aug1_img1, Aug2_img1), axis=0)

            # Numpy to torch
            img = torch.from_numpy(img)

            # Randomize the augmentations
            shuffle_idx = torch.randperm(len(img))
            img = img[shuffle_idx, :, :, :]

            label = np.array(label)
            label = torch.from_numpy(label)
            label = label.repeat(img.shape[0])

            # Change Tensor Dimension to N x C x H x W
            img = img.permute(0, 3, 1, 2)
        else:
            # Numpy to torch
            img = np.array(img)
            img = torch.from_numpy(img)

            target = np.array(label)
            target = torch.from_numpy(target)

            # Change Tensor Dimension to N x C x H x W
            img = img.permute(2, 0, 1)

        return img, label


class DatasetBreastPathQ_SSLtrain(Dataset):

    """ BreastPathQ consistency training / validation  """

    def __init__(self, datalist, image_size, transform=None, maxsize=10000):

        self.image_size = image_size
        self.transform = transform
        self.datalist = datalist
        
        if len(self.datalist) > maxsize:
            self.datalist = random.sample(self.datalist, maxsize)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img_path, label = self.datalist[index]
        
        img = Image.open(img_path).convert('RGB').reduce(2)

        if self.transform:
            image = self.transform(img)

            if isinstance(image, tuple):
                img = image[0]
                target = image[1]

                # Numpy to torch
                img = np.array(img)
                img = torch.from_numpy(img)
                target = np.array(target)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

            else:
                # Numpy to torch
                img = np.array(image)
                img = torch.from_numpy(img)

                target = np.array(label)
                target = torch.from_numpy(target)

                # Change Tensor Dimension to N x C x H x W
                img = img.permute(2, 0, 1)

        return img, target
    

class DatasetBreastPathQ_eval:

    def __init__(self, dataset_path, image_size, transform=None):

        """
        BreastPathQ dataset: test
        """

        self.image_size = image_size
        self.transform = transform

        dataset_pathA = os.path.join(dataset_path, "TestSetSherine/")
        dataset_pathB = os.path.join(dataset_path, "TestSetSharon/")

        self.datalist = []
        data_pathsA = glob.glob(dataset_pathA + "*.h5")
        data_pathsB = glob.glob(dataset_pathB + "*.h5")

        with tqdm(enumerate(sorted(zip(data_pathsA, data_pathsB))), disable=True) as t:
            for wj, (data_pathA, data_pathB) in t:

                dataA = h5py.File(data_pathA)
                dataB = h5py.File(data_pathB)

                data_patches = dataA['x'][:]

                cls_idA = dataA['y'][:]
                cls_idB = dataB['y'][:]

                for idx in range(len(data_patches)):
                    self.datalist.append((data_patches[idx], cls_idA[idx], cls_idB[idx]))

    def __len__(self):

        return len(self.datalist)

    def __getitem__(self, index):

        np_data = self.datalist[index][0]
        np_data = np.transpose(np_data, (1, 2, 0))
        img = Image.fromarray((np_data * 255).astype(np.uint8))

        # label assignment
        labelA = self.datalist[index][1]
        labelB = self.datalist[index][2]

        if self.transform:
            img = self.transform(img)
            img = np.array(img)
            img = torch.from_numpy(img)

            labelA = np.array(labelA)
            labelA = torch.from_numpy(labelA)

            labelB = np.array(labelB)
            labelB = torch.from_numpy(labelB)

            # Change Tensor Dimension to N x C x H x W
            img = img.permute(2, 0, 1)

        return img, labelA, labelB


class DatasetBreastPathQ_eval(Dataset):

    """ BreastPathQ consistency training / validation  """

    def __init__(self, dataset_path, transform=None):
        pass

    def __len__(self):
        return None

    def __getitem__(self, index):
        return None
    

class TransformFix(object):

    """" Weak and strong augmentation for consistency training """

    def __init__(self, image_size, N):
        self.weak = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=image_size)])
        self.strong = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=image_size),
                                          RandAugment(n=N, m=10)])  # (7) - standard value for BreastPathQ/Camelyon16/Kather

    def __call__(self, x):
        weak_img = self.weak(x)
        # weak_img.show()
        strong_img = self.strong(x)
        # strong_img.show()
        return weak_img, strong_img


def load_dataset(args, TRAIN_PARAMS):

    if args.dataset in ["chulille", "dlbclmorph"]:
        labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

    if args.dataset == "chulille":
        assert not args.val_image_pth is None, "For dataset chulille, argument `val_image_pth` must be provided"
        
        train_data = []
        val_data = []
        for data_list, data_path in ((train_data, args.train_image_pth), 
                                     (val_data ,args.val_image_pth)):
            for p in os.listdir(data_path):
                y = labels[int(p)]
                label = TRAIN_PARAMS['class_to_label'][args.dataset][y]
                associate_label = lambda i : (i, label)
                list_of_patches = glob.glob(os.path.join(data_path, p, '*', '*.png'))
                data_list.extend(list(map(associate_label, list_of_patches)))
    elif  args.dataset == "dlbclmorph":
        data = []
        for fold in os.listdir(args.train_image_pth):
            if fold != args.fold:
                for p in os.listdir(os.path.join(args.train_image_pth, fold)):
                    y = labels[int(p)]
                    label = TRAIN_PARAMS['class_to_label'][args.dataset][y]
                    associate_label = lambda i : (i, label)
                    list_of_patches = glob.glob(os.path.join(args.train_image_pth, fold, p, '*.png'))
                    data.extend(list(map(associate_label, list_of_patches)))
            
        n = len(data) - int(args.validation_split * len(data))
        random.shuffle(data)
        train_data, val_data = data[:n], data[n:]
    elif  args.dataset == "bci":
        data = []
        for img in glob.glob(os.path.join(args.train_image_pth, "train", "*.png")):
            y = os.path.splitext(os.path.split(img)[1])[0].split('_')[-1]
            data.append((img, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
        
        n = len(data) - int(args.validation_split * len(data))
        random.shuffle(data)
        train_data, val_data = data[:n], data[n:]
    
    return train_data, val_data