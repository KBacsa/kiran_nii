import os
import torch
import torch.utils.data as data
from torchvision.transforms import functional as TF
import numpy as np
from math import pi
import random

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# class
def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(root, dataframe):
    
    images = []
    
    for index, row in dataframe.iterrows():
        image_path = os.path.join(root, row['subDirectory_filePath'])
        label = row['expression']
        x = int(row['face_x'])
        y = (row['face_y'])
        h = row['face_height']
        w = row['face_width']
        images.append((image_path, x, y, h, w, label))

    return images
            
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class AffectFolder(data.Dataset):

    def __init__(self, root, dataframe, loader=default_loader, transform=None, target_transform=None):

        imgs = make_dataset(root, dataframe)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.dataframe = dataframe
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = imgs
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (video, target) where target is class_index of the target class.
        """
        
        image_path, x, y, h, w, label = self.imgs[index]

        image = self.loader(image_path)
        
        # crop 
        image.crop((y, y+h, x, x+w))
        
        if self.transform is not None:     
            image = self.transform(image)
            
        return image, label

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

class FolderFER(data.Dataset):

    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (video, target) where target is class_index of the target class.
        """

        dtpoints = self.data.iloc[index]
        pixels = np.array([int(s) for s in dtpoints['pixels'].split(' ')]).reshape([48,48])
        image = Image.fromarray(np.uint8(pixels) , 'L')
        rgbimg = Image.new("RGB", image.size)
        rgbimg.paste(image)
        
        if self.transform is not None:      
            #image = self.transform(image)
            rgbimg = self.transform(rgbimg)
        
        label = dtpoints['emotion']
        
        #return image, label 
        return rgbimg, label 

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str  