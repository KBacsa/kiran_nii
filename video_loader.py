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


def make_dataset(dir, image_folder, label_folder, fold, indexing, are_subjects):
    
    images = []
    
    # get all emotion directories
    dirs = []
    data_dir = os.path.join(dir, image_folder, fold)
    
    for root, directories, filenames in os.walk(data_dir):
        dirs.append(directories)

    dirs = [x for x in dirs if x]
    subjects = dirs[0]
    sequences = dirs[1:]

    if are_subjects:
        subjects = dirs[0]
        sequences = dirs[1:]
        for subject in range(len(subjects)):
            for sequence in sequences[subject]:

                # check if emotion has indeed been detected on subject
                label_path = os.path.join(dir, label_folder, subjects[subject], sequence)

                if os.listdir(label_path) != []:
                    filename = os.listdir(label_path)[0]
                    with open(os.path.join(label_path, filename), 'r') as content_file:
                        content = content_file.read()
                        content = content.replace(' ', '')
                        content = content.replace('\n', '')
                        label = int(float(content))

                    item = (os.path.join(data_dir, subjects[subject], sequence), label - indexing)
                    images.append(item)
    else:
        sequences = dirs[0]
        for sequence in sequences:
            # check if emotion has indeed been detected on subject
            label_path = os.path.join(dir, label_folder, sequence)

            if os.listdir(label_path) != []:
                filename = os.listdir(label_path)[0]
                with open(os.path.join(label_path, filename), 'r') as content_file:
                    content = content_file.read()
                    content = content.replace(' ', '')
                    content = content.replace('\n', '')
                    label = int(float(content))

            item = (os.path.join(data_dir, sequence), label - indexing)
            images.append(item)
    
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


class VideoFolder(data.Dataset):

    def __init__(self, root, image_folder, label_folder, landmark_folder,
                 fold, classes, phase, n_frames, n_landmarks, img_type='png', 
                 transform=None, target_transform=None, indexing=0,
                 are_subjects=True, loader=default_loader):

        imgs = make_dataset(root, image_folder, label_folder, fold, indexing, are_subjects)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.landmark_folder = landmark_folder
        self.fold = fold
        self.phase = phase
        self.imgs = imgs
        self.img_type = img_type
        self.classes = classes
        self.n_frames = n_frames
        self.n_landmarks = 2 * n_landmarks
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.transform = transform
        self.target_transform = target_transform
        self.indexing = indexing
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (video, target) where target is class_index of the target class.
        """
        
        sequence_path, target = self.imgs[index]
        files = os.listdir(sequence_path)
        
        video = torch.FloatTensor()
        
        frames = []
        landmarks = np.zeros(self.n_landmarks * len(files))
        idx_frame = 0
        nose = int(self.n_landmarks / 2 - 1)
        noise_mean = 0
        noise_std = 0.01
        possible_angles = [-15, -10, -5, -0, 0, 5, 10, 15]
        beta = - pi/10
        gamma = pi/10
        theta = np.random.uniform(beta, gamma)
        angle = random.choice(possible_angles)
        flip = bool(random.getrandbits(1))
        
        samples = []
          
        for idx_frame, frame_name in enumerate(sorted(files)):
            frame_path = os.path.join(sequence_path, frame_name)
            frame = self.loader(frame_path)
            
            if self.transform is not None:      
                frame = self.transform(frame)
                # data augmentation for training
                if self.phase == 'train':
                    if flip:
                        frame = TF.hflip(frame)   
                    frame = TF.rotate(frame, angle)
                frame = TF.to_grayscale(frame)
                frame = TF.to_tensor(frame)
                #frame = TF.normalize(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            center = frame.size()[-1] / 2
            frames.append(frame)
            
            # read landmarks
            landmarks_path = frame_path.replace('.' + self.img_type, '_landmarks.npy')
            landmarks_path = landmarks_path.replace(self.image_folder, self.landmark_folder)
            
            
            # check if landmarks have been indeed detected
            if os.path.isfile(landmarks_path):  
                samples.append(idx_frame)
                frame_landmarks = np.load(landmarks_path).astype(float)
                landmarks_x = frame_landmarks[::2]
                landmarks_y = frame_landmarks[1::2]

                # rotation
                # x' = cos(theta) * (x - x0) - sin(theta) * (y - y0) + x0
                # y' = sin(theta) * (x - x0) + cos(theta) * (y - y0) + y0

                # normalize
                landmarks_x = (landmarks_x - landmarks_x[nose]) / landmarks_x.std()
                landmarks_y = (landmarks_y - landmarks_y[nose]) / landmarks_y.std()

                if self.phase == 'train':
                    if flip:
                        landmarks_x = -landmarks_x

                    # add gaussian noise
                    frame_landmarks[::2] = landmarks_x + np.random.normal(noise_mean, noise_std, landmarks_x.shape)
                    frame_landmarks[1::2] = landmarks_y + np.random.normal(noise_mean, noise_std, landmarks_y.shape)

                    # rotation data augmentation
                    frame_landmarks[::2] = landmarks_x * np.cos(theta) - landmarks_y * np.sin(theta)
                    frame_landmarks[1::2] = landmarks_x * np.sin(theta) + landmarks_y * np.cos(theta)

                else:
                    frame_landmarks[::2] = landmarks_x
                    frame_landmarks[1::2] = landmarks_y

                landmarks[idx_frame*self.n_landmarks:(idx_frame+1)*self.n_landmarks] = frame_landmarks
               
            # pad
            else:
                landmarks[idx_frame*self.n_landmarks:(idx_frame+1)*self.n_landmarks] = np.zeros(self.n_landmarks)
            
        # subsample frames
        #sampling = np.linspace(0, len(frames)-1, self.n_frames, dtype=int)
        sampling = np.sort(np.random.choice(samples, self.n_frames, replace=True))
        frames = [frames[i] for i in sampling]
        landmarks = np.array_split(landmarks, len(landmarks)/self.n_landmarks)
        landmarks = [landmarks[i] for i in sampling]
        landmarks = np.concatenate(landmarks, axis=0)
        
        video = torch.stack(frames)
            
        if self.target_transform is not None:
            target = self.target_transform(target)  
            
        return video, target, landmarks

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