import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file).resize((512,512), Image.Resampling.LANCZOS)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)

# class SemanticSegmentationDataset(Dataset):
#     """Image (semantic) segmentation dataset."""

#     def __init__(self, root_dir, feature_extractor, train=True):
#         """
#         Args:
#             root_dir (string): Root directory of the dataset containing the images + annotations.
#             feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
#             train (bool): Whether to load "training" or "validation" images + annotations.
#         """
#         self.root_dir = root_dir
#         self.feature_extractor = feature_extractor
#         self.train = train

#         self.img_dir = os.path.join(self.root_dir, "images")
#         self.ann_dir = os.path.join(self.root_dir, "masks")

#         # read images
#         image_file_names = []
#         for root, dirs, files in os.walk(self.img_dir):
#             image_file_names.extend(files)
#         self.images = sorted(image_file_names)
#         print(len(self.images))

#         # read annotations
#         annotation_file_names = []
#         for root, dirs, files in os.walk(self.ann_dir):
#             annotation_file_names.extend(files)
#         self.annotations = sorted(annotation_file_names)
#         print(len(self.annotations))

#         assert len(self.images) == len(
#             self.annotations), "There must be as many images as there are segmentation maps"

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):

#         image = Image.open(os.path.join(self.img_dir, self.images[idx]))
#         segmentation_map = Image.open(os.path.join(
#             self.ann_dir, self.annotations[idx])).convert('L')

#         # randomly crop + pad both image and segmentation map to same size
#         encoded_inputs = self.feature_extractor(
#             image, segmentation_map, return_tensors="pt")

#         for k, v in encoded_inputs.items():
#             encoded_inputs[k].squeeze_()  # remove batch dimension

#         return encoded_inputs


class iiscmed(Dataset):

    def __init__(self,root, co_transform, subset='train'):
        
        self.images_root = os.path.join(root, f'{subset}/images')
        self.labels_root = os.path.join(root, f'{subset}/masks')

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        
        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS
        print(f"len(self.filenames) = {len(self.filenames)}")
        print(f"len(self.filenamesGt) = {len(self.filenamesGt)}")

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path('', filename, ''), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path('', filenameGt, ''), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class ACDC(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'rgb_anon/')
        self.labels_root = os.path.join(root, 'gt/')
        
        self.adverse_cons=['fog/','night/','rain/','snow/']
        self.filenames = []
        self.filenamesGt = []
        # print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        for cons in self.adverse_cons:
            images_root = self.images_root + cons + subset
            labels_root = self.labels_root + cons + subset   
            filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_root)) for f in fn if is_image(f)]


            #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
            #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
            filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(labels_root)) for f in fn if is_label(f)]
            self.filenames.extend(filenames)
            self.filenamesGt.extend(filenamesGt)
        self.filenames.sort()
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class NYUv2(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'images/')
        self.labels_root = os.path.join(root, 'labels40/')
        # self.images_root = os.path.join(root, 'RGB/')
        # self.labels_root = os.path.join(root, 'Label/')
        self.subset_ls = f'{root}{subset}.txt'
        print (self.images_root)
        with open(self.subset_ls) as f:
            dir_subset = [line.strip() for line in f.readlines()]
        self.filenames = [os.path.join(self.images_root,f'{dp}.jpg') for dp in dir_subset]
        self.filenamesGt = [os.path.join(self.labels_root,f'{dp}.png') for dp in dir_subset]
        self.filenames.sort()
        self.filenamesGt.sort()
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        label=label.point(lambda p: p-1)
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)