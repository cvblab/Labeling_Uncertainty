import numpy as np
import os
import cv2
from PIL import Image
import random
from matplotlib import pyplot as plt
import skimage.transform
import skimage.util
import torch
from tensorflow.keras.utils import to_categorical
from random import randint
import pandas as pd


class MILDataset(object):

    def __init__(self, dir_images, data_frame, classes, bag_id='Name_Image', input_shape=(3, 224, 224),
                 data_augmentation=False, images_on_ram=False, channel_first=True):

        self.dir_images = dir_images
        self.data_frame = data_frame
        self.classes = classes
        self.bag_id = bag_id
        self.data_augmentation = data_augmentation
        self.input_shape = input_shape
        self.images_on_ram = images_on_ram
        self.channel_first = channel_first

        self.images_folder = []
        [self.images_folder.extend(os.listdir(idir)) for idir in dir_images]

        # Regiones de interes
        self.images = pd.read_csv('../data/' + 'HUSC_HCUV_RI.csv', dtype=str, delimiter=';')
        self.images = self.images.values.tolist()
        self.images = [item for sublist in self.images for item in sublist]

        # Filter wrong files
        self.images = [ID for ID in self.images if ID != 'Thumbs.db']

        # Filter patches whose slide is not in the dataframe
        idx = np.in1d([ID.split('_')[0] + '_' + ID.split('_')[1] for ID in self.images], self.data_frame[self.bag_id])
        images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
        self.images = images

        # Filter non-present patches
        idx = np.in1d(self.images, self.images_folder)
        images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
        self.images = images

        # Filter slides in the dataframe whose patches are not in the images folder
        self.data_frame = self.data_frame[
            np.in1d(self.data_frame[self.bag_id], [ID.split('_')[0] + '_' + ID.split('_')[1] for ID in images])]

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.D = dict()
        for i, item in enumerate([ID.split('_')[0] + '_' + ID.split('_')[1] for ID in self.images]):
            if item not in self.D:
                self.D[item] = [i]
            else:
                self.D[item].append(i)

        self.GT = self.data_frame['GT'].values / 100
        self.y = self.data_frame[self.classes].values / 100
        self.indexes = np.arange(len(self.images))

        if self.images_on_ram:

            # Pre-allocate images
            self.X = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)

            # Load, and normalize images
            print('[INFO]: Training on ram: Loading images')
            for i in np.arange(len(self.indexes)):
                print(str(i) + '/' + str(len(self.indexes)), end='\r')

                ID = self.images[self.indexes[i]]

                # Load image
                if 'HCUV' in ID:
                    x = Image.open(os.path.join(self.dir_images[0], ID))
                elif 'HUSC' in ID:
                    x = Image.open(os.path.join(self.dir_images[1], ID))
                x = np.asarray(x)

                # Normalization
                x = self.image_normalization(x)
                self.X[self.indexes[i], :, :, :] = x

            print('[INFO]: Images loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.images[self.indexes[index]]

        if self.images_on_ram:
            x = np.squeeze(self.X[self.indexes[index], :, :, :])
        else:
            # Load image
            if 'HCUV' in ID:
                x = Image.open(os.path.join(self.dir_images[0], ID))
            elif 'HUSC' in ID:
                x = Image.open(os.path.join(self.dir_images[1], ID))

            x = np.asarray(x)
            # Normalization
            x = self.image_normalization(x)

        # data augmentation
        if self.data_augmentation:
            x_augm = self.image_transformation(x.copy())
        else:
            x_augm = None

        return x, x_augm

    def image_transformation(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def image_normalization(self, x):
        # image resize
        x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if self.channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x

    def plot_image(self, x, norm_intensity=False):
        # channel first
        if self.channel_first:
            x = np.transpose(x, (1, 2, 0))
        if norm_intensity:
            x = x / 255.0

        plt.imshow(x)
        plt.axis('off')
        plt.show()


class MILDataGenerator(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, max_instances=512, labeler='annotator',
                 type_labels='hard'):

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        """

        'Internal states initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.labeler = labeler  # 'annotator','expert'
        self.type_labels = type_labels  # 'hard','soft'

        self._idx = 0
        self._reset()

    def __len__(self):

        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):

        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]

        # Get bag-level label
        if self.labeler == 'annotator':
            Y = df_row[self.dataset.classes].to_list()
            Y = np.expand_dims(np.array(Y), 0) / 100
        elif self.labeler == 'expert':
            Y = np.array(df_row['GT'])
            Y = np.expand_dims(to_categorical(Y, num_classes=len(self.dataset.classes)), 0)

        if self.labeler == 'annotator' and self.type_labels == 'hard':
            Y = np.where(Y[0, :] == np.max(Y))
            if len(Y[0])>1:
                value=randint(0, len(Y[0])-1)
                Y = to_categorical(Y[0][value], num_classes=len(self.dataset.classes))
                Y = np.expand_dims(Y, 0)
            else:
                Y = to_categorical(Y[0], num_classes=len(self.dataset.classes))

            # Add batch dimension
            Y = np.expand_dims(Y, 0)

        # Select instances from bag
        ID = list(df_row[[self.dataset.bag_id]].values)[0]
        images_id = self.dataset.D[ID]

        # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)
        # Minimum number os patches in a slide (by precaution).
        if len(images_id) < 4:
            images_id.extend(images_id)

        self.instances_indexes = images_id

        # Load images and include into the batch
        X = []
        X_augm = []
        for i in images_id:
            x, x_augm = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)

        # Update bag index iterator
        self._idx += self.batch_size

        return np.array(X).astype('float32'), np.array(Y).astype('float32')

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0
