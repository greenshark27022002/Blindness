
import numpy as np
import pandas as pd

import torch
import torchvision
from tensorflow.keras.optimizers import Adam
from torchvision import transforms, datasets
from torch.utils.data import Dataset

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

from tqdm import tqdm_notebook as tqdm

import random
import time
import sys
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
print('Train Size = {}'.format(len(train_csv)))
print('Test Size = {}'.format(len(test_csv)))

##### CLASS DISTRIBUTION
# plot
fig = plt.figure(figsize = (15, 5))
plt.hist(train_csv['diagnosis'])
plt.title('Class Distribution')
plt.ylabel('Number of examples')
plt.xlabel('Diagnosis')