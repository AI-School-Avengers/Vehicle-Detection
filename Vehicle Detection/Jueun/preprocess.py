import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import random
from tqdm import tqdm
import shutil

class CustomDataset(Dataset):
    CLASSES = (
        "__background__", "Car", "Truck", "Bus", "Etc vehicle", "Bike", "License",  # ?_ background 있어야하나
    )

    def __init__(self, data_dir, mode, transforms=None):     # mode->train/test
        # data 정의
        self.all_img_data = sorted(glob.glob(os.path.join(data_dir, mode,"images","*")))
        self.all_label_data = sorted(glob.glob(os.path.join(data_dir, mode,"annotations","*")))
        self.transforms = transforms

        if len(self.all_img_data) != len(self.all_label_data):
            raise Exception("Data Error")

        # class 정의
        pass
    def __getitem__(self, idx):
        # target(box/labels/name?) 만들기
        # img와 target return
        # image읽어오기
        pass
    def __len__(self):
        pass
