import torch
from torchvision.datasets.vision import VisionDataset
import transforms as T
import glob
from PIL import Image
import random
import os
import xml.etree.ElementTree as ET

#Car (승용차), Truck (트럭), Bus (버스), Etc vehicle (기타 차량-덤프트럭, 레미콘 등 건설용차량), Bike( 이륜 차) License(번호판)

class ConvertCarPlatetoCOCO(object):
    CLASSES = (
        "__background__", "Car", "Truck", "Bus", "Etc vehicle", "Bike", "License",
    )

    def __call__(self, image, annotation_path):

        boxes = []
        classes = []
        # for annotation_file in annotation_files:
        # xml_path = os.path.join(annotation_path, annotation_files)

        doc = ET.parse(annotation_path)

        root = doc.getroot()

        # root = dom.documentElement
        # objects = dom.getElementsBytagName('object')
        # objects = dom.getElementsBytagName('filename')
        filename = root.findtext("filename")

        for object in root.iter("object"):

            xmin = int(float(object.find("bndbox").findtext("xmin")))
            ymin = int(float(object.find("bndbox").findtext("ymin")))
            xmax = int(float(object.find("bndbox").findtext("xmax")))
            ymax = int(float(object.find("bndbox").findtext("ymax")))
            bbox = [xmin, ymin, xmax, ymax]

            boxes.append(bbox)
            classes.append(self.CLASSES.index(object.findtext("name")))


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)


        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8)  # convert filename in int8

        return image, target

class CarPlateDetection(VisionDataset):
    def __init__(self, img_folder, all_images_path, all_labels_path, image_set, transforms = None):
        print("############################## \n","CarPlateDetection Init")
        # print(img_folder, image_set, transforms)

        
        self.all_images_path = all_images_path
        self.all_labels_path = all_labels_path

        self._transforms = transforms

    def __getitem__(self, idx):
        data_path = self.all_images_path[idx]
        label_path = self.all_labels_path[idx]

        img = Image.open(data_path)
        img = img.convert("RGB") # image BGR -> RGB convert

        if self._transforms is not None:
            img, target = self._transforms(img, label_path)
            
        return img, target
    
    def get_height_and_width(self, idx):
        data_path = self.all_images_path[idx]

        img = Image.open(data_path)
        return img.size

    def __len__(self):
        return len(self.all_images_path)

# get CarPlate dataset
def get_CarPlate(root, image_set, transforms):
    t = [ConvertCarPlatetoCOCO()]
    print(t)
    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    print("Split CarPlate Data")
    images_train_path, images_test_path, labels_train_path, labels_test_path = get_CarPlate_image_path_split_list(img_folder=root, test_size=0.2)

    print("train data count:", len(images_train_path), "/ test data count:", len(images_test_path))

    if image_set == "train":
        print("Create Train Dataset Car Plate Detection")
        dataset = CarPlateDetection(img_folder=root, all_images_path=images_train_path, 
        all_labels_path=labels_train_path, image_set=image_set, transforms=transforms)
    else:
        print("Create Test Dataset Car Plate Detection")
        dataset = CarPlateDetection(img_folder=root, all_images_path=images_test_path, 
        all_labels_path=labels_test_path, image_set=image_set, transforms=transforms)
    print("###############################")

    
    return dataset

# CarPlate data split
def get_CarPlate_image_path_split_list(img_folder, test_size= 0.2, seed=0):
    all_images_path = sorted(glob.glob(os.path.join(img_folder, "images", "*")))
    all_labels_path = sorted(glob.glob(os.path.join(img_folder, "annotations", "*")))

    if len(all_images_path) != len(all_labels_path):
        raise Exception("Split Data Failed")

    if test_size < 0 or test_size > 1:
        raise Exception("Split Test Size 0 ~ 1")
        
    total_data = list(zip(all_images_path, all_labels_path))
    random.seed(seed)
    random.shuffle(total_data)

    images_path, labels_path = map(list, zip(*total_data))

    divide_index = int(len(all_images_path) * (1-test_size)) # train index / test index

    x_train = images_path[:divide_index]
    x_test = images_path[divide_index:]

    y_train = labels_path[:divide_index]
    y_test = labels_path[divide_index:]

    return x_train, x_test, y_train, y_test