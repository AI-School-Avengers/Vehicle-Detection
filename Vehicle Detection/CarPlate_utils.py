import torch
from torchvision.datasets.vision import VisionDataset
import transforms as T
import glob
from PIL import Image
import random
import os
import xml.etree.ElementTree as ET


# Car (승용차), Truck (트럭), Bus (버스), Etc vehicle (기타 차량-덤프트럭, 레미콘 등 건설용차량), Bike( 이륜 차) License(번호판)

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
            xmin = float(object.find("bndbox").findtext("xmin"))
            ymin = float(object.find("bndbox").findtext("ymin"))
            xmax = float(object.find("bndbox").findtext("xmax"))
            ymax = float(object.find("bndbox").findtext("ymax"))
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
    def __init__(self, img_folder, all_images_path, all_labels_path, image_set, transforms=None):
        print("############################## \n", "CarPlateDetection Init")
        # print(img_folder, image_set, transforms)

        self.all_images_path = all_images_path
        self.all_labels_path = all_labels_path

        self._transforms = transforms

    def __getitem__(self, idx):
        data_path = self.all_images_path[idx]
        label_path = self.all_labels_path[idx]

        img = Image.open(data_path)
        img = img.convert("RGB")  # images BGR -> RGB convert

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
    images_train_path, images_test_path, labels_train_path, labels_test_path = get_CarPlate_image_path_list(
        img_folder=root)

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
def get_CarPlate_image_path_list(img_folder, seed=0):
    train_images_path = sorted(glob.glob(os.path.join(img_folder, "train/images", "*")))
    train_labels_path = sorted(glob.glob(os.path.join(img_folder, "train/annotations", "*")))
    test_images_path = sorted(glob.glob(os.path.join(img_folder, "test/images", "*")))
    test_labels_path = sorted(glob.glob(os.path.join(img_folder, "test/annotations", "*")))

    if len(train_images_path) != len(train_labels_path):
        raise Exception("Train Data Error")

    if len(test_images_path) != len(test_labels_path):
        raise Exception("Test Data Error")

    total_data = list(zip(train_images_path, train_labels_path))
    random.seed(seed)
    random.shuffle(total_data)

    x_train, y_train = map(list, zip(*total_data))
    x_test, y_test = list(test_images_path), list(test_labels_path)

    return x_train, x_test, y_train, y_test