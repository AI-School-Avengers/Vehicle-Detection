import torch
from torchvision.datasets.vision import VisionDataset
import transforms as T
import glob
from PIL import Image
import random
import os
import xml.etree.ElementTree as ET


# Car (승용차), Truck (트럭), Bus (버스), Etc vehicle (기타 차량-덤프트럭, 레미콘 등 건설용차량), Bike( 이륜 차) License(번호판)

class ConvertVehicleToCOCO(object):
    CLASSES = (
        "__background__", "Car", "Truck", "Bus", "Etc vehicle", "Bike", "License",  # ?_ background 있어야하나
    )

    def __call__(self, image, anno_dir):        # __call__()는 instance가 호출될 때 실행 -> class의 객체 함수처럼 호출 가능하도록 한다
        boxes = []                              # bounding boxs
        classes = []                            # bounding box마다의 라벨 정보

        doc = ET.parse(anno_dir)                # parse xml file
        root = doc.getroot()                    # get root node
        filename = root.findtext("filename")    # annotaton 처리한 jpg filename

        for object in root.iter("object"):      # root 이하의 object tag들 순회
            xmin = float(object.find("bndbox").findtext("xmin"))
            ymin = float(object.find("bndbox").findtext("ymin"))
            xmax = float(object.find("bndbox").findtext("xmax"))
            ymax = float(object.find("bndbox").findtext("ymax"))
            bbox = [xmin, ymin, xmax, ymax]      # (leftTop, rightBottom)
            boxes.append(bbox)
            classes.append(self.CLASSES.index(object.findtext("name")))

        # torch.Tensor 타입으로 변환
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        classes = torch.as_tensor(classes)
        names = torch.tensor([ord(i) for i in list(filename)], dtype = torch.int8)  # ord(): int8타입 -> Tensor 타입으로 변환

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['name'] =  names

        print("run ConvertVehicleToCOCO")

        return image, target


class CarPlateDetection(VisionDataset):
    def __init__(self, all_images_path, all_labels_path, transforms=None):
        print("############################## \n", "CarPlateDetection Init")
        # print(img_folder, image_set, transforms)

        self.all_images_path = all_images_path
        self.all_labels_path = all_labels_path

        self._transforms = transforms
        #_transforms :  <transforms.Compose object at 0x00000287B6C98E50>  train 일때 aug
        #_transforms :  <transforms.Compose object at 0x00000287B6C5DA00> test 일때 aug

    def __getitem__(self, idx):
        # 한개씩가져와
        data_path = self.all_images_path[idx]
        label_path = self.all_labels_path[idx]
        # 이미지 읽고
        img = Image.open(data_path)
        img = img.convert("RGB")  # images BGR -> RGB convert

        # transform 적용
        if self._transforms is not None:
            img, target = self._transforms(img, label_path)

        return img, target

    def get_height_and_width(self, idx):
        data_path = self.all_images_path[idx]

        img = Image.open(data_path)
        return img.size

    def __len__(self):
        return len(self.all_images_path)


# get vehicle dataset
def get_vehicle(root, image_set, transforms):  # transforms  = get_transform(True,data_augmentation) = presets.DetectionPreset
    t = [ConvertVehicleToCOCO()]

    # transform 설정한거임 걍
    if transforms is not None:   # <presets.DetectionPresetTrain object at 0x00000287B6C98E20>
        t.append(transforms)    # [<vehicle_utils.ConvertVehicleToCOCO object at 0x00000287B6C98F10>, <presets.DetectionPresetTrain object at 0x00000287B6C98E20>]
    transforms = T.Compose(t)   # <transforms.Compose object at 0x00000287B6C98E50>

    print("Split CarPlate Data")
    images_train_path, images_test_path, labels_train_path, labels_test_path = get_CarPlate_image_path_list(
        img_folder=root)

    print("train data count:", len(images_train_path), "/ test data count:", len(images_test_path))

    if image_set == "train":
        print("Create Train Dataset Car Plate Detection")
        dataset = CarPlateDetection(all_images_path=images_train_path,
                                    all_labels_path=labels_train_path, transforms=transforms)
    else:
        print("Create Test Dataset Car Plate Detection")
        dataset = CarPlateDetection( all_images_path=images_test_path,
                                    all_labels_path=labels_test_path, transforms=transforms)
        print("dataset : " ,type(dataset))
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
        print(test_images_path)
        print(test_labels_path)
        raise Exception("Test Data Error")

    total_data = list(zip(train_images_path, train_labels_path))
    random.seed(seed)
    random.shuffle(total_data)

    x_train, y_train = map(list, zip(*total_data))
    x_test, y_test = list(test_images_path), list(test_labels_path)
    return x_train, x_test, y_train, y_test