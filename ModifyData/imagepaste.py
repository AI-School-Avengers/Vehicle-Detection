import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

data_dir = 'D:\\img'
all_image_data_path = sorted(glob.glob(os.path.join(data_dir,"*")))
total_data = list(all_image_data_path)
data_length = len(total_data)
print(data_length)
for i in range(1, 65, 4):
    img1 = cv2.imread(total_data[i], 1)
    img2 = cv2.imread(total_data[i+1], 1)
    img3 = cv2.imread(total_data[i+2], 1)
    img4 = cv2.imread(total_data[i+3], 1)

    img1 = cv2.resize(img1, (480,282))
    img2 = cv2.resize(img2, (480,282))
    img3 = cv2.resize(img3, (480,282))
    img4 = cv2.resize(img4, (480,282))

    addh1 = cv2.hconcat([img1, img2])
    addh2 = cv2.hconcat([img3, img4])
    addv = cv2.vconcat([addh1, addh2])


    cv2.imwrite('D:/img/{}.jpg'.format(i), addv)
    i = 4*i+1

    # cv2.imshow('imgv',addv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()