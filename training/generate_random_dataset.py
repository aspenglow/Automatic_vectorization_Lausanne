import cv2
import numpy as np
import os

input_path = "data0/"
save_path = "data/"
img_path = "images/"
label_path = "labels/"

if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(save_path + img_path):
    os.mkdir(save_path + img_path)
if not os.path.exists(save_path + label_path):
    os.mkdir(save_path + label_path)

img_lists = os.listdir(save_path + img_path)
label_lists = os.listdir(save_path + label_path)

for img_name in img_lists:
    os.remove(save_path + img_path + img_name)
for label_name in label_lists:
    os.remove(save_path + label_path + label_name)



def crop_bbox(img_name, label_name, Crop_N=300, stride=1024, L=2048):
    img = cv2.imread(input_path + img_path + img_name)
    label = cv2.imread(input_path + label_path + label_name)
    # get shape
    H, W, C = img.shape

    count = 0
    # each crop
    for i in range(Crop_N):
        x1 = np.random.randint(W - L)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L)
        # get right bottom x of crop bounding box
        x2 = x1 + L
        # get right bottom y of crop bounding box
        y2 = y1 + L

        # crop bounding box
        img_crop = img[y1:y2,x1:x2]
        label_crop = label[y1:y2,x1:x2]
        img_crop_name = img_name[:-4] + "_" + str(count) + img_name[-4:]
        label_crop_name = label_name[:-4] + "_" + str(count) + label_name[-4:]
        cv2.imwrite(save_path + img_path + img_crop_name, img_crop)
        cv2.imwrite(save_path + label_path + label_crop_name, label_crop)
        count += 1

img_lists = os.listdir(input_path + img_path)
for img_name in img_lists:
    label_name = img_name[:-4] + ".png"
    print("generating dataset from " + img_name)
    crop_bbox(img_name, label_name, stride=512, L=1024)
print("success")


