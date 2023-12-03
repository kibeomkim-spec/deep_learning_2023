import re
import json
import os
import csv
import shutil
import csv
import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from glob import glob
from sklearn.model_selection import train_test_split

feature = {"집전체" : 0, "지붕" : 0, "집벽" : 0, "문" : 0, "창문" : 0, "굴뚝" : 0, "연기" : 0, "울타리" : 0, "길" : 0, "연못" : 0, "산" : 0, "나무" : 0, "꽃" : 0, "잔디" : 0, "태양" : 0, "나무전체" : 0, "기둥" : 0, "수관" : 0, "가지" : 0, "뿌리" : 0, "나뭇잎" : 0, "꽃" : 0, "열매" : 0, "그네" : 0, "새" : 0, "다람쥐" : 0, "구름" : 0, "달" : 0, "별" : 0, "사람전체" : 0, "머리" : 0, "얼굴" : 0, "눈" : 0, "코" : 0, "입" : 0, "귀" : 0, "머리카락" : 0, "목" : 0, "상체" : 0, "팔" : 0, "손" : 0, "다리" : 0, "발" : 0, "단추" : 0, "주머니" : 0, "운동화" : 0, "여자구두" : 0, "머리카락" : 0, "남자구두" : 0}

keys = list(feature.keys())

# interpret the title of file
def interpret_list(filename):
  name = ['나무', '남자사람', '여자사람', '집']
  global_name = ['tree', 'man', 'woman', 'house']
  sex = ['남', '여']
  global_sex = ['boy', 'girl']
  strings = filename.split('_')
  strings[0] = [global_name[i] for i in range(len(name)) if name[i] == strings[0]][0]
  strings[2] = [global_sex[i] for i in range(len(sex)) if sex[i] == strings[2]][0]
  return "_".join(strings)

# save the yolo format from original json file
def save_as_yolo_format(destination_folder, json_data):
  filename = pd.json_normalize(json_data['meta'])["label_path"][0].replace('./', '')
  filename = filename.replace('.json', '.jpg')
  img_width = int(json_data['meta']['img_resolution'][:4])
  img_height = int(json_data['meta']['img_resolution'][-4:])
  
  yolov5_format_list = []
  for i in json_data['annotations']['bbox']:
    region_num = [j for j in range(len(feature)) if i['label'] == keys[j]][0]
    xcentre = float(i['x']) /img_width
    ycentre = float(i['y']) /img_height
    bbox_width = float(i['w']) /img_width
    bbox_height = float(i['h']) /img_height
    yolov5_format = [region_num, xcentre, ycentre, bbox_width, bbox_height]
    yolov5_format_list.append(yolov5_format)
    
    file = open(os.path.join(destination_folder, re.sub(r'[^.]+$', 'txt', filename)), 'w', newline='')
    with file:
        write = csv.writer(file, delimiter= ' ')
        write.writerows(yolov5_format_list)
        
def bbox_visualization(img_path, jpegs, jsons):
  img = cv2.imread(os.path.join(img_path, jpegs))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  draw_img = Image.fromarray(img)
  draw = ImageDraw.Draw(draw_img)

  img_width = img.shape[1]
  img_height = img.shape[0]
  
  with open(os.path.join(img_path, jsons), encoding="utf-8-sig", errors="ignore") as f:
    csv_reader = csv.reader(f, delimiter=' ')
    for row in csv_reader:
        xcentre = float(row[1]) * img_width
        ycentre = float(row[2]) * img_height
        bbox_width = float(row[3]) * img_width
        bbox_height = float(row[4]) * img_height
        x0 = xcentre
        x1 = xcentre + bbox_width
        y0 = ycentre
        y1 = ycentre + bbox_height
        draw.rectangle([x0, y0, x1, y1], outline='blue', width=2)

  plt.figure(figsize=(20, 10))
  plt.imshow(draw_img)

def copy_file(txt_file, src, dst):
  file_list = open(txt_file, "r");
  file_list = file_list.read().split(sep="\n")
  
  for filename in os.listdir(src):
    if filename in file_list :
      shutil.copy(os.path.join(src, filename), os.path.join(dst, filename))
  
training_images_folder = os.path.join(os.getcwd(), "dataset", "export", "images")
training_labels_folder = os.path.join(os.getcwd(), "dataset", "export", "labels")

jpegs = []
jsons = []

for f in os.listdir(training_images_folder):
  if f[-4:] == ".jpg":
    jpegs.append(f)

for f in os.listdir(training_labels_folder):
  if f[-5:] == ".json":
    jsons.append(f)
    
path = os.path.join(os.getcwd(), "dataset", "export", "labels")
for j in jsons:
    with open(os.path.join(training_labels_folder, j), encoding="utf-8-sig", errors="ignore") as json_file:
        json_data = json.load(json_file)
    save_as_yolo_format(path, json_data)

# rename the file_name to English  
file_list = os.listdir(path)

for file in file_list:
  src = os.path.join(path, file)
  dst = os.path.join(path, interpret_list(file))
  os.rename(src, dst)
  
# Randomly extract 1000 image data
house_list = random.sample(glob(os.path.join(os.getcwd(), "house_*.jpg")), 1000)
tree_list = random.sample(glob(os.path.join(os.getcwd(), "tree_*.jpg")), 1000)
man_list = random.sample(glob(os.path.join(os.getcwd(), "man_*.jpg")), 1000)
woman_list = random.sample(glob(os.path.join(os.getcwd(), "woman_*.jpg")), 1000)
img_list = house_list + tree_list + man_list + woman_list

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)

with open('yolov5/dataset/export/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')
  
with open('yolov5/dataset/export/val.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')
  
bbox_visualization(src, jpegs[100], jsons[100])
