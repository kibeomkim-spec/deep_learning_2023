import csv
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def set_centroid(path):
  new_text_content = ''
  with open(path, 'r', encoding="utf-8-sig", errors="ignore") as f:
      csv_reader = csv.reader(f, delimiter=' ')
      for row in csv_reader:
        class_number = row[0]
        x = float(row[1]) * img_width
        y = float(row[2]) * img_height
        bbox_width = float(row[3]) * img_width
        bbox_height = float(row[4]) * img_height
        xcentre = (x + bbox_width / 2) / img_width
        ycentre = (y + bbox_height / 2) / img_height
        new_text_content +=  " ".join([class_number, str(xcentre), str(ycentre), str(bbox_width / img_width), str(bbox_height / img_height)]) + "\n"

  with open(path, "w") as f:
    f.write(new_text_content)

def draw_bounding_box(path, file_name):
  try :
    img_path = os.path.join(path, "images", file_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    draw_img = Image.fromarray(img)
    draw = ImageDraw.Draw(draw_img)

    img_width = img.shape[1]
    img_height = img.shape[0]

    file_name = file_name.replace("jpg", "txt")
    labels_path = os.path.join(path, "labels", file_name)

    with open(labels_path, encoding="utf-8-sig", errors="ignore") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            xcentre = float(row[1]) * img_width
            ycentre = float(row[2]) * img_height
            bbox_width = float(row[3]) * img_width
            bbox_height = float(row[4]) * img_height
            x0 = xcentre - bbox_width / 2
            x1 = xcentre + bbox_width / 2
            y0 = ycentre - bbox_height / 2
            y1 = ycentre + bbox_height / 2
            draw.rectangle([x0, y0, x1, y1], outline='blue', width=2)

  except :
    img_path = os.path.join(file_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    draw_img = Image.fromarray(img)
    draw = ImageDraw.Draw(draw_img)

    img_width = img.shape[1]
    img_height = img.shape[0]

    file_name = file_name.replace("jpg", "txt")
    labels_path = os.path.join(file_name)

    with open(labels_path, encoding="utf-8-sig", errors="ignore") as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            xcentre = float(row[1]) * img_width
            ycentre = float(row[2]) * img_height
            bbox_width = float(row[3]) * img_width
            bbox_height = float(row[4]) * img_height
            x0 = xcentre - bbox_width / 2
            x1 = xcentre + bbox_width / 2
            y0 = ycentre - bbox_height / 2
            y1 = ycentre + bbox_height / 2
            draw.rectangle([x0, y0, x1, y1], outline='blue', width=2)
  
  plt.figure(figsize=(20, 10))
  plt.imshow(draw_img)
  
set_centroid("test.txt") 
draw_bounding_box("","test.jpg")
    