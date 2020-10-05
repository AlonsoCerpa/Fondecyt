import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
from skimage import draw
from skimage import measure
from PIL import Image
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt
from detectron2.utils.visualizer import ColorMode
import os
import random
import cv2
from detectron2.data import build_detection_test_loader
from detectron2.evaluation.evaluator import *
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
import numpy as np
from skimage import draw
import copy
from PIL import ImageDraw

from tkinter import *
from PIL import ImageTk, Image

img_id = 3000
xfact = 1.02
yfact = 1.02
dilate_dist = 3

def create_mask(mask_image):
    height, width = mask_image.shape
    mask_array = np.zeros((height+2, width+2), dtype=bool)
    indices = np.argwhere(mask_image)
    indices += 1
    rows, cols = zip(*indices)
    mask_array[rows, cols] = True
    mask = Image.fromarray(mask_array)
    return mask

def polygon2mask_aux(mask, polygon):
    polygon = np.asarray(polygon)
    vertex_row_coords, vertex_col_coords = polygon.T
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, mask.shape)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def process_mask_and_bbox(multi_poly):
    new_mask = np.zeros((576, 1024), dtype=np.bool)
    for polygon in multi_poly:
        polygon2mask_aux(new_mask, polygon.exterior.coords)
    new_mask = create_mask(new_mask)
    contours = measure.find_contours(new_mask, 0.5, positive_orientation='low')
    polygons = []
    for contour in contours:
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=True)
        polygons.append(poly)
    multi_poly = MultiPolygon(polygons)
    segmentations = []
    for polygon in multi_poly:
        segmentation = np.array(polygon.exterior.coords)
        temp = np.copy(segmentation[:,0])
        segmentation[:,0] = segmentation[:,1]
        segmentation[:,1] = temp
        segmentation = segmentation.ravel().tolist()
        segmentations.append(segmentation)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (y, x, height, width)

    return segmentations, bbox

def save_file_with_new_masks(data, ann, img_id, segmentations, bbox, name_file):
    new_data = copy.deepcopy(data)
    images = data["images"]
    new_images = []
    new_images.append(images[img_id-1])
    new_data["images"] = new_images
    ann["segmentation"] = segmentations
    ann["bbox"] = bbox
    new_data["annotations"] = [ann]

    with open(name_file, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, ensure_ascii=False, indent=4)

with open('coco_dataset_train.json') as json_file:
    data = json.load(json_file)
    annotations = data["annotations"]
    
    ann = copy.deepcopy(annotations[img_id-1])
    segms = ann["segmentation"]
    polygons = []
    for segm in segms:
        segm = np.asarray(segm)
        segm = np.reshape(segm, (-1, 2))
        temp = np.copy(segm[:, 0])
        segm[:, 0] = segm[:, 1]
        segm[:, 1] = temp
        poly = Polygon(segm)
        polygons.append(poly)
    multi_poly = MultiPolygon(polygons)
    list_polys_dilated_round = []
    list_polys_dilated_flat = []
    list_polys_dilated_square = []
    for poly in multi_poly:
        list_polys_dilated_round.append(poly.buffer(dilate_dist, cap_style=1))
        list_polys_dilated_flat.append(poly.buffer(dilate_dist, cap_style=2))
        list_polys_dilated_square.append(poly.buffer(dilate_dist, cap_style=3))
    multi_poly_dilated_round = MultiPolygon(list_polys_dilated_round)
    multi_poly_dilated_flat = MultiPolygon(list_polys_dilated_flat)
    multi_poly_dilated_square = MultiPolygon(list_polys_dilated_square)
    multi_poly_scaled = scale(multi_poly, xfact, yfact)

    segmentations_scaled, bbox_scaled = process_mask_and_bbox(multi_poly_scaled)
    segmentations_dilated_round, bbox_dilated_round = process_mask_and_bbox(multi_poly_dilated_round)
    segmentations_dilated_flat, bbox_dilated_flat = process_mask_and_bbox(multi_poly_dilated_flat)
    segmentations_dilated_square, bbox_dilated_square = process_mask_and_bbox(multi_poly_dilated_square)

    save_file_with_new_masks(data, ann, img_id, segmentations_scaled, bbox_scaled, 'coco_dataset_train_visualizer_scale.json')
    save_file_with_new_masks(data, ann, img_id, segmentations_dilated_round, bbox_dilated_round, 'coco_dataset_train_visualizer_dilated_round.json')
    save_file_with_new_masks(data, ann, img_id, segmentations_dilated_flat, bbox_dilated_flat, 'coco_dataset_train_visualizer_dilated_flat.json')
    save_file_with_new_masks(data, ann, img_id, segmentations_dilated_square, bbox_dilated_square, 'coco_dataset_train_visualizer_dilated_square.json')

root = Tk()
root.title("Image Viewer")


register_coco_instances("UCSP Fondecyt Dataset train", {}, "coco_dataset_train.json", "blurred_images")
register_coco_instances("UCSP Fondecyt Dataset train 2", {}, "coco_dataset_train.json", "new_images")

dataset_dicts = get_detection_dataset_dicts(["UCSP Fondecyt Dataset train"])
dataset_dicts2 = get_detection_dataset_dicts(["UCSP Fondecyt Dataset train 2"])
dataset_metadata = MetadataCatalog.get("UCSP Fondecyt Dataset train")

print(dataset_dicts[img_id-1]["file_name"])
my_img0 = ImageTk.PhotoImage(Image.open(dataset_dicts2[img_id-1]["file_name"][:-4] + "_w.png"))
my_img1 = ImageTk.PhotoImage(Image.open(dataset_dicts[img_id-1]["file_name"]))
#img = cv2.imread(dataset_dicts[img_id-1]["file_name"])
#visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata)
#vis = visualizer.draw_dataset_dict(dataset_dicts[img_id-1])
#my_img2 = ImageTk.PhotoImage(Image.fromarray(vis.get_image()))
img_orig = Image.open(dataset_dicts[img_id-1]["file_name"])
draw = ImageDraw.Draw(img_orig)
for list_coords in dataset_dicts[img_id-1]["annotations"][0]["segmentation"]:
    draw.polygon(list_coords)
my_img2 = ImageTk.PhotoImage(img_orig)



register_coco_instances("UCSP Fondecyt Dataset visualizer scaled", {}, "coco_dataset_train_visualizer_scale.json", "blurred_images")

dataset_dicts = get_detection_dataset_dicts(["UCSP Fondecyt Dataset visualizer scaled"])
dataset_metadata = MetadataCatalog.get("UCSP Fondecyt Dataset visualizer scaled")

#img = cv2.imread(dataset_dicts[0]["file_name"])
#visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, instance_mode=ColorMode.IMAGE_BW)
#vis = visualizer.draw_dataset_dict(dataset_dicts[0])
#my_img3 = ImageTk.PhotoImage(Image.fromarray(vis.get_image()))
img_orig = Image.open(dataset_dicts[0]["file_name"])
draw = ImageDraw.Draw(img_orig)
for list_coords in dataset_dicts[0]["annotations"][0]["segmentation"]:
    draw.polygon(list_coords)
my_img3 = ImageTk.PhotoImage(img_orig)



register_coco_instances("UCSP Fondecyt Dataset visualizer dilated round", {}, "coco_dataset_train_visualizer_dilated_round.json", "blurred_images")

dataset_dicts = get_detection_dataset_dicts(["UCSP Fondecyt Dataset visualizer dilated round"])
dataset_metadata = MetadataCatalog.get("UCSP Fondecyt Dataset visualizer dilated round")

#img = cv2.imread(dataset_dicts[0]["file_name"])
#visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata)
#vis = visualizer.draw_dataset_dict(dataset_dicts[0])
#my_img4 = ImageTk.PhotoImage(Image.fromarray(vis.get_image()))
img_orig = Image.open(dataset_dicts[0]["file_name"])
draw = ImageDraw.Draw(img_orig)
for list_coords in dataset_dicts[0]["annotations"][0]["segmentation"]:
    draw.polygon(list_coords)
my_img4 = ImageTk.PhotoImage(img_orig)


register_coco_instances("UCSP Fondecyt Dataset visualizer dilated flat", {}, "coco_dataset_train_visualizer_dilated_flat.json", "blurred_images")

dataset_dicts = get_detection_dataset_dicts(["UCSP Fondecyt Dataset visualizer dilated flat"])
dataset_metadata = MetadataCatalog.get("UCSP Fondecyt Dataset visualizer dilated flat")

#img = cv2.imread(dataset_dicts[0]["file_name"])
#visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata)
#vis = visualizer.draw_dataset_dict(dataset_dicts[0])
#my_img5 = ImageTk.PhotoImage(Image.fromarray(vis.get_image()))
img_orig = Image.open(dataset_dicts[0]["file_name"])
draw = ImageDraw.Draw(img_orig)
for list_coords in dataset_dicts[0]["annotations"][0]["segmentation"]:
    draw.polygon(list_coords)
my_img5 = ImageTk.PhotoImage(img_orig)



register_coco_instances("UCSP Fondecyt Dataset visualizer dilated square", {}, "coco_dataset_train_visualizer_dilated_square.json", "blurred_images")

dataset_dicts = get_detection_dataset_dicts(["UCSP Fondecyt Dataset visualizer dilated square"])
dataset_metadata = MetadataCatalog.get("UCSP Fondecyt Dataset visualizer dilated square")

#img = cv2.imread(dataset_dicts[0]["file_name"])
#visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata)
#vis = visualizer.draw_dataset_dict(dataset_dicts[0])
#my_img6 = ImageTk.PhotoImage(Image.fromarray(vis.get_image()))
img_orig = Image.open(dataset_dicts[0]["file_name"])
draw = ImageDraw.Draw(img_orig)
for list_coords in dataset_dicts[0]["annotations"][0]["segmentation"]:
    draw.polygon(list_coords)
my_img6 = ImageTk.PhotoImage(img_orig)



image_list = [my_img0, my_img1, my_img2, my_img3, my_img4, my_img5, my_img6]

my_label = Label(image=image_list[0])
my_label.grid(row=0, column=0, columnspan=3)

def forward(image_number):
    global my_label
    global button_forward
    global button_back

    my_label.grid_forget()
    my_label = Label(image=image_list[image_number-1])
    button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
    button_back = Button(root, text="<<", command=lambda: back(image_number-1))

    if image_number == len(image_list):
        button_forward = Button(root, text=">>", state=DISABLED)

    my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)
    button_forward.grid(row=1, column=2)

def back(image_number):
    global my_label
    global button_forward
    global button_back

    my_label.grid_forget()
    my_label = Label(image=image_list[image_number-1])
    button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
    button_back = Button(root, text="<<", command=lambda: back(image_number-1))

    if image_number == 1:
        button_back = Button(root, text="<<", state=DISABLED)

    my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)
    button_forward.grid(row=1, column=2)

button_back = Button(root, text="<<", command=lambda: back, state=DISABLED)
button_exit = Button(root, text="EXIT PROGRAM", command=root.quit)
button_forward = Button(root, text=">>", command=lambda: forward(2))

button_back.grid(row=1, column=0)
button_exit.grid(row=1, column=1)
button_forward.grid(row=1, column=2)

root.mainloop()

