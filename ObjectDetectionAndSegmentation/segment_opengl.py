from PIL import Image
import numpy as np                                 
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import json
import os
from datetime import datetime
from datetime import date
import random

def create_mask(mask_image):
    width, height = mask_image.size

    mask = Image.new('1', (width+2, height+2))

    for x in range(width):
        for y in range(height):
            alpha = mask_image.getpixel((x,y))[3]

            # If the pixel is not transparent
            if alpha == 255:
                # Set the pixel value to 1 (default is 0), accounting for padding
                mask.putpixel((x+1, y+1), 1)

    return mask

def create_mask_annotation(mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=True)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def main():
    backgrounds = 100 - 8 + 1
    positions_per_model = 11
    width = 1024
    height = 576

    train_perc = 1
    
    is_crowd = 0
    image_id_train = 1
    annotation_id_train = 1
    image_id_test = 1
    annotation_id_test = 1

    main_path = "new_images"
    list_categories = os.listdir(main_path)
    list_categories.sort()

    annotations_train = []
    annotations_test = []
    images_train = []
    images_test = []
    categories = []

    for category_id, category in enumerate(list_categories, 1):
        category_dict = {"supercategory": "pieza", "id": category_id, "name": category}
        categories.append(category_dict)
        for pos_model in range(positions_per_model):
            for background in range(backgrounds):
                image_name = "img" + str(pos_model) + "_background" + str(background) + ".png"
                print(image_name)
                    
                img = Image.open(main_path + "/" + category + "/" + image_name)       
                
                mask = create_mask(img)
                annotation = create_mask_annotation(mask, image_id_train, category_id, annotation_id_train, is_crowd)


                image_name = "img" + str(pos_model) + "_background" + str(background) + "_w.png"
                print(category + "/" + image_name)

                if random.random() < train_perc:
                    image_dict = {"id": image_id_train, "license": 1, "coco_url": "None", "flickr_url": "None", "width": width, "height": height, "file_name": category + "/" + image_name, "date_captured": str(datetime.now())}
                
                    annotation["image_id"] = image_id_train
                    annotation["id"] = annotation_id_train

                    images_train.append(image_dict)
                    annotations_train.append(annotation)

                    image_id_train += 1
                    annotation_id_train += 1
                else:
                    image_dict = {"id": image_id_test, "license": 1, "coco_url": "None", "flickr_url": "None", "width": width, "height": height, "file_name": category + "/" + image_name, "date_captured": str(datetime.now())}
                
                    annotation["image_id"] = image_id_test
                    annotation["id"] = annotation_id_test

                    images_test.append(image_dict)
                    annotations_test.append(annotation)

                    image_id_test += 1
                    annotation_id_test += 1
                
    coco_dataset_train = {}
    coco_dataset_test = {}

    coco_dataset_train["info"] = {"description": "FONDECYT DATASET UCSP", "url": "None", "version": "1.0", "year": str(datetime.now().year), "contributor": "Team Fondecyt UCSP", "date_created": str(date.today())}
    coco_dataset_train["licenses"] = [{"url": "http://creativecommons.org/licenses/by/2.0/", "id": 1, "name": "Attribution License"}]
    coco_dataset_train["images"] = images_train
    coco_dataset_train["annotations"] = annotations_train
    coco_dataset_train["categories"] = categories

    with open('coco_dataset_train.json', 'w', encoding='utf-8') as f:
        json.dump(coco_dataset_train, f, ensure_ascii=False, indent=4)

    coco_dataset_test["info"] = {"description": "FONDECYT DATASET UCSP", "url": "None", "version": "1.0", "year": str(datetime.now().year), "contributor": "Team Fondecyt UCSP", "date_created": str(date.today())}
    coco_dataset_test["licenses"] = [{"url": "http://creativecommons.org/licenses/by/2.0/", "id": 1, "name": "Attribution License"}]
    coco_dataset_test["images"] = images_test
    coco_dataset_test["annotations"] = annotations_test
    coco_dataset_test["categories"] = categories

    with open('coco_dataset_test.json', 'w', encoding='utf-8') as f:
        json.dump(coco_dataset_test, f, ensure_ascii=False, indent=4)

main()