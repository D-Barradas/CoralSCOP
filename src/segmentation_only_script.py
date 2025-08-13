# Segmentation-only script derived from segmetation_ocr_script.py
import sys
sys.path.append('./manipulation_library.py')
from manipulation_library import *
import os
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

# Segmentation utility functions

def get_center(bbox):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return np.array([x_center, y_center])

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def find_close_bounding_boxes(seg_bboxes, threshold):
    # Dummy function for compatibility, not used without OCR
    return [], []

def find_unpaired_seg_bboxes(seg_bboxes, close_bboxes_indices):
    paired_seg_indices = {pair[0] for pair in close_bboxes_indices}
    unpaired_seg_indices = [idx for idx in range(len(seg_bboxes)) if idx not in paired_seg_indices]
    return unpaired_seg_indices

def use_sorted_mask(image, masks):
    cropped_image_list = []
    for i in range(len(masks)):
        x, y, width, height = masks[i]['bbox']
        image_b, masked_pixels = background_to_black(image=image, index=i , masks=masks)
        cropped_image = image_b[int(y):int(y+height), int(x):int(x+width)]
        cropped_image_list.append(cropped_image)
    return cropped_image_list

def process_images_and_use_sorted_mask(image, masks):
    cropped_image_list  = use_sorted_mask( image=image , masks=masks )
    return cropped_image_list

def get_user_input():
    parser = argparse.ArgumentParser(description="Process images and segment objects.")
    parser.add_argument('main_path', type=str, help='The main path to the images.')
    parser.add_argument('rotate_option', type=str, choices=['clockwise', 'counterclockwise', 'none'], help='Rotate image? (clockwise/counterclockwise/none)')
    try:
        args = parser.parse_args()
        return args.main_path, args.rotate_option
    except argparse.ArgumentError:
        parser.print_usage()
        sys.exit(1)

main_path, rotate_option = get_user_input()

image_files = glob.glob(os.path.join(main_path, '*'))

mask_generator = load_sam_model(model_type="vit_b")

for image_file in image_files:
    image = get_image(image_file)
    # Support various image extensions (e.g., .jpg, .jpeg, .png)
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    print (f'Processing image: {image_name}')

    output_dir = f'../data/interim/{image_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir ,exist_ok=True)

    if rotate_option == 'clockwise':
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_option == 'counterclockwise':
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # else: no rotation

    masks = mask_generator.generate(image=image)

    seg_bboxes = []
    for i in range(len(masks)):
        x, y, width, height = masks[i]['bbox']
        seg_bboxes.append ( np.array([x, y, x+width, y+height]) )

    # Plot segmentation bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    colors = plt.get_cmap('tab20', len(seg_bboxes))
    for idx, seg_bbox in enumerate(seg_bboxes):
        color = colors(idx)
        rect_seg = patches.Rectangle((seg_bbox[0], seg_bbox[1]), seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1], linewidth=2, edgecolor=color, facecolor='none', linestyle='dashed', label=f'Seg {idx}')
        ax.add_patch(rect_seg)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.savefig(f'{output_dir}/bounding_boxes_{image_name}.png')

    # Save cropped mask images
    cropped_image_list = process_images_and_use_sorted_mask(image, masks)
    for idx, image_segment in enumerate(cropped_image_list):
        image_segment = cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{output_dir}/image_index_{idx}.jpg',image_segment)

    # # Save unpaired masks (all masks, since no OCR)
    # for idx, mask in enumerate(masks):
    #     x, y, width, height = mask['bbox']
    #     image_b, masked_pixels = background_to_black(image=image, index=idx , masks=masks)
    #     cropped_image = image_b[int(y):int(y+height), int(x):int(x+width)]
    #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(f'{output_dir}/image_unpaired_{idx}.jpg',cropped_image)
