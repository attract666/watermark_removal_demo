import cv2,torch,torch.nn,sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def merge_mask(mask1, mask2):
    merged_mask = np.maximum(mask1, mask2)
    return merged_mask

def SASOWAB_1(predictor):
    input_box = np.array([0, 0, 250, 100])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    color = np.array([0/255, 0/255, 0/255, 1])
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * color.reshape(1, 1, -1)

    white_color = np.array([1, 1, 1, 1])
    black_color = np.array([0, 0, 0, 1])
    mask_image[masks[0] > 0] = white_color
    mask_image[masks[0] == 0] = black_color

    print('one step finish')

    return mask_image

def SASOWAB_2(predictor):
    input_box = np.array([830, 600, 1080, 720])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    color = np.array([0/255, 0/255, 0/255, 1])
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * color.reshape(1, 1, -1)

    white_color = np.array([1, 1, 1, 1])
    black_color = np.array([0, 0, 0, 1])
    mask_image[masks[0] > 0] = white_color
    mask_image[masks[0] == 0] = black_color

    print('two step finish')

    return mask_image

def segment_frame(input_path, output_path):
    model_Path = '../segment-anything-main/checkpoint'
    models = {'h':['%s/sam_vit_h_4b8939.pth' % model_Path,'vit_h']}

    model = models['h']
    sam_checkpoint = model[0]
    model_type = model[1]
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # onr step
    image = cv2.imread(os.path.join(input_path,f'frame00299.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask1 = SASOWAB_1(predictor)

    # two step
    image = cv2.imread(os.path.join(input_path,f'frame00396.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask2 = SASOWAB_2(predictor)

    merged_mask = merge_mask(mask1, mask2)

    rgba_frame_mask = (merged_mask * 255).astype(np.uint8)
    Image.fromarray(rgba_frame_mask, 'RGBA').save(os.path.join(output_path, f'frame_mask.png'))


input_path = 'your_picture_folder'
output_path = 'frame_mask_folder'
segment_frame(input_path, output_path)
