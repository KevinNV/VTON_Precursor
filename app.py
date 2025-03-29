# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 21:57:07 2025

@author: kevin
"""

import gradio as gr
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import OrderedDict
from sklearn.cluster import KMeans
from transforms import transform_logits, get_affine_transform
from networks.AugmentCE2P import resnet101

# ----------------------------- CONFIG -----------------------------
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': [
            'Background','Hat','Hair','Glove','Sunglasses','Upper-clothes',
            'Dress','Coat','Socks','Pants','Jumpsuits','Scarf','Skirt',
            'Face','Left-arm','Right-arm','Left-leg','Right-leg','Left-shoe','Right-shoe'
        ]
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': [
            'Background','Head','Torso','Upper Arms','Lower Arms','Upper Legs','Lower Legs'
        ]
    }
}

# ----------------------------- HELPERS -----------------------------
def make_transform(input_size):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
    ])

def load_schp_model(model_path, dataset='lip'):
    num_classes = dataset_settings[dataset]['num_classes']
    model = resnet101(num_classes=num_classes, pretrained=None)
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_state[k.replace('module.', '')] = v
    model.load_state_dict(new_state, strict=False)
    model.cuda()
    model.eval()
    return model

# ----------------------------- INFERENCE -----------------------------
def schp_infer(model, bgr_image, dataset='lip'):
    input_size = dataset_settings[dataset]['input_size']
    aspect_ratio = float(input_size[1]) / input_size[0]
    H, W = bgr_image.shape[:2]
    center = np.array([(W - 1)/2, (H - 1)/2], dtype=np.float32)
    new_w, new_h = W, H
    if new_w > aspect_ratio * new_h:
        new_h = new_w / aspect_ratio
    else:
        new_w = new_h * aspect_ratio
    scale = np.array([new_w, new_h], dtype=np.float32)

    trans = get_affine_transform(center, scale, 0, input_size)
    warped_img = cv2.warpAffine(
        bgr_image, trans, (input_size[1], input_size[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    tform = make_transform(input_size)
    rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    tensor = tform(Image.fromarray(rgb)).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs[0][-1][0].unsqueeze(0)
        upsample = torch.nn.Upsample(size=tuple(input_size), mode='bilinear', align_corners=True)
        upsample_output = upsample(logits).squeeze(0).permute(1,2,0).cpu().numpy()

    logits_result = transform_logits(upsample_output, center, scale, W, H, input_size)
    parsing_map = np.argmax(logits_result, axis=2).astype(np.uint8)
    return parsing_map

# ----------------------------- CATEGORY & CORRECTION -----------------------------
def color_similarity(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def get_dominant_colors(img_region, n_clusters=2):
    if img_region.size == 0:
        return []

    pixels = img_region.reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_clusters:
        return []

    pixels = img_region.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(pixels)
    return kmeans.cluster_centers_

def correct_body_confusion_kmeans(person_img, parsing_map):
    corrected = parsing_map.copy()
    
    CONFUSION_PAIRS = {
       9: [16, 17],       # Pants vs Legs
      # 5: [14, 15],       # Shirt vs Arms
    }

    for garment_label, body_labels in CONFUSION_PAIRS.items():
        garment_mask = (parsing_map == garment_label)
        body_mask = np.isin(parsing_map, body_labels)

        garment_colors = get_dominant_colors(person_img[garment_mask], n_clusters=5)
        body_colors = get_dominant_colors(person_img[body_mask], n_clusters=5)

        if len(garment_colors) == 0 or len(body_colors) == 0:
            continue

        sim_scores = []
        for gc in garment_colors:
            for bc in body_colors:
                sim_scores.append(color_similarity(gc, bc))

        avg_sim = np.mean(sim_scores)
        print(f"Avg color diff ({garment_label} vs {body_labels}):", round(avg_sim, 2))

        if avg_sim > 50:
            print(f"Heuristic: Reassigning {body_labels} to {garment_label}")
            for lbl in body_labels:
                corrected[corrected == lbl] = garment_label

    return corrected


def deduce_garment_category_lip(garment_map):
    from collections import Counter
    ctr = Counter(garment_map.flatten())
    def c(lbl): return ctr.get(lbl, 0)
    label_candidates = {
        'top': c(5) + c(7),
        'bottom': c(9) + c(12),
        'full': c(6) + c(10),
        'shoes': c(18) + c(19),
        'socks': c(8) + c(18) + c(19),
        'hat': c(1) + c(2),
        'scarf': c(11),
        'glove': c(3),
        'sunglasses': c(4)
    }
    cat, size = max(label_candidates.items(), key=lambda x: x[1])
    return 'unknown' if size < 10 else cat

def get_remove_labels(category):
    if category == 'full': return [5,6,7,9,10,12]
    elif category == 'top': return [5,7]
    elif category == 'bottom': return [9,12]
    elif category == 'shoes': return [18,19]
    elif category == 'socks': return [8,18,19]
    elif category == 'hat': return [1,2]
    elif category == 'scarf': return [11]
    elif category == 'glove': return [3]
    elif category == 'sunglasses': return [4]
    return []

def fill_pascal_regions_with_overlap(pascal_map, coverage_mask, person_lip_map, target_labels=[2,3,4,5,6], threshold=0.45):
    filled = coverage_mask.copy()
    for lbl in target_labels:
        part_mask = (pascal_map == lbl).astype(np.uint8)
        if part_mask.sum() == 0:
            continue
        overlap = np.logical_and(part_mask, coverage_mask > 0).astype(np.uint8)
        if overlap.sum() / part_mask.sum() < threshold:
            continue
        
        if lbl == 6:
            shoe_mask = np.isin(person_lip_map, [18, 19]).astype(np.uint8)
            shoes_overlap = np.logical_and(shoe_mask, coverage_mask > 0).sum()
            shoe_ratio = shoes_overlap / part_mask.sum()
            if shoe_ratio < 0.1:  # Only fill if at least 10% of the region has shoe pixels
                continue
            
        if lbl == 4:  # Lower Arms
            glove_mask = (person_lip_map == 3).astype(np.uint8)
            gloves_on_arm = np.logical_and(glove_mask, coverage_mask > 0).sum()
            glove_ratio = gloves_on_arm / part_mask.sum()
            if glove_ratio < 0.1:
                continue

        filled[part_mask > 0] = 255
        
    return filled

# ----------------------------- MAIN FUNCTION -----------------------------
lip_model = load_schp_model('models/exp-schp-201908261155-lip.pth', dataset='lip')
pascal_model = load_schp_model('models/exp-schp-201908270938-pascal-person-part.pth', dataset='pascal')

def process_images(person_img_pil, garment_img_pil):
    person_img = cv2.cvtColor(np.array(person_img_pil), cv2.COLOR_RGB2BGR)
    garment_img = cv2.cvtColor(np.array(garment_img_pil), cv2.COLOR_RGB2BGR)

    garment_map = schp_infer(lip_model, garment_img, dataset='lip')
    garment_cat = deduce_garment_category_lip(garment_map)

    person_lip_map = schp_infer(lip_model, person_img, dataset='lip')
    person_lip_map = correct_body_confusion_kmeans(person_img, person_lip_map)

    person_pascal_map = schp_infer(pascal_model, person_img, dataset='pascal')

    labels_to_remove = get_remove_labels(garment_cat)
    coverage_mask = np.isin(person_lip_map, labels_to_remove).astype(np.uint8) * 255
    coverage_mask = fill_pascal_regions_with_overlap(person_pascal_map, coverage_mask, person_lip_map)

    result = person_img.copy()
    result[coverage_mask > 0] = (128, 128, 128)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)

# ----------------------------- GRADIO UI -----------------------------
demo = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="pil", label="Person Image"),
        gr.Image(type="pil", label="Garment Image")
    ],
    outputs=gr.Image(type="pil", label="Masked Output"),
    title="Garment Overlay Mask Generator",
    description="Upload a person and garment image to generate a garment area mask on the person image"
)

if __name__ == "__main__":
    demo.launch()
