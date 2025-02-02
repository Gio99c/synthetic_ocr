import json
import os
import numpy as np
from difflib import SequenceMatcher
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# Define paths
DATASET_DIR = "synthetic_dataset"
RESULTS_DIR = "ocr_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Surya OCR models
detection_model, detection_processor = load_det_model(), load_det_processor()
recognition_model, recognition_processor = load_rec_model(), load_rec_processor()

# Function to compute Character Error Rate (CER)
def cer(gt, pred):
    return 1 - SequenceMatcher(None, gt, pred).ratio()

# Function to compute Word Error Rate (WER)
def wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    return 1 - SequenceMatcher(None, gt_words, pred_words).ratio()

# Function to compute Intersection over Union (IoU) for bounding boxes
def iou(boxA, boxB):
    if not boxA or not boxB:
        return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Iterate over the dataset
all_cer, all_wer, all_iou = [], [], []
image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".png")]

for img_file in image_files:
    img_path = os.path.join(DATASET_DIR, img_file)
    json_path = os.path.join(DATASET_DIR, img_file.replace(".png", ".json"))
    
    if not os.path.exists(json_path):
        continue
    
    # Load ground truth
    with open(json_path, "r") as f:
        annotation = json.load(f)
    gt_text = annotation["text"]
    gt_bbox = annotation["bbox"]
    
    # Read image and apply Surya OCR
    img = Image.open(img_path)
    ocr_results = run_ocr([img], [['en']], detection_model, detection_processor, recognition_model, recognition_processor)

    extracted_text = " ".join([line.text for line in ocr_results[0].text_lines]).strip()
    pred_bboxes = [line.bbox for line in ocr_results[0].text_lines]
    
    # Convert multiple bounding boxes to a single one (merge them)
    pred_bbox = [
        min([b[0] for b in pred_bboxes]),
        min([b[1] for b in pred_bboxes]),
        max([b[2] for b in pred_bboxes]),
        max([b[3] for b in pred_bboxes])
    ] if pred_bboxes else None
    
    # Calculate CER and WER
    text_cer = cer(gt_text, extracted_text)
    text_wer = wer(gt_text, extracted_text)
    
    # Calculate IoU for bounding boxes
    box_iou = iou(gt_bbox, pred_bbox)
    
    # Store results
    all_cer.append(text_cer)
    all_wer.append(text_wer)
    all_iou.append(box_iou)
    
    # Save results
    result_path = os.path.join(RESULTS_DIR, img_file.replace(".png", "_result.json"))
    with open(result_path, "w") as f:
        json.dump({"extracted_text": extracted_text, "CER": text_cer, "WER": text_wer, "IoU": box_iou}, f, indent=4)
    
    print(f"Processed {img_file}: CER={text_cer:.3f}, WER={text_wer:.3f}, IoU={box_iou:.3f}")

# Compute averages
print(f"Average CER: {np.mean(all_cer):.3f}")
print(f"Average WER: {np.mean(all_wer):.3f}")
print(f"Average IoU: {np.mean(all_iou):.3f}")
