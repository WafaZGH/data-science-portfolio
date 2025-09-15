# 🍝 GluciPred — From a plate photo to estimated carbs

Live app: https://data-science-portfolio-mkpsgolueu7crorkflsug7.streamlit.app/

API docs: https://data-jed-api-glucipred.hf.space/docs

Direct API endpoint: https://data-jed-api-glucipred.hf.space/predict/image

GluciPred turns a meal photo into an estimated carbohydrate count. It segments foods, converts mask areas into per-item weights, and computes carbs using CIQUAL and a curated Glycemic Index (GI) table. Goal: a fast, transparent estimate for day-to-day glucose management.

# What the app does

Upload a plate photo

Segment foods (plate/fork kept as context)

Convert mask share → grams (calibrated with empty-plate images)

Link labels to CIQUAL and GI; compute per-item and total carbs

Show segmented image, results table (label, %, grams, carbs, confidence) + CSV export

# Data used

FoodSeg103 plus our labeled images; JPG images with YOLO TXT masks

Empty-plate set for geometric calibration (pixels → plate dimensions)

CIQUAL nutrition table and a cleaned GI dataset (with an embedded variant for matching)

Train / validation / test splits

# Model overview

YOLOv8-Seg, 109 classes (including plate/fork)

Labels as YOLO segmentation TXT

Dataset YAML = the model’s “map” (paths to train/val/test, ordered class names)

# Weight estimation (pixels → grams)

Compute each food’s percent of the plate

Distribute a user-chosen total weight across foods by mask share

Use CIQUAL per-100 g values to derive carbs; add GI where available

# Repository tour

configs — dataset/training configuration (YAML: paths, class names, splits)

data/lookup — CIQUAL and GI tables

models — instructions and link to the trained checkpoint

notebooks — training and inference/demo

src/glucy_pred — inference, weight estimation, nutrition/GI loaders

streamlit — web UI calling the API, with a simple debug mode
