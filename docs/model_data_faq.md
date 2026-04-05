# Model & Data FAQ

## Was the model trained on the downloaded data in this repo?
No.

The current implementation uses a **pretrained YOLO model** for inference (`YOLO_MODEL_PATH`, default `yolov8n.pt`).
The downloaded open-source clip is used as **runtime input data** for demo/inference, not for training.

## Did we bring real data?
Yes.

The project includes an open-source data manifest and downloader:
- `data/metadata/open_source_video_sources.csv`
- `scripts/download_open_source_data.py`

That gives real video input for the pipeline.

## If training is required by interviewer
For this assignment scope, full custom training is usually optional unless explicitly requested.
If needed, add:
1. Labeled train/val split (bbox + activity interval labels)
2. Fine-tuning pipeline (`ultralytics` train config)
3. Evaluation metrics (mAP + activity accuracy/F1)
4. Model artifact versioning and inference model switch
