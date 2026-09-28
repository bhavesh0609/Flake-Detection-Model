# GaSe Flake Analyzer — Free Streamlit Deployment

This deploys the existing GaSe inference pipeline as a browser application.

## Pipeline

```text
Upload image
    ↓
YOLOv8n-seg
    ↓
Mask deduplication
    ↓
24 optical features
    ↓
Saved StandardScaler + RBF-SVM
    ↓
Class 1–4
    ↓
Professional prediction image
    ↓
Download PNG
```

## 1. Required model files

Put your trained files here:

```text
models/
├── best.pt
└── gase_appearance_svm_175_trainval.joblib
```

`best.pt` must be the trained **YOLOv8n-seg** weights used by your notebook.

The SVM file must be the saved model from:

```text
gase_pipeline_output_175/appearance_model/
gase_appearance_svm_175_trainval.joblib
```

## 2. Test locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 3. Free public deployment

Streamlit Community Cloud is a free hosted option.

1. Create a GitHub repository.
2. Put these files in it:

```text
app.py
requirements.txt
README.md
models/best.pt
models/gase_appearance_svm_175_trainval.joblib
```

3. Go to:

```text
https://share.streamlit.io
```

4. Sign in with GitHub.
5. Choose the repository and `app.py`.
6. Deploy.

Your app will receive a public `streamlit.app` URL.

## 4. Website behavior

The user only needs to:

```text
Open website
    ↓
Upload image
    ↓
Click Predict
    ↓
View result
    ↓
Download prediction PNG
```

No training is performed on the website.

## 5. Important

This app uses CPU inference so the deployment remains on the free Streamlit Community Cloud tier.

Because YOLOv8n is still a neural-network model, inference may take a few seconds on CPU. That is the tradeoff for a completely free hosted deployment.

For your 200×200 microscopy images, the application keeps the notebook's 640-pixel YOLO inference setting for consistency.

## 6. Updating the model

When a better trained YOLO model is available, replace:

```text
models/best.pt
```

with the new trained weights.

When a new appearance classifier is available, replace:

```text
models/gase_appearance_svm_175_trainval.joblib
```

The website code does not need to change.

## 7. Output

The generated result is a PNG containing:

```text
Original microscope image | Prediction | Prediction details
```

The same PNG is displayed in the browser and offered through the download button.
