from pathlib import Path
import io
import os
import tempfile
import uuid

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GaSe Flake Analyzer",
    page_icon="🔬",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

YOLO_WEIGHTS = MODEL_DIR / "best.pt"
APPEARANCE_MODEL_PATH = (
    MODEL_DIR
    / "gase_appearance_svm_175_trainval.joblib"
)

# Keep these aligned with the notebook.
YOLO_IMGSZ = 640
YOLO_CONF = 0.15
YOLO_IOU = 0.50
YOLO_MAX_DET = 50

MIN_MASK_AREA = 20
MIN_CLASS_PROB = 0.40
MIN_CLASS_MARGIN = 0.03

CLASS_NAMES = {
    0: "Class 1",
    1: "Class 2",
    2: "Class 3",
    3: "Class 4",
}

# Same notebook display colors.
# OpenCV uses BGR.
CLASS_COLORS_BGR = {
    0: (255, 255, 255),  # Class 1 = white
    1: (255, 0, 0),      # Class 2 = blue
    2: (0, 255, 0),      # Class 3 = green
    3: (0, 0, 255),      # Class 4 = red
}


# ============================================================
# 2. LOAD TRAINED MODELS ONCE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            "YOLO weights were not found.\n\n"
            "Expected:\n"
            f"{YOLO_WEIGHTS}"
        )

    if not APPEARANCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Appearance classifier was not found.\n\n"
            "Expected:\n"
            f"{APPEARANCE_MODEL_PATH}"
        )

    yolo = YOLO(str(YOLO_WEIGHTS))
    appearance = joblib.load(
        APPEARANCE_MODEL_PATH
    )

    return yolo, appearance


try:
    yolo_model, appearance_model = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# 3. EXACT NOTEBOOK FEATURE EXTRACTION
# ============================================================

def safe_stats(arr):
    arr = arr.astype(np.float32)

    mean = arr.reshape(
        -1,
        arr.shape[-1]
    ).mean(axis=0)

    std = arr.reshape(
        -1,
        arr.shape[-1]
    ).std(axis=0)

    return mean, std


def feature_from_bgr_pixels(pixels_bgr):
    """
    Same 24-feature representation used by the notebook:

    RGB mean/std          = 6
    HSV mean/std          = 6
    Lab mean/std          = 6
    Chromaticity mean/std = 6
    """

    pixels_bgr = (
        pixels_bgr
        .reshape(-1, 1, 3)
        .astype(np.uint8)
    )

    rgb = (
        pixels_bgr[..., ::-1]
        .astype(np.float32)
    )

    hsv = cv2.cvtColor(
        pixels_bgr,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    lab = cv2.cvtColor(
        pixels_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    denom = np.maximum(
        rgb.sum(
            axis=2,
            keepdims=True
        ),
        1.0
    )

    chroma = rgb / denom

    rm, rs = safe_stats(rgb)
    hm, hs = safe_stats(hsv)
    lm, ls = safe_stats(lab)
    cm, cs = safe_stats(chroma)

    features = np.concatenate([
        rm, rs,
        hm, hs,
        lm, ls,
        cm, cs,
    ])

    return features.astype(
        np.float32
    )


# ============================================================
# 4. YOLO INFERENCE
# ============================================================

def yolo_candidates(
    image_bgr
):
    """
    Run the trained one-class YOLOv8n-seg model.
    """

    result = yolo_model.predict(
        image_bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
        verbose=False,
        retina_masks=True,
        device="cpu",
    )

    raw = []

    if (
        not result
        or result[0].masks is None
    ):
        return raw

    r = result[0]

    polygons = r.masks.xy

    if r.boxes is not None:
        confidences = (
            r.boxes.conf
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
    else:
        confidences = (
            [0.0] * len(polygons)
        )

    h, w = image_bgr.shape[:2]

    for idx, polygon in enumerate(
        polygons
    ):

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        points = (
            np.round(polygon)
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )

        if len(points) < 3:
            continue

        cv2.fillPoly(
            mask,
            [points],
            1
        )

        area = int(
            mask.sum()
        )

        if area < MIN_MASK_AREA:
            continue

        raw.append({
            "det_index": idx,
            "mask": mask,
            "area_px": area,
            "yolo_confidence": (
                float(confidences[idx])
                if idx < len(confidences)
                else 0.0
            ),
        })

    return raw


# ============================================================
# 5. NOTEBOOK MASK DEDUPLICATION
# ============================================================

def mask_iou(
    mask_a,
    mask_b
):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    intersection = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    return (
        float(intersection / union)
        if union
        else 0.0
    )


def containment_ratio(
    mask_a,
    mask_b
):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    denom = a.sum()

    return (
        float(
            np.logical_and(a, b).sum()
            / denom
        )
        if denom
        else 0.0
    )


def deduplicate_mask_candidates(
    candidates,
    iou_thr=0.60,
    containment_thr=0.80
):
    """
    Same logic as the notebook's main inference path.
    """

    ordered = sorted(
        candidates,
        key=lambda d: d["yolo_confidence"],
        reverse=True
    )

    kept = []

    for candidate in ordered:

        duplicate = False

        for previous in kept:

            iou = mask_iou(
                candidate["mask"],
                previous["mask"]
            )

            candidate_is_smaller = (
                candidate["area_px"]
                <= previous["area_px"]
            )

            containment = (
                containment_ratio(
                    candidate["mask"],
                    previous["mask"]
                )
                if candidate_is_smaller
                else
                containment_ratio(
                    previous["mask"],
                    candidate["mask"]
                )
            )

            if (
                iou >= iou_thr
                or
                containment >= containment_thr
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(
                candidate
            )

    return kept


# ============================================================
# 6. FEATURE EXTRACTION FROM DETECTED MASK
# ============================================================

def feature_from_mask(
    image_bgr,
    mask
):
    m = (
        mask.astype(np.uint8) > 0
    ).astype(np.uint8)

    area = int(
        m.sum()
    )

    if area == 0:
        return None, 0

    if area >= 25:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        eroded = cv2.erode(
            m,
            kernel,
            iterations=1
        )

        if eroded.sum() >= max(
            8,
            int(0.20 * area)
        ):
            m = eroded

    pixels = image_bgr[
        m.astype(bool)
    ]

    if len(pixels) < MIN_MASK_AREA:
        return None, 0

    features = feature_from_bgr_pixels(
        pixels
    )

    return (
        features,
        int(m.sum())
    )


# ============================================================
# 7. CLASSIFICATION
# ============================================================

def classify_mask(
    image_bgr,
    mask,
    yolo_confidence
):
    features, area = feature_from_mask(
        image_bgr,
        mask
    )

    if (
        features is None
        or area < MIN_MASK_AREA
    ):
        return None

    probabilities = (
        appearance_model
        .predict_proba(
            features.reshape(1, -1)
        )[0]
    )

    order = np.argsort(
        probabilities
    )[::-1]

    best = int(
        order[0]
    )

    second = int(
        order[1]
    )

    p1 = float(
        probabilities[best]
    )

    p2 = float(
        probabilities[second]
    )

    margin = (
        p1 - p2
    )

    accepted = (
        p1 >= MIN_CLASS_PROB
        and
        margin >= MIN_CLASS_MARGIN
    )

    return {
        "class_id": best,
        "class_name": CLASS_NAMES[best],
        "probability": p1,
        "second_probability": p2,
        "margin": margin,
        "accepted": accepted,
        "area_px": area,
        "yolo_confidence": float(
            yolo_confidence
        ),
        "mask": mask.copy(),
    }


# ============================================================
# 8. DRAWING
# ============================================================

def text_fit(
    image,
    text,
    org,
    font_scale=0.45,
    color=(20, 20, 20),
    thickness=1
):
    cv2.putText(
        image,
        str(text),
        (
            int(org[0]),
            int(org[1])
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_perimeter(
    image,
    mask,
    class_id,
    thickness=2
):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:

        if cv2.contourArea(
            contour
        ) < MIN_MASK_AREA:
            continue

        cv2.polylines(
            image,
            [contour],
            True,
            CLASS_COLORS_BGR[
                int(class_id)
            ],
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def add_number_marker(
    image,
    centroid,
    number
):
    cx, cy = centroid

    cv2.circle(
        image,
        (int(cx), int(cy)),
        7,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        image,
        (int(cx), int(cy)),
        7,
        (30, 30, 30),
        1,
    )

    text_fit(
        image,
        number,
        (
            int(cx) - 3,
            int(cy) + 4
        ),
        font_scale=0.32,
        color=(20, 20, 20),
        thickness=1,
    )


def build_result_image(
    original,
    detections
):
    """
    Create the downloadable image in the same three-panel
    format used by the project:

        Original microscope image
        Prediction
        Prediction details
    """

    h, w = original.shape[:2]

    HEADER_H = 30
    GAP = 18
    PANEL_W = 245

    total_h = max(
        HEADER_H + h,
        HEADER_H + 42 * max(
            len(detections),
            1
        ) + 40
    )

    total_w = (
        w
        + GAP
        + w
        + GAP
        + PANEL_W
    )

    canvas = np.full(
        (
            total_h,
            total_w,
            3
        ),
        248,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    canvas[
        HEADER_H:
        HEADER_H + h,
        0:w
    ] = original

    # --------------------------------------------------------
    # Prediction image
    # --------------------------------------------------------

    prediction = original.copy()

    ordered = sorted(
        detections,
        key=lambda d: (
            d["centroid"][1],
            d["centroid"][0]
        )
    )

    for det in ordered:

        draw_perimeter(
            prediction,
            det["mask"],
            det["class_id"],
            thickness=2,
        )

    for idx, det in enumerate(
        ordered,
        start=1
    ):
        add_number_marker(
            prediction,
            det["centroid"],
            str(idx),
        )

    seg_x = (
        w + GAP
    )

    canvas[
        HEADER_H:
        HEADER_H + h,
        seg_x:
        seg_x + w
    ] = prediction

    # --------------------------------------------------------
    # Headers
    # --------------------------------------------------------

    text_fit(
        canvas,
        "Original microscope image",
        (10, 20),
        font_scale=0.46,
        color=(35, 35, 35),
    )

    text_fit(
        canvas,
        "Prediction",
        (seg_x + 6, 20),
        font_scale=0.46,
        color=(35, 35, 35),
    )

    # --------------------------------------------------------
    # Details panel
    # --------------------------------------------------------

    panel_x = (
        seg_x
        + w
        + GAP
    )

    panel = canvas[
        0:
        total_h,
        panel_x:
        panel_x + PANEL_W
    ]

    cv2.rectangle(
        panel,
        (0, 0),
        (
            PANEL_W - 1,
            total_h - 1
        ),
        (210, 210, 210),
        1,
    )

    text_fit(
        panel,
        "Prediction details",
        (12, 22),
        font_scale=0.50,
    )

    text_fit(
        panel,
        f"Accepted: {len(ordered)}",
        (12, 42),
        font_scale=0.36,
        color=(70, 70, 70),
    )

    if not ordered:

        text_fit(
            panel,
            "NO ACCEPTED",
            (12, 78),
            font_scale=0.43,
            color=(0, 0, 150),
        )

        text_fit(
            panel,
            "DETECTION",
            (12, 98),
            font_scale=0.43,
            color=(0, 0, 150),
        )

        text_fit(
            panel,
            "Manual review required.",
            (12, 124),
            font_scale=0.33,
            color=(80, 80, 80),
        )

        return canvas

    label_start_y = 78
    label_step = 48

    for idx, det in enumerate(
        ordered,
        start=1
    ):

        y = (
            label_start_y
            + (idx - 1)
            * label_step
        )

        class_id = int(
            det["class_id"]
        )

        color = CLASS_COLORS_BGR[
            class_id
        ]

        # Color swatch
        cv2.rectangle(
            panel,
            (12, y - 10),
            (32, y + 10),
            color,
            -1,
        )

        cv2.rectangle(
            panel,
            (12, y - 10),
            (32, y + 10),
            (30, 30, 30),
            1,
        )

        text_fit(
            panel,
            f"F{idx:02d}  {det['class_name']}",
            (42, y),
            font_scale=0.43,
        )

        text_fit(
            panel,
            f"p={det['probability']:.3f}",
            (42, y + 16),
            font_scale=0.31,
            color=(75, 75, 75),
        )

        text_fit(
            panel,
            f"YOLO={det['yolo_confidence']:.3f}",
            (42, y + 29),
            font_scale=0.30,
            color=(95, 95, 95),
        )

        cv2.line(
            panel,
            (12, y + 38),
            (PANEL_W - 12, y + 38),
            (220, 220, 220),
            1,
        )

    return canvas


# ============================================================
# 9. COMPLETE PREDICTION
# ============================================================

def predict_image(
    image_rgb
):
    image_rgb = np.ascontiguousarray(
        image_rgb
    ).astype(
        np.uint8
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    raw = yolo_candidates(
        image_bgr
    )

    kept = deduplicate_mask_candidates(
        raw
    )

    detections = []

    for candidate in kept:

        result = classify_mask(
            image_bgr,
            candidate["mask"],
            candidate["yolo_confidence"],
        )

        if result is None:
            continue

        if not result["accepted"]:
            continue

        ys, xs = np.where(
            candidate["mask"] > 0
        )

        if len(xs) == 0:
            continue

        result["centroid"] = (
            float(xs.mean()),
            float(ys.mean()),
        )

        detections.append(
            result
        )

    # Top-to-bottom, then left-to-right.
    detections.sort(
        key=lambda d: (
            d["centroid"][1],
            d["centroid"][0],
        )
    )

    output_bgr = build_result_image(
        image_bgr,
        detections
    )

    output_rgb = cv2.cvtColor(
        output_bgr,
        cv2.COLOR_BGR2RGB
    )

    output_rgb = np.ascontiguousarray(
        output_rgb
    )

    ok, encoded = cv2.imencode(
        ".png",
        output_bgr
    )

    if not ok:
        raise RuntimeError(
            "Could not encode result as PNG."
        )

    png_bytes = encoded.tobytes()

    return (
        output_rgb,
        png_bytes,
        detections,
        len(raw),
        len(kept),
    )


# ============================================================
# 10. USER INTERFACE
# ============================================================

st.title(
    "🔬 GaSe Flake Analyzer"
)

st.write(
    "Upload a GaSe microscope image to segment flakes "
    "and classify their optical appearance."
)

st.info(
    "Inference only — the deployed application does not "
    "train or modify the models."
)

uploaded_file = st.file_uploader(
    "Upload microscope image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
    ],
)

if uploaded_file is not None:

    try:

        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        input_array = np.array(
            pil_image
        )

        st.subheader(
            "Input"
        )

        st.image(
            input_array,
            width="stretch"
        )

        if st.button(
            "Predict",
            type="primary",
        ):

            with st.spinner(
                "Running YOLOv8n + appearance classifier..."
            ):

                (
                    output_image,
                    png_bytes,
                    detections,
                    raw_count,
                    kept_count,
                ) = predict_image(
                    input_array
                )

            st.subheader(
                "Prediction"
            )

            st.image(
                output_image,
                width="stretch"
            )

            st.download_button(
                label="⬇️ Download prediction PNG",
                data=png_bytes,
                file_name=(
                    "gase_flake_prediction.png"
                ),
                mime="image/png",
            )

            st.subheader(
                "Prediction details"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Raw YOLO candidates",
                raw_count,
            )

            col2.metric(
                "After deduplication",
                kept_count,
            )

            col3.metric(
                "Accepted flakes",
                len(detections),
            )

            if detections:

                rows = []

                for i, det in enumerate(
                    detections,
                    start=1
                ):
                    rows.append({
                        "Flake":
                            f"F{i:02d}",
                        "Class":
                            det["class_name"],
                        "Probability":
                            f"{det['probability']:.3f}",
                        "YOLO confidence":
                            f"{det['yolo_confidence']:.3f}",
                    })

                st.table(
                    rows
                )

            else:

                st.warning(
                    "No flake passed the current "
                    "acceptance thresholds."
                )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

else:

    st.caption(
        "Supported image formats: JPG, JPEG, PNG, BMP, TIFF."
    )
