
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image, ImageOps
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mask_detector.h5"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Mask", "No Mask"]

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="FaceGuard | AI Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# PREMIUM BALANCED THEME
# ==================================================

st.markdown("""
<style>

.stApp {
    background: #cbd6e2 !important;
    color: #142d46 !important;
}

[data-testid="stMain"] {
    background: #cbd6e2 !important;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp label,
.stApp strong,
.stApp li {
    color: #142d46;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #142d46 !important;
    border-right: 1px solid #34516d;
}

[data-testid="stSidebar"] * {
    color: #edf4fb !important;
}

[data-testid="stSidebar"] hr {
    border-color: #46617a !important;
}

/* HERO */

.hero {
    background: linear-gradient(120deg, #142d46, #245b7a);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 7px 20px rgba(20,45,70,.18);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 34px;
    font-weight: 750;
    margin: 0 0 10px 0;
}

.hero p {
    color: #e4edf6 !important;
    font-size: 15px;
    margin: 0;
}

/* SECTION HEADINGS */

.section-title {
    color: #142d46 !important;
    font-size: 23px;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 13px;
}

/* UPLOADER */

[data-testid="stFileUploader"] {
    background: #e4ebf3 !important;
    border: 2px dashed #6485a4 !important;
    border-radius: 15px !important;
    padding: 18px !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div {
    color: #142d46 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button {
    background: #142d46 !important;
    color: #ffffff !important;
    border: 1px solid #142d46 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

/* BUTTONS */

.stButton button {
    background: #245b7a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: #e4ebf3 !important;
    border: 1px solid #9aafc2 !important;
    border-radius: 13px !important;
    padding: 17px !important;
    box-shadow: 0 3px 10px rgba(20,45,70,.07);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #263e55 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #142d46 !important;
    font-weight: 800 !important;
}

/* RESULT CONTAINERS */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #e4ebf3 !important;
    border: 1px solid #9aafc2 !important;
    border-radius: 14px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3,
[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] strong {
    color: #142d46 !important;
}

/* PROGRESS BAR */

[data-testid="stProgress"] {
    background: #b8c7d6 !important;
    border-radius: 10px !important;
}

[data-testid="stProgress"] > div > div {
    background: #245b7a !important;
    border-radius: 10px !important;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    font-weight: 700 !important;
}

/* CAPTIONS */

[data-testid="stCaptionContainer"] p {
    color: #405970 !important;
}

/* DIVIDERS */

hr {
    border-color: #aebfd0 !important;
}

/* LIGHT COLOR SCHEME */

:root {
    color-scheme: light !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    color-scheme: light !important;
}

/* FOOTER */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_mask_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# ==================================================
# LOAD FACE DETECTOR
# ==================================================

@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if detector.empty():
        raise RuntimeError(
            "Could not load Haar Cascade."
        )

    return detector


# ==================================================
# FACE DETECTION
# ==================================================

def detect_faces(image_rgb, detector):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    enhanced_gray = cv2.equalizeHist(gray)

    attempts = [
        (gray, 2),
        (enhanced_gray, 2),
        (gray, 1)
    ]

    for test_image, neighbors in attempts:

        faces = detector.detectMultiScale(
            test_image,
            scaleFactor=1.05,
            minNeighbors=neighbors,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            return faces

    return []


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

def preprocess_face(face_rgb):

    face = cv2.resize(
        face_rgb,
        IMG_SIZE
    )

    face = face.astype("float32")
    face = preprocess_input(face)

    return np.expand_dims(
        face,
        axis=0
    )


# ==================================================
# LIVE WEBCAM PROCESSOR
# ==================================================

class FaceMaskProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_mask_model()
        self.detector = load_face_detector()

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror webcam
        img = cv2.flip(img, 1)

        # Face detection
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40)
        )

        mask_count = 0
        no_mask_count = 0

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            if face.size == 0:
                continue

            # BGR to RGB
            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # Correct MobileNetV2 preprocessing
            face_input = preprocess_face(
                face_rgb
            )

            prediction = self.model.predict(
                face_input,
                verbose=0
            )[0]

            predicted_class = int(
                np.argmax(prediction)
            )

            confidence = (
                float(prediction[predicted_class]) * 100
            )

            label = CLASS_NAMES[predicted_class]

            if predicted_class == 0:

                color = (74, 131, 22)
                mask_count += 1

            else:

                color = (72, 61, 212)
                no_mask_count += 1

            text = f"{label.upper()} | {confidence:.1f}%"

            # Bounding box
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                color,
                3
            )

            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness
            )

            label_y = max(y - 10, text_h + 15)

            cv2.rectangle(
                img,
                (x, label_y - text_h - 10),
                (x + text_w + 12, label_y + baseline),
                color,
                -1
            )

            cv2.putText(
                img,
                text,
                (x + 6, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # Live status
        cv2.putText(
            img,
            f"Faces: {len(faces)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 100, 180),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 😷 FaceGuard")

    st.caption(
        "AI-Powered Face Mask Detection"
    )

    st.divider()

    st.markdown("### Project Information")

    st.write("**Model:** MobileNetV2")
    st.write("**Framework:** TensorFlow")
    st.write("**Image Processing:** OpenCV")
    st.write("**Interface:** Streamlit")

    st.divider()

    st.markdown("### Model Classes")

    st.markdown("🟢 Mask")
    st.markdown("🔴 No Mask")

    st.divider()

    st.markdown("### Detection Workflow")

    st.write("1. Upload Image")
    st.write("2. Detect Faces")
    st.write("3. Predict Mask Status")
    st.write("4. Live Webcam Detection")
    st.write("5. View Results")

    st.divider()

    st.caption("FaceGuard | AI Project")


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>😷 Face Mask Detection System</h1>
    <p>
        Intelligent face detection and mask classification
        powered by Deep Learning and Computer Vision.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# IMAGE UPLOAD
# ==================================================

st.markdown(
    '<div class="section-title">📤 Upload Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    key="faceguard_upload"
)


# ==================================================
# HOME SCREEN
# ==================================================

if uploaded_file is None:

    st.info(
        "Upload an image to start face mask detection."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "MobileNetV2")
    col2.metric("Classes", "2")
    col3.metric("Input Size", "224 × 224")

    st.markdown("""
    <div class="result-card">
        <h3>🚀 How FaceGuard Works</h3>
        <p>📤 Upload an image.</p>
        <p>🔍 Detect faces using OpenCV.</p>
        <p>🧠 Classify every detected face using MobileNetV2.</p>
        <p>📊 View prediction and confidence scores.</p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# IMAGE PROCESSING
# ==================================================

else:

    image_rgb = None

    try:

        # Read uploaded image
        image = Image.open(uploaded_file)

        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        image_rgb = np.array(image)

        # Show original image immediately
        st.markdown(
            '<div class="section-title">🖼️ Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image_rgb,
            caption="Original Uploaded Image",
            use_container_width=True,
            channels="RGB",
            output_format="PNG"
        )

        # Load model and face detector
        detector = load_face_detector()
        model = load_mask_model()

        # Detect faces
        faces = detect_faces(
            image_rgb,
            detector
        )

        # Detection overview
        st.markdown(
            '<div class="section-title">📊 Detection Overview</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("Faces Detected", len(faces))
        metric2.metric("Image Width", f"{image_rgb.shape[1]} px")
        metric3.metric("Image Height", f"{image_rgb.shape[0]} px")

        # Always create result image
        result_image = image_rgb.copy()
        results = []

        if len(faces) == 0:

            st.warning(
                "No faces detected. Try a clear front-facing image."
            )

        else:

            for index, (x, y, w, h) in enumerate(faces):

                face_rgb = image_rgb[y:y+h, x:x+w]

                if face_rgb.size == 0:
                    continue

                face_input = preprocess_face(face_rgb)

                prediction = model.predict(
                    face_input,
                    verbose=0
                )[0]

                predicted_class = int(
                    np.argmax(prediction)
                )

                confidence = (
                    float(prediction[predicted_class]) * 100
                )

                label = CLASS_NAMES[predicted_class]

                mask_score = float(prediction[0]) * 100
                no_mask_score = float(prediction[1]) * 100

                color = (
                    (22, 131, 74)
                    if predicted_class == 0
                    else (212, 61, 72)
                )

                # Draw bounding box
                cv2.rectangle(
                    result_image,
                    (x, y),
                    (x+w, y+h),
                    color,
                    3
                )

                # Label
                label_text = (
                    f"{label.upper()} | {confidence:.2f}%"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2

                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text,
                    font,
                    font_scale,
                    thickness
                )

                label_y = y - 12

                if label_y - text_h - 12 < 0:
                    label_y = y + text_h + 18

                cv2.rectangle(
                    result_image,
                    (x, label_y - text_h - 12),
                    (x + text_w + 16, label_y + baseline - 5),
                    color,
                    -1
                )

                cv2.putText(
                    result_image,
                    label_text,
                    (x + 8, label_y - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

                results.append({
                    "face": index + 1,
                    "label": label,
                    "confidence": confidence,
                    "mask_score": mask_score,
                    "no_mask_score": no_mask_score
                })

        # Classification results
        if results:

            st.markdown(
                '<div class="section-title">🎯 Classification Results</div>',
                unsafe_allow_html=True
            )

            mask_count = sum(
                1 for r in results
                if r["label"] == "Mask"
            )

            no_mask_count = sum(
                1 for r in results
                if r["label"] == "No Mask"
            )

            c1, c2 = st.columns(2)

            c1.metric("😷 Mask Detected", mask_count)
            c2.metric("⚠️ No Mask Detected", no_mask_count)

            for result in results:

                is_mask = result["label"] == "Mask"

                status = (
                    "😷 MASK DETECTED"
                    if is_mask
                    else "⚠️ NO MASK DETECTED"
                )

                with st.container(border=True):

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader(
                            f"Face {result['face']}"
                        )

                    with col2:
                        if is_mask:
                            st.success(status)
                        else:
                            st.error(status)

                    st.divider()

                    st.metric(
                        "Prediction Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                    st.progress(
                        min(result["confidence"] / 100, 1.0)
                    )

                    score1, score2 = st.columns(2)

                    score1.metric(
                        "😷 Mask",
                        f"{result['mask_score']:.2f}%"
                    )

                    score2.metric(
                        "⚠️ No Mask",
                        f"{result['no_mask_score']:.2f}%"
                    )

        # Detection output - always display
        st.markdown(
            '<div class="section-title">🖼️ Detection Output</div>',
            unsafe_allow_html=True
        )

        original_col, result_col = st.columns(2)

        with original_col:

            st.image(
                image_rgb,
                caption="Original Image",
                use_container_width=True,
                channels="RGB",
                output_format="PNG"
            )

        with result_col:

            st.image(
                result_image,
                caption="AI Detection Result",
                use_container_width=True,
                channels="RGB",
                output_format="PNG"
            )

        st.caption(
            "Predictions are model estimates and may not always be correct."
        )

    except Exception as error:

        st.error(
            f"Could not process image: {error}"
        )

        st.exception(error)

        if image_rgb is not None:

            st.markdown("### Original Uploaded Image")

            st.image(
                image_rgb,
                caption="Original Image",
                use_container_width=True,
                channels="RGB",
                output_format="PNG"
            )

# ==================================================
# LIVE WEBCAM DETECTION
# ==================================================

st.divider()

st.markdown(
    '<div class="section-title">📷 Live Webcam Detection</div>',
    unsafe_allow_html=True
)

st.write(
    "Start your webcam to detect faces and predict mask status in real time."
)


webrtc_streamer(
    key="faceguard-live",
    video_processor_factory=FaceMaskProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    },
    async_processing=True
)


st.caption(
    "Allow camera permission in your browser to start live detection."
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "FaceGuard | Face Mask Detection | Python • TensorFlow • OpenCV • Streamlit"
)
