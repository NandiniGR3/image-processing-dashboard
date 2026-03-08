import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ===============================
# Folders
# ===============================
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===============================
# Utils
# ===============================
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ===============================
# Load SSD Face Detector
# ===============================
PROTO = "models/deploy.prototxt"
MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

ssd_net = None
if os.path.exists(PROTO) and os.path.exists(MODEL):
    ssd_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

# ===============================
# Face Detection (SSD)
# ===============================
def detect_faces(img, conf=0.5):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > conf:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# ===============================
# Face Blur
# ===============================
def blur_faces(img):
    out = img.copy()
    faces = detect_faces(img)
    for x, y, w, h in faces:
        roi = out[y:y+h, x:x+w]
        if roi.size != 0:
            roi = cv2.GaussianBlur(roi, (51, 51), 0)
            out[y:y+h, x:x+w] = roi
    return out, faces

# ===============================
# Cartoonify (Bilateral Filter)
# ===============================
def cartoonify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 9
    )

    color = img.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# ===============================
# ASCII Conversion (FIXED)
# ===============================
ASCII_CHARS = "@#S%?*+;:,. "

def image_to_ascii(img, width=120):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_h = int(h / w * width * 0.55)
    resized = cv2.resize(gray, (width, new_h))

    ascii_img = ""
    for row in resized:
        ascii_img += "".join(
            ASCII_CHARS[int(p) * len(ASCII_CHARS) // 256]
            for p in row
        ) + "\n"

    return ascii_img, resized

def ascii_to_image(grid, scale=8):
    img = cv2.resize(
        grid,
        (grid.shape[1]*scale, grid.shape[0]*scale),
        interpolation=cv2.INTER_NEAREST
    )
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config("Image Processing Mini Project", layout="wide")
st.title("Image Processing  Dashboard")

feature = st.sidebar.radio(
    "Select Feature",
    [
        "Cartoonify",
        "Edge Detection",
        "Face Detection",
        "Face Blur",
        "ASCII Art",
        "ASCII Reconstruction",
        "Live Face Detection",
        "Live Face Blur"
    ]
)

input_mode = st.sidebar.radio("Input Mode", ["Upload Image", "Capture Image"])

image = None
if input_mode == "Upload Image":
    file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")
else:
    cam = st.camera_input("Capture Image")
    if cam:
        image = Image.open(cam).convert("RGB")

# ===============================
# IMAGE FEATURES (SIDE BY SIDE)
# ===============================
if image:
    img = pil_to_cv(image)

    if feature == "Cartoonify":
        out = cartoonify(img)

    elif feature == "Edge Detection":
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif feature == "Face Detection":
        out = img.copy()
        faces = detect_faces(img)
        for x, y, w, h in faces:
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)

    elif feature == "Face Blur":
        out, _ = blur_faces(img)

    elif feature == "ASCII Art":
        ascii_txt, grid = image_to_ascii(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.text_area("ASCII Output", ascii_txt, height=400)

        name = f"ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = f"uploads/{name}"
        with open(path, "w") as f:
            f.write(ascii_txt)

        st.download_button("Download ASCII", ascii_txt, name)
        st.stop()

    elif feature == "ASCII Reconstruction":
        ascii_txt, grid = image_to_ascii(img)
        out = ascii_to_image(grid)

    else:
        out = None

    if out is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(cv_to_pil(out), caption="Output", use_container_width=True)

        name = f"{feature.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = f"uploads/processed/{name}"
        cv2.imwrite(path, out)
        with open(path, "rb") as f:
            st.download_button("Download Output", f, name)

# ===============================
# LIVE FEATURES
# ===============================
if feature == "Live Face Detection":
    cap = cv2.VideoCapture(0)
    frame_box = st.empty()
    stop = st.button("Stop")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()

if feature == "Live Face Blur":
    cap = cv2.VideoCapture(0)
    frame_box = st.empty()
    stop = st.button("Stop")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = blur_faces(frame)
        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
