import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ===============================
# Folder setup
# ===============================
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===============================
# Utility conversions
# ===============================
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ===============================
# Load SSD face detector
# ===============================
PROTO = "models/deploy.prototxt"
MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

ssd = None
if os.path.exists(PROTO) and os.path.exists(MODEL):
    ssd = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

# ===============================
# Face detection (SSD)
# ===============================
def detect_faces(img, conf=0.5):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300,300),
        (104.0,177.0,123.0)
    )
    ssd.setInput(blob)
    detections = ssd.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf:
            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)
            boxes.append((x1,y1,x2-x1,y2-y1))
    return boxes

# ===============================
# Face blur
# ===============================
def blur_faces(img, conf):
    out = img.copy()
    boxes = detect_faces(img, conf)
    for x,y,w,h in boxes:
        roi = out[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (51,51), 0)
        out[y:y+h, x:x+w] = roi
    return out, boxes

# ===============================
# Cartoonify (Snapchat style)
# ===============================
def snapchat_cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )
    color = img.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 75, 75)
    return cv2.bitwise_and(color, color, mask=edges)

# ===============================
# ASCII utilities
# ===============================
ASCII_CHARS = "@#S%?*+;:,. "

def image_to_ascii(img, width=120):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_h = int(h/w * width * 0.55)
    resized = cv2.resize(gray, (width, new_h))

    ascii_text = ""
    for row in resized:
        ascii_text += "".join(
            ASCII_CHARS[p*len(ASCII_CHARS)//256] for p in row
        ) + "\n"
    return ascii_text, resized

def ascii_to_image(grid, scale=8):
    recon = cv2.resize(
        grid,
        (grid.shape[1]*scale, grid.shape[0]*scale),
        interpolation=cv2.INTER_NEAREST
    )
    return cv2.cvtColor(recon, cv2.COLOR_GRAY2BGR)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config("Image Mini Project", layout="wide")
st.title("Image Processing Mini Project (SSD Based)")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio(
    "",
    ["Upload Image", "Capture Image", "Live Face Detection", "Live Face Blur"]
)

st.sidebar.header("Feature")
feature = st.sidebar.radio(
    "",
    [
        "Cartoonify (Snapchat)",
        "Edge Detection",
        "Face Detection",
        "Face Blur",
        "ASCII Art",
        "ASCII Reconstruction"
    ]
)

conf = st.sidebar.slider("Face Confidence", 0.1, 0.9, 0.5)

# ===============================
# Live modes
# ===============================
if input_mode == "Live Face Detection":
    cap = cv2.VideoCapture(0)
    frame = st.image([])
    stop = st.button("Stop")
    while cap.isOpened() and not stop:
        ret, img = cap.read()
        boxes = detect_faces(img, conf)
        for x,y,w,h in boxes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        frame.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cap.release()
    st.stop()

if input_mode == "Live Face Blur":
    cap = cv2.VideoCapture(0)
    frame = st.image([])
    stop = st.button("Stop")
    while cap.isOpened() and not stop:
        ret, img = cap.read()
        out,_ = blur_faces(img, conf)
        frame.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    cap.release()
    st.stop()

# ===============================
# Static image input
# ===============================
image = None
if input_mode == "Upload Image":
    f = st.file_uploader("Upload Image", ["jpg","png","jpeg"])
    if f:
        image = Image.open(f).convert("RGB")

elif input_mode == "Capture Image":
    cam = st.camera_input("Capture")
    if cam:
        image = Image.open(cam).convert("RGB")

# ===============================
# Processing
# ===============================
if image:
    img = pil_to_cv(image)
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_column_width=True)

    if feature == "Cartoonify (Snapchat)":
        out = snapchat_cartoon(img)
        col2.image(cv_to_pil(out), caption="Cartoonified")

    elif feature == "Edge Detection":
        edges = cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),100,200)
        col2.image(edges, caption="Edges")

    elif feature == "Face Detection":
        out = img.copy()
        boxes = detect_faces(img, conf)
        for x,y,w,h in boxes:
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
        col2.image(cv_to_pil(out), caption=f"Faces: {len(boxes)}")

    elif feature == "Face Blur":
        out,_ = blur_faces(img, conf)
        col2.image(cv_to_pil(out), caption="Blurred Faces")

    elif feature == "ASCII Art":
        ascii_text, grid = image_to_ascii(img)
        col2.text_area("ASCII Output", ascii_text, height=400)

        txt_name = f"ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        txt_path = f"uploads/{txt_name}"
        with open(txt_path,"w") as f:
            f.write(ascii_text)

        st.download_button("Download ASCII", ascii_text, txt_name)

    elif feature == "ASCII Reconstruction":
        ascii_text, grid = image_to_ascii(img)
        recon = ascii_to_image(grid)
        col2.image(cv_to_pil(recon), caption="Reconstructed Image")

        img_name = f"ascii_recon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = f"uploads/processed/{img_name}"
        cv2.imwrite(path, recon)

        with open(path,"rb") as f:
            st.download_button("Download Image", f, img_name)
