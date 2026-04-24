import streamlit as st  # UI framework
from ultralytics import YOLO
from PIL import Image   # Python Imaging Library (PIL)- Used to open,processing and display images
import numpy as np
from solutions import disease_info
import os
import gdown

MODEL_PATH = "../models/best.pt"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/drive/folders/1sPxGUlLNJildK55f83mITWqRnl33eVwN?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load Model
# -------------------------------
model = YOLO("../models/best.pt")   # load trained model(best.pt), this is our AI brain 


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Tomato Plant Disease Detection", layout="centered")

st.title("🌿Tomato Plant Disease Detection")   # Big heading on UI
st.write("Upload a leaf image to detect disease and get solution.")   # Simple text description

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])  # Creates file upload button, Accepts only image formats

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # This converts the uploaded file into an image object, Opens image using Pillow
    st.image(image, caption="Uploaded Image", use_container_width=True)  # st.image(image, width=500) instead, Displays uploaded image on UI

    # Convert image to numpy
    img_array = np.array(image)

    # -------------------------------
    # YOLO Prediction
    # -------------------------------
    results = model(img_array)  # Runs model on image, Returns: bounding boxes, class labels, confidence scores  of multiple images

    # Plot result image (Draws bounding boxes + labels on image)
    result_img = results[0].plot() # select only the first image

    st.image(result_img, caption="Detection Result", use_container_width=True)  # Displays output image 

    # -------------------------------
    # Extract Predictions
    # -------------------------------
    st.subheader("🔍 Detected Diseases")

    boxes = results[0].boxes

    if boxes is not None:   # Check if detection exists
        found = False

        for box in boxes:   # all detected objects in that image, loop through each detected object
            # Why [0]? Because box.cls is stored as a tensor (array-like), even for one value.
            cls_id = int(box.cls[0])   # box.cls - class ID tensor (not a simple number)
            confidence = float(box.conf[0])

            label = model.names[cls_id]  # Converts class ID → class name

            # 🔥 FIX: normalize string
            label_clean = label.strip().lower()
            st.write("DEBUG:", label_clean)

            st.write(f"🌿 {label} (Confidence: {confidence:.2f})")

            # Get solution
            info = disease_info.get(label_clean)

            if info:
                st.markdown("### 🦠 Disease Description")  # Used to display formatted text using Markdown
                st.info(info["description"])  # Shows blue info box

                st.markdown("### 🛡️ Prevention Tips")
                st.warning(info["prevention"])  # Shows yellow warning box

                st.markdown("### 💊 Recommended Medicine")
                st.success(info["medicine"])   # Shows green success box

            else:
                st.error("No information available for this disease.")

    else:
        st.warning("No objects detected.")