import streamlit as st
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Object Counting",
    page_icon="🍂",
    layout="wide"
)

st.title("🍂 Object Detection & Counting")
st.write("Upload an image and detect individual objects.")

# -----------------------------
# 2. Sidebar Parameters
# -----------------------------
st.sidebar.header("Image Processing Settings")

st.sidebar.subheader("HSV Threshold")

h_min = st.sidebar.slider("Hue Min", 0, 179, 5)
h_max = st.sidebar.slider("Hue Max", 0, 179, 45)

s_min = st.sidebar.slider("Saturation Min", 0, 255, 35)
v_min = st.sidebar.slider("Value Min", 0, 255, 35)

st.sidebar.subheader("Morphological Processing")

kernel_size = st.sidebar.slider(
    "Kernel Size", 1, 21, 5, step=2
)

min_area = st.sidebar.slider(
    "Minimum Object Area (pixels)",
    10, 10000, 300, step=50
)

# -----------------------------
# 3. Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # -----------------------------
    # 4. Color Space Conversion
    # -----------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # -----------------------------
    # 5. Morphological Operations
    # -----------------------------
    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    # Remove small noise
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    # Fill small holes
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    # -----------------------------
    # 6. Connected Component Analysis
    # -----------------------------
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    result = img.copy()
    object_count = 0

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        object_count += 1

        # Draw bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Draw object number
        cv2.putText(
            result,
            str(object_count),
            (x, max(y - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # -----------------------------
    # 7. Display Results
    # -----------------------------
    st.subheader("Detection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Original Image")
        st.image(img, use_container_width=True)

    with col2:
        st.write("Binary Mask")
        st.image(mask, clamp=True, use_container_width=True)

    with col3:
        st.write("Detected Objects")
        st.image(result, use_container_width=True)

    # -----------------------------
    # 8. Counting Result
    # -----------------------------
    st.divider()

    st.metric(
        label="Total Objects Detected",
        value=object_count
    )

else:
    st.info("Please upload an image to start object counting.")
