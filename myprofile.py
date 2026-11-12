
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Object Counting",
    page_icon="🍂",
    layout="wide"
)

st.title("Object Detection & Counting")

# -----------------------------
# 2. Sidebar
# -----------------------------
st.sidebar.header("Processing Settings")

threshold = st.sidebar.slider(
    "Red Color Threshold",
    0, 255, 100
)

min_area = st.sidebar.slider(
    "Minimum Object Area",
    10, 10000, 300
)

# -----------------------------
# 3. Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image using PIL
    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    # -----------------------------
    # 4. Color Segmentation
    # -----------------------------

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Simple red color detection
    mask = (
        (R > threshold) &
        (R > G * 1.2) &
        (R > B * 1.2)
    )

    # -----------------------------
    # 5. Morphological Processing
    # -----------------------------

    structure = np.ones((3, 3))

    mask = ndimage.binary_opening(
        mask, structure=structure
    )

    mask = ndimage.binary_closing(
        mask, structure=structure
    )

    # -----------------------------
    # 6. Connected Component
    # -----------------------------

    labels, num_objects = ndimage.label(mask)

    result = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(result)

    object_count = 0

    for i in range(1, num_objects + 1):

        component = (labels == i)

        area = np.sum(component)

        if area < min_area:
            continue

        y, x = np.where(component)

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        object_count += 1

        # Bounding Box
        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=3
        )

        # Object Number
        draw.text(
            (x_min, max(0, y_min - 15)),
            str(object_count),
            fill="yellow"
        )

    # -----------------------------
    # 7. Display Results
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Binary Mask")
        st.image(mask.astype(np.uint8) * 255,
                 clamp=True,
                 use_container_width=True)

    with col3:
        st.subheader("Detection")
        st.image(result, use_container_width=True)

    st.divider()

    st.metric(
        "Total Objects Detected",
        object_count
    )
