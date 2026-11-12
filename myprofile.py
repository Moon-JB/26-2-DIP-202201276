import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

st.title("Object counting")

# Sidebar
st.sidebar.header("Image Upload")

uploaded_file = st.sidebar.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # ==========================================
    # STEP 1. HPF & Background Removal
    # ==========================================

    st.header("Step 1. Edge Enhancement & Background Removal")

    # RGB to Grayscale
    gray = np.mean(img, axis=2).astype(float)

    # Sidebar parameters
    hpf_sigma = st.sidebar.slider(
        "HPF Sigma", 0.5, 10.0, 2.0, step=0.5
    )

    bg_sigma = st.sidebar.slider(
        "Background Sigma", 5.0, 100.0, 30.0, step=5.0
    )

    hpf_strength = st.sidebar.slider(
        "HPF Strength", 0.0, 3.0, 1.0, step=0.1
    )

    # High-pass filtering
    low_freq = ndimage.gaussian_filter(
        gray, sigma=hpf_sigma
    )

    high_freq = gray - low_freq

    # Background estimation
    background = ndimage.gaussian_filter(
        gray, sigma=bg_sigma
    )

    # Background subtraction + HPF enhancement
    corrected = (
        gray - background + 128
        + hpf_strength * high_freq
    )

    corrected = np.clip(
        corrected, 0, 255
    ).astype(np.uint8)

    hpf_image = np.clip(
        128 + high_freq * 3, 0, 255
    ).astype(np.uint8)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            hpf_image,
            caption="High-pass Filter",
            clamp=True,
            use_container_width=True
        )

    with col3:
        st.image(
            corrected,
            caption="Background Removed",
            clamp=True,
            use_container_width=True
        )

    # ==========================================
    # STEP 2. Thresholding & Morphological Opening
    # ==========================================

    st.header("Step 2. Morphological Processing")

    threshold = st.sidebar.slider(
        "Threshold",
        0, 255, 135
    )

    polarity = st.sidebar.selectbox(
        "Object Intensity",
        ["Brighter than background", "Darker than background"]
    )

    kernel_size = st.sidebar.slider(
        "Opening Kernel Size",
        1, 15, 3, step=2
    )

    min_area = st.sidebar.slider(
        "Minimum Object Area",
        10, 10000, 300, step=50
    )

    # Binary thresholding
    if polarity == "Brighter than background":
        binary = corrected > threshold
    else:
        binary = corrected < threshold

    # Morphological Opening
    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=bool
    )

    opened = ndimage.binary_opening(
        binary,
        structure=kernel
    )

    # Connected Component Labeling
    labels, num_objects = ndimage.label(opened)

    # ==========================================
    # STEP 3. Parameter Optimization & Counting
    # ==========================================

    st.header("Step 3. Object Detection & Counting")

    result = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(result)

    object_count = 0

    for i in range(1, num_objects + 1):

        component = (labels == i)

        area = np.sum(component)

        if area < min_area:
            continue

        y, x = np.where(component)

        x_min = x.min()
        x_max = x.max()

        y_min = y.min()
        y_max = y.max()

        object_count += 1

        # Draw Bounding Box
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

    # Display intermediate results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            binary.astype(np.uint8) * 255,
            caption="Binary Threshold",
            clamp=True,
            use_container_width=True
        )

    with col2:
        st.image(
            opened.astype(np.uint8) * 255,
            caption="After Opening",
            clamp=True,
            use_container_width=True
        )

    with col3:
        st.image(
            result,
            caption="Object Detection",
            use_container_width=True
        )

    # Final counting result
    st.divider()

    st.metric(
        label="Total Objects Detected",
        value=object_count
    )

