import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

st.set_page_config(page_title="Object counting", layout="wide")
st.title("Object counting")


# ==========================================
# Processing functions (no assumption about object color)
# ==========================================

def to_working_res(image, max_side):
    """Downscale large photos so filters run fast; counting is unaffected."""
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return image


def large_median(ch, size):
    """Large median filter done cheaply: downsample -> median -> upsample."""
    f = max(1, int(size // 8))
    small = ch[::f, ::f]
    k = max(3, int(size / f) | 1)
    out = ndimage.median_filter(small, size=k)
    return ndimage.zoom(out, (ch.shape[0] / out.shape[0], ch.shape[1] / out.shape[1]), order=1)


def contrast_map(img, use_color, polarity, denoise, bg_size):
    """
    How different each pixel is from its LOCAL background.
    - use_color=True : Euclidean distance in RGB -> any color that differs from the
                       background counts (no specific hue is assumed).
    - use_color=False: grayscale brightness difference only.
    """
    f = img.astype(float)
    if not use_color:
        f = f.mean(axis=2, keepdims=True)

    # Small median: removes pixel-level grain, keeps edges of larger objects
    if denoise > 1:
        f = np.stack([ndimage.median_filter(f[..., c], size=denoise)
                      for c in range(f.shape[2])], axis=2)

    # Background = large median (unlike a Gaussian, it is not pulled toward objects)
    bg = np.stack([large_median(f[..., c], bg_size) for c in range(f.shape[2])], axis=2)
    diff = f - bg

    if polarity == "Brighter than background":
        return np.clip(diff.mean(axis=2), 0, None)
    if polarity == "Darker than background":
        return np.clip(-diff.mean(axis=2), 0, None)
    return np.sqrt((diff ** 2).sum(axis=2))  # either direction


def texture_normalize(d, tex_size):
    """
    Adaptive noise suppression = robust local z-score.
    score = (d - local median) / local MAD
    In busy/textured regions the local spread (MAD) is large, so their responses shrink.
    An object on a calm background stays a strong outlier.
    """
    med = large_median(d, tex_size)
    mad = large_median(np.abs(d - med), tex_size)
    floor = 0.5 * np.median(mad) + 1e-6  # avoid exploding in perfectly flat areas
    return (d - med) / (1.4826 * np.maximum(mad, floor))


def global_normalize(d):
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    return (d - med) / (1.4826 * mad + 1e-6)


@st.cache_data(show_spinner="Computing contrast map...")
def compute_score(img, use_color, polarity, denoise, bg_size, use_texture, tex_size):
    d = contrast_map(img, use_color, polarity, denoise, bg_size)
    score = texture_normalize(d, tex_size) if use_texture else global_normalize(d)
    return d, score


def to_display(arr, vmax):
    return np.clip(arr / (vmax + 1e-6) * 255, 0, 255).astype(np.uint8)


# ==========================================
# Sidebar
# ==========================================

st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

st.sidebar.header("Step 1. Contrast")
max_side = st.sidebar.select_slider(
    "Working resolution (long side, px)", [600, 800, 1000, 1200, 1600, 2000], value=1200,
    help="Large photos are resized to this size first. Higher = slower but keeps tiny objects.")
use_color = st.sidebar.checkbox(
    "Use color contrast", value=True,
    help="Measures how far each pixel's color is from the local background color. "
         "No specific object color is assumed. Turn off for grayscale images.")
polarity = st.sidebar.selectbox(
    "Object Intensity",
    ["Either (different from background)", "Brighter than background", "Darker than background"])
denoise = st.sidebar.slider("Denoise median size", 1, 9, 3, step=2,
                            help="Removes pixel-level grain. Keep it smaller than the smallest object.")
bg_size = st.sidebar.slider("Background window", 15, 301, 61, step=2,
                            help="Should be clearly LARGER than one object.")

st.sidebar.header("Adaptive noise suppression")
use_texture = st.sidebar.checkbox("Suppress textured background", value=True)
tex_size = st.sidebar.slider("Texture window", 9, 151, 31, step=2,
                             help="Region size used to measure local background noise.")

st.sidebar.header("Step 2. Segmentation")
z_thresh = st.sidebar.slider(
    "Threshold (robust z-score)", 1.0, 30.0, 8.0, step=0.5,
    help="How many 'noise units' above the local background a pixel must be. "
         "This scale is comparable across images, unlike a 0–255 threshold.")
open_k = st.sidebar.slider("Opening kernel size", 1, 15, 3, step=2)
close_k = st.sidebar.slider("Closing kernel size", 1, 15, 3, step=2)

st.sidebar.header("Step 3. Counting")
min_area = st.sidebar.number_input("Minimum object area (px)", 1, 100000, 20)
max_area = st.sidebar.number_input("Maximum object area (px, 0 = no limit)", 0, 10000000, 0)
use_clump = st.sidebar.checkbox(
    "Estimate touching objects", value=False,
    help="A blob much larger than the typical object is counted as round(area / median area).")


# ==========================================
# Main
# ==========================================

if uploaded_file is not None:

    image = to_working_res(Image.open(uploaded_file).convert("RGB"), max_side)
    img = np.array(image)

    # ---------- STEP 1 ----------
    st.header("Step 1. Background Removal & Adaptive Noise Suppression")

    d, score = compute_score(img, use_color, polarity, denoise, bg_size, use_texture, tex_size)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption=f"Original ({img.shape[1]}×{img.shape[0]})", use_container_width=True)
    with col2:
        st.image(to_display(d, np.percentile(d, 99.5)),
                 caption="Contrast from local background", use_container_width=True)
    with col3:
        st.image(to_display(np.clip(score, 0, None), 2 * z_thresh),
                 caption="Noise-normalized score" if use_texture else "Globally normalized score",
                 use_container_width=True)

    # ---------- STEP 2 ----------
    st.header("Step 2. Thresholding & Morphological Processing")

    binary = score > z_thresh
    opened = ndimage.binary_opening(binary, np.ones((open_k, open_k), bool)) if open_k > 1 else binary
    closed = ndimage.binary_closing(opened, np.ones((close_k, close_k), bool)) if close_k > 1 else opened
    mask = ndimage.binary_fill_holes(closed)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(binary.astype(np.uint8) * 255, caption="Binary Threshold", use_container_width=True)
    with col2:
        st.image(opened.astype(np.uint8) * 255, caption="After Opening", use_container_width=True)
    with col3:
        st.image(mask.astype(np.uint8) * 255, caption="After Closing + Hole Filling",
                 use_container_width=True)

    # ---------- STEP 3 ----------
    st.header("Step 3. Object Detection & Counting")

    labels, num = ndimage.label(mask)
    areas_all = np.bincount(labels.ravel())[1:] if num else np.array([], int)
    slices = ndimage.find_objects(labels)

    keep = [(sl, a) for sl, a in zip(slices, areas_all)
            if a >= min_area and (max_area == 0 or a <= max_area)]
    kept_areas = np.array([a for _, a in keep])
    unit = np.median(kept_areas) if len(kept_areas) else 1

    result = image.copy()
    draw = ImageDraw.Draw(result)
    lw = max(2, img.shape[1] // 400)
    total = 0
    for i, (sl, a) in enumerate(keep, 1):
        x0, x1 = sl[1].start, sl[1].stop - 1
        y0, y1 = sl[0].start, sl[0].stop - 1
        n_here = int(max(1, round(a / unit))) if use_clump else 1
        total += n_here
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2],
                       outline="red" if n_here == 1 else "cyan", width=lw)
        draw.text((x0, max(0, y0 - 12)), str(i) if n_here == 1 else f"{i}(×{n_here})", fill="yellow")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(result, caption="Object Detection (cyan = estimated touching objects)",
                 use_container_width=True)
    with col2:
        st.metric("Total Objects Detected", total)
        st.caption(f"{len(keep)} blobs kept · {num - len(keep)} removed by area filter")
        if len(areas_all):
            st.subheader("Blob area distribution")
            st.caption("Use this to set the minimum area: noise is the pile of tiny blobs on the left.")
            bins = np.unique(np.geomspace(1, max(areas_all.max(), 2), 25).astype(int))
            hist, edges = np.histogram(areas_all, bins=bins)
            st.bar_chart(pd.DataFrame({"blobs": hist}, index=pd.Index(edges[:-1], name="area (px)")))

else:
    st.info("Upload an image from the sidebar.")
