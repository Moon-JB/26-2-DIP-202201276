import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

st.set_page_config(page_title="Object counting", layout="wide")
st.title("Object counting")


# ==========================================================
# Helpers (no assumption about object color or shape)
# All window sizes scale with ONE parameter: object size S.
# ==========================================================

def to_working_res(image, max_side):
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return image


def _resize_to(a, shape):
    return ndimage.zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


def large_median(ch, size):
    """Large median filter done on a downsampled copy (fast)."""
    f = max(1, int(size // 8))
    out = ndimage.median_filter(ch[::f, ::f], size=max(3, int(size / f) | 1))
    return _resize_to(out, ch.shape)


def norm_conv(x, w, sigma):
    """Gaussian average using only pixels where w is True (fast, on a downsampled grid)."""
    f = max(1, int(sigma // 4))
    xs, ws = x[::f, ::f], w[::f, ::f].astype(float)
    num = ndimage.gaussian_filter(xs * ws, sigma / f)
    den = ndimage.gaussian_filter(ws, sigma / f)
    return _resize_to(num / np.maximum(den, 1e-6), x.shape)


def contrast(f, bg, polarity):
    diff = f - bg
    if polarity == "Brighter than background":
        return np.clip(diff.mean(axis=2), 0, None)
    if polarity == "Darker than background":
        return np.clip(-diff.mean(axis=2), 0, None)
    return np.sqrt((diff ** 2).sum(axis=2))


def shadow_mask(f, bg, chroma_tol=0.03):
    """Shadow = same chromaticity as the local background, only darker (color-agnostic)."""
    s_f = f.sum(axis=2, keepdims=True) + 1e-6
    s_b = bg.sum(axis=2, keepdims=True) + 1e-6
    chroma = np.sqrt(((f / s_f - bg / s_b) ** 2).sum(axis=2))
    return (chroma < chroma_tol) & (s_f[..., 0] / s_b[..., 0] < 0.97)


@st.cache_data(show_spinner="Step 1: estimating background...")
def compute_score(img, S, use_color, polarity, suppress_texture, ignore_shadow, z_low, n_iter=3):
    """
    1) Rough background = large median (window 6S)
    2) Iterate: mark objects found so far -> re-estimate background & noise
       using ONLY background pixels. Large/dense objects no longer leak into
       the background, so objects stay solid instead of only edges surviving.
    3) score = contrast / local noise  (robust z-score)
    """
    f = img.astype(float)
    if not use_color:
        f = f.mean(axis=2, keepdims=True)
    k = max(1, int(0.06 * S)) | 1                      # denoise ~ 6% of object size
    if k > 1:
        f = np.stack([ndimage.median_filter(f[..., c], k) for c in range(f.shape[2])], axis=2)

    C = f.shape[2]
    bg_size, noise_size = 6 * S, 4 * S
    bg = np.stack([large_median(f[..., c], bg_size) for c in range(C)], axis=2)
    d = contrast(f, bg, polarity)

    if suppress_texture:
        # Texture level from a robust local MAD (works while objects cover < half the window)
        med0 = large_median(d, noise_size)
        mad0 = large_median(np.abs(d - med0), noise_size)
        tex = 1.4826 * np.maximum(mad0, 0.5 * np.median(mad0) + 1e-6)
        score = (d - med0) / tex
    else:
        tex = None
        score = (d - np.median(d)) / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-6)

    for _ in range(n_iter):
        obj = ndimage.binary_dilation(score > z_low, iterations=max(2, S // 10))
        w = ~ndimage.binary_fill_holes(obj)
        if w.mean() < 0.05:
            break
        bg = np.stack([norm_conv(f[..., c], w, bg_size / 3) for c in range(C)], axis=2)
        d = contrast(f, bg, polarity)
        g = 1.2533 * d[w].mean() + 1e-6                # mean|d| -> sigma
        noise = np.clip(1.2533 * norm_conv(d, w, noise_size / 3), 0.5 * g, None)
        if suppress_texture:
            noise = np.maximum(noise, tex)
        score = d / noise

    shadow = shadow_mask(f, bg) if (ignore_shadow and C == 3) else np.zeros(d.shape, bool)
    score = np.where(shadow, 0, score)
    return d, score, shadow


def hysteresis(score, high, low):
    """Grow from confident seeds (> high) into connected weaker pixels (> low).
    Keeps each object in one piece without letting noise in."""
    lab, n = ndimage.label(score > low)
    if n == 0:
        return lab > 0
    seeds = np.unique(lab[score > high])
    return np.isin(lab, seeds[seeds > 0])


def edge_map(img, sigma):
    f = img.astype(float)
    return np.max([ndimage.gaussian_gradient_magnitude(f[..., c], sigma) for c in range(3)], axis=0)


def split_by_edges(img, mask, S, edge_pct):
    """Touching objects are usually separated by a color/brightness boundary.
    Cut each blob along its strongest internal edges -> seeds; every blob pixel
    then goes to its nearest seed inside the same blob."""
    g = edge_map(img, max(1.0, 0.03 * S))
    if not mask.any():
        return ndimage.label(mask)[0]
    inner = mask & (g < np.percentile(g[mask], edge_pct))
    inner = ndimage.binary_erosion(inner, iterations=max(1, int(0.08 * S)))
    inner = ndimage.binary_opening(inner, np.ones((3, 3), bool))
    seeds, ns = ndimage.label(inner)
    if ns:
        a = np.bincount(seeds.ravel())
        good = a >= 0.08 * S * S
        good[0] = False
        seeds = (np.cumsum(good) * good)[seeds]
        ns = int(good.sum())
    comp, nc = ndimage.label(mask)
    if ns == 0:
        return comp
    _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    nearest = seeds[iy, ix]
    seed_comp = np.zeros(ns + 1, int)
    seed_comp[seeds[seeds > 0]] = comp[seeds > 0]
    lab = np.where(mask & (seed_comp[nearest] == comp), nearest, 0)
    orphan = mask & (lab == 0)
    o, _ = ndimage.label(orphan)
    lab[orphan] = o[orphan] + ns
    return lab


def merge_small(lab, min_frac=0.4, n_iter=3):
    """Re-attach fragments smaller than min_frac × median piece to the neighbor
    they share the longest border with (undoes over-splitting)."""
    lab = lab.copy()
    for _ in range(n_iter):
        areas = np.bincount(lab.ravel())
        valid = areas[1:][areas[1:] > 0]
        if len(valid) < 2:
            break
        med = np.median(valid)
        slices = ndimage.find_objects(lab)
        changed = False
        for i in np.argsort(areas):
            if i == 0 or not (0 < areas[i] < min_frac * med) or slices[i - 1] is None:
                continue
            sl = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in slices[i - 1])
            sub = lab[sl]
            m = sub == i
            if not m.any():
                continue
            ring = ndimage.binary_dilation(m, iterations=2) & ~m
            nb = sub[ring]
            nb = nb[(nb > 0) & (nb != i)]
            if nb.size:
                sub[m] = np.bincount(nb).argmax()
                changed = True
        if not changed:
            break
    return lab


@st.cache_data(show_spinner="Steps 2–3: segmenting & splitting...")
def segment(img, score, S, z_high, z_low, do_split, edge_pct):
    mask0 = hysteresis(score, z_high, z_low)
    k = max(1, int(0.1 * S)) | 1                       # morphology ~ 10% of object size
    mask = mask0
    if k > 1:
        mask = ndimage.binary_opening(mask, np.ones((k, k), bool))
        mask = ndimage.binary_closing(mask, np.ones((k, k), bool))
    mask = ndimage.binary_fill_holes(mask)
    lab = split_by_edges(img, mask, S, edge_pct) if do_split else ndimage.label(mask)[0]
    lab = merge_small(lab)
    return mask0, mask, lab


def to_display(arr, vmax):
    return np.clip(arr / (vmax + 1e-6) * 255, 0, 255).astype(np.uint8)


def colorize(lab):
    rng = np.random.default_rng(0)
    pal = rng.integers(60, 256, (lab.max() + 1, 3)).astype(np.uint8)
    pal[0] = 0
    return pal[lab]


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

st.sidebar.header("Scale")
max_side = st.sidebar.select_slider(
    "Working resolution (long side, px)", [600, 800, 1000, 1200, 1600], value=1000)
obj_size = st.sidebar.slider(
    "Approx. object size (px)", 5, 200, 40,
    help="Rough width of ONE object at the working resolution. Every window "
         "(background, noise, morphology, splitting, min area) scales from this. "
         "After a run, the measured median size is shown under the count.")

st.sidebar.header("Step 1. Contrast")
use_color = st.sidebar.checkbox("Use color contrast", value=True,
                                help="Distance from the local background color; no specific color assumed.")
polarity = st.sidebar.selectbox(
    "Object Intensity",
    ["Either (different from background)", "Brighter than background", "Darker than background"])
ignore_shadow = st.sidebar.checkbox("Ignore shadows", value=True,
                                    help="Removes regions with the background's hue that are only darker.")
suppress_texture = st.sidebar.checkbox(
    "Suppress textured background", value=False,
    help="ON for busy/patterned backgrounds (pavement, fabric). "
         "OFF for plain backgrounds, especially with densely packed objects — "
         "a dense cluster looks like 'texture' and would be suppressed.")

st.sidebar.header("Step 2. Segmentation")
z_high = st.sidebar.slider("Seed threshold (z)", 2.0, 30.0, 10.0, step=0.5,
                           help="Pixels this many noise units above background start an object.")
z_low = st.sidebar.slider("Grow threshold (z)", 1.0, 15.0, 4.0, step=0.5,
                          help="Objects grow from seeds into connected pixels above this. "
                               "Lower = fuller objects (less fragmentation).")

st.sidebar.header("Step 3. Counting")
do_split = st.sidebar.checkbox("Split touching objects", value=True)
edge_pct = st.sidebar.slider("Split sensitivity", 50, 95, 70, step=5,
                             help="Lower = cut along more edges (more splitting).")
min_area_frac = st.sidebar.slider("Minimum area (× object size²)", 0.02, 1.0, 0.15, step=0.01)


# ==========================================================
# Main
# ==========================================================

if uploaded_file is not None:

    image = to_working_res(Image.open(uploaded_file).convert("RGB"), max_side)
    img = np.array(image)
    S = int(obj_size)

    # ---------- STEP 1 ----------
    st.header("Step 1. Background Removal & Noise Normalization")
    d, score, shadow = compute_score(img, S, use_color, polarity, suppress_texture, ignore_shadow, z_low)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption=f"Original ({img.shape[1]}×{img.shape[0]})", use_container_width=True)
    with col2:
        st.image(to_display(d, np.percentile(d, 99.5)), caption="Contrast from local background",
                 use_container_width=True)
    with col3:
        st.image(to_display(np.clip(score, 0, None), 2 * z_high),
                 caption="Noise-normalized score (shadows removed)" if ignore_shadow
                 else "Noise-normalized score", use_container_width=True)

    # ---------- STEP 2 ----------
    st.header("Step 2. Hysteresis Thresholding & Morphology")
    mask0, mask, lab = segment(img, score, S, z_high, z_low, do_split, edge_pct)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(mask0.astype(np.uint8) * 255, caption="Hysteresis threshold", use_container_width=True)
    with col2:
        st.image(mask.astype(np.uint8) * 255, caption="Opening + Closing + Hole filling",
                 use_container_width=True)
    with col3:
        st.image(colorize(lab), caption="Objects after splitting (one color each)",
                 use_container_width=True)

    # ---------- STEP 3 ----------
    st.header("Step 3. Object Detection & Counting")
    areas = np.bincount(lab.ravel())[1:]
    min_area = min_area_frac * S * S
    result = image.copy()
    draw = ImageDraw.Draw(result)
    lw = max(2, img.shape[1] // 400)
    kept = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None or areas[i] < min_area:
            continue
        kept.append(areas[i])
        x0, x1, y0, y1 = sl[1].start, sl[1].stop - 1, sl[0].start, sl[0].stop - 1
        draw.rectangle([x0, y0, x1, y1], outline="red", width=lw)
        draw.text((x0 + 2, y0 + 1), str(len(kept)), fill="yellow")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(result, caption="Object Detection", use_container_width=True)
    with col2:
        st.metric("Total Objects Detected", len(kept))
        if kept:
            med_size = int(np.sqrt(np.median(kept)))
            st.caption(f"Measured median object size ≈ {med_size} px "
                       f"(slider: {S} px). If these differ a lot, move the slider closer.")
            st.subheader("Object area distribution")
            bins = np.unique(np.geomspace(max(1, min(areas)), max(areas.max(), 2), 25).astype(int))
            hist, edges = np.histogram(areas, bins=bins)
            st.bar_chart(pd.DataFrame({"objects": hist}, index=pd.Index(edges[:-1], name="area (px)")))

else:
    st.info("Upload an image from the sidebar.")
