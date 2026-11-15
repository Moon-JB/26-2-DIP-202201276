"""
General-purpose object counting (Streamlit)

Design goal: decent counts on MANY kinds of images with the same defaults,
rather than a perfect count on one image. Nothing is tuned to a specific image:
object size, background and threshold are all estimated from each image.

Pipeline
  Step 1  Background = dominant color, smoothly varying (lighting/vignetting)
          Contrast   = color distance from that background
          Shadows / cracks / joints (same hue as background, only darker) removed
  Step 2  Object size estimated automatically (typical blob thickness)
          Background re-estimated at the object scale with objects masked out
          Otsu threshold + hysteresis, morphology scaled to the object size
  Step 3  Blobs kept only if (a) clearly above threshold on average,
          (b) not a thin line, (c) not tiny relative to the typical object.
          Touching objects are counted from their area (area / typical area).
"""
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

st.set_page_config(page_title="Object counting", layout="wide")
st.title("Object counting")


# ==========================================================
# Utilities
# ==========================================================

def to_working_res(image, max_side):
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return image


def _resize_to(a, shape):
    return ndimage.zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


def masked_smooth(x, w, sigma):
    """Gaussian average using only pixels where w is True (holes filled from surroundings)."""
    f = max(1, int(sigma // 4))
    xs, ws = x[::f, ::f], w[::f, ::f].astype(float)
    num = ndimage.gaussian_filter(xs * ws, sigma / f, mode="nearest")
    den = ndimage.gaussian_filter(ws, sigma / f, mode="nearest")
    return _resize_to(num / np.maximum(den, 1e-6), x.shape)


def otsu(v):
    hist, edges = np.histogram(v, bins=256)
    p = hist / max(hist.sum(), 1)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m = np.cumsum(p * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (m[-1] * w0 - m) ** 2 / (w0 * (1 - w0))
    return float(c[np.nanargmax(sb)])


def disk(r):
    r = max(0, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


# ==========================================================
# Step 1: background & contrast
# ==========================================================

def estimate_background(f, sigma, obj_mask=None, n_iter=3):
    """
    Background = the dominant color, allowed to vary smoothly.
    - Without a mask: start from the most common color (works even when
      objects cover about half the image, unlike a local median filter).
    - With a mask: use only pixels outside the known objects.
    Then iteratively keep pixels close to the current background and re-smooth.
    """
    C = f.shape[2]
    if obj_mask is None:
        q = np.clip(ndimage.gaussian_filter(f, (2, 2, 0)) // 16, 0, 15).astype(int)
        code = (q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]) if C == 3 else q[..., 0]
        seed = code == np.bincount(code.ravel()).argmax()
        c0 = f[seed].mean(axis=0)
        dist = np.sqrt(((f - c0) ** 2).sum(axis=2))
        w = dist < max(otsu(dist), np.percentile(dist[seed], 99))
    else:
        w = ~obj_mask
    bg = np.stack([masked_smooth(f[..., c], w, sigma) for c in range(C)], axis=2)
    for _ in range(n_iter):
        d = np.sqrt(((f - bg) ** 2).sum(axis=2))
        dn = d[w]
        level = np.median(dn) + 1.4826 * np.median(np.abs(dn - np.median(dn))) + 1e-6
        w_new = d < 3 * level
        if obj_mask is not None:
            w_new &= ~obj_mask
        if w_new.mean() < 0.05:
            break
        w = w_new
        bg = np.stack([masked_smooth(f[..., c], w, sigma) for c in range(C)], axis=2)
    return bg


def contrast_map(f, bg, polarity):
    diff = f - bg
    if polarity == "Brighter than background":
        return np.clip(diff.mean(axis=2), 0, None)
    if polarity == "Darker than background":
        return np.clip(-diff.mean(axis=2), 0, None)
    return np.sqrt((diff ** 2).sum(axis=2))


def darker_same_hue(f, bg, tol=0.035):
    """Shadows, cracks, tile joints: same chromaticity as the background, only darker."""
    if f.shape[2] != 3:
        return np.zeros(f.shape[:2], bool)
    sf = f.sum(axis=2, keepdims=True) + 1e-6
    sb = bg.sum(axis=2, keepdims=True) + 1e-6
    chroma = np.sqrt(((f / sf - bg / sb) ** 2).sum(axis=2))
    return (chroma < tol) & (sf[..., 0] < 0.97 * sb[..., 0])


# ==========================================================
# Step 2: object size, threshold, morphology
# ==========================================================

def hysteresis(d, high, low):
    """Seeds above `high`, grown into connected pixels above `low` (keeps objects whole)."""
    lab, n = ndimage.label(d > low)
    if n == 0:
        return lab > 0
    seeds = np.unique(lab[d > high])
    return np.isin(lab, seeds[seeds > 0])


def typical_thickness(mask):
    """
    Object scale = thickness (2 × largest inscribed radius) that holds the most
    'pixels × thickness'. Thin lines (cracks) and tiny specks get little weight,
    and a clump of touching objects still has single-object thickness.
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    idx = np.arange(1, n + 1)
    th = 2 * ndimage.maximum(ndimage.distance_transform_edt(mask), lab, idx)
    area = ndimage.sum(mask, lab, idx)
    ok = th >= 3
    if not ok.any():
        return None
    th, area = th[ok], area[ok]
    bins = np.geomspace(3, th.max() + 1, 25)
    h, e = np.histogram(th, bins=bins, weights=area * th)
    j = int(h.argmax())
    return float(np.sqrt(e[j] * e[j + 1]))


@st.cache_data(show_spinner="Analyzing image...")
def analyze(img, polarity, use_color, ignore_shadows, size_override, sens,
            min_rel, min_contrast, count_clumps):
    f = img.astype(float)
    if not use_color:
        f = f.mean(axis=2, keepdims=True)
    f = np.stack([ndimage.median_filter(f[..., c], 3) for c in range(f.shape[2])], axis=2)
    H, W = f.shape[:2]
    use_shadow = ignore_shadows and use_color

    # ---- Pass 1: global smooth background -> rough objects -> object size
    bg = estimate_background(f, sigma=min(H, W) / 8)
    d = contrast_map(f, bg, polarity)
    if use_shadow:
        d = np.where(darker_same_hue(f, bg), 0, d)
    rough = d > otsu(d[d > 0]) if (d > 0).any() else d > 0
    S = size_override or typical_thickness(rough) or min(H, W) / 20
    S = float(np.clip(S, 4, min(H, W) / 3))

    # ---- Pass 2: background at the object scale, objects masked out
    obj = ndimage.binary_fill_holes(ndimage.binary_dilation(rough, disk(max(1, 0.2 * S))))
    bg = estimate_background(f, sigma=max(8, 3 * S), obj_mask=obj if obj.mean() < 0.7 else None)
    d = contrast_map(f, bg, polarity)
    shadow = darker_same_hue(f, bg) if use_shadow else np.zeros(d.shape, bool)
    d = np.where(shadow, 0, d)
    t = otsu(d[d > 0]) * sens if (d > 0).any() else 1.0
    mask0 = hysteresis(d, t, 0.5 * t)

    # Morphology scaled to the object: opening removes lines/specks thinner than ~S/3
    mask = ndimage.binary_opening(mask0, disk(S / 6)) if S / 6 >= 1 else mask0
    mask = ndimage.binary_closing(mask, disk(max(1, S / 10)))
    mask = ndimage.binary_fill_holes(mask)

    # ---- Step 3: blob filtering relative to the typical object
    lab, n = ndimage.label(mask)
    areas = np.bincount(lab.ravel())[1:] if n else np.array([], int)
    objects, typical = [], None
    kept_lab = np.zeros_like(lab)
    if n:
        idx = np.arange(1, n + 1)
        th = 2 * ndimage.maximum(ndimage.distance_transform_edt(mask), lab, idx)
        mean_c = ndimage.mean(d, lab, idx) / t
        # (a) clearly above threshold on average  (b) not a thin line
        cand = (mean_c >= min_contrast) & (th >= 0.35 * S)
        # typical area from candidates with single-object thickness
        single = cand & (th > 0.6 * S) & (th < 1.6 * S)
        pool = areas[single] if single.any() else (areas[cand] if cand.any() else areas)
        typical = float(np.median(pool))
        for i, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None or not cand[i]:
                continue
            ratio = areas[i] / typical
            if ratio < min_rel:                       # (c) not tiny
                continue
            k = int(max(1, round(ratio))) if (count_clumps and ratio >= 1.7) else 1
            objects.append(dict(box=(sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1),
                                area=int(areas[i]), count=k))
            kept_lab[sl][lab[sl] == i + 1] = len(objects)
    return dict(S=S, threshold=t, contrast=d, shadow=shadow, mask0=mask0, mask=mask,
                kept=kept_lab, areas=areas, typical=typical, objects=objects,
                total=int(sum(o["count"] for o in objects)))


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

st.sidebar.header("Basic")
max_side = st.sidebar.select_slider(
    "Working resolution (long side, px)", [800, 1000, 1200, 1600], value=1200,
    help="Large photos are resized first. Raise it if objects are very small.")
polarity = st.sidebar.selectbox(
    "Object Intensity",
    ["Either (different from background)", "Brighter than background", "Darker than background"])
use_color = st.sidebar.checkbox("Use color contrast", value=True,
                                help="Color distance from the background (no specific color assumed). "
                                     "Turn off for grayscale images.")
ignore_shadows = st.sidebar.checkbox(
    "Ignore shadows / cracks / joints", value=True,
    help="Removes regions with the background's hue that are only darker. "
         "Turn off if your objects are darker versions of the background color.")

with st.sidebar.expander("Advanced (defaults work for most images)"):
    size_override = st.number_input("Object size (px, 0 = auto)", 0, 500, 0,
                                    help="Width of one object at working resolution.")
    sens = st.slider("Threshold scale", 0.5, 2.0, 1.0, step=0.05,
                     help="< 1: more sensitive (faint objects), > 1: stricter.")
    min_contrast = st.slider("Minimum object contrast (× threshold)", 1.0, 2.5, 1.25, step=0.05,
                             help="Rejects textured patches and grain whose average contrast is "
                                  "barely above the threshold.")
    min_rel = st.slider("Minimum size (× typical object area)", 0.05, 1.0, 0.25, step=0.05)
    count_clumps = st.checkbox("Count touching objects by area", value=True,
                               help="A blob ≥1.7× the typical area counts as round(area / typical).")


# ==========================================================
# Main
# ==========================================================

if uploaded_file is not None:
    image = to_working_res(Image.open(uploaded_file).convert("RGB"), max_side)
    img = np.array(image)
    r = analyze(img, polarity, use_color, ignore_shadows, float(size_override) or None,
                sens, min_rel, min_contrast, count_clumps)

    # ---------- STEP 1 ----------
    st.header("Step 1. Background Removal & Contrast")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption=f"Original ({img.shape[1]}×{img.shape[0]})", use_container_width=True)
    with col2:
        st.image(to_display(r["contrast"], 2 * r["threshold"]),
                 caption="Contrast from background", use_container_width=True)
    with col3:
        ov = img.copy()
        ov[r["shadow"]] = (0.4 * ov[r["shadow"]] + 0.6 * np.array([0, 120, 255])).astype(np.uint8)
        st.image(ov, caption="Ignored shadows / dark lines (blue)", use_container_width=True)

    # ---------- STEP 2 ----------
    st.header("Step 2. Thresholding & Morphology")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(r["mask0"].astype(np.uint8) * 255,
                 caption=f"Otsu + hysteresis (threshold = {r['threshold']:.1f})", use_container_width=True)
    with col2:
        st.image(r["mask"].astype(np.uint8) * 255,
                 caption=f"Opening / closing at object scale (size ≈ {r['S']:.0f} px)",
                 use_container_width=True)
    with col3:
        st.image(colorize(r["kept"]), caption="Kept objects (one color each)", use_container_width=True)

    # ---------- STEP 3 ----------
    st.header("Step 3. Object Detection & Counting")
    result = image.copy()
    draw = ImageDraw.Draw(result)
    lw = max(2, img.shape[1] // 400)
    for i, o in enumerate(r["objects"], 1):
        color = "red" if o["count"] == 1 else "cyan"
        draw.rectangle(o["box"], outline=color, width=lw)
        label = str(i) if o["count"] == 1 else f"{i} (×{o['count']})"
        draw.text((o["box"][0] + 2, o["box"][1] + 1), label, fill="yellow")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(result, caption="Red = single object, cyan = touching objects counted by area",
                 use_container_width=True)
    with col2:
        st.metric("Total Objects Detected", r["total"])
        st.caption(f"{len(r['objects'])} blobs · auto object size ≈ {r['S']:.0f} px · "
                   f"typical area ≈ {r['typical'] or 0:.0f} px")
        if len(r["objects"]):
            st.subheader("Kept object areas")
            a = np.array([o["area"] for o in r["objects"]])
            bins = np.unique(np.geomspace(a.min(), max(a.max(), a.min() + 1), 20).astype(int))
            hist, edges = np.histogram(a, bins=bins) if len(bins) > 1 else (np.array([len(a)]), np.array([a.min(), a.min()]))
            st.bar_chart(pd.DataFrame({"objects": hist}, index=pd.Index(edges[:-1], name="area (px)")))
else:
    st.info("Upload an image from the sidebar.")
