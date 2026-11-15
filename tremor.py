"""
레이저 포인터 기반 손떨림 분석 모듈.

처리 순서
  1) 프레임별 레이저 점 검출 (R-G 차이 + 밝기) -> 서브픽셀 중심
  2) 레이저 '발사 시점' = 측정 화면(마지막 장면과 유사)에서 레이저가
     연속 MIN_RUN 프레임 이상 검출되기 시작한 프레임 (칠판에 비친 레이저 제외)
  3) 분석 구간 = 발사 + start_offset_s  ~  + window_s
  4) 배경(레이저 제거) 영상 = 분석 구간 프레임들의 중앙값
       -> 종이 영역, 모눈 간격(px), 검은 기준점 검출
  5) px -> mm (모눈 간격) -> mrad (촬영 거리)
  6) 특징값: 떨림 RMS, 주파수 피크, 대역 파워 비율, 조준 오차, 흩어짐 타원
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
import cv2
from scipy import signal

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x/2.x 호환

PROC_WIDTH = 1920        # 검출용 해상도 (4K 영상은 절반으로 축소)
MIN_RUN = 6              # 발사 판정용 연속 검출 프레임 수 (~0.2 s @30fps)
SCENE_THR = 0.9          # 마지막 장면과의 정규화 상관 (칠판 장면 제외용)
N_BG = 15                # 배경(레이저 제거)용 프레임 수
TREMOR_BAND = (3.0, 12.0)
PD_BAND = (3.5, 6.5)
ET_BAND = (4.0, 8.0)
PHYS_BAND = (8.0, 12.0)


# --------------------------------------------------------------------------------------
# 1. 레이저 검출
# --------------------------------------------------------------------------------------
def detect_laser(small_bgr: np.ndarray):
    """축소 프레임에서 레이저 점 중심(x, y, area) 반환. 없으면 None."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(small_bgr)]
    redness = r - g
    mask = ((redness > 60) & (r > 150)).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    if area < 8 or area > 20000:
        return None
    # 주변을 조금 넓혀 밝기 가중 중심 (흰 포화 코어 + 마젠타 헤일로 모두 반영)
    pad = 6
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, small_bgr.shape[1]), min(y + h + pad, small_bgr.shape[0])
    roi = small_bgr[y0:y1, x0:x1].astype(np.float32)
    wgt = np.clip(roi[..., 2] - roi[..., 1], 0, None) + np.clip(roi.sum(2) - 600, 0, None)
    wgt[wgt < 40] = 0
    s = wgt.sum()
    if s <= 0:
        return None
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    return (float((wgt * xx).sum() / s + x0), float((wgt * yy).sum() / s + y0), int(area))


def find_onset(present: np.ndarray, min_run: int = MIN_RUN) -> int | None:
    run = 0
    for i, p in enumerate(present):
        run = run + 1 if p else 0
        if run >= min_run:
            return i - min_run + 1
    return None


# --------------------------------------------------------------------------------------
# 2. 배경 분석: 종이 영역, 모눈 간격, 기준점
# --------------------------------------------------------------------------------------
def paper_mask(bg_bgr: np.ndarray) -> np.ndarray:
    """흰 종이(약간 푸른 회색) vs 나무 배경(주황 계열)을 색상으로 분리."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(bg_bgr)]
    v = bg_bgr.max(2)
    m = ((b - r) > -2) & (v > 90)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if num <= 1:
        return np.ones(bg_bgr.shape[:2], bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def _autocorr(prof: np.ndarray) -> np.ndarray:
    prof = prof.astype(np.float32)
    prof = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 15).ravel()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    return ac / ac[0] if ac[0] > 0 else ac


def grid_pitch(gray: np.ndarray, pmask: np.ndarray, lo: int = 8, hi: int = 150) -> float:
    """모눈 간격(px). 가로·세로 프로파일 자기상관을 합친 뒤 기본 주기를 찾는다."""
    ys, xs = np.where(pmask)
    y0, y1 = np.percentile(ys, [15, 85]).astype(int)
    x0, x1 = np.percentile(xs, [15, 85]).astype(int)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    acx, acy = _autocorr(patch.mean(0)), _autocorr(patch.mean(1))
    L = min(len(acx), len(acy), hi + 2)
    if L <= lo + 3:
        return np.nan
    ac = acx[:L] + acy[:L]
    pk, props = signal.find_peaks(ac[lo:L], prominence=0.05)
    if len(pk) == 0:
        return np.nan
    pk = pk + lo
    prom = props["prominences"]
    p0 = int(pk[prom >= 0.3 * prom.max()].min())       # 기본 주기 후보 (배수 피크 오인 방지)
    # 정밀화: k*p0 근처 피크들의 위치를 k로 나눠 평균
    ests = []
    for k in range(1, (L - 2) // p0 + 1):
        c = k * p0
        w = max(2, p0 // 4)
        a, b = max(c - w, 1), min(c + w, L - 2)
        j = a + int(np.argmax(ac[a:b + 1]))
        y_1, y0_, y1_ = ac[j - 1], ac[j], ac[j + 1]
        den = y_1 - 2 * y0_ + y1_
        d = 0.5 * (y_1 - y1_) / den if den != 0 else 0.0
        ests.append((j + d) / k)
    return float(np.median(ests))


def find_reference_dot(gray: np.ndarray, pmask: np.ndarray, pitch: float):
    """종이 안의 검고 둥근 점 (격자선/축선/삼각 눈금은 열림 연산으로 제거)."""
    inner = cv2.erode(pmask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    paper_med = np.median(gray[inner]) if inner.any() else 180
    dark = ((gray < paper_med * 0.5) & inner).astype(np.uint8) * 255
    k = max(3, int(round(pitch * 0.2)) | 1) if np.isfinite(pitch) else 5
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    num, lab, stats, cents = cv2.connectedComponentsWithStats(dark)
    ys, xs = np.where(pmask)
    pc = np.array([xs.mean(), ys.mean()])
    best = None
    amin = (0.15 * pitch) ** 2 * np.pi if np.isfinite(pitch) else 30
    amax = (1.5 * pitch) ** 2 * np.pi if np.isfinite(pitch) else 5000
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if not (amin < a < amax):
            continue
        if not (0.6 < w / h < 1.6) or a / (w * h) < 0.55:
            continue
        d = np.linalg.norm(cents[i] - pc)
        if best is None or d < best[0]:
            best = (d, cents[i])
    return None if best is None else (float(best[1][0]), float(best[1][1]))


# --------------------------------------------------------------------------------------
# 3. 결과 구조
# --------------------------------------------------------------------------------------
@dataclass
class TremorResult:
    fps: float
    n_frames: int
    onset_s: float | None
    win_start_s: float
    win_end_s: float
    px_per_mm: float
    ref_px: tuple | None
    ref_source: str                       # 'dot' | 'mean'
    detect_rate: float                    # 분석 구간 내 레이저 검출 비율
    # 특징값
    tremor_rms_mm: float
    tremor_rms_mrad: float | None
    peak_freq_hz: float
    peak_prominence_db: float             # 1/f 배경 대비 피크 높이
    peak_rms_mm: float                    # 주 피크(±1 Hz) 성분의 RMS
    peak_rms_mrad: float | None
    peak_power_ratio: float               # 피크 대역 파워 / 3-12 Hz 전체 파워
    pd_band_ratio: float
    et_band_ratio: float
    phys_band_ratio: float
    drift_rms_mm: float                   # < 3 Hz 성분 (자발적 보정)
    mean_offset_mm: tuple | None          # 기준점 대비 평균 위치 (기준점 있을 때만)
    ellipse95_area_mm2: float
    # 시계열 (그래프용)
    t: np.ndarray = field(repr=False, default=None)
    x_mm: np.ndarray = field(repr=False, default=None)
    y_mm: np.ndarray = field(repr=False, default=None)
    xf_mm: np.ndarray = field(repr=False, default=None)
    yf_mm: np.ndarray = field(repr=False, default=None)
    freqs: np.ndarray = field(repr=False, default=None)
    psd: np.ndarray = field(repr=False, default=None)
    psd_bg: np.ndarray = field(repr=False, default=None)
    preview_bgr: np.ndarray = field(repr=False, default=None)
    warnings: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("t", "x_mm", "y_mm", "xf_mm", "yf_mm", "freqs", "psd", "psd_bg", "preview_bgr"):
            d.pop(k, None)
        return d


# --------------------------------------------------------------------------------------
# 4. 메인 파이프라인
# --------------------------------------------------------------------------------------
def _thumb(fr):
    g = cv2.cvtColor(cv2.resize(fr, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return ((g - g.mean()) / (g.std() + 1e-6)).astype(np.float16)


def scan_video(path: str, progress_cb=None) -> dict:
    """1차 패스: 전체 프레임에서 레이저 좌표(검출 해상도)와 장면 썸네일 수집."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다 (코덱/파일 확인).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pts, thumbs, scale = [], [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if scale is None:
            scale = min(1.0, PROC_WIDTH / fr.shape[1])
        small = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else fr
        det = detect_laser(small)
        pts.append((det[0], det[1]) if det else (np.nan, np.nan))
        thumbs.append(_thumb(fr))
        if progress_cb and n_total:
            progress_cb(min(len(pts) / n_total, 1.0))
    cap.release()
    if not pts:
        raise RuntimeError("영상에서 프레임을 읽지 못했습니다.")
    P = np.array(pts, float)
    n = len(P)
    tgt = np.median(np.array(thumbs[int(n * 0.7):], np.float32), 0)
    tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-6)
    scene = np.array([float((th.astype(np.float32) * tgt).mean()) for th in thumbs])  # 측정 화면과의 유사도
    return dict(fps=float(fps), scale=scale, pts=P, scene=scene, n=n)


def find_onset_scene(pts, scene, min_run=MIN_RUN, scene_thr=SCENE_THR):
    ok = (~np.isnan(pts[:, 0])) & (scene >= scene_thr)
    return find_onset(ok, min_run)


def analyze_video(path: str, window_s: float = 7.0, start_offset_s: float = 1.0,
                  grid_mm: float = 5.0, distance_cm: float | None = None,
                  onset_override_s: float | None = None, progress_cb=None,
                  scan: dict | None = None) -> TremorResult:
    scan = scan or scan_video(path, progress_cb)
    fps, scale, P, scene, n = scan["fps"], scan["scale"], scan["pts"].copy(), scan["scene"], scan["n"]
    warnings = []

    if onset_override_s is not None:
        onset = int(round(onset_override_s * fps))
    else:
        onset = find_onset_scene(P, scene)
        if onset is None:
            raise RuntimeError("측정 화면에서 레이저 점을 찾지 못했습니다. 밝기/노출이나 레이저 색을 확인하세요.")
    win0 = onset + int(round(start_offset_s * fps))
    win1 = win0 + int(round(window_s * fps))
    if win0 >= n - 5:
        raise RuntimeError("발사 시점 이후 영상이 너무 짧습니다.")
    if win1 > n:
        warnings.append(f"영상 길이가 부족합니다: 요청 {window_s:.1f}s 중 {(n - win0) / fps:.2f}s만 사용")
        win1 = n

    seg = P[win0:win1].copy()
    det_rate = float(np.mean(~np.isnan(seg[:, 0])))
    if det_rate < 0.9:
        warnings.append(f"분석 구간에서 레이저 검출률이 낮습니다 ({det_rate * 100:.0f}%)")
    if np.mean(scene[win0:win1] < SCENE_THR) > 0.05:
        warnings.append("분석 구간 중 카메라/장면이 바뀐 프레임이 있습니다 (삼각대 고정 확인)")
    for c in range(2):
        v = seg[:, c]
        m = np.isnan(v)
        if m.all():
            raise RuntimeError("분석 구간에서 레이저가 검출되지 않았습니다.")
        if m.any():
            v[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), v[~m])

    # ---- 2차 패스: 분석 구간 프레임 N_BG장으로 배경(레이저 제거) 생성 ----
    # 메모리 절약: uint8 배열 하나에 바로 채우고, 중앙값은 행 단위로 나눠 계산
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, win0)
    step = max(1, (win1 - win0) // N_BG)
    stack = None
    k = 0
    last = min(win1 - win0, step * N_BG)
    for j in range(last):
        ok, fr = cap.read()
        if not ok:
            break
        if j % step == 0 and k < N_BG:
            sm = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else fr
            if stack is None:
                stack = np.empty((N_BG,) + sm.shape, np.uint8)
            stack[k] = sm
            k += 1
        del fr
    cap.release()
    if stack is None or k == 0:
        raise RuntimeError("분석 구간 프레임을 다시 읽지 못했습니다.")
    stack = stack[:k]
    bg = np.empty(stack.shape[1:], np.uint8)
    for r0 in range(0, bg.shape[0], 64):
        bg[r0:r0 + 64] = np.median(stack[:, r0:r0 + 64], axis=0).astype(np.uint8)
    del stack
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    pmask = paper_mask(bg)
    pitch = grid_pitch(gray, pmask)
    if not np.isfinite(pitch):
        raise RuntimeError("모눈 간격을 측정하지 못했습니다 (종이가 화면에 충분히 보이는지 확인).")
    px_per_mm = pitch / grid_mm        # 검출 해상도 기준
    ref = find_reference_dot(gray, pmask, pitch)
    if ref is None:
        ref_source = "mean"
        warnings.append("기준점(검은 점)을 찾지 못해 레이저 평균 위치를 기준으로 사용했습니다. "
                        "떨림 주파수·세기에는 영향이 없고, 조준 오차만 계산되지 않습니다.")
        ref_xy = seg.mean(0)
    else:
        ref_source = "dot"
        ref_xy = np.array(ref)

    t = (np.arange(win0, win1) - onset) / fps
    x = (seg[:, 0] - ref_xy[0]) / px_per_mm
    y = -(seg[:, 1] - ref_xy[1]) / px_per_mm        # 위쪽 +
    feats = compute_features(x, y, fps, distance_cm)

    # ---- 미리보기 이미지 (배경 + 궤적 + 기준점 + 종이 윤곽) ----
    prev = bg.copy()
    contours, _ = cv2.findContours(pmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(prev, contours, -1, (255, 200, 0), 2)
    cv2.polylines(prev, [np.round(seg).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
    col = (0, 200, 0) if ref_source == "dot" else (255, 140, 0)
    cv2.drawMarker(prev, (int(ref_xy[0]), int(ref_xy[1])), col, cv2.MARKER_CROSS, 40, 2)

    return TremorResult(
        fps=float(fps), n_frames=n, onset_s=onset / fps,
        win_start_s=win0 / fps, win_end_s=win1 / fps,
        px_per_mm=float(px_per_mm / scale), ref_px=tuple((ref_xy / scale).round(1).tolist()),
        ref_source=ref_source, detect_rate=det_rate,
        mean_offset_mm=(float(x.mean()), float(y.mean())) if ref_source == "dot" else None,
        t=t, x_mm=x, y_mm=y, preview_bgr=prev, warnings=warnings, **feats)


def _band_power(f, P, band):
    m = (f >= band[0]) & (f <= band[1])
    return float(_trapz(P[m], f[m])) if m.sum() > 1 else 0.0


def compute_features(x, y, fps, distance_cm=None) -> dict:
    nyq = fps / 2
    hi = min(TREMOR_BAND[1], nyq * 0.9)
    b, a = signal.butter(3, [TREMOR_BAND[0], hi], btype="band", fs=fps)
    xf, yf = signal.filtfilt(b, a, x), signal.filtfilt(b, a, y)
    bl, al = signal.butter(3, TREMOR_BAND[0], btype="low", fs=fps)
    xl, yl = signal.filtfilt(bl, al, signal.detrend(x)), signal.filtfilt(bl, al, signal.detrend(y))

    rms = float(np.sqrt(np.mean(xf ** 2 + yf ** 2)))
    drift = float(np.sqrt(np.mean(xl ** 2 + yl ** 2)))

    nper = min(len(x), int(round(4 * fps)))   # 4 s 창 -> 0.25 Hz 분해능
    f, Px = signal.welch(signal.detrend(x), fs=fps, nperseg=nper, noverlap=nper // 2, nfft=4 * nper)
    _, Py = signal.welch(signal.detrend(y), fs=fps, nperseg=nper, noverlap=nper // 2, nfft=4 * nper)
    P = Px + Py
    # 1/f(드리프트) 배경을 로그-로그 직선으로 근사 -> 그 위로 솟은 피크를 떨림으로 판단
    fit = (f >= 1.0) & (f <= min(14.0, nyq * 0.95))
    coef = np.polyfit(np.log10(f[fit]), np.log10(P[fit] + 1e-12), 1)
    bgfit = 10 ** np.polyval(coef, np.log10(np.maximum(f, 1e-3)))
    resid_db = 10 * np.log10((P + 1e-12) / bgfit)
    band = (f >= TREMOR_BAND[0]) & (f <= hi)
    j = np.argmax(resid_db[band])
    fpk = float(f[band][j])
    prom_db = float(resid_db[band][j])
    tot = _band_power(f, P, (TREMOR_BAND[0], hi)) or 1e-12
    pk_rms = float(np.sqrt(_band_power(f, P, (fpk - 1.0, fpk + 1.0))))
    pk_ratio = _band_power(f, P, (fpk - 1.0, fpk + 1.0)) / tot

    # 95% 신뢰 타원 면적 (원신호)
    cov = np.cov(np.vstack([x, y]))
    ell = float(np.pi * 5.991 * np.sqrt(max(np.linalg.det(cov), 0)))

    return dict(
        tremor_rms_mm=rms,
        tremor_rms_mrad=(rms / (distance_cm * 10) * 1000) if distance_cm else None,
        peak_freq_hz=fpk, peak_prominence_db=prom_db,
        peak_rms_mm=pk_rms, peak_rms_mrad=(pk_rms / (distance_cm * 10) * 1000) if distance_cm else None, peak_power_ratio=float(pk_ratio),
        pd_band_ratio=_band_power(f, P, PD_BAND) / tot,
        et_band_ratio=_band_power(f, P, ET_BAND) / tot,
        phys_band_ratio=_band_power(f, P, (PHYS_BAND[0], min(PHYS_BAND[1], hi))) / tot,
        drift_rms_mm=drift, ellipse95_area_mm2=ell,
        xf_mm=xf, yf_mm=yf, freqs=f, psd=P, psd_bg=bgfit,
    )


# --------------------------------------------------------------------------------------
# 5. 판정 (규칙 기반, 연구·교육용)
# --------------------------------------------------------------------------------------
@dataclass
class Criteria:
    amp_thr: float = 1.0            # 병적 떨림 진폭 기준 (주 피크 RMS, mrad / 거리 미입력시 mm로 해석)
    prom_thr_db: float = 6.0        # 1/f 배경 대비 피크 돌출 기준
    path_band: tuple = (3.5, 8.0)   # 병적 떨림 주파수 대역
    pd_max_freq: float = 6.5        # 파킨슨 안정 떨림 상한 주파수
    rest_dominance: float = 1.2     # 이완/수축 진폭비가 이 값 이상이면 '안정 시 우세'


def _amp(r: TremorResult):
    return r.peak_rms_mrad if r.peak_rms_mrad is not None else r.peak_rms_mm


def is_pathological(r: TremorResult, c: Criteria) -> tuple[bool, list]:
    checks = [
        (c.path_band[0] <= r.peak_freq_hz <= c.path_band[1],
         f"주 피크 {r.peak_freq_hz:.1f} Hz 가 {c.path_band[0]}–{c.path_band[1]} Hz 대역"),
        (r.peak_prominence_db >= c.prom_thr_db,
         f"피크 돌출 {r.peak_prominence_db:.1f} dB ≥ {c.prom_thr_db} dB"),
        (_amp(r) >= c.amp_thr,
         f"피크 진폭 {_amp(r):.2f} ≥ {c.amp_thr}"),
    ]
    return all(ok for ok, _ in checks), checks


def classify(rest: TremorResult | None, contr: TremorResult | None, c: Criteria = Criteria()) -> dict:
    out = {"label": None, "reasons": [], "detail": {}}
    pr = is_pathological(rest, c) if rest else (False, [])
    pc = is_pathological(contr, c) if contr else (False, [])
    out["detail"] = {"이완": pr[1], "수축": pc[1]}
    ar = _amp(rest) if rest else None
    ac = _amp(contr) if contr else None
    ratio = (ar / ac) if (ar is not None and ac) else None
    out["rest_contr_ratio"] = ratio

    if not pr[0] and not pc[0]:
        out["label"] = "정상 범위"
        if contr is not None and 8.0 <= contr.peak_freq_hz <= 12.0:
            out["reasons"].append(f"수축 시 {contr.peak_freq_hz:.1f} Hz 피크는 정상 생리적 떨림(8–12 Hz) 대역")
        out["reasons"].append("두 조건 모두 병적 떨림 기준(주파수·돌출·진폭)을 동시에 만족하지 않음")
    elif pr[0] and (not pc[0] or (ratio is not None and ratio >= c.rest_dominance)) \
            and rest.peak_freq_hz <= c.pd_max_freq:
        out["label"] = "파킨슨병 패턴 의심"
        out["reasons"].append(f"이완(안정) 시 {rest.peak_freq_hz:.1f} Hz 떨림이 두드러짐")
        if ratio is not None:
            out["reasons"].append(f"이완/수축 진폭비 {ratio:.2f} (안정 시 떨림 우세)")
    elif pc[0] and (not pr[0] or (ratio is not None and ratio < c.rest_dominance)):
        out["label"] = "본태성 떨림(수전증) 패턴 의심"
        out["reasons"].append(f"수축(자세 유지) 시 {contr.peak_freq_hz:.1f} Hz 떨림이 두드러짐")
        if ratio is not None:
            out["reasons"].append(f"이완/수축 진폭비 {ratio:.2f} (자세·동작 시 떨림 우세)")
    else:
        out["label"] = "판단 보류 (혼합 패턴)"
        out["reasons"].append("두 조건 모두 병적 떨림 기준을 만족하거나 패턴이 엇갈림")
    if rest is None or contr is None:
        out["reasons"].append("한 조건만 입력되어 이완/수축 비교 없이 판정함 (신뢰도 낮음)")
    return out
