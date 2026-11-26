"""
[벽 실험용] 레이저 포인터 손떨림 분석 모듈 — wall_tremor.py
모눈종이 대신 벽에 그은 10 cm 가로선·세로선으로 길이 축척을 정한다.

처리 순서
  1) 프레임별 레이저 점 검출 (R-G 차이 + 밝기) -> 서브픽셀 중심
  2) 레이저 '발사 시점' = 측정 화면에서 레이저가 연속 MIN_RUN 프레임 이상 검출되기 시작한 프레임
  3) 분석 구간 = 발사 + start_offset_s  ~  + window_s
  4) 배경(레이저 제거) 영상 = 분석 구간 프레임들의 중앙값
       -> 검은 가로선·세로선 검출, 각 선의 픽셀 길이 측정
  5) 가로 축척(px/mm) = 가로선 길이 / 실제 길이, 세로 축척 = 세로선 길이 / 실제 길이
     (가로·세로를 따로 재므로 카메라가 약간 비스듬해도 보정됨)
     두 선이 교차하면 교차점을 조준 기준점으로 사용
  6) mm -> mrad (촬영 거리·높이 차) / 특징값: 떨림 RMS, 주파수 피크, 조준 오차 등
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
# 2. 배경 분석: 10 cm 가로선·세로선 검출
# --------------------------------------------------------------------------------------
LINE_KERNEL = 41          # 이보다 짧은 획은 선으로 보지 않음 (검출 해상도 px)


def _segments(mask: np.ndarray, horizontal: bool):
    """선 마스크에서 선분 후보 [(길이, 두께, p1, p2, 중심, 어두운정도)] 반환. 주축(PCA) 방향 끝점 사이 길이."""
    out = []
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 3 * LINE_KERNEL:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        c = pts.mean(0)
        u = np.linalg.svd(pts - c, full_matrices=False)[2][0]     # 주축 방향
        if (abs(u[0]) < 0.94) if horizontal else (abs(u[1]) < 0.94):   # 약 ±20° 이내만
            continue
        proj = (pts - c) @ u
        L_ext = float(proj.max() - proj.min() + 1)      # 잉크 끝에서 끝까지
        thick = a / L_ext
        if L_ext < 6 * thick:
            continue
        # 펜 중심을 0과 10 cm 눈금에 맞춰 그으면 잉크가 양 끝에서 두께/2씩 더 나감 -> 두께만큼 뺌
        L = L_ext - thick
        out.append(dict(length=L, thick=float(thick), p1=c + u * proj.min(), p2=c + u * proj.max(), center=c))
    return out


def _dist_to_seg(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return float(np.hypot(*(p - (a + t * ab))))


def find_scale_lines(gray: np.ndarray, near_xy):
    """벽의 검은 가로선·세로선을 찾는다. 레이저 위치에 가장 가까운 선을 각각 선택.
    반환: dict(h=선분 or None, v=선분 or None, cross=교차점 or None)"""
    bgl = cv2.medianBlur(gray, 61)                          # 국소 벽 밝기
    diff = bgl.astype(np.int16) - gray.astype(np.int16)
    dark = (diff > 45).astype(np.uint8) * 255
    hmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_KERNEL, 1)))
    vmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_KERNEL)))
    near = np.asarray(near_xy, float)
    res = {}
    for key, m, hz in (("h", hmask, True), ("v", vmask, False)):
        segs = _segments(m, hz)
        res[key] = min(segs, key=lambda s_: _dist_to_seg(near, s_["p1"], s_["p2"])) if segs else None
    res["cross"] = None
    if res["h"] and res["v"]:
        h, v = res["h"], res["v"]
        # 두 직선의 교점
        d1, d2 = h["p2"] - h["p1"], v["p2"] - v["p1"]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) > 1e-6:
            t1, t2 = np.linalg.solve(A, v["p1"] - h["p1"])
            if -0.02 <= t1 <= 1.02 and -0.02 <= t2 <= 1.02:     # 두 선분이 실제로 교차할 때만
                res["cross"] = h["p1"] + t1 * d1
    return res


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
    px_per_mm_x: float                    # 가로 축척 (원본 영상 px/mm)
    px_per_mm_y: float                    # 세로 축척
    scale_source: str                     # 'lines' | 'manual'
    line_len_px: tuple | None             # (가로선, 세로선) 픽셀 길이 (원본 해상도)
    ref_px: tuple | None
    ref_source: str                       # 'cross'(선 교차점) | 'mean'
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
    descent: bool = False                 # 처음 수평 위치 -> 이완 위치 하강이 검출됐는지
    move_s: float | None = None           # 하강 시작 시각
    arrive_s: float | None = None         # 이완 위치 도착 시각 (분석 구간 기준)
    drop_mm: float | None = None          # 처음 위치와 이완 위치의 세로 차이 (아래 +)
    height_used_cm: float = 0.0           # 각도 계산에 쓴 높이 차이
    height_source: str = "manual"         # 'auto'(하강으로 자동 계산) | 'manual'
    elev_angle_deg: float | None = None   # 레이저가 아래로 기운 각도 = atan(하강 / 수평거리)
    slant_cm: float | None = None         # 손~이완 위치 경사 거리 = sqrt(D^2 + 하강^2)
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
def find_descent(P: np.ndarray, onset: int, fps: float, frame_h: int):
    """처음 수평으로 겨눈 위치에서 팔을 이완해 레이저가 아래로 내려가는 구간을 찾는다.
    반환: dict(init_xy, move_idx, arrive_idx, final_level) 또는 None(하강 없음)
      - init_xy   : 처음 수평 위치 대표값 (움직이기 전 구간의 중앙값, 검출 해상도 px)
      - move_idx  : 하강 시작 프레임
      - arrive_idx: 이완 위치 도착 프레임 (여기서부터 분석 구간을 잡음)"""
    pres = np.flatnonzero(~np.isnan(P[:, 0]))
    pres = pres[pres >= onset]
    if len(pres) < int(3 * fps):
        return None
    last = pres[-1]
    xy = P[onset:last + 1].copy()
    for c in range(2):                                   # 끊긴 프레임 보간
        v = xy[:, c]; m = np.isnan(v)
        if m.any():
            v[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), v[~m])
    k = max(3, int(round(0.4 * fps)) | 1)               # 0.4 s 중앙값 필터: 떨림 제거, 큰 움직임만 남김
    ys = signal.medfilt(xy[:, 1], k)
    i0 = int(round(0.2 * fps))                           # 발사 직후 0.2 s는 버림
    thr = max(40.0, 0.04 * frame_h)                      # 하강으로 볼 최소 이동량 (px)
    y0 = np.median(ys[i0:i0 + max(3, int(0.3 * fps))])
    final = float(np.median(ys[-int(min(3.0 * fps, len(ys) // 3)):]))   # 마지막 구간 = 이완 위치
    if final - y0 < 3 * thr:                             # 충분히 내려가지 않음 -> 하강 없음
        return None
    moved = np.flatnonzero(ys[i0:] - y0 > thr)
    if len(moved) == 0:
        return None
    mv = i0 + int(moved[0])
    init_seg = xy[i0:max(i0 + 1, mv - int(0.1 * fps))]
    init_xy = np.median(init_seg, axis=0)
    # 도착 = 전체 하강의 90 %를 지난 뒤, 0.2 s 동안 움직임이 작아진 첫 시점 (최대 1 s 더 기다림)
    cross = np.flatnonzero(ys[mv:] >= y0 + 0.9 * (final - y0))
    if len(cross) == 0:
        return None
    a0 = mv + int(cross[0])
    w = max(2, int(round(0.2 * fps)))
    arrive = a0
    for j in range(a0, min(a0 + int(fps), len(ys) - w)):
        if abs(ys[j + w] - ys[j]) < thr * 0.5:
            arrive = j
            break
    return dict(init_xy=init_xy, init_n=len(init_seg), move_idx=onset + mv, arrive_idx=onset + arrive,
                final_level=final)


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
            frame_h0 = fr.shape[0]
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
    return dict(fps=float(fps), scale=scale, pts=P, scene=scene, n=n, h=frame_h0)


def find_onset_scene(pts, scene, min_run=MIN_RUN, scene_thr=SCENE_THR):
    ok = (~np.isnan(pts[:, 0])) & (scene >= scene_thr)
    return find_onset(ok, min_run)


def analyze_video(path: str, window_s: float = 7.0, start_offset_s: float = 1.0,
                  distance_cm: float | None = None, height_cm: float = 0.0, height_mode: str = "auto",
                  line_h_mm: float = 100.0, line_v_mm: float = 100.0,
                  scale_mode: str = "lines", manual_ppm_x: float | None = None, manual_ppm_y: float | None = None,
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
    desc = find_descent(P, onset, fps, frame_h=int(round(scan.get("h", 1080) * scale))) if height_mode == "auto" else None
    base = desc["arrive_idx"] if desc else onset        # 하강이 있으면 이완 위치 도착 시점이 기준
    win0 = base + int(round(start_offset_s * fps))
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
    laser_c = seg.mean(0)
    lines = None
    ref = None
    if scale_mode == "manual":
        if not (manual_ppm_x and manual_ppm_y):
            raise RuntimeError("직접 입력 모드: 가로·세로 px/mm 값을 입력하세요.")
        ppm_x, ppm_y = manual_ppm_x * scale, manual_ppm_y * scale     # 원본 해상도 입력 -> 검출 해상도
    else:
        lines = find_scale_lines(gray, laser_c)
        h, v = lines["h"], lines["v"]
        if h is None and v is None:
            raise RuntimeError("벽의 가로선·세로선을 찾지 못했습니다. 진한 검은 선이 화면에 잘 보이는지 확인하거나 "
                               "'직접 입력' 모드를 쓰세요.")
        ppm_x = h["length"] / line_h_mm if h else None
        ppm_y = v["length"] / line_v_mm if v else None
        if ppm_x is None:
            ppm_x = ppm_y
            warnings.append("가로선을 찾지 못해 세로선 축척을 가로에도 적용했습니다.")
        if ppm_y is None:
            ppm_y = ppm_x
            warnings.append("세로선을 찾지 못해 가로선 축척을 세로에도 적용했습니다.")
        if h and v and abs(ppm_x / ppm_y - 1) > 0.05:
            warnings.append(f"가로·세로 축척 차이가 {abs(ppm_x / ppm_y - 1) * 100:.1f}% 입니다. "
                            "카메라가 벽에 비스듬하거나 선 길이가 다를 수 있습니다.")
        for nm, sg in (("가로선", h), ("세로선", v)):
            if sg and sg["length"] < 150:
                warnings.append(f"{nm}이 {sg['length'] / scale:.0f} px로 짧아 축척 오차가 클 수 있습니다.")
        if lines["cross"] is not None:
            ref = lines["cross"]
    if ref is None:
        ref_source = "mean"
        ref_xy = laser_c
    else:
        ref_source = "cross"
        ref_xy = np.array(ref)

    t = (np.arange(win0, win1) - onset) / fps
    x = (seg[:, 0] - ref_xy[0]) / ppm_x
    y = -(seg[:, 1] - ref_xy[1]) / ppm_y        # 위쪽 +
    drop_mm = None
    h_used, h_src = (height_cm or 0.0), "manual"
    if desc is not None:
        final_xy = seg.mean(0)
        drop_mm = float((final_xy[1] - desc["init_xy"][1]) / ppm_y)   # 화면 아래 = +
        h_used, h_src = drop_mm / 10.0, "auto"
        if desc["init_n"] < int(0.3 * fps):
            warnings.append("처음 수평 위치를 유지한 시간이 짧아(0.3 s 미만) 시작 위치 대표값이 부정확할 수 있습니다.")
    elif height_mode == "auto":
        h_used, h_src = 0.0, "auto"   # 하강 없음 = 처음부터 수평으로 겨눈 상태 (예: 수축 조건)
    elev = slant = None
    if distance_cm:
        elev = float(np.degrees(np.arctan2(abs(h_used), distance_cm)))
        slant = float(np.hypot(distance_cm, h_used))
    feats = compute_features(x, y, fps, distance_cm, h_used)

    # ---- 미리보기 이미지 (배경 + 검출된 선 + 궤적 + 기준점) ----
    prev = bg.copy()
    if lines is not None:
        for key, col in (("h", (255, 0, 255)), ("v", (255, 160, 0))):
            sg = lines[key]
            if sg:
                p1, p2 = tuple(np.round(sg["p1"]).astype(int)), tuple(np.round(sg["p2"]).astype(int))
                cv2.line(prev, p1, p2, col, 3, cv2.LINE_AA)
                for p in (p1, p2):
                    cv2.circle(prev, p, 8, col, 2, cv2.LINE_AA)
    cv2.polylines(prev, [np.round(seg).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
    col = (0, 200, 0) if ref_source == "cross" else (0, 140, 255)
    cv2.drawMarker(prev, (int(ref_xy[0]), int(ref_xy[1])), col, cv2.MARKER_CROSS, 40, 2)
    if desc is not None:
        p0 = (int(desc["init_xy"][0]), int(desc["init_xy"][1]))
        p1 = (int(seg[:, 0].mean()), int(seg[:, 1].mean()))
        cv2.line(prev, p0, (p0[0], p1[1]), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(prev, p0, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
        cv2.putText(prev, f"drop {drop_mm:.0f} mm", (p0[0] + 12, (p0[1] + p1[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    ll = None
    if lines is not None:
        ll = tuple(round(lines[k]["length"] / scale, 1) if lines[k] else None for k in ("h", "v"))
    return TremorResult(
        fps=float(fps), n_frames=n, onset_s=onset / fps,
        win_start_s=win0 / fps, win_end_s=win1 / fps,
        px_per_mm_x=float(ppm_x / scale), px_per_mm_y=float(ppm_y / scale), scale_source=scale_mode,
        line_len_px=ll, ref_px=tuple((ref_xy / scale).round(1).tolist()),
        ref_source=ref_source, detect_rate=det_rate,
        mean_offset_mm=(float(x.mean()), float(y.mean())) if ref_source == "cross" else None,
        descent=desc is not None, move_s=(desc["move_idx"] / fps) if desc else None,
        arrive_s=(desc["arrive_idx"] / fps) if desc else None, drop_mm=drop_mm,
        height_used_cm=float(h_used), height_source=h_src, elev_angle_deg=elev, slant_cm=slant,
        t=t, x_mm=x, y_mm=y, preview_bgr=prev, warnings=warnings, **feats)


def _band_power(f, P, band):
    m = (f >= band[0]) & (f <= band[1])
    return float(_trapz(P[m], f[m])) if m.sum() > 1 else 0.0


def angle_scale(distance_cm, height_cm=0.0):
    """종이 위 변위(mm) -> 손의 회전각(mrad) 환산 계수 (x, y).
    distance_cm: 레이저에서 종이 벽까지의 수평(수직 입사) 거리 D
    height_cm  : 레이저 높이와 표적 높이의 차이 h (부호 무관)
      경사 거리 L = sqrt(D^2 + h^2), 앙각 θ = atan(h / D)
      가로: dx = L·dα          -> dα = dx / L
      세로: dy = L·dα / cosθ   -> dα = dy·cosθ / L  (비스듬히 맞아 세로로 늘어남)"""
    if not distance_cm:
        return None
    D = distance_cm * 10.0
    h = (height_cm or 0.0) * 10.0
    L = float(np.hypot(D, h))
    cos_t = D / L
    return 1000.0 / L, 1000.0 * cos_t / L


def compute_features(x, y, fps, distance_cm=None, height_cm=0.0) -> dict:
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

    # 각도(mrad) 환산: 가로·세로에 서로 다른 계수 적용
    sc = angle_scale(distance_cm, height_cm)
    if sc:
        kx, ky = sc
        rms_mrad = float(np.sqrt(np.mean((kx * xf) ** 2 + (ky * yf) ** 2)))
        pk_mrad = float(np.sqrt(kx ** 2 * _band_power(f, Px, (fpk - 1.0, fpk + 1.0))
                                + ky ** 2 * _band_power(f, Py, (fpk - 1.0, fpk + 1.0))))
    else:
        rms_mrad = pk_mrad = None

    # 95% 신뢰 타원 면적 (원신호)
    cov = np.cov(np.vstack([x, y]))
    ell = float(np.pi * 5.991 * np.sqrt(max(np.linalg.det(cov), 0)))

    return dict(
        tremor_rms_mm=rms,
        tremor_rms_mrad=rms_mrad,
        peak_freq_hz=fpk, peak_prominence_db=prom_db,
        peak_rms_mm=pk_rms, peak_rms_mrad=pk_mrad, peak_power_ratio=float(pk_ratio),
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
