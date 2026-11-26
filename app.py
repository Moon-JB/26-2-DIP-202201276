"""
레이저 포인터 손떨림 분석기 (Streamlit)
- 이완/수축 영상을 업로드하면 레이저 점을 추적해 떨림 주파수·세기를 계산하고
  정상 / 본태성 떨림(수전증) / 파킨슨병 패턴을 규칙 기반으로 판정합니다.
- 연구·교육용 선별 도구이며 의학적 진단을 대신하지 않습니다.
"""
import gc
import hashlib
import os
import tempfile

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tremor  # noqa: E402

st.set_page_config(page_title="레이저 손떨림 분석기", page_icon="🔴", layout="wide")

CONDS = {"rest": "이완", "contr": "수축"}


# ------------------------------------------------------------------------------------------
# 캐시: 같은 파일은 다시 디코딩하지 않음 (1차 스캔이 가장 오래 걸림)
# ------------------------------------------------------------------------------------------
def _file_key(upload) -> str:
    """영상 전체를 복사·해시하지 않고 (이름, 크기, 앞 1 MB)로 식별 -> 메모리 절약"""
    buf = upload.getbuffer()            # 복사 없는 memoryview
    head = bytes(buf[: 1 << 20])
    return hashlib.md5(head + f"{upload.name}{len(buf)}".encode()).hexdigest()


def _save_temp(upload, key: str) -> str:
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    path = os.path.join(tempfile.gettempdir(), f"tremor_{key[:12]}{suffix}")
    if not os.path.exists(path):
        buf = upload.getbuffer()
        with open(path, "wb") as f:
            for i in range(0, len(buf), 8 << 20):     # 8 MB씩 나눠 쓰기
                f.write(buf[i:i + (8 << 20)])
    return path


def cached_scan(file_hash: str, path: str, label: str):
    """같은 파일은 다시 디코딩하지 않도록 session_state에 스캔 결과 저장.
    (st.cache_data 안에서 진행률 표시줄을 갱신하면 Streamlit 오류가 나므로 직접 캐시)"""
    cache = st.session_state.setdefault("scan_cache", {})
    if file_hash in cache:
        return cache[file_hash]
    bar = st.progress(0.0, text=f"{label} 영상 프레임 분석 중…")
    scan = tremor.scan_video(
        path, progress_cb=lambda p: bar.progress(p, text=f"{label} 영상 프레임 분석 중… {p * 100:.0f}%"))
    bar.empty()
    if len(cache) >= 4:              # 메모리 보호: 오래된 것부터 삭제
        cache.pop(next(iter(cache)))
    cache[file_hash] = scan
    return scan


def run_analysis(upload, settings, onset_override, label):
    h = _file_key(upload)
    path = _save_temp(upload, h)
    scan = cached_scan(h, path, label)
    res = tremor.analyze_video(
        path, window_s=settings["window_s"], start_offset_s=settings["offset_s"],
        grid_mm=settings["grid_mm"], distance_cm=settings["distance_cm"] or None,
        height_cm=settings["height_cm"], height_mode=settings["height_mode"], scale_mode=settings["scale_mode"],
        mark_mm=settings["mark_mm"], manual_px_per_mm=settings["manual_ppm"],
        onset_override_s=onset_override, scan=scan)
    return res, scan


# ------------------------------------------------------------------------------------------
# 그래프
# ------------------------------------------------------------------------------------------
def fig_timeline(scan, res):
    t = np.arange(scan["n"]) / scan["fps"]
    present = ~np.isnan(scan["pts"][:, 0])
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.plot(t, scan["scene"], color="#555", lw=1, label="scene similarity")
    ax.fill_between(t, 0, present * 1.0, color="#e4572e", alpha=0.25, step="mid", label="laser detected")
    ax.axvspan(res.win_start_s, res.win_end_s, color="#2e86ab", alpha=0.2, label="analysis window")
    ax.axvline(res.onset_s, color="#e4572e", ls="--", lw=1)
    if res.descent:
        ax.axvspan(res.move_s, res.arrive_s, color="#f6ae2d", alpha=0.35, label="descent (relaxing)")
    ax.axhline(tremor.SCENE_THR, color="#999", ls=":", lw=0.8)
    ax.set_ylim(-0.1, 1.1); ax.set_xlabel("time (s)"); ax.set_xlim(0, t[-1])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def fig_signals(res):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    lim = max(np.abs(res.x_mm).max(), np.abs(res.y_mm).max()) * 1.1
    ax[0].plot(res.x_mm, res.y_mm, lw=0.6, color="#2e86ab")
    ax[0].plot(0, 0, "k+", ms=14, mew=2)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim); ax[0].set_aspect("equal")
    ax[0].set_title("trajectory (mm, + = reference)"); ax[0].grid(alpha=0.3)

    ax[1].plot(res.t, res.xf_mm, lw=0.8, label="x"); ax[1].plot(res.t, res.yf_mm, lw=0.8, label="y")
    ax[1].set_title(f"tremor component {tremor.TREMOR_BAND[0]:.0f}-{tremor.TREMOR_BAND[1]:.0f} Hz (mm)")
    ax[1].set_xlabel("time since laser onset (s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    f, P = res.freqs, res.psd
    ax[2].semilogy(f, P, color="#2e86ab", label="PSD (x+y)")
    ax[2].semilogy(f, res.psd_bg, "--", color="#999", label="1/f background")
    ax[2].axvspan(*tremor.PD_BAND, color="#f6ae2d", alpha=0.2, label="PD rest 3.5-6.5")
    ax[2].axvspan(*tremor.PHYS_BAND, color="#86bbd8", alpha=0.25, label="physiological 8-12")
    ax[2].axvline(res.peak_freq_hz, color="#e4572e", lw=1.2)
    ax[2].set_xlim(0, min(15, res.fps / 2)); ax[2].set_xlabel("frequency (Hz)")
    ax[2].set_title(f"spectrum  (peak {res.peak_freq_hz:.2f} Hz)"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_compare(results):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    colors = {"rest": "#2e86ab", "contr": "#e4572e"}
    for k, r in results.items():
        ax.semilogy(r.freqs, r.psd, color=colors[k], label=f"{'rest' if k == 'rest' else 'contraction'} ({r.peak_freq_hz:.1f} Hz)")
    ax.axvspan(*tremor.PD_BAND, color="#f6ae2d", alpha=0.15)
    ax.axvspan(*tremor.PHYS_BAND, color="#86bbd8", alpha=0.2)
    ax.set_xlim(0, 15); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD (mm²/Hz)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def metrics_row(res, has_dist):
    c = st.columns(5)
    if has_dist:
        c[0].metric("떨림 세기 (RMS)", f"{res.tremor_rms_mrad:.2f} mrad", f"{res.tremor_rms_mm:.2f} mm", delta_color="off")
    else:
        c[0].metric("떨림 세기 (RMS)", f"{res.tremor_rms_mm:.2f} mm")
    c[1].metric("주 떨림 주파수", f"{res.peak_freq_hz:.2f} Hz", f"돌출 {res.peak_prominence_db:.1f} dB", delta_color="off")
    amp = f"{res.peak_rms_mrad:.2f} mrad" if has_dist else f"{res.peak_rms_mm:.2f} mm"
    c[2].metric("주 피크 진폭", amp)
    c[3].metric("느린 흔들림 (<3 Hz)", f"{res.drift_rms_mm:.1f} mm")
    if res.mean_offset_mm is not None:
        c[4].metric("조준 오차", f"{np.hypot(*res.mean_offset_mm):.1f} mm")
    else:
        c[4].metric("95% 흩어짐 면적", f"{res.ellipse95_area_mm2:.0f} mm²")


def summary_rows(results, meta, verdict):
    rows = []
    for k, r in results.items():
        s = r.summary()
        s.pop("preview_bgr", None)
        row = {"이름": meta["name"], "차수": meta["trial"], "조건": CONDS[k], **s,
               "판정": verdict["label"] if verdict else ""}
        rows.append(row)
    return pd.DataFrame(rows)


def traj_csv(results):
    frames = []
    for k, r in results.items():
        frames.append(pd.DataFrame({"condition": CONDS[k], "t_s": r.t, "x_mm": r.x_mm, "y_mm": r.y_mm,
                                    "x_tremor_mm": r.xf_mm, "y_tremor_mm": r.yf_mm}))
    return pd.concat(frames).to_csv(index=False).encode("utf-8-sig")


# ------------------------------------------------------------------------------------------
# 사이드바: 설정
# ------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("분석 설정")
    offset_s = st.number_input("레이저 발사 후 제외 시간 (s)", 0.0, 5.0, 1.0, 0.1)
    window_s = st.number_input("분석 구간 길이 (s)", 2.0, 30.0, 7.0, 0.5)
    scale_label = st.radio("길이 기준 (축척)", ["모눈종이 자동", "벽 표시점 2개", "직접 입력 (px/mm)"],
                           help="떨림 주파수는 축척과 무관합니다. 축척은 진폭(mm·mrad)에만 영향을 줍니다.")
    scale_mode = {"모눈종이 자동": "grid", "벽 표시점 2개": "marks", "직접 입력 (px/mm)": "manual"}[scale_label]
    grid_mm, mark_mm, manual_ppm = 5.0, 100.0, None
    if scale_mode == "grid":
        grid_mm = st.number_input("모눈 한 칸 (mm)", 1.0, 20.0, 5.0, 0.5)
    elif scale_mode == "marks":
        mark_mm = st.number_input("두 표시점 사이 실제 거리 (mm)", 5.0, 1000.0, 100.0, 1.0,
                                  help="길수록 정확합니다. 10 mm 표시는 오차가 수 % 생길 수 있어요.")
    else:
        manual_ppm = st.number_input("축척 (원본 영상 px / mm)", 0.1, 100.0, 4.5, 0.01)
    distance_cm = st.number_input("레이저–종이 벽 수평 거리 (cm, 0 = 미입력)", 0.0, 2000.0, 426.0, 1.0,
                                  help="입력하면 떨림을 각도(mrad)로 환산해 거리와 무관하게 비교합니다.")
    h_label = st.radio("높이 차이 (각도 보정)", ["자동: 처음 수평 위치 − 이완 위치", "직접 입력"],
                       help="자동: 처음 수평으로 겨눈 레이저 위치와 팔을 이완해 내려온 위치의 세로 차이를 "
                            "높이 차이로 보고, 수평 거리와 피타고라스로 각도를 계산합니다. "
                            "하강이 없으면(수축 조건 등) 0 cm로 계산합니다.")
    height_mode = "auto" if h_label.startswith("자동") else "manual"
    height_cm = 0.0
    if height_mode == "manual":
        height_cm = st.number_input("레이저 높이 − 표적 높이 (cm)", -500.0, 500.0, 0.0, 1.0)
        if distance_cm:
            import math
            st.caption(f"경사 거리 {math.hypot(distance_cm, height_cm):.1f} cm · "
                       f"앙각 {math.degrees(math.atan2(abs(height_cm), distance_cm)):.1f}°")
    else:
        st.caption("하강이 있으면 분석 구간은 '이완 위치 도착 후' 기준으로 잡힙니다.")
    with st.expander("판정 기준 (고급)"):
        unit = "mrad" if distance_cm else "mm"
        amp_thr = st.number_input(f"병적 떨림 진폭 기준 ({unit})", 0.05, 50.0, 1.0 if distance_cm else 4.0, 0.05)
        prom_thr = st.number_input("피크 돌출 기준 (dB)", 0.0, 30.0, 6.0, 0.5)
        band_lo, band_hi = st.slider("병적 떨림 주파수 대역 (Hz)", 2.0, 14.0, (3.5, 8.0), 0.5)
        pd_max = st.number_input("파킨슨 안정 떨림 상한 (Hz)", 3.0, 10.0, 6.5, 0.5)
        dom = st.number_input("안정 시 우세 판단 진폭비 (이완/수축)", 0.5, 5.0, 1.2, 0.1)
        st.caption("기본값은 문헌의 주파수 대역과 정상인 시험 데이터로 정한 잠정값입니다. "
                   "데이터가 쌓이면 조정하세요.")
settings = dict(offset_s=offset_s, window_s=window_s, grid_mm=grid_mm, scale_mode=scale_mode,
                mark_mm=mark_mm, manual_ppm=manual_ppm, distance_cm=distance_cm, height_cm=height_cm,
                height_mode=height_mode)
criteria = tremor.Criteria(amp_thr=amp_thr, prom_thr_db=prom_thr, path_band=(band_lo, band_hi),
                           pd_max_freq=pd_max, rest_dominance=dom)

# ------------------------------------------------------------------------------------------
# 본문
# ------------------------------------------------------------------------------------------
st.title("🔴 레이저 포인터 손떨림 분석기")
st.caption("모눈종이 가운데 기준점에 레이저를 비춘 영상을 이완·수축 조건별로 올리면 "
           "떨림 주파수와 세기를 계산하고 떨림 패턴을 판정합니다. **연구·교육용 도구이며 진단을 대신하지 않습니다.**")

c1, c2 = st.columns(2)
name = c1.text_input("피험자 이름 / ID", "")
trial = c2.text_input("실험 차수", "1차")

u1, u2 = st.columns(2)
uploads = {
    "rest": u1.file_uploader("이완 영상", type=["mp4", "mov", "avi", "mkv", "m4v"], key="up_rest"),
    "contr": u2.file_uploader("수축 영상", type=["mp4", "mov", "avi", "mkv", "m4v"], key="up_contr"),
}

with st.expander("발사 시점 수동 지정 (자동 검출이 틀렸을 때)"):
    st.caption("0으로 두면 자동 검출. 아래 타임라인 그래프의 빨간 점선이 자동 검출된 발사 시점입니다.")
    o1, o2 = st.columns(2)
    manual = {"rest": o1.number_input("이완 영상 발사 시점 (s)", 0.0, 600.0, 0.0, 0.1),
              "contr": o2.number_input("수축 영상 발사 시점 (s)", 0.0, 600.0, 0.0, 0.1)}

if not any(uploads.values()):
    st.info("이완·수축 영상을 올려주세요. 한 조건만 올려도 분석은 되지만, 판정은 두 조건을 비교할 때 더 정확합니다.")
    st.stop()

if st.button("분석 시작", type="primary"):
    st.session_state["run"] = True
if not st.session_state.get("run"):
    st.stop()

results, scans = {}, {}
for k, up in uploads.items():
    if up is None:
        continue
    try:
        res, scan = run_analysis(up, settings, manual[k] or None, CONDS[k])
        results[k], scans[k] = res, scan
        gc.collect()
    except Exception as e:  # noqa: BLE001
        st.error(f"{CONDS[k]} 영상 분석 실패: {e}")

if not results:
    st.stop()

# ---- 종합 판정 ----
verdict = tremor.classify(results.get("rest"), results.get("contr"), criteria)
color = {"정상 범위": "green", "파킨슨병 패턴 의심": "red", "본태성 떨림(수전증) 패턴 의심": "orange"}.get(verdict["label"], "gray")
st.subheader("종합 판정")
st.markdown(f"### :{color}[{verdict['label']}]")
for r_ in verdict["reasons"]:
    st.markdown(f"- {r_}")
if verdict.get("rest_contr_ratio") is not None:
    st.caption(f"이완/수축 주 피크 진폭비: {verdict['rest_contr_ratio']:.2f}")

with st.expander("판정 기준 충족 여부"):
    for cond, checks in verdict["detail"].items():
        if checks:
            st.markdown(f"**{cond}**")
            for ok, txt in checks:
                st.markdown(f"{'✅' if ok else '⬜'} {txt}")
    st.caption("병적 떨림 = 주파수 대역 · 피크 돌출 · 진폭 세 조건을 모두 만족할 때")

if len(results) == 2:
    show(fig_compare(results))

# ---- 조건별 상세 ----
has_dist = bool(distance_cm)
tabs = st.tabs([CONDS[k] for k in results])
for tab, (k, res) in zip(tabs, results.items()):
    with tab:
        for w in res.warnings:
            st.warning(w)
        metrics_row(res, has_dist)
        a, b = st.columns([1, 1.4])
        with a:
            st.image(cv2.cvtColor(res.preview_bgr, cv2.COLOR_BGR2RGB),
                     caption="검출 결과: 하늘색=종이, 보라 원=벽 표시점, 빨간선=레이저 궤적, 초록 +=기준점(주황 +=평균 위치)")
        with b:
            show(fig_timeline(scans[k], res))
            if res.descent:
                st.info(f"수평 → 이완 하강 {res.drop_mm:.1f} mm 검출 ({res.move_s:.2f}–{res.arrive_s:.2f}s) → 높이 차이 "
                        f"{res.height_used_cm:.1f} cm" + (f", 앙각 {res.elev_angle_deg:.2f}°, 경사 거리 {res.slant_cm:.1f} cm"
                                                           if res.elev_angle_deg is not None else ""))
            else:
                st.caption(f"높이 차이 {res.height_used_cm:.1f} cm ({'하강 없음 → 수평' if res.height_source == 'auto' else '직접 입력'})")
            st.caption(f"레이저 발사 {res.onset_s:.2f}s → 분석 구간 {res.win_start_s:.2f}–{res.win_end_s:.2f}s · "
                       f"{res.fps:.1f} fps · 축척 {res.px_per_mm:.2f} px/mm ({res.scale_source}) · 검출률 {res.detect_rate * 100:.0f}%")
        show(fig_signals(res))

# ---- 내보내기 ----
st.subheader("결과 내보내기")
meta = {"name": name, "trial": trial}
df = summary_rows(results, meta, verdict)
st.dataframe(df[["조건", "peak_freq_hz", "peak_prominence_db", "peak_rms_mm", "peak_rms_mrad",
                 "tremor_rms_mm", "tremor_rms_mrad", "drift_rms_mm", "ellipse95_area_mm2", "drop_mm", "height_used_cm", "elev_angle_deg", "onset_s"]],
             hide_index=True)
fname = f"{name or 'subject'}_{trial}".replace(" ", "_")
d1, d2 = st.columns(2)
d1.download_button("요약 CSV", df.to_csv(index=False).encode("utf-8-sig"), f"{fname}_summary.csv", "text/csv")
d2.download_button("궤적 CSV (프레임별)", traj_csv(results), f"{fname}_trajectory.csv", "text/csv")
