import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import subprocess
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.shared import Cm
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Fish Movement Analyzer", page_icon="🐟", layout="wide")

FISH = {
    "WHITE": {"label": "White fish (green circle)", "bgr": (0, 255, 0),
              "ranges": [([35, 45, 35], [95, 255, 255])]},
    "BLACK": {"label": "Black fish (red circle)", "bgr": (0, 0, 255),
              "ranges": [([0, 55, 45], [12, 255, 255]),
                         ([168, 55, 45], [179, 255, 255])]},
    "ORANGE": {"label": "Orange fish (orange circle)", "bgr": (0, 165, 255),
               "ranges": [([5, 65, 65], [30, 255, 255])]}
}

def make_mask(frame, fish):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in FISH[fish]["ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    # Small morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def candidates_for_frame(frame, fish, min_area, min_circularity):
    mask = make_mask(frame, fish)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0 or area < min_area:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if circularity >= min_circularity and 0.55 <= aspect <= 1.8:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            candidates.append((cx, cy, float(circularity), float(area)))
    return candidates

def track_one(video_path, total_frames, seed_frame, seed_xy, fish,
              max_jump, min_area, min_circularity):
    positions = [None] * total_frames
    confidence = [None] * total_frames
    positions[seed_frame] = tuple(map(int, seed_xy))
    confidence[seed_frame] = 1.0

    for direction in (-1, 1):
        previous = tuple(map(int, seed_xy))
        indices = range(seed_frame - 1, -1, -1) if direction == -1 else range(seed_frame + 1, total_frames)
        cap = cv2.VideoCapture(video_path)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            candidates = candidates_for_frame(frame, fish, min_area, min_circularity)
            if not candidates:
                continue
            scored = []
            for cx, cy, circ, area in candidates:
                dist = float(np.hypot(cx - previous[0], cy - previous[1]))
                if dist <= max_jump:
                    # Prefer nearby, round candidates; distance is the main factor.
                    score = dist - 8.0 * circ
                    scored.append((score, cx, cy, circ))
            if scored:
                scored.sort(key=lambda z: z[0])
                _, cx, cy, circ = scored[0]
                positions[idx] = (cx, cy)
                confidence[idx] = circ
                previous = (cx, cy)
        cap.release()
    return positions, confidence

def make_df(tracks, fps, total_frames):
    rows = []
    for i in range(total_frames):
        row = {"Frame": i, "Time_seconds": i / fps}
        for fish in FISH:
            p = tracks[fish]["positions"][i]
            row[fish + "_X"] = p[0] if p is not None else np.nan
            row[fish + "_Y"] = p[1] if p is not None else np.nan
            row[fish + "_Detected"] = p is not None
        rows.append(row)
    return pd.DataFrame(rows)

def interpolate_tracks(df):
    out = df.copy()
    for fish in FISH:
        for axis in ("X", "Y"):
            col = f"{fish}_{axis}"
            out[col] = out[col].interpolate(method="linear", limit_area="inside")
    return out

def calculate_summary(original, completed, fps):
    rows, speed_series = [], {}
    for fish in FISH:
        x, y = completed[f"{fish}_X"], completed[f"{fish}_Y"]
        dx, dy = x.diff(), y.diff()
        dist = np.sqrt(dx**2 + dy**2)
        pair_completed = x.notna() & x.shift(1).notna()
        est_dist = dist.where(pair_completed)
        est_speed = est_dist * fps

        ox, oy = original[f"{fish}_X"], original[f"{fish}_Y"]
        observed_pair = ox.notna() & oy.notna() & ox.shift(1).notna() & oy.shift(1).notna()
        obs_dist = dist.where(observed_pair)
        obs_speed = obs_dist * fps
        speed_series[fish] = est_speed

        rows.append({
            "Fish": fish,
            "Detected frames": int(ox.notna().sum()),
            "Missing frames": int(ox.isna().sum()),
            "Interpolated frames": int(x.notna().sum() - ox.notna().sum()),
            "Estimated path distance (pixels)": float(est_dist.sum(skipna=True)),
            "Estimated mean speed (pixels/s)": float(est_speed.mean(skipna=True)),
            "Estimated maximum speed (pixels/s)": float(est_speed.max(skipna=True)),
            "Observed mean speed (pixels/s)": float(obs_speed.mean(skipna=True)),
            "Observed valid intervals": int(obs_speed.notna().sum())
        })
    return pd.DataFrame(rows), speed_series

def draw_trajectory(original, completed, path):
    colors = {"WHITE": "green", "BLACK": "red", "ORANGE": "orange"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for fish, color in colors.items():
        ax.plot(completed[f"{fish}_X"], completed[f"{fish}_Y"],
                color=color, linewidth=1.8, label=f"{fish} fish (completed)")
        valid = original[f"{fish}_X"].notna()
        ax.scatter(original.loc[valid, f"{fish}_X"],
                   original.loc[valid, f"{fish}_Y"],
                   s=10, color=color, alpha=0.8, label=f"{fish} detections")
    ax.set_title("Fish trajectories (interpolated gaps shown as connecting lines)")
    ax.set_xlabel("X coordinate (pixels)")
    ax.set_ylabel("Y coordinate (pixels)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def draw_speed_graph(df, speed_series, path):
    colors = {"WHITE": "green", "BLACK": "red", "ORANGE": "orange"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for fish, series in speed_series.items():
        ax.plot(df["Time_seconds"], series, color=colors[fish],
                linewidth=1.2, label=fish)
    ax.set_title("Estimated swimming speed over time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (pixels/second)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def export_tracked_video(video_path, tracks, out_path, fps, width, height):
    cap = cv2.VideoCapture(video_path)
    temp_mp4v = out_path.replace(".mp4", "_temp.mp4")
    writer = cv2.VideoWriter(temp_mp4v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for fish in FISH:
            p = tracks[fish]["positions"][idx]
            if p is not None:
                x, y = map(int, p)
                cv2.circle(frame, (x, y), 8, FISH[fish]["bgr"], 2)
                cv2.putText(frame, fish, (x + 10, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, FISH[fish]["bgr"], 2)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    # Try H.264 browser-compatible conversion; keep MP4V file if ffmpeg unavailable.
    try:
        subprocess.run(["ffmpeg", "-y", "-i", temp_mp4v, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_mp4v)
    except Exception:
        os.replace(temp_mp4v, out_path)

def make_report(summary, original, completed, trajectory_img, speed_img, out_docx, fps):
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(29.7), Cm(21)
    sec.left_margin = sec.right_margin = Cm(1.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)

    title = doc.add_heading("Fish Movement and Trajectory Analysis", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("White, black and orange fish — video-based marker tracking")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("1. Method", level=1)
    doc.add_paragraph(
        f"The video was analysed at {fps:.3f} frames per second. The user selected "
        "the centre of each coloured circle in a reference frame. OpenCV HSV colour "
        "segmentation and contour shape filtering were used to follow the nearest "
        "matching marker in both directions from the reference frame."
    )
    doc.add_paragraph(
        "Missing positions between detected frames were filled using linear "
        "interpolation for a visually continuous estimated trajectory. No positions "
        "were extrapolated outside the first and last detections. Interpolated points "
        "are estimates and are not direct observations."
    )
    doc.add_paragraph(
        "Distances are reported in pixels and speeds in pixels per second. "
        "Conversion to physical units requires a known scale in the video."
    )

    doc.add_heading("2. Completed trajectories", level=1)
    doc.add_picture(trajectory_img, width=Inches(8.8))
    doc.add_heading("3. Estimated speed over time", level=1)
    doc.add_picture(speed_img, width=Inches(8.8))

    doc.add_heading("4. Summary results", level=1)
    t = doc.add_table(rows=1, cols=len(summary.columns))
    t.style = "Table Grid"
    for j, c in enumerate(summary.columns):
        t.rows[0].cells[j].text = c
    for _, row in summary.iterrows():
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)

    doc.add_heading("5. Frame-by-frame data", level=1)
    detail = completed[["Frame", "Time_seconds"]].copy()
    for fish in FISH:
        detail[f"{fish}_X"] = completed[f"{fish}_X"]
        detail[f"{fish}_Y"] = completed[f"{fish}_Y"]
        detail[f"{fish}_Original_detection"] = original[f"{fish}_X"].notna()
    # Split into manageable tables if many frames
    cols = list(detail.columns)
    t2 = doc.add_table(rows=1, cols=len(cols))
    t2.style = "Table Grid"
    for j, c in enumerate(cols):
        t2.rows[0].cells[j].text = c
    for _, row in detail.iterrows():
        cells = t2.add_row().cells
        for j, c in enumerate(cols):
            val = row[c]
            if pd.isna(val):
                cells[j].text = "-"
            elif isinstance(val, (float, np.floating)):
                cells[j].text = f"{val:.2f}"
            else:
                cells[j].text = str(val)

    doc.add_heading("6. Limitations", level=1)
    doc.add_paragraph(
        "This application tracks the added coloured circle markers, not the fish body "
        "independently. The marker must remain aligned with the same fish for the "
        "coordinates to represent fish movement. Nearest-candidate matching can fail "
        "when markers overlap, disappear, change appearance or move farther than the "
        "maximum allowed frame-to-frame distance. Long interpolated gaps are especially "
        "uncertain and should be reviewed in the tracked video."
    )
    doc.save(out_docx)

st.title("🐟 Fish Movement Analyzer")
st.caption("Upload a video, select the three coloured markers, track their paths, and export results.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0
    cap.release()

    st.success(f"Video loaded: {total_frames} frames | {fps:.2f} FPS | {width} × {height} | {duration:.2f} seconds")

    if total_frames <= 0:
        st.error("Could not read this video. Please try another video file.")
        st.stop()

    st.subheader("1. Select a reference frame")
    seed_frame = st.slider("Choose a frame where all three coloured circles are visible",
                           0, total_frames - 1, min(int(2 * fps), total_frames - 1))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        st.error("Could not read the selected frame.")
        st.stop()

    st.write("Select a fish below, then click the centre of its coloured circle in the image.")
    fish_choice = st.selectbox("Fish marker to select", list(FISH.keys()),
                               format_func=lambda k: FISH[k]["label"])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    clicked = streamlit_image_coordinates(rgb, key=f"image_{seed_frame}_{fish_choice}")

    if "seeds" not in st.session_state:
        st.session_state.seeds = {}
    if clicked is not None:
        st.session_state.seeds[fish_choice] = (int(clicked["x"]), int(clicked["y"]))

    seed_cols = st.columns(3)
    for col, fish in zip(seed_cols, FISH):
        with col:
            if fish in st.session_state.seeds:
                st.success(f"{fish}: {st.session_state.seeds[fish]}")
            else:
                st.warning(f"{fish}: not selected")

    st.subheader("2. Tracking settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_jump = st.slider("Maximum movement per frame (pixels)", 10, 250, 100)
    with c2:
        min_area = st.slider("Minimum contour area", 10, 2000, 80)
    with c3:
        min_circularity = st.slider("Minimum circularity", 0.10, 0.95, 0.30, 0.05)

    if st.button("Start tracking", type="primary", disabled=len(st.session_state.seeds) != 3):
        tracks = {}
        progress = st.progress(0)
        status = st.empty()
        for n, fish in enumerate(FISH):
            status.write(f"Tracking {fish} fish...")
            pos, conf = track_one(video_path, total_frames, seed_frame,
                                  st.session_state.seeds[fish], fish,
                                  max_jump, min_area, min_circularity)
            tracks[fish] = {"positions": pos, "confidence": conf}
            progress.progress((n + 1) / 3)

        original = make_df(tracks, fps, total_frames)
        completed = interpolate_tracks(original)
        summary, speed_series = calculate_summary(original, completed, fps)

        trajectory_img = os.path.join(tempfile.gettempdir(), "fish_trajectory.png")
        speed_img = os.path.join(tempfile.gettempdir(), "fish_speed.png")
        draw_trajectory(original, completed, trajectory_img)
        draw_speed_graph(completed, speed_series, speed_img)

        tracked_video = os.path.join(tempfile.gettempdir(), "fish_tracked_output.mp4")
        export_tracked_video(video_path, tracks, tracked_video, fps, width, height)

        csv_path = os.path.join(tempfile.gettempdir(), "fish_tracking_data.csv")
        summary_csv = os.path.join(tempfile.gettempdir(), "fish_tracking_summary.csv")
        docx_path = os.path.join(tempfile.gettempdir(), "fish_tracking_report.docx")
        export_df = completed.copy()
        for fish in FISH:
            export_df[f"{fish}_Estimated_speed_pixels_sec"] = speed_series[fish]
        export_df.to_csv(csv_path, index=False)
        summary.to_csv(summary_csv, index=False)
        make_report(summary, original, completed, trajectory_img, speed_img, docx_path, fps)

        st.session_state.analysis = {
            "original": original, "completed": completed, "summary": summary,
            "trajectory_img": trajectory_img, "speed_img": speed_img,
            "tracked_video": tracked_video, "csv_path": csv_path,
            "summary_csv": summary_csv, "docx_path": docx_path
        }
        st.success("Analysis completed!")

    if "analysis" in st.session_state:
        a = st.session_state.analysis
        st.divider()
        st.subheader("3. Analysis results")
        st.dataframe(a["summary"].round(2), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.image(a["trajectory_img"], caption="Completed trajectories")
        with right:
            st.image(a["speed_img"], caption="Estimated speed over time")
        st.subheader("4. Tracked video")
        st.video(a["tracked_video"])
        st.subheader("5. Download results")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            with open(a["docx_path"], "rb") as f:
                st.download_button("Download Word report", f, file_name="Fish_Trajectory_Report.docx")
        with d2:
            with open(a["csv_path"], "rb") as f:
                st.download_button("Download tracking CSV", f, file_name="Fish_Tracking_Data.csv")
        with d3:
            with open(a["summary_csv"], "rb") as f:
                st.download_button("Download summary CSV", f, file_name="Fish_Tracking_Summary.csv")
        with d4:
            with open(a["tracked_video"], "rb") as f:
                st.download_button("Download tracked MP4", f, file_name="Fish_Tracked_Video.mp4")

else:
    st.info("Upload a video to begin. For best results, use a clear video where each coloured marker remains visible.")
    st.markdown("""
    **Workflow**
    1. Upload video.
    2. Choose a frame and click the centre of the green, red and orange circles.
    3. Adjust tracking settings if needed.
    4. Start tracking and review the output.
    5. Download the Word report, CSV files and tracked MP4.
    """)
