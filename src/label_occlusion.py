"""
======================================================================
  Occlusion Ground-Truth Labelling Tool  —  TMRB Revision (Step 2.1)
======================================================================
Labels each (sampled) frame of a surgical video as:
    1 = OCCLUDED   (instrument overlaps / covers the ROI)
    0 = NOT occluded

Controls (OpenCV window must be in focus):
  1  or  O   →  label current frame as OCCLUDED
  0  or  N   →  label current frame as NOT OCCLUDED
  Right / D  →  next frame (without labelling)
  Left  / A  →  previous frame
  ]          →  jump +10 frames
  [          →  jump -10 frames
  S          →  save progress now (auto-saves every 50 labels too)
  Q / ESC    →  quit and save

Usage:
  python label_occlusion.py                   # labels all videos in VIDEO_PATHS
  python label_occlusion.py video_1.mp4       # label a single video
  python label_occlusion.py --resume          # resume from existing CSV

Output:
  occlusion_gt.csv  with columns:
      video_name, frame_index, timestamp_sec, label
      label: 1=occluded, 0=not_occluded, -1=skipped/unlabelled
======================================================================
"""

import cv2
import csv
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG — edit these as needed
# ─────────────────────────────────────────────

VIDEO_PATHS = [
    "E:/Antigravity/TMRB/video_1.mp4",
    "E:/Antigravity/TMRB/video_2.mp4",
    # Add video_3.mp4 and video_4.mp4 here when available:
    # "E:/Antigravity/TMRB/video_3.mp4",
    # "E:/Antigravity/TMRB/video_4.mp4",
]

OUTPUT_CSV = "E:/Antigravity/TMRB/occlusion_gt.csv"

# Sample every Nth frame to keep labelling manageable
# For video_1 (766 frames @ 30 FPS):  STEP=3  → ~255 frames to label
# For video_2 (12848 frames @ 50 FPS): STEP=25 → ~514 frames to label
# You can override per video in STEP_OVERRIDE below
DEFAULT_STEP = 5
STEP_OVERRIDE = {
    "video_1.mp4": 3,
    "video_2.mp4": 25,
}

# Display scale (0.5 = half size; reduce if your screen is small)
DISPLAY_SCALE = 0.55

# Auto-save every N labels
AUTOSAVE_EVERY = 50

# ─────────────────────────────────────────────
#  COLOUR CONSTANTS
# ─────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREY   = (120, 120, 120)
CYAN   = (220, 200, 0)


def load_existing_labels(csv_path):
    """Load already-saved labels so we can resume."""
    labels = {}  # key: (video_name, frame_index) → label
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["video_name"], int(row["frame_index"]))
            labels[key] = int(row["label"])
    return labels


def save_all_labels(csv_path, all_rows):
    """Write all label rows to the CSV (overwrites)."""
    if not all_rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_name", "frame_index", "timestamp_sec", "label"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[SAVED]  {len(all_rows)} rows → {csv_path}")


def draw_overlay(frame, video_name, frame_idx, total_frames, fps, label,
                 n_labelled, n_total, step):
    """Draw a rich HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # ── Top bar background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Title / video name ──
    cv2.putText(frame, f"VIDEO: {video_name}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)

    # ── Frame counter ──
    ts = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame,
                f"Frame {frame_idx}/{total_frames - 1}  |  {ts:.2f}s  |  step={step}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # ── Progress bar ──
    bar_x, bar_y, bar_w, bar_h = 12, 70, w - 24, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GREY, -1)
    prog = int(bar_w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + prog, bar_y + bar_h), CYAN, -1)

    # ── Label count ──
    cv2.putText(frame, f"Labelled: {n_labelled}/{n_total}",
                (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # ── Current label badge ──
    if label == 1:
        badge_col = RED
        badge_txt = "  OCCLUDED  "
    elif label == 0:
        badge_col = GREEN
        badge_txt = " NOT OCCLUDED "
    else:
        badge_col = GREY
        badge_txt = "  UNLABELLED  "

    badge_x = w - 280
    cv2.rectangle(frame, (badge_x - 10, 8), (w - 10, 50), badge_col, -1)
    cv2.putText(frame, badge_txt, (badge_x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # ── Bottom controls hint ──
    hint_y = h - 12
    cv2.rectangle(frame, (0, h - 36), (w, h), BLACK, -1)
    cv2.putText(frame,
                " [1/O]=Occluded  [0/N]=Not Occluded  [</>]=prev/next  [[/]]=jump10  [S]=Save  [Q/ESC]=Quit",
                (8, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1)

    return frame


def label_video(video_path, existing_labels, all_rows):
    """
    Main labelling loop for a single video.
    Returns updated all_rows list.
    """
    video_name = Path(video_path).name
    step = STEP_OVERRIDE.get(video_name, DEFAULT_STEP)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return all_rows

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w_orig       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build the list of frame indices we will label
    frame_indices = list(range(0, total_frames, step))
    n_total = len(frame_indices)

    # Build per-frame label dict from existing CSV rows
    frame_labels = {}
    for key, lbl in existing_labels.items():
        if key[0] == video_name:
            frame_labels[key[1]] = lbl

    print(f"\n{'='*60}")
    print(f"  Video : {video_name}")
    print(f"  Frames: {total_frames}  |  FPS: {fps}  |  Res: {w_orig}x{h_orig}")
    print(f"  Step  : every {step} frames  →  {n_total} frames to label")
    already = sum(1 for k, v in frame_labels.items() if v != -1)
    print(f"  Already labelled: {already}/{n_total}")
    print(f"{'='*60}")

    # Start at first unlabelled frame
    ptr = 0
    for i, fi in enumerate(frame_indices):
        if fi not in frame_labels or frame_labels[fi] == -1:
            ptr = i
            break

    display_w = int(w_orig * DISPLAY_SCALE)
    display_h = int(h_orig * DISPLAY_SCALE)

    cv2.namedWindow("Occlusion Labeller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Occlusion Labeller", display_w, display_h + 50)

    cached_frame_idx = -1
    cached_frame     = None
    labels_since_save = 0

    while True:
        frame_idx = frame_indices[ptr]

        # ── Load frame (with caching) ──
        if frame_idx != cached_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, raw = cap.read()
            if not ret:
                print(f"[WARN] Could not read frame {frame_idx}")
                raw = (255 * __import__('numpy').ones((h_orig, w_orig, 3),
                                                     dtype='uint8'))
            cached_frame     = raw.copy()
            cached_frame_idx = frame_idx

        frame = cached_frame.copy()
        current_label = frame_labels.get(frame_idx, -1)
        n_labelled = sum(1 for v in frame_labels.values() if v != -1)

        # Resize for display
        disp = cv2.resize(frame, (display_w, display_h + 50))

        # Draw overlay
        disp = draw_overlay(
            disp, video_name, frame_idx, total_frames, fps,
            current_label, n_labelled, n_total, step
        )

        cv2.imshow("Occlusion Labeller", disp)

        # ── Key handling ──
        key = cv2.waitKey(0) & 0xFF

        # Label: OCCLUDED
        if key in (ord('1'), ord('o'), ord('O')):
            frame_labels[frame_idx] = 1
            labels_since_save += 1
            ptr = min(ptr + 1, n_total - 1)

        # Label: NOT OCCLUDED
        elif key in (ord('0'), ord('n'), ord('N')):
            frame_labels[frame_idx] = 0
            labels_since_save += 1
            ptr = min(ptr + 1, n_total - 1)

        # Navigate: next (no label)
        elif key in (ord('d'), ord('D'), 83, 3):   # D or Right arrow
            ptr = min(ptr + 1, n_total - 1)

        # Navigate: previous
        elif key in (ord('a'), ord('A'), 81, 2):   # A or Left arrow
            ptr = max(ptr - 1, 0)

        # Jump +10
        elif key == ord(']'):
            ptr = min(ptr + 10, n_total - 1)

        # Jump -10
        elif key == ord('['):
            ptr = max(ptr - 10, 0)

        # Save
        elif key in (ord('s'), ord('S')):
            _flush_rows(all_rows, video_name, frame_labels, frame_indices, fps)
            save_all_labels(OUTPUT_CSV, all_rows)
            labels_since_save = 0

        # Quit
        elif key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break

        # Auto-save
        if labels_since_save >= AUTOSAVE_EVERY:
            _flush_rows(all_rows, video_name, frame_labels, frame_indices, fps)
            save_all_labels(OUTPUT_CSV, all_rows)
            labels_since_save = 0

    # Final flush
    _flush_rows(all_rows, video_name, frame_labels, frame_indices, fps)
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Done with {video_name}.")
    labelled = sum(1 for v in frame_labels.values() if v != -1)
    occluded = sum(1 for v in frame_labels.values() if v == 1)
    not_occ  = sum(1 for v in frame_labels.values() if v == 0)
    print(f"  Total labelled : {labelled}/{n_total}")
    print(f"  Occluded       : {occluded}")
    print(f"  Not occluded   : {not_occ}")
    print(f"  Unlabelled     : {n_total - labelled}")

    return all_rows


def _flush_rows(all_rows, video_name, frame_labels, frame_indices, fps):
    """Merge frame_labels into all_rows (in-place update)."""
    existing = {(r["video_name"], int(r["frame_index"])): i
                for i, r in enumerate(all_rows)}

    for fi in frame_indices:
        lbl = frame_labels.get(fi, -1)
        ts  = round(fi / fps, 3) if fps > 0 else 0
        key = (video_name, fi)
        if key in existing:
            all_rows[existing[key]]["label"] = lbl
        else:
            all_rows.append({
                "video_name" : video_name,
                "frame_index": fi,
                "timestamp_sec": ts,
                "label"      : lbl,
            })


def print_summary(all_rows):
    """Print overall labelling summary."""
    from collections import Counter
    print("\n" + "="*60)
    print("  OVERALL LABELLING SUMMARY")
    print("="*60)
    by_video = {}
    for r in all_rows:
        vn = r["video_name"]
        by_video.setdefault(vn, []).append(int(r["label"]))

    for vn, labels in by_video.items():
        c = Counter(labels)
        total = len(labels)
        labelled = total - c.get(-1, 0)
        print(f"\n  {vn}")
        print(f"    Total frames (sampled) : {total}")
        print(f"    Labelled               : {labelled}")
        print(f"    Occluded (1)           : {c.get(1, 0)}")
        print(f"    Not occluded (0)       : {c.get(0, 0)}")
        print(f"    Unlabelled (-1)        : {c.get(-1, 0)}")

    total_labelled = sum(1 for r in all_rows if int(r["label"]) != -1)
    print(f"\n  TOTAL labelled rows: {total_labelled}/{len(all_rows)}")
    print(f"  Saved to: {OUTPUT_CSV}")
    print("="*60)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Occlusion ground-truth labelling tool for surgical videos."
    )
    parser.add_argument("videos", nargs="*",
                        help="Paths to video files. Defaults to VIDEO_PATHS in script.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing occlusion_gt.csv")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    parser.add_argument("--step", type=int, default=None,
                        help="Override frame step for all videos (overrides STEP_OVERRIDE dict)")
    args = parser.parse_args()

    # Apply overrides
    if args.output:
        OUTPUT_CSV = args.output
    if args.step:
        DEFAULT_STEP = args.step
        STEP_OVERRIDE.clear()

    videos_to_label = args.videos if args.videos else VIDEO_PATHS

    print("\n" + "="*60)
    print("  OCCLUSION GROUND-TRUTH LABELLING TOOL")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Videos to label: {len(videos_to_label)}")
    for v in videos_to_label:
        exists = "✓" if os.path.exists(v) else "✗ NOT FOUND"
        print(f"    [{exists}] {v}")
    print("="*60)

    # Load existing labels (for resume)
    existing_labels = load_existing_labels(OUTPUT_CSV) if args.resume else {}
    if existing_labels:
        print(f"  Resuming: loaded {len(existing_labels)} existing labels.")

    # Start with existing rows so we don't lose them on save
    all_rows = []
    if existing_labels:
        # Rebuild rows from existing_labels
        for (vn, fi), lbl in existing_labels.items():
            all_rows.append({
                "video_name"   : vn,
                "frame_index"  : fi,
                "timestamp_sec": 0,  # will be recalculated on flush
                "label"        : lbl,
            })

    # Label each video
    for vpath in videos_to_label:
        if not os.path.exists(vpath):
            print(f"\n[SKIP] File not found: {vpath}")
            continue
        all_rows = label_video(vpath, existing_labels, all_rows)
        # Save after each video
        save_all_labels(OUTPUT_CSV, all_rows)
        # Ask if user wants to continue to the next video
        if len(videos_to_label) > 1 and vpath != videos_to_label[-1]:
            print("\nPress Enter to continue to the next video, or Ctrl+C to stop...")
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                break

    # Final save & summary
    save_all_labels(OUTPUT_CSV, all_rows)
    print_summary(all_rows)
    print("\n[DONE] Labelling session complete.\n")
