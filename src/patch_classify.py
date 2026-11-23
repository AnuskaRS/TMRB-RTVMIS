"""Patch classify_frame in occlusion_analysis.py to skip depth check for PF ROI."""
from pathlib import Path

path = Path("E:/Antigravity/TMRB/occlusion_analysis.py")
src  = path.read_text(encoding="utf-8")

OLD = '''def classify_frame(res, delta):
    """
    Hybrid 3-D depth-aware occlusion rule (Eq. 12):
      occlusion(t) = 1  iff  overlap(t) >= delta  AND  z_instr(t) <= z_roi(t) + DEPTH_MARGIN

    If either depth value is NaN the depth clause defaults to False
    (conservative: no false positives from missing depth).
    """
    ov = res.get("overlap", 0.0)
    zi = res.get("z_instr", float('nan'))
    zr = res.get("z_roi",   float('nan'))
    overlap_ok = ov >= delta
    if np.isnan(zi) or np.isnan(zr):
        depth_ok = False
    else:
        depth_ok = zi <= zr + DEPTH_MARGIN
    return 1 if (overlap_ok and depth_ok) else 0'''

# Check exact content
if "z_instr(t) <= z_roi(t) + DEPTH_MARGIN" in src:
    print("Found old classify_frame (uses delta symbol in docstring, written as >=).")
else:
    # try locating the function any way
    print("Note: exact old string not found. Showing lines 346-362:")
    for i, line in enumerate(src.splitlines()[345:363], 346):
        print(f"{i}: {repr(line)}")
    raise SystemExit("Cannot patch — check line content above")

NEW = '''def classify_frame(res, delta):
    """
    Hybrid 3-D depth-aware occlusion rule (Eq. 12):
      occlusion(t) = 1  iff  overlap(t) >= delta  AND  depth_ok(t)

    depth_ok logic
    --------------
    roi_source = 'live': z_instr <= z_roi + DEPTH_MARGIN.
                         If either depth is NaN -> depth_ok = False (conservative).
    roi_source = 'pf'  : ROI is occluded; its depth cannot be read.
                         Depth ordering is trivial (instrument is covering the ROI),
                         so depth_ok = True.  Only spatial overlap is required.
    roi_source = 'none': overlap = 0 -> classifier never fires.
    """
    ov = res.get("overlap", 0.0)
    overlap_ok = ov >= delta

    roi_src = res.get("roi_source", "live")
    if roi_src == "pf":
        # Instrument is trivially anterior (it covers the ROI).
        # Depth comparison is not meaningful here — use overlap alone.
        depth_ok = True
    else:
        zi = res.get("z_instr", float('nan'))
        zr = res.get("z_roi",   float('nan'))
        if np.isnan(zi) or np.isnan(zr):
            depth_ok = False
        else:
            depth_ok = zi <= zr + DEPTH_MARGIN

    return 1 if (overlap_ok and depth_ok) else 0'''

patched = src.replace(OLD, NEW)
if patched == src:
    print("ERROR: replacement did not match. Dumping actual lines 346-362:")
    for i, line in enumerate(src.splitlines()[345:363], 346):
        print(f"{i}: {repr(line)}")
else:
    path.write_text(patched, encoding="utf-8")
    print("classify_frame patched successfully.")
