import os
import glob
import re
import cv2
import numpy as np

BASE_DIR = r"e:\Antigravity\TMRB\Depth calibration"

def detect_aruco_robust(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    
    dicts = [
        ('ARUCO_ORIGINAL', cv2.aruco.DICT_ARUCO_ORIGINAL),
        ('4X4_50', cv2.aruco.DICT_4X4_50),
        ('4X4_100', cv2.aruco.DICT_4X4_100),
        ('5X5_50', cv2.aruco.DICT_5X5_50),
        ('6X6_50', cv2.aruco.DICT_6X6_50),
        ('APRILTAG_36h11', cv2.aruco.DICT_APRILTAG_36h11)
    ]
    
    for dname, dval in dicts:
        dictionary = cv2.aruco.getPredefinedDictionary(dval)
        for p_step in [3, 5, 8]:
            params = cv2.aruco.DetectorParameters()
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 63
            params.adaptiveThreshWinSizeStep = p_step
            params.minMarkerPerimeterRate = 0.003
            params.polygonalApproxAccuracyRate = 0.03
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            
            for g in [gray, clahe]:
                corners, ids, _ = detector.detectMarkers(g)
                if ids is not None and len(ids) > 0:
                    c = corners[0][0]
                    cx = float(c[:, 0].mean())
                    cy = float(c[:, 1].mean())
                    marker_id = int(ids.flatten()[0])
                    return {'id': marker_id, 'dict': dname, 'cx': cx, 'cy': cy, 'corners': c}
    return None

def scan_all():
    for trial in ['trial1', 'trial2']:
        path = os.path.join(BASE_DIR, trial)
        print(f"\n================ Scanning {trial} ================")
        rgb_files = sorted(glob.glob(os.path.join(path, "rgb_*.png")))
        for rgb_path in rgb_files:
            idx = re.search(r"\d+", os.path.basename(rgb_path)).group(0)
            info_path = os.path.join(path, f"depth_info_{idx}.txt")
            with open(info_path) as f:
                text = f.read()
            d_match = re.search(r"Marker Depth \(mm\):\s*(\d+)", text)
            c_match = re.search(r"Marker Center:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
            gt_d = int(d_match.group(1))
            info_cx, info_cy = int(c_match.group(1)), int(c_match.group(2))
            
            img_bgr = cv2.imread(rgb_path)
            det = detect_aruco_robust(img_bgr)
            
            raw_path = os.path.join(path, f"depth_raw_{idx}.png")
            raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
            raw_at_info = raw_img[info_cy, info_cx] if raw_img is not None else -1
            
            if det:
                dist = np.sqrt((det['cx'] - info_cx)**2 + (det['cy'] - info_cy)**2)
                raw_at_aruco = raw_img[int(det['cy']), int(det['cx'])] if raw_img is not None else -1
                print(f"Frame {idx}: Info Center=({info_cx:4d},{info_cy:4d}), GT={gt_d:4d}mm, Raw@Info={raw_at_info:4d}mm | ArUco ({det['dict']}) Center=({det['cx']:6.1f},{det['cy']:6.1f}), Raw@ArUco={raw_at_aruco:4d}mm | Offset={dist:6.1f} px")
            else:
                print(f"Frame {idx}: Info Center=({info_cx:4d},{info_cy:4d}), GT={gt_d:4d}mm, Raw@Info={raw_at_info:4d}mm | ArUco: NOT DETECTED AUTOMATICALLY")

if __name__ == "__main__":
    scan_all()
