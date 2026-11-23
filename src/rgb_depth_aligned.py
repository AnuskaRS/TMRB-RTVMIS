#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse

import os

save_dir = "."

img_id = 1
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

arucoParams = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    arucoDict,
    arucoParams
)

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
args = parser.parse_args()
alpha = args.alpha

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight


fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.Camera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
disparityOut.setStreamName("disp")
queueNames.append("disp")

#Properties
rgbCamSocket = dai.CameraBoardSocket.CAM_A

camRgb.setBoardSocket(rgbCamSocket)
camRgb.setSize(1280, 720)
camRgb.setFps(fps)

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(rgbCamSocket)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise
left.setResolution(monoResolution)
left.setCamera("left")
left.setFps(fps)
right.setResolution(monoResolution)
right.setCamera("right")
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(rgbCamSocket)

# Linking
camRgb.video.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.depth.link(disparityOut.input)

camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    camRgb.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    depthWindowName = "depth"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

    # -----------------------------
    # For smoothing
    # -----------------------------
    depth_history = []

    while True:

        latestPacket = {"rgb": None, "disp": None}

        queueEvents = device.getQueueEvents(("rgb", "disp"))

        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        ####################################################
        # STEP 1 : READ DEPTH
        ####################################################
        if latestPacket["disp"] is not None:

            frameDepth = latestPacket["disp"].getFrame()

            depthVis = cv2.normalize(
                frameDepth,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            depthVis = depthVis.astype(np.uint8)
            depthVis = cv2.applyColorMap(depthVis, cv2.COLORMAP_JET)

            cv2.imshow(depthWindowName, depthVis)

        ####################################################
        # STEP 2 : RGB + ARUCO
        ####################################################
        if latestPacket["rgb"] is not None:

            frameRgb = latestPacket["rgb"].getCvFrame()

            corners, ids, rejected = detector.detectMarkers(frameRgb)

            if ids is not None and 'frameDepth' in locals():

                cv2.aruco.drawDetectedMarkers(frameRgb, corners, ids)

                for marker in corners:

                    pts = marker[0]

                    cx = int(np.mean(pts[:,0]))
                    cy = int(np.mean(pts[:,1]))

                    cv2.circle(frameRgb, (cx, cy), 5, (0,0,255), -1)

                    h, w = frameDepth.shape

                    if 0 <= cx < w and 0 <= cy < h:

                        ###################################################
                        # Larger ROI (50x50)
                        ###################################################

                        x1 = max(cx - 25, 0)
                        x2 = min(cx + 25, w)

                        y1 = max(cy - 25, 0)
                        y2 = min(cy + 25, h)

                        roi = frameDepth[y1:y2, x1:x2]

                        ###################################################
                        # Remove invalid pixels
                        ###################################################

                        valid = roi[(roi > 100) & (roi < 5000)]

                        if len(valid) > 50:

                            ###################################################
                            # Median depth
                            ###################################################

                            depth_mm = int(np.median(valid))

                            ###################################################
                            # Moving average over last 30 frames
                            ###################################################

                            depth_history.append(depth_mm)

                            if len(depth_history) > 30:
                                depth_history.pop(0)

                            smooth_depth = int(np.mean(depth_history))

                            print("--------------------------------")
                            print("Marker Depth :", smooth_depth, "mm")
                            print("Valid Pixels :", len(valid))

                            cv2.putText(
                                frameRgb,
                                f"{smooth_depth} mm",
                                (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,255,0),
                                2
                            )

                        else:

                            cv2.putText(
                                frameRgb,
                                "Invalid Depth",
                                (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,0,255),
                                2
                            )

            cv2.imshow(rgbWindowName, frameRgb)

        key = cv2.waitKey(1) & 0xFF

        # Press C to capture
        if key == ord('c'):

            rgb_file = f"rgb_{img_id:03d}.png"
            depth_vis_file = f"depth_vis_{img_id:03d}.png"
            depth_raw_file = f"depth_raw_{img_id:03d}.png"
            info_file = f"depth_info_{img_id:03d}.txt"

            cv2.imwrite(rgb_file, frameRgb)
            cv2.imwrite(depth_vis_file, depthVis)
            cv2.imwrite(depth_raw_file, frameDepth)

            with open(info_file, "w") as f:
                f.write(f"Marker Depth (mm): {smooth_depth}\n")
                f.write(f"Marker Center: ({cx}, {cy})\n")

            print("--------------------------------")
            print("Capture Saved")
            print(rgb_file)
            print(depth_vis_file)
            print(depth_raw_file)
            print(info_file)

            img_id += 1

        elif key == ord('q'):
            break