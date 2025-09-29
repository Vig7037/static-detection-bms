import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create folders to save images
os.makedirs("RGB", exist_ok=True)
os.makedirs("Depth", exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

count = 0
try:
    print("Press 's' to save a frame, 'q' to quit")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RGB + Depth', images)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # save frame
            cv2.imwrite(f"RGB/color_{count}.png", color_image)
            cv2.imwrite(f"Depth/depth_{count}.png", depth_colormap)
            print(f"Saved frame {count}")
            count += 1
        elif key == ord('q'):  # quit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
