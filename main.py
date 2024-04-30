import sys
import pyrealsense2 as rs
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


# model = YOLO("yolov8n-seg.pt")
model = YOLO("yolov8s-seg.pt")
if torch.cuda.is_available():
    model.to('cuda')

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize an object tracker (simple for demonstration)
object_tracker = {}
object_filters = {}

resultMap= {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# Object Types and Colors
object_colors = {
    'person': (255, 0, 0), 'bicycle': (0, 255, 0), 'car': (0, 0, 255), 'motorcycle': (255, 255, 0),
    'bus': (255, 0, 255), 'train': (0, 255, 255), 'truck': (128, 128, 0), 'traffic light': (128, 0, 128),
    'stop sign': (0, 128, 128), # Continue for all object types...
}

# Cropping parameters
CROP_MARGIN_T = 90
CROP_MARGIN_B = 90
CROP_MARGIN_L = 100
CROP_MARGIN_R = 140



# *** Camera Calibration Parameters (Get these for your RealSense) ***
fx = 615.25  # Focal length (x) - Example values, replace with your camera's
fy = 615.95  # Focal length (y)
cx = 310.09  # Principle point (x)
cy = 244.7   # Principle point (y)
distortion = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients

# *** Bird's-eye view parameters ***
bev_height = 480  # Height of the bird's-eye view image (pixels)
bev_width = 640   # Width of the bird's-eye view image
bev_ground_level = 2.0  # Assumed ground level height (meters)


# Object tracking deque
object_data = deque(maxlen=10)  # Stores last 10 detected objects
# Initialize a dictionary to store position history for each track ID
position_history = defaultdict(lambda: deque(maxlen=5))


def calculate_birdseye_position(distance, max_distance=9):
    # Simple linear mapping: max distance at top (y=0), zero at bottom (y=479)
    print(f"this is te coordinate{(1 - distance / max_distance) * (bev_height-1)}")
    return int((1 - distance / max_distance) * (bev_height-1))

def calculate_partition_lines(max_distance=9):
    partition_lines = []
    for distance in range(1, max_distance + 1):
        y_position = calculate_birdseye_position(distance)
        partition_lines.append(y_position)
    return partition_lines


def calculate_center_of_gravity(mask):
    """Calculates the center of gravity (CoG) for an object segmentation mask.

    Args:
        mask: A NumPy array representing the object's segmentation mask.

    Returns:
        A tuple (x, y) representing the center of gravity coordinates.
    """
    M = cv2.moments(mask)
    if M['m00'] != 0: # Ensure the mask has non-zero area
        cog_x = int(M["m10"] / M["m00"])
        cog_y = int(M["m01"] / M["m00"])
        return cog_x, cog_y
    else:
        return None  # Handle cases where the mask has zero area
# Define a function to calculate the average position from a deque
def get_average_position(positions):
    if len(positions) == 0:
        return None
    positions_array = np.array(positions)
    avg_position = np.mean(positions_array, axis=0)
    return int(avg_position[0]), int(avg_position[1])

# Kalman Filter for object tracking


def overlay_image(image, scaleFactor = 10):


    # Check if the image has an alpha channel
    if image.shape[2] == 4:
        # Split the image into RGB and Alpha channels
        bgr, alpha = image[..., :3], image[..., 3]
        # Convert alpha channel to a binary mask where 255 is fully opaque
        mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]

        # Resize the image and mask
        image_resized_width = bev_width // scaleFactor
        image_aspect_ratio = bgr.shape[0] / bgr.shape[1]
        image_resized_height = int(image_resized_width * image_aspect_ratio)
        image_resized = cv2.resize(bgr, (image_resized_width, image_resized_height))
        mask_resized = cv2.resize(mask, (image_resized_width, image_resized_height))

        return image_resized, mask_resized, (image_resized_height, image_resized_width)



def creating_roi_on_birdseye_with_image(birdseye_frame, image_resized, mask_resized, image_resized_dimension, x_position= None, y_position= None):
    if x_position == None:
        x_position = (bev_width - image_resized_dimension[1]) // 2
        y_position = bev_height - image_resized_dimension[0]
    else:
        y_position = max(y_position - image_resized_dimension[0] // 2, 0)
        x_position = max(x_position - image_resized_dimension[1] // 2, 0)

        # Boundary checks (with adjustments)
    roi_top = max(0, y_position)  # Adjust top if out-of-bounds
    roi_bottom = min(bev_height, y_position + image_resized_dimension[0])  # Adjust bottom
    roi_left = max(0, x_position)  # Adjust left
    roi_right = min(bev_width, x_position + image_resized_dimension[1])  # Adjust right

    # Calculate visible portion of the image
    visible_image_height = roi_bottom - roi_top
    visible_image_width = roi_right - roi_left

    # Create the ROI in the birdseye frame
    roi = birdseye_frame[roi_top:roi_bottom, roi_left:roi_right]

    # Slice the image and mask to match the visible ROI
    image_to_overlay = image_resized[0:visible_image_height, 0:visible_image_width]
    mask_to_overlay = mask_resized[0:visible_image_height, 0:visible_image_width]

    # Overlay the visible portion
    roi[np.where(mask_to_overlay == 255)] = image_to_overlay[np.where(mask_to_overlay == 255)]
    birdseye_frame[roi_top:roi_bottom, roi_left:roi_right] = roi

def objectInit(imgPath, scaleFactor= None):
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: '{imgPath}' not found.")

    if scaleFactor == None:
        return overlay_image(img)


    return overlay_image(img, scaleFactor)

# Stroke history parameters
STROKE_HISTORY_LENGTH = 10  # Adjust as needed
STROKE_FADEOUT_TIME = 0.2  # Time in seconds for full fadeout

# Initialize stroke history
stroke_history = defaultdict(lambda: deque(maxlen=STROKE_HISTORY_LENGTH))
def calculate_stroke_opacity(timestamp):
    """Calculates the opacity for a stroke based on its age.

    Args:
        timestamp: The datetime.datetime object when the stroke point was added.

    Returns:
        A float between 0.0 and 1.0 representing the opacity.
    """
    time_since = datetime.datetime.now() - timestamp[2]
    fade_progress = min(1.0, time_since.total_seconds() / STROKE_FADEOUT_TIME)
    return 1.0 - fade_progress

if __name__ == "__main__":

    birdseye_frame = np.full((bev_height, bev_width, 3), 255, dtype=np.uint8)  # Fill the frame with white
    # Load the car image
    carpath = "car4.png"
    personPath = "person_icon.png"
    otherObjectsPath = "otherObjects.png"

    car_resized, car_mask_resized, car_resized_dimension = objectInit(carpath, scaleFactor=8)
    person_resized, person_mask_resized, person_resized_dimension = objectInit(personPath, scaleFactor=10)
    otherObjects_resized,  otherObjects_mask_resized, otherObjects_dimension = objectInit(otherObjectsPath)


    try:
        while True:
            # Get frames from RealSense
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays for processing
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # *** Apply Median Filtering to Depth Image ***
            # depth_image = cv2.medianBlur(depth_image, 1)  # Apply median filter with a 5x5 kernel

            # Crop depth image
            h, w = depth_image.shape
            cropped_depth = depth_image[CROP_MARGIN_T:h - CROP_MARGIN_B, CROP_MARGIN_L:w - CROP_MARGIN_R]

            # Apply colormap to cropped depth image
            cropped_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cropped_depth, alpha=0.33), cv2.COLORMAP_JET)
            cropped_depth_colormap = cv2.resize(cropped_depth_colormap, (640, 480))

            birdseye_frame.fill(255)  # Clear previous frame
            cv2.line(birdseye_frame, (bev_width//2, 0), (bev_width//2, bev_height), (0, 0, 0), 2)  # Line of sight


            # Perform object detection with YOLOv8 (on the original color_image)
            annotator = Annotator(color_image, pil=False, line_width=2)
            # Perform object detection with YOLOv8 (on the original color_image)
            results = model.track(color_image, persist=True)



            creating_roi_on_birdseye_with_image(birdseye_frame, car_resized, car_mask_resized, car_resized_dimension,
                                                x_position=None, y_position=None)



            # Draw partition lines and labels
            partition_lines = calculate_partition_lines()
            for i, line_y in enumerate(partition_lines):
                cv2.line(birdseye_frame, (0, line_y), (bev_width, line_y), (0,0,0), 1)
                cv2.putText(birdseye_frame, f"{i + 1}m", (10, line_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,0), 1)

            # Process detections
            if results[0].boxes.id is not None and results[0].masks is not None:
                #boxesClassVals = r.boxes.cls
                masks = results[0].masks.xy
                track_ids = results[0].boxes.id.int().cpu().tolist()
                labels = results[0].boxes.cls

                for mask, track_id, label_rep in zip(masks, track_ids, labels ):
                    annotator.seg_bbox(mask=mask,
                                       mask_color=colors(track_id, True),
                                       track_label=str(track_id))

                    # Calculate center of gravity (CoG)
                    cog_x, cog_y = calculate_center_of_gravity(mask)



                    if cog_x is not None:  # Only process if CoG is valid
                        # Get distance from the depth frame

                        scale_x = cropped_depth.shape[1] / cropped_depth_colormap.shape[1]
                        scale_y = cropped_depth.shape[0] / cropped_depth_colormap.shape[0]

                        cog_x_final = int(cog_x * scale_x)
                        cog_y_final = int(cog_y * scale_y)

                        cog_x_adjusted = cog_x_final + CROP_MARGIN_L
                        cog_y_adjusted = cog_y_final + CROP_MARGIN_T

                        distance = depth_frame.get_distance(cog_x_adjusted, cog_y_adjusted)

                        label = resultMap[int(label_rep)]
                        cv2.putText(color_image,
                                    f"{label} - {distance:.2f}m",
                                    (cog_x, cog_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        cv2.rectangle(color_image, (cog_x - 5, cog_y - 5), (cog_x + 5, cog_y + 5), (0, 255, 0),
                                      2)  # Mark CoG on color image

                        # Update position history
                        timestamp = datetime.datetime.now()
                        stroke_history[track_id].append((cog_x, calculate_birdseye_position(distance), timestamp))

                        # Update position history for this track ID
                        position_history[track_id].append((cog_x, calculate_birdseye_position(distance)))

                        # Get the averaged position
                        avg_cog_x, avg_birdseye_y = get_average_position(position_history[track_id])

                        # Use averaged positions for plotting
                        if label == 'person': #if person show image icon
                            #

                            creating_roi_on_birdseye_with_image(birdseye_frame, person_resized, person_mask_resized,
                                                                person_resized_dimension,
                                                                avg_cog_x, avg_birdseye_y)
                        else: #if not person do not show image icon
                            creating_roi_on_birdseye_with_image(birdseye_frame, otherObjects_resized, otherObjects_mask_resized,
                                                                otherObjects_dimension,
                                                                avg_cog_x, avg_birdseye_y)


                        color = object_colors.get(label, (128, 128, 128))
                        cv2.circle(birdseye_frame, (avg_cog_x, avg_birdseye_y), 7, color, -1)
                        cv2.line(birdseye_frame, (avg_cog_x, avg_birdseye_y), (320, 480), color, 2)
                        for i in range(1, len(stroke_history[track_id])):
                            point1, point2, timestamp = stroke_history[track_id][i - 1], stroke_history[track_id][i], \
                                                        stroke_history[track_id][i]
                            print("point1", point1)
                            print("point2", point2)
                            print("timestamp", timestamp)
                            opacity = calculate_stroke_opacity(timestamp)
                            print("opacity", opacity)
                            color_with_opacity = tuple(
                                [int(c * opacity) for c in color] + [opacity])  # Adjust opacity
                            print("color_with_opacity", opacity)
                            cv2.line(birdseye_frame, (point1[0], point1[1]), (point2[0], point2[1]), color_with_opacity, thickness=2)

                        # Resize images for side-by-side display
            # birdseye_frame[y_position:y_position + car_resized_height,x_position:x_position + car_resized_width] = car_resized
            resized_color = cv2.resize(annotator.result(), (512, 384))
            resized_depth_colormap = cv2.resize(cropped_depth_colormap, (512, 384))
            resized_birdview = cv2.resize(birdseye_frame, (512, 384))

            combined_image = np.hstack((resized_color, resized_depth_colormap, resized_birdview))
            cv2.imshow('RealSense Color & Depth & Birds Eye', combined_image)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
