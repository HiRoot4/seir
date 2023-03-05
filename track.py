from sort import Sort
from ultralytics import YOLO
import ultralytics
from tempfile import NamedTemporaryFile
import cv2 as cv
import numpy as np

mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
# model = YOLO("models/yolov8n.pt")
model = ultralytics.YOLO("yolov8m.pt")
cap = cv.VideoCapture("/home/hiroot/Desktop/seirpy/bet.mp4")
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / 8)
color = np.random.randint(0, 255, (10000, 3))


def nms2(bounding_boxes, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([])

    # Bounding boxes
    boxes = bounding_boxes

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = bounding_boxes[:, 4]

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(bounding_boxes[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / \
            (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes)


with NamedTemporaryFile(suffix=".png", prefix="frame_", dir="/dev/shm") as file:
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("No frames grabbed!")
            continue

        dets = []
        cv.imwrite(file.name, frame)
        f2 = cv.imread(file.name)
        results = model.predict(file.name)

        for res in results:
            if len(res) > 0:
                dets = res["det"]
        # dets = NMS(dets1)

        dets1 = dets
        
        dets2 = nms2(dets1, 0.3)

        trackers = mot_tracker.update(dets2)
        print(dets)
        print("________________________")
        print(dets1)
        print("________________________")
        print(dets2)
        print("________________________")
        print(trackers)

        for j, det in enumerate(dets1):
            if not np.isnan(det[0]):
                frame = cv.rectangle(
                    frame,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    (0, 0, 0),
                    3,
                )

        for j, det in enumerate(dets2):
            if not np.isnan(det[0]):
                frame = cv.rectangle(
                    frame,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    (255, 255, 255),
                    2,
                )

        for tracker in trackers:
            frame = cv.rectangle(
                frame,
                (int(tracker[0]), int(tracker[1])),
                (int(tracker[2]), int(tracker[3])),
                color[int(tracker[4])].tolist(),
                3,
            )
    
            cv.imshow("frame", frame)
            k = cv.waitKey(0)  # & 0xFF
            if k == 27:
                break