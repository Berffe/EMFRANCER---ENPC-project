"""
Decision logic based on YOLO detections.

Given a YOLO Results object `r`, we:
- Parse detections (class name, conf, bbox).
- Define a landing zone ROI in the image.
- Decide if:
    - landing_spot is found in the ROI
    - any obstacles are in that ROI
- Return compact flags to send via MAVLink.
"""
from typing import List, Dict, Tuple

# ----------------- HELPERS -----------------

# Detection confidence threshold
CONF_THRESH = 0.5

def parse_detections(r) -> List[Dict]:
    """
    Convert YOLO Results `r` into a list of simple dicts:
    [
        {
            "class": class_name,
            "conf": float,
            "bbox": [x1, y1, x2, y2],
        },
        ...
    ]
    All coords in pixel space of the original image.
    """
    dets: List[Dict] = []

    if r.boxes is None or len(r.boxes) == 0:
        return dets

    # xyxy: [N, 4], cls: [N], conf: [N]
    boxes = r.boxes.xyxy.cpu().numpy()      # x1, y1, x2, y2
    clss = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, clss, confs):
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = box
        class_name = r.names[int(cls_id)]
        dets.append(
            {
                "class": class_name,
                "conf": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    return dets

def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy

def point_in_box(x: float, y: float, box: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def landing_zone(bbox: List[float], CAMERA_RESOLUTION: Tuple[int, int]) -> Tuple[float, float, float, float]:
	limx, limy = CAMERA_RESOLUTION
	x1, y1, x2, y2 = bbox
	roi_x = 0.4*(x2 - x1)
	roi_y = 0.4*(y2 - y1)

	nx1 = max(0, x1 - roi_x/2)
	nx2 = min(limx, x1 + roi_x/2)
	ny1 = max(0, y1 - roi_y/2)
	ny2 = min(limy, y1 + roi_y/2)

	return nx1, nx2, ny1, ny2

# ----------------- DECISION MAKING -----------------

def course_of_action(r, flags, CAMERA_RESOLUTION):
	dets = parse_detections(r)
	landing_found = False

	# obstacles_in_landing = 0
	# person_found = False
	# vehicle_found = False

	for det in dets:
		cls = det["class"]

		# Check if landing spot is in frame
		if cls == "landing":
			landing_found = True
			zone = landing_zone(r, CAMERA_RESOLUTION)
			break

	for det in dets:
		cls = det["class"]
		bbox = det["bbox"]
		cx, cy = bbox_center(bbox)

		# Simple flags
		if cls == "obstacle" and point_in_box(cx, cy, zone):
			obstacles_in_landing += 1
	
	landing_safe = landing_found and (obstacles_in_landing == 0)

	return {
		"landing_found": landing_found,
		"landing_safe": landing_safe,
		"obstacles_in_landing": obstacles_in_landing,
		}