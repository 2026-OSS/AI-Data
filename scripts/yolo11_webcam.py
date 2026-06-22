#!/usr/bin/env python3
import argparse
from collections import Counter, deque
import time
from pathlib import Path


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts/yolo11-v15/weights/best.pt"
)
DEFAULT_HAND_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/hand-landmarker/hand_landmarker.task"
)


def import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is not installed. Install it in the run environment with:\n"
            "  pip install opencv-python ultralytics\n"
            "or, if you use conda:\n"
            "  conda run -n <env> pip install opencv-python ultralytics"
        ) from exc


def import_yolo():
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it in the run environment with:\n"
            "  pip install ultralytics opencv-python\n"
            "or, if you use conda:\n"
            "  conda run -n <env> pip install ultralytics opencv-python"
        ) from exc


def parse_source(source):
    return int(source) if str(source).isdigit() else source


def resolve_model_path(path):
    path = Path(path)
    if path.is_dir():
        for candidate in (path / "best.pt", path / "weights/best.pt"):
            if candidate.exists():
                return candidate
    return path


def load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "mediapipe is not installed. Install it in the run environment with:\n"
            "  pip install mediapipe\n"
            "or:\n"
            "  conda run -n pytorch_mnist pip install mediapipe"
        ) from exc
    if not hasattr(mp, "solutions"):
        try:
            import mediapipe.solutions as solutions
        except ImportError:
            solutions = None
        if solutions is not None:
            mp.solutions = solutions
    return mp


def create_hand_tracker(mp, args):
    if hasattr(mp, "solutions"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=args.hand_conf,
            min_tracking_confidence=args.hand_track_conf,
        )
        return {
            "api": "solutions",
            "hands": hands,
            "solutions": mp.solutions,
            "drawer": mp.solutions.drawing_utils,
            "styles": mp.solutions.drawing_styles,
        }

    hand_model = Path(args.hand_model)
    if not hand_model.exists():
        raise SystemExit(f"Hand landmarker model not found: {hand_model}")

    try:
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(hand_model),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=args.hand_conf,
            min_hand_presence_confidence=args.hand_conf,
            min_tracking_confidence=args.hand_track_conf,
        )
        hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
    except AttributeError as exc:
        raise SystemExit(
            "This mediapipe installation does not provide either "
            "mediapipe.solutions.hands or mediapipe.tasks.vision.HandLandmarker.\n"
            "Install a compatible package in the active environment:\n"
            "  pip install --upgrade mediapipe"
        ) from exc

    return {
        "api": "tasks",
        "hands": hands,
        "module": mp,
    }


def result_to_detections(result):
    detections = []
    if result is None:
        return detections

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        detections.append(
            {
                "class_id": cls,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            }
        )
    return detections


def detection_label(detection, names):
    class_id = detection["class_id"]
    return names.get(class_id, str(class_id))


def best_detection_for_label(detections, label, names, preferred_detection=None):
    matches = [detection for detection in detections if detection_label(detection, names) == label]
    if not matches:
        return preferred_detection
    if preferred_detection in matches:
        return preferred_detection
    return max(matches, key=lambda detection: detection["confidence"])


def draw_detections(frame, detections, names, pointed_detection=None):
    for detection in detections:
        cls = detection["class_id"]
        conf = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        label = f"{names.get(cls, cls)} {conf:.2f}"
        is_pointed = detection is pointed_detection
        color = (0, 210, 255) if is_pointed else (40, 220, 40)
        thickness = 4 if is_pointed else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        y_text = max(y1, text_h + baseline + 6)
        cv2.rectangle(
            frame,
            (x1, y_text - text_h - baseline - 6),
            (x1 + text_w + 8, y_text + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, y_text - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def find_index_fingertip(mp, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mp["api"] == "tasks":
        module = mp["module"]
        mp_image = module.Image(
            image_format=module.ImageFormat.SRGB,
            data=rgb,
        )
        results = mp["hands"].detect(mp_image)
        if not results.hand_landmarks:
            return None, False

        landmark = results.hand_landmarks[0][8]
        height, width = frame.shape[:2]
        fingertip = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(mp["display"], fingertip, 9, (0, 210, 255), -1)
        cv2.circle(mp["display"], fingertip, 13, (0, 0, 0), 2)
        return fingertip, True

    results = mp["hands"].process(rgb)
    if not results.multi_hand_landmarks:
        return None, False

    hand_landmarks = results.multi_hand_landmarks[0]
    mp["drawer"].draw_landmarks(
        mp["display"],
        hand_landmarks,
        mp["solutions"].hands.HAND_CONNECTIONS,
        mp["styles"].get_default_hand_landmarks_style(),
        mp["styles"].get_default_hand_connections_style(),
    )
    landmark = hand_landmarks.landmark[8]
    height, width = frame.shape[:2]
    fingertip = (int(landmark.x * width), int(landmark.y * height))
    cv2.circle(mp["display"], fingertip, 9, (0, 210, 255), -1)
    cv2.circle(mp["display"], fingertip, 13, (0, 0, 0), 2)
    return fingertip, True


def point_distance_to_box(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0

    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def detection_at_point(detections, point, margin=0):
    if point is None:
        return None

    matches = [detection for detection in detections if point_distance_to_box(point, detection["bbox"]) <= margin]
    if not matches:
        return None

    return min(
        matches,
        key=lambda detection: (
            point_distance_to_box(point, detection["bbox"]),
            -detection["confidence"],
        ),
    )


def update_pointing_vote(samples, detection, names, now, args, last_pointed_at):
    if detection is None:
        if last_pointed_at is None or now - last_pointed_at > args.point_reset_seconds:
            samples.clear()
            return None, last_pointed_at, 0.0
        return None, last_pointed_at, current_progress(samples, now, args.point_hold_seconds)

    class_id = detection["class_id"]
    label = names.get(class_id, str(class_id))
    samples.append((now, label, detection["confidence"]))
    while samples and now - samples[0][0] > args.point_hold_seconds:
        samples.popleft()

    progress = current_progress(samples, now, args.point_hold_seconds)
    if progress < 1.0:
        return None, now, progress

    counts = Counter(label for _, label, _ in samples)
    selected_label, _ = max(
        counts.items(),
        key=lambda item: (
            item[1],
            sum(conf for _, label, conf in samples if label == item[0]) / item[1],
        ),
    )
    return selected_label, now, progress


def current_progress(samples, now, hold_seconds):
    if not samples:
        return 0.0
    return min((now - samples[0][0]) / hold_seconds, 1.0)


def draw_status(frame, inference_ms, hand_detected, pointed_label, selected_label, progress, hand_mode):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 124), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.82, frame, 0.18, 0, frame)

    if hand_mode == "off":
        status = "YOLO11 webcam detection"
        detail = "MediaPipe finger tracking: off"
    else:
        status = "MediaPipe: finger not detected"
        detail = "Touch an object for 3 seconds"
        if hand_detected:
            status = "MediaPipe: finger detected"
        if pointed_label:
            detail = f"Touching: {pointed_label}  {progress * 100:.0f}%"
        if selected_label:
            detail = f"Most recognized for 3s: {selected_label}"

    color = (60, 220, 60) if selected_label else (0, 210, 255)
    cv2.putText(frame, status, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        detail,
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{inference_ms:.0f} ms   c: clear   q/esc: quit",
        (18, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def run(args):
    global cv2

    cv2 = import_cv2()
    YOLO = import_yolo()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    mp_module = load_mediapipe() if args.hand_mode == "point" else None
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/source: {args.source}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Using model: {model_path}")
    print(f"Using source: {args.source}")
    print(f"Classes: {model.names}")
    print(f"Hand mode: {args.hand_mode}")
    if args.hand_mode == "point":
        print(f"Pointing hold: {args.point_hold_seconds:.1f}s")
    print("Press q or Esc to quit.")

    frame_count = 0
    last_result = None
    inference_ms = 0.0
    pointing_samples = deque()
    last_pointed_at = None
    selected_label = None
    hand_tracker = None
    if args.hand_mode == "point":
        hand_tracker = create_hand_tracker(mp_module, args)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            display = frame.copy()

            if frame_count % max(args.infer_stride, 1) == 0:
                start = time.perf_counter()
                last_result = model.predict(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device or None,
                    verbose=False,
                )[0]
                inference_ms = (time.perf_counter() - start) * 1000

            detections = result_to_detections(last_result)
            pointed_label = None
            pointed_detection = None
            progress = 0.0
            hand_detected = False
            display_detections = detections
            display_detection = None
            if args.hand_mode == "point":
                mp_context = dict(hand_tracker)
                mp_context["display"] = display
                fingertip, hand_detected = find_index_fingertip(mp_context, frame)
                pointed_detection = detection_at_point(detections, fingertip, args.point_margin)
                if pointed_detection is not None:
                    pointed_label = detection_label(pointed_detection, model.names)

                voted_label, last_pointed_at, progress = update_pointing_vote(
                    pointing_samples,
                    pointed_detection,
                    model.names,
                    time.perf_counter(),
                    args,
                    last_pointed_at,
                )
                if voted_label is not None:
                    selected_label = voted_label

                display_detection = best_detection_for_label(
                    detections,
                    selected_label,
                    model.names,
                    preferred_detection=pointed_detection,
                )
                if display_detection is None:
                    display_detection = pointed_detection
                display_detections = [display_detection] if display_detection is not None else []

            draw_detections(display, display_detections, model.names, display_detection)
            draw_status(
                display,
                inference_ms,
                hand_detected,
                pointed_label,
                selected_label,
                progress,
                args.hand_mode,
            )
            cv2.imshow(args.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                pointing_samples.clear()
                selected_label = None
            elif key in (ord("q"), 27):
                break
    finally:
        if hand_tracker is not None:
            hand_tracker["hands"].close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a YOLO11 model from a webcam or iPhone camera.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a YOLO11 .pt model or artifact directory.",
    )
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--device", default="", help="Device, e.g. cpu, mps, or 0.")
    parser.add_argument("--infer-stride", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--hand-mode", choices=("off", "point"), default="point", help="Use off for plain detection, point for fingertip selection.")
    parser.add_argument("--point-hold-seconds", type=float, default=3.0, help="Seconds to vote while pointing.")
    parser.add_argument("--point-reset-seconds", type=float, default=0.7, help="Seconds before clearing votes when not pointing.")
    parser.add_argument("--point-margin", type=int, default=50, help="Pixel margin around boxes that still counts as pointing.")
    parser.add_argument("--hand-conf", type=float, default=0.4, help="MediaPipe hand detection confidence.")
    parser.add_argument("--hand-track-conf", type=float, default=0.4, help="MediaPipe hand tracking confidence.")
    parser.add_argument("--hand-model", default=str(DEFAULT_HAND_MODEL_PATH), help="MediaPipe task hand landmarker model.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera capture width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera capture height.")
    parser.add_argument("--window-name", default="yolo11-webcam", help="OpenCV window name.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
