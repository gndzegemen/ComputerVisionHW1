import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YOLO_PATH = MODEL_DIR / "yolov8n-seg.pt"
TARGET_PATH = MODEL_DIR / "target_face.npy"

BLUR_KERNEL = 61
MATCH_THRESHOLD = 0.5
CONFIRM_FRAMES = 3
TARGET_MEMORY = 10

BLUR_METHODS = ["Gaussian", "Mean", "Pixelation"]
WINDOW = "Anonymizer"
MODE_TRACKBAR = "Mode 1-3"
BLUR_TRACKBAR = "Blur 0-2"


def download_if_missing(url, path):
    if not path.exists():
        print("Downloading:", path.name)
        urllib.request.urlretrieve(url, str(path))


download_if_missing(YUNET_URL, YUNET_PATH)
download_if_missing(SFACE_URL, SFACE_PATH)

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

face_detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.8)
face_recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
person_model = YOLO(str(YOLO_PATH))


def apply_separable(image, row_matrix, col_matrix):
    channels = image.astype(np.float32).transpose(2, 0, 1)
    result = row_matrix @ channels @ col_matrix.T
    return np.clip(result.transpose(1, 2, 0), 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def gaussian_matrix(n, k):
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    index = np.arange(n)
    distance = index[None, :] - index[:, None]
    weights = np.exp(-distance ** 2 / (2 * sigma ** 2)) * (np.abs(distance) <= k // 2)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def gaussian_blur(image, k):
    h, w = image.shape[:2]
    return apply_separable(image, gaussian_matrix(h, k), gaussian_matrix(w, k))


@lru_cache(maxsize=None)
def mean_matrix(n, k):
    index = np.arange(n)
    distance = index[None, :] - index[:, None]
    weights = (np.abs(distance) <= k // 2).astype(np.float32)
    return weights / weights.sum(axis=1, keepdims=True)


def mean_blur(image, k):
    h, w = image.shape[:2]
    return apply_separable(image, mean_matrix(h, k), mean_matrix(w, k))


def pixelate(image, k):
    h, w = image.shape[:2]
    padded = np.pad(image, ((0, -h % k), (0, -w % k), (0, 0)), mode="edge")
    ph, pw = padded.shape[:2]
    blocks = padded.reshape(ph // k, k, pw // k, k, 3).mean(axis=(1, 3)).astype(np.uint8)
    return np.repeat(np.repeat(blocks, k, axis=0), k, axis=1)[:h, :w]


def blur(image, method):
    if method == "Gaussian":
        return gaussian_blur(image, BLUR_KERNEL)
    if method == "Mean":
        return mean_blur(image, BLUR_KERNEL)
    return pixelate(image, BLUR_KERNEL)


def keep_sharp(original, blurred, mask):
    return np.where(mask[:, :, None] > 0, original, blurred)


def detect_faces(frame):
    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    return [] if faces is None else list(faces)


def face_embedding(frame, face):
    aligned_face = face_recognizer.alignCrop(frame, face)
    return face_recognizer.feature(aligned_face)


def face_similarity(embedding1, embedding2):
    return face_recognizer.match(embedding1, embedding2, cv2.FaceRecognizerSF_FR_COSINE)


def detect_people(frame):
    result = person_model(frame, classes=[0], verbose=False)[0]
    masks = []
    if result.masks is not None:
        for outline in result.masks.xy:
            mask = np.zeros(frame.shape[:2], np.uint8)
            if len(outline) > 2:
                cv2.fillPoly(mask, [outline.astype(np.int32)], 255)
            masks.append(mask)
    return masks


def face_mask(frame):
    mask = np.zeros(frame.shape[:2], np.uint8)
    for face in detect_faces(frame):
        x, y, w, h = face[:4].astype(int)
        center = (x + w // 2, y + h // 2)
        axes = (int(w * 0.6), int(h * 0.75))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask, ""


def people_mask(frame):
    mask = np.zeros(frame.shape[:2], np.uint8)
    people = detect_people(frame)
    for person in people:
        mask = cv2.bitwise_or(mask, person)
    return mask, f"People: {len(people)}"


class TargetPerson:
    def __init__(self):
        self.embedding = np.load(str(TARGET_PATH)) if TARGET_PATH.exists() else None
        self.last_center = None
        self.matched_frames = 0
        self.missed_frames = 0

    def register(self, frame):
        faces = detect_faces(frame)
        if not faces:
            print("No face found.")
            return
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        self.embedding = face_embedding(frame, largest_face)
        np.save(str(TARGET_PATH), self.embedding)
        print("Target registered.")

    def clear(self):
        self.embedding = None
        self.last_center = None
        self.matched_frames = 0
        self.missed_frames = 0
        TARGET_PATH.unlink(missing_ok=True)
        print("All registered people cleared.")

    def find_target_face(self, frame):
        best_score, best_center = 0.0, None
        for face in detect_faces(frame):
            score = face_similarity(self.embedding, face_embedding(frame, face))
            if score > MATCH_THRESHOLD and score > best_score:
                x, y, w, h = face[:4].astype(int)
                best_score, best_center = score, (x + w // 2, y + h // 2)
        return best_center, best_score

    def mask(self, frame):
        mask = np.zeros(frame.shape[:2], np.uint8)
        if self.embedding is None:
            return mask, "No target - press 'r' to register"

        center, score = self.find_target_face(frame)

        self.matched_frames = self.matched_frames + 1 if center is not None else 0
        confirmed = self.matched_frames >= CONFIRM_FRAMES
        if confirmed:
            self.last_center, self.missed_frames = center, 0
        else:
            self.missed_frames += 1
            if self.missed_frames > TARGET_MEMORY:
                self.last_center = None
        if self.last_center is None:
            return mask, "Target not found"

        h, w = frame.shape[:2]
        cx = min(max(self.last_center[0], 0), w - 1)
        cy = min(max(self.last_center[1], 0), h - 1)
        info = f"Target similarity: {score:.2f}" if confirmed else "Target (tracking)"
        for person in detect_people(frame):
            if person[cy, cx] > 0:
                return person, info

        cv2.circle(mask, (cx, cy), 80, 255, -1)
        return mask, info


def draw_text(image, text, y, color=(0, 255, 0)):
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save_comparison(frame, mask):
    panels = []
    for method in BLUR_METHODS:
        panel = keep_sharp(frame, blur(frame, method), mask)
        cv2.putText(panel, method, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        panels.append(panel)
    path = OUTPUT_DIR / f"comparison_{int(time.time())}.png"
    cv2.imwrite(str(path), np.hstack(panels))
    print("Comparison saved:", path)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        print("Could not open video source:", source)
        return

    target = TargetPerson()

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar(MODE_TRACKBAR, WINDOW, 1, 3, lambda value: None)
    cv2.createTrackbar(BLUR_TRACKBAR, WINDOW, 0, len(BLUR_METHODS) - 1, lambda value: None)
    cv2.setTrackbarMin(MODE_TRACKBAR, WINDOW, 1)

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        mode = cv2.getTrackbarPos(MODE_TRACKBAR, WINDOW)
        method = BLUR_METHODS[cv2.getTrackbarPos(BLUR_TRACKBAR, WINDOW)]
        start = time.time()

        if mode == 1:
            mask, info = face_mask(frame)
        elif mode == 2:
            mask, info = people_mask(frame)
        else:
            mask, info = target.mask(frame)

        blur_start = time.time()
        blurred = blur(frame, method)
        blur_ms = (time.time() - blur_start) * 1000

        result = keep_sharp(frame, blurred, mask)
        fps = 1.0 / max(time.time() - start, 1e-6)

        draw_text(result, f"Mode {mode} | {method}: {blur_ms:.1f} ms | FPS: {fps:.1f}", 25)
        if info:
            draw_text(result, info, 50, color=(0, 255, 255))
        draw_text(result, "r: register  x: clear  c: compare  s: snapshot  q: quit",
                  result.shape[0] - 10, color=(255, 255, 255))
        cv2.imshow(WINDOW, result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            target.register(frame)
        elif key == ord("x"):
            target.clear()
        elif key == ord("c"):
            save_comparison(frame, mask)
        elif key == ord("s"):
            path = OUTPUT_DIR / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(path), result)
            print("Snapshot saved:", path)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
