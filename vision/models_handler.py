"""
Vision Models Handler
Centralized model management for all vision tasks
Using OpenCV DNN for object detection (no torch dependency)
"""
import numpy as np
import cv2
import base64
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseVisionModel:
    """Base class for all vision models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.device = "cpu"

    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Process image and return results"""
        raise NotImplementedError


class ObjectDetector(BaseVisionModel):
    """Object detection using YOLOv3 via OpenCV DNN"""

    COCO_LABELS = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(self):
        super().__init__("detector")
        self.model = None  # Will store the DNN net
        self.output_layers = None

    def load_model(self):
        """Load YOLOv3 model using OpenCV DNN"""
        try:
            # Try to download models if not exists
            self._ensure_model_files()

            # Load YOLOv3
            weights_path = self._get_model_path("yolov3.weights")
            config_path = self._get_model_path("yolov3.cfg")

            if not os.path.exists(weights_path):
                print(f"YOLO weights not found at {weights_path}, using fallback detector")
                return False

            self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            self.output_layers = [layer.name for layer in self.model.getLayerNames()
                                  if layer.type == 'Region']

            print("Loaded: YOLOv3 Object Detector")
            return True
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            return False

    def _ensure_model_files(self):
        """Ensure model files exist"""
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)

    def _get_model_path(self, filename: str) -> str:
        """Get path to model file"""
        return str(Path(__file__).parent / "models" / filename)

    def process(self, image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        if self.model is None:
            # Try to load if not loaded
            if not self.load_model():
                return {"error": "Model not loaded. Please download yolov3.weights", "success": False}

        try:
            h, w = image.shape[:2]

            # Create blob
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.model.setInput(blob)

            # Get outputs
            outputs = self.model.forward(self.output_layers if self.output_layers
                                         else [layer.name for layer in self.model.getLayerNames()])

            # Parse detections
            boxes, scores, labels = [], [], []

            # YOLO outputs from 3 scales
            for output in outputs:
                for detection in output:
                    scores_det = detection[5:]
                    class_id = np.argmax(scores_det)
                    confidence = scores_det[class_id]

                    if confidence > threshold:
                        center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                        box_w, box_h = int(detection[2] * w), int(detection[3] * h)

                        x = int(center_x - box_w / 2)
                        y = int(center_y - box_h / 2)

                        boxes.append([x, y, box_w, box_h])
                        scores.append(float(confidence))
                        labels.append(self.COCO_LABELS[class_id] if class_id < len(self.COCO_LABELS) else f"class_{class_id}")

            # Apply non-max suppression
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.4)
                if len(indices) > 0:
                    boxes = [boxes[i] for i in indices.flatten()]
                    scores = [scores[i] for i in indices.flatten()]
                    labels = [labels[i] for i in indices.flatten()]

            # Draw boxes
            result_img = image.copy()
            for box, score, label in zip(boxes, scores, labels):
                x, y, box_w, box_h = box
                cv2.rectangle(result_img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                cv2.putText(result_img, f"{label}: {score:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', result_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "detections": boxes,
                "scores": scores,
                "labels": labels,
                "count": len(boxes),
                "image": img_b64
            }
        except Exception as e:
            return {"error": str(e), "success": False}


class SimpleDetector(BaseVisionModel):
    """Simple fallback detector using OpenCV Haar Cascades"""

    def __init__(self):
        super().__init__("detector")
        self.face_cascade = None

    def load_model(self):
        """Load Haar cascade for face detection"""
        try:
            # Try OpenCV's built-in cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("Loaded: Face Detector (Haar Cascade)")
                return True
        except Exception as e:
            print(f"Failed to load cascade: {e}")
        return False

    def process(self, image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        if self.face_cascade is None:
            if not self.load_model():
                return {"error": "No detector available", "success": False}

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            boxes, scores, labels = [], [], []
            result_img = image.copy()

            for (x, y, w, h) in faces:
                boxes.append([int(x), int(y), int(w), int(h)])
                scores.append(1.0)
                labels.append("face")
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_img, "face", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', result_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "detections": boxes,
                "scores": scores,
                "labels": labels,
                "count": len(boxes),
                "image": img_b64
            }
        except Exception as e:
            return {"error": str(e), "success": False}


class GestureRecognizer(BaseVisionModel):
    """Hand gesture recognition"""

    def __init__(self):
        super().__init__("gesture")
        self.mp_hands = None

    def process(self, image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        try:
            import mediapipe as mp

            # Initialize MediaPipe if needed
            if self.mp_hands is None:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image_rgb)

            gestures = []
            result_img = image.copy()

            if results.multi_hand_landmarks:
                import mediapipe as mp_drawing
                import mediapipe as mp_styles

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        result_img,
                        hand_landmarks,
                        mp_styles.HAND_CONNECTIONS
                    )

                    # Simple gesture classification
                    fingers_extended = sum([
                        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,
                        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
                        hand_landmarks.landmark[16].y < hand_landmarks[14].y,
                        hand_landmarks.landmark[20].y < hand_landmarks[18].y,
                    ])

                    if fingers_extended == 0:
                        gesture = "Closed Fist"
                    elif fingers_extended == 4:
                        gesture = "Open Palm"
                    elif fingers_extended == 1 and hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                        gesture = "Pointing"
                    else:
                        gesture = f"Hand ({fingers_extended} fingers)"

                    gestures.append({"hand_id": idx, "gesture": gesture})

            _, buffer = cv2.imencode('.jpg', result_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "hands_detected": len(gestures),
                "gestures": gestures,
                "image": img_b64
            }
        except ImportError:
            return {"error": "MediaPipe not installed. Run: uv pip install mediapipe", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}


class StyleTransfer(BaseVisionModel):
    """Style transfer using OpenCV"""

    STYLES = ["sketch", "watercolor", "oil", "anime"]

    def __init__(self):
        super().__init__("style_transfer")

    def process(self, image: np.ndarray, style: str = "sketch", strength: float = 0.8) -> Dict[str, Any]:
        try:
            if style == "sketch":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                inverted = 255 - gray
                blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
                inverted_blurred = 255 - blurred
                result = cv2.divide(gray, inverted_blurred, scale=256.0)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            elif style == "watercolor":
                result = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
            elif style == "oil":
                result = cv2.xphoto.oilPainting(image, 7, 1) if hasattr(cv2, 'xphoto') else image
            elif style == "anime":
                result = cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
                result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
            else:
                result = image

            if strength < 1.0:
                result = cv2.addWeighted(image, 1 - strength, result, strength, 0)

            _, buffer = cv2.imencode('.jpg', result)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return {"success": True, "style": style, "image": img_b64}
        except Exception as e:
            return {"error": str(e), "success": False}


class ActionRecognizer(BaseVisionModel):
    """Simple action recognition using optical flow"""

    def __init__(self):
        super().__init__("action")
        self.frame_buffer = []
        self.buffer_size = 16

    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        try:
            self.frame_buffer.append(image.copy())
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

            if len(self.frame_buffer) < 2:
                return {"success": True, "status": "buffering", "frames_collected": len(self.frame_buffer)}

            # Calculate optical flow
            gray = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.frame_buffer[-2], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sum(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

            if magnitude < 5000:
                action = "standing still"
            elif magnitude < 15000:
                action = "walking"
            else:
                action = "running"

            return {
                "success": True,
                "action": action,
                "motion_intensity": float(magnitude),
                "frames_collected": len(self.frame_buffer)
            }
        except Exception as e:
            return {"error": str(e), "success": False}


# Model Registry
_models: Dict[str, BaseVisionModel] = {}


def load_models():
    """Load all vision models"""
    global _models

    # Object Detector - try YOLO first, then fallback to simple detector
    detector = ObjectDetector()
    if detector.load_model():
        _models["detector"] = detector
    else:
        # Use simple fallback detector
        simple_detector = SimpleDetector()
        if simple_detector.load_model():
            _models["detector"] = simple_detector

    # Gesture Recognizer
    try:
        import mediapipe as mp
        gesture = GestureRecognizer()
        _models["gesture"] = gesture
        print("Loaded: Gesture Recognizer (MediaPipe)")
    except ImportError:
        print("MediaPipe not available for gesture recognition")

    # Style Transfer (always available - uses OpenCV)
    _models["style_transfer"] = StyleTransfer()
    print("Loaded: Style Transfer")

    # Action Recognizer
    _models["action"] = ActionRecognizer()
    print("Loaded: Action Recognizer")


def get_model(name: str) -> Optional[BaseVisionModel]:
    """Get model by name"""
    return _models.get(name)


def list_models() -> List[Dict[str, str]]:
    """List all available models"""
    return [{"name": name, "status": "loaded" if model.model or model.name == "style_transfer" else "unloaded"}
            for name, model in _models.items()]


def process_image(image_data: bytes, model_name: str, **kwargs) -> Dict[str, Any]:
    """Process image with specified model"""
    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image", "success": False}

    # Get model
    model = get_model(model_name)
    if model is None:
        return {"error": f"Model {model_name} not found", "success": False}

    # Process
    return model.process(image, **kwargs)