"""
Vision Models Handler
Centralized model management for all vision tasks
"""
import numpy as np
import cv2
import base64
from typing import Dict, Any, Optional, List
import torch


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
    """Object detection using DETR"""

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
        self.processor = None

    def process(self, image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not loaded", "success": False}

        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            from PIL import Image

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process
            inputs = self.processor(
                images=Image.fromarray(image_rgb),
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            from transformers import DetrImageProcessor
            target_sizes = torch.tensor([image_rgb.shape[:2]])
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]

            boxes, scores, labels = [], [], []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                boxes.append(box.cpu().numpy().tolist())
                scores.append(float(score.cpu().numpy()))
                label_idx = int(label.cpu().numpy())
                labels.append(self.COCO_LABELS[label_idx] if label_idx < len(self.COCO_LABELS) else f"class_{label_idx}")

            # Draw boxes
            result_img = image.copy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_img, f"{label}: {score:.2f}", (x1, y1 - 10),
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

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image_rgb)

            gestures = []
            result_img = image.copy()

            if results.multi_hand_landmarks:
                from mediapipe.framework.formats import landmark_pb2
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
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
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
            return {"error": "MediaPipe not installed", "success": False}
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

    # Object Detector
    try:
        from transformers import DetrImageProcessor, DetrForObjectDetection
        detector = ObjectDetector()
        detector.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        detector.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        detector.model.eval()
        _models["detector"] = detector
        print("Loaded: Object Detector")
    except Exception as e:
        print(f"Failed to load detector: {e}")

    # Gesture Recognizer
    try:
        import mediapipe as mp
        gesture = GestureRecognizer()
        gesture.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        _models["gesture"] = gesture
        print("Loaded: Gesture Recognizer")
    except ImportError:
        print("MediaPipe not available for gesture recognition")
    except Exception as e:
        print(f"Failed to load gesture: {e}")

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