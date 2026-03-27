import { Model } from '../types/model';

export const TASK_MODELS: Record<string, Model[]> = {
  detection: [
    {
      id: 'detector',
      name: 'Object Detector',
      description: 'YOLOv8-based universal object detector',
      type: 'detector',
      model_name: 'detector',
    },
    {
      id: 'yolov8n',
      name: 'YOLOv8 Nano',
      description: 'Fast and lightweight object detection',
      type: 'detector',
      model_name: 'yolov8n',
    },
  ],
  gesture: [
    {
      id: 'gesture',
      name: 'Gesture Recognizer',
      description: 'Hand gesture recognition model',
      type: 'recognizer',
      model_name: 'gesture',
    },
  ],
  style_transfer: [
    {
      id: 'style_transfer',
      name: 'Style Transfer',
      description: 'Artistic style transfer (sketch, cartoon, oil painting)',
      type: 'style_transfer',
      model_name: 'style_transfer',
    },
  ],
  action: [
    {
      id: 'action',
      name: 'Action Recognition',
      description: 'Human action recognition from video',
      type: 'action',
      model_name: 'action',
    },
  ],
};

export const STYLE_OPTIONS = [
  { id: 'sketch', name: 'Sketch' },
  { id: 'cartoon', name: 'Cartoon' },
  { id: 'oil_painting', name: 'Oil Painting' },
  { id: 'watercolor', name: 'Watercolor' },
];