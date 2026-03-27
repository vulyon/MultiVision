export interface Model {
  id: string;
  name: string;
  description: string;
  type: 'detector' | 'recognizer' | 'style_transfer' | 'action';
  model_name: string;
}

export interface InferenceResult {
  success: boolean;
  detections?: Detection[];
  image?: string;
  error?: string;
  processing_time?: number;
}

export interface Detection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export type InputSource = 'image' | 'video' | 'camera';

export type TaskType = 'detection' | 'gesture' | 'style_transfer' | 'action';