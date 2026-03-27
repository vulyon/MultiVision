import { Model, InferenceResult } from '../types/model';

export async function fetchModels(): Promise<Model[]> {
  const response = await fetch('/api/models');
  const data = await response.json();
  return data.models || [];
}

export async function runInference(
  imageData: string | File,
  modelName: string,
  options: {
    threshold?: number;
    style?: string;
  } = {}
): Promise<InferenceResult> {
  const formData = new FormData();

  if (typeof imageData === 'string') {
    // Base64 encoded image
    formData.append('image', imageData);
  } else {
    // File upload
    formData.append('file', imageData);
  }

  formData.append('model', modelName);
  if (options.threshold !== undefined) {
    formData.append('threshold', options.threshold.toString());
  }
  if (options.style) {
    formData.append('style', options.style);
  }

  const response = await fetch('/api/infer', {
    method: 'POST',
    body: formData,
  });

  return response.json();
}

export async function processCameraFrame(
  frameData: string,
  modelName: string,
  threshold: number = 0.5
): Promise<InferenceResult> {
  const response = await fetch('/api/camera', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      frame: frameData,
      model: modelName,
      threshold: threshold,
    }),
  });

  return response.json();
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix if present
      const base64 = result.includes(',') ? result.split(',')[1] : result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}