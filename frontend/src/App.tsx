import { useState, useRef, useEffect } from 'react';
import {
  Eye,
  Camera,
  ImageIcon,
  Video,
  Zap,
  Cpu,
  Activity,
  Upload,
  Loader2,
} from 'lucide-react';
import { Model, Detection, InputSource, TaskType, InferenceResult } from './types/model';
import { TASK_MODELS, STYLE_OPTIONS } from './config/models';
import { runInference, processCameraFrame, fileToBase64 } from './services/modelService';

function App() {
  const [taskType, setTaskType] = useState<TaskType>('detection');
  const [inputSource, setInputSource] = useState<InputSource>('image');
  const [selectedModel, setSelectedModel] = useState<string>('detector');
  const [selectedStyle, setSelectedStyle] = useState<string>('sketch');
  const [threshold, setThreshold] = useState<number>(0.5);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [processingTime, setProcessingTime] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const models = TASK_MODELS[taskType] || [];

  useEffect(() => {
    if (models.length > 0) {
      setSelectedModel(models[0].model_name);
    }
  }, [taskType]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const captureFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0);
      const frameData = canvas.toDataURL('image/jpeg').split(',')[1];
      return frameData;
    }
    return null;
  };

  const handleProcess = async () => {
    setIsProcessing(true);
    const startTime = Date.now();

    try {
      let inferenceResult: InferenceResult;

      if (inputSource === 'camera' && isCameraActive) {
        const frameData = await captureFrame();
        if (frameData) {
          inferenceResult = await processCameraFrame(frameData, selectedModel, threshold);
        } else {
          throw new Error('Failed to capture frame');
        }
      } else if (inputSource === 'image' && imageFile) {
        const base64Image = await fileToBase64(imageFile);
        inferenceResult = await runInference(base64Image, selectedModel, {
          threshold,
          style: selectedStyle,
        });
      } else if (inputSource === 'image' && imagePreview) {
        const base64Image = imagePreview.split(',')[1];
        inferenceResult = await runInference(base64Image, selectedModel, {
          threshold,
          style: selectedStyle,
        });
      } else {
        throw new Error('No image to process');
      }

      const endTime = Date.now();
      setProcessingTime((endTime - startTime) / 1000);
      setResult(inferenceResult);
    } catch (err) {
      setResult({
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLiveDetection = async () => {
    if (!isCameraActive) return;

    try {
      const frameData = await captureFrame();
      if (frameData) {
        const inferenceResult = await processCameraFrame(frameData, selectedModel, threshold);
        setResult(inferenceResult);
      }
    } catch (err) {
      console.error('Live detection error:', err);
    }
  };

  useEffect(() => {
    let interval: number;
    if (isCameraActive && inputSource === 'camera') {
      interval = window.setInterval(handleLiveDetection, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isCameraActive, inputSource, selectedModel, threshold]);

  const renderDetections = () => {
    if (!result?.detections || result.detections.length === 0) {
      return <p className="text-muted-foreground">No detections</p>;
    }

    return (
      <div className="space-y-2">
        {result.detections.map((detection: Detection, idx: number) => (
          <div
            key={idx}
            className="flex items-center justify-between p-2 rounded bg-secondary"
          >
            <span className="font-medium">{detection.label}</span>
            <span className="text-sm text-muted-foreground">
              {(detection.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-2">
            <Eye className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold">Neural Vision Engine</h1>
          </div>
          <p className="text-muted-foreground mt-1">
            Universal Object Detection & Recognition Platform
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            {/* Task Type */}
            <div className="bg-card rounded-lg border p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Task Type
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'detection', icon: Eye, label: 'Detection' },
                  { id: 'gesture', icon: Camera, label: 'Gesture' },
                  { id: 'style_transfer', icon: ImageIcon, label: 'Style' },
                  { id: 'action', icon: Activity, label: 'Action' },
                ].map((task) => (
                  <button
                    key={task.id}
                    onClick={() => setTaskType(task.id as TaskType)}
                    className={`flex items-center gap-2 p-2 rounded-md transition-colors ${
                      taskType === task.id
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-secondary hover:bg-secondary/80'
                    }`}
                  >
                    <task.icon className="h-4 w-4" />
                    <span className="text-sm">{task.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Input Source */}
            <div className="bg-card rounded-lg border p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Upload className="h-4 w-4" />
                Input Source
              </h3>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'image', icon: ImageIcon, label: 'Image' },
                  { id: 'video', icon: Video, label: 'Video' },
                  { id: 'camera', icon: Camera, label: 'Camera' },
                ].map((source) => (
                  <button
                    key={source.id}
                    onClick={() => {
                      setInputSource(source.id as InputSource);
                      if (source.id === 'camera') {
                        startCamera();
                      } else {
                        stopCamera();
                      }
                    }}
                    className={`flex flex-col items-center gap-1 p-3 rounded-md transition-colors ${
                      inputSource === source.id
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-secondary hover:bg-secondary/80'
                    }`}
                  >
                    <source.icon className="h-5 w-5" />
                    <span className="text-sm">{source.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Selection */}
            <div className="bg-card rounded-lg border p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                Model
              </h3>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 rounded-md border bg-background"
              >
                {models.map((model) => (
                  <option key={model.id} value={model.model_name}>
                    {model.name}
                  </option>
                ))}
              </select>
              {models.find((m) => m.model_name === selectedModel) && (
                <p className="text-sm text-muted-foreground mt-2">
                  {models.find((m) => m.model_name === selectedModel)?.description}
                </p>
              )}
            </div>

            {/* Style Selection (for style transfer) */}
            {taskType === 'style_transfer' && (
              <div className="bg-card rounded-lg border p-4">
                <h3 className="font-semibold mb-3">Style</h3>
                <select
                  value={selectedStyle}
                  onChange={(e) => setSelectedStyle(e.target.value)}
                  className="w-full p-2 rounded-md border bg-background"
                >
                  {STYLE_OPTIONS.map((style) => (
                    <option key={style.id} value={style.id}>
                      {style.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Threshold */}
            <div className="bg-card rounded-lg border p-4">
              <h3 className="font-semibold mb-3">Confidence Threshold</h3>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-center mt-1">{(threshold * 100).toFixed(0)}%</div>
            </div>

            {/* Image Upload */}
            {inputSource === 'image' && (
              <div className="bg-card rounded-lg border p-4">
                <h3 className="font-semibold mb-3">Upload Image</h3>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="w-full p-2 rounded-md border bg-background"
                />
              </div>
            )}
          </div>

          {/* Right Panel - Display */}
          <div className="lg:col-span-2 space-y-6">
            {/* Image/Video Display */}
            <div className="bg-card rounded-lg border p-4">
              <div className="aspect-video relative bg-muted rounded-lg overflow-hidden flex items-center justify-center">
                {inputSource === 'camera' ? (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className={`w-full h-full object-contain ${isCameraActive ? 'block' : 'hidden'}`}
                    />
                    {!isCameraActive && (
                      <div className="text-center text-muted-foreground">
                        <Camera className="h-12 w-12 mx-auto mb-2" />
                        <p>Click Camera to start</p>
                      </div>
                    )}
                  </>
                ) : imagePreview ? (
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center text-muted-foreground">
                    <ImageIcon className="h-12 w-12 mx-auto mb-2" />
                    <p>Upload an image to start</p>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>

              {/* Process Button */}
              <div className="mt-4 flex gap-2">
                <button
                  onClick={handleProcess}
                  disabled={
                    isProcessing ||
                    (!imageFile && !imagePreview && inputSource !== 'camera')
                  }
                  className="flex-1 flex items-center justify-center gap-2 bg-primary text-primary-foreground p-3 rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Zap className="h-5 w-5" />
                      Process
                    </>
                  )}
                </button>
                {isCameraActive && (
                  <button
                    onClick={stopCamera}
                    className="px-4 py-3 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90"
                  >
                    Stop Camera
                  </button>
                )}
              </div>
            </div>

            {/* Results */}
            <div className="bg-card rounded-lg border p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Results</h3>
                {processingTime > 0 && (
                  <span className="text-sm text-muted-foreground">
                    Processing time: {processingTime.toFixed(2)}s
                  </span>
                )}
              </div>

              {result ? (
                <div className="space-y-4">
                  {result.success ? (
                    <>
                      {result.image && (
                        <div className="aspect-video relative bg-muted rounded-lg overflow-hidden">
                          <img
                            src={`data:image/jpeg;base64,${result.image}`}
                            alt="Result"
                            className="w-full h-full object-contain"
                          />
                        </div>
                      )}
                      <div className="border-t pt-4">
                        <h4 className="font-medium mb-2">Detections</h4>
                        {renderDetections()}
                      </div>
                    </>
                  ) : (
                    <div className="p-4 bg-destructive/10 text-destructive rounded-lg">
                      Error: {result.error}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  Run inference to see results
                </p>
              )}
            </div>

            {/* Performance Stats */}
            <div className="bg-card rounded-lg border p-4">
              <h3 className="font-semibold mb-3">Performance</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-secondary rounded-lg">
                  <div className="text-2xl font-bold text-primary">
                    {processingTime > 0 ? processingTime.toFixed(2) : '--'}
                  </div>
                  <div className="text-sm text-muted-foreground">Seconds</div>
                </div>
                <div className="text-center p-3 bg-secondary rounded-lg">
                  <div className="text-2xl font-bold text-primary">
                    {result?.detections?.length || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">Objects</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;