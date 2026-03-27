"""
Vision Views
Django views for API and page rendering
"""
import json
import base64
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from .models_handler import process_image, list_models, get_model


def index(request):
    """Main page"""
    return render(request, 'vision/index.html')


@require_http_methods(["GET"])
def api_models(request):
    """List available models"""
    models = list_models()
    return JsonResponse({"models": models})


@csrf_exempt
@require_http_methods(["POST"])
def api_infer(request):
    """Process image inference"""
    try:
        # Get model name
        model_name = request.POST.get('model', 'detector')

        # Get threshold
        threshold = float(request.POST.get('threshold', 0.5))

        # Get style for style_transfer
        style = request.POST.get('style', 'sketch')

        # Handle uploaded file
        if 'file' in request.FILES:
            image_data = request.FILES['file'].read()
        elif 'image' in request.POST:
            # Base64 encoded image
            img_str = request.POST['image']
            image_data = base64.b64decode(img_str)
        else:
            return JsonResponse({"error": "No image provided", "success": False}, status=400)

        # Process
        result = process_image(image_data, model_name, threshold=threshold, style=style)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e), "success": False}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_camera(request):
    """Process camera frame via WebSocket/JSON"""
    try:
        data = json.loads(request.body)
        model_name = data.get('model', 'detector')
        frame_data = data.get('frame', '')

        # Decode base64 frame
        image_data = base64.b64decode(frame_data)

        # Get threshold
        threshold = float(data.get('threshold', 0.5))

        # Process
        result = process_image(image_data, model_name, threshold=threshold)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e), "success": False}, status=500)


def health(request):
    """Health check"""
    return JsonResponse({"status": "ok", "models": list_models()})