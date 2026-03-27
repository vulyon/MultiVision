"""
Main URL Configuration for MultiVision
"""
from django.contrib import admin
from django.urls import path, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from vision import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # API routes
    path('api/models', views.api_models, name='api_models'),
    path('api/infer', views.api_infer, name='api_infer'),
    path('api/camera', views.api_camera, name='api_camera'),
    path('health', views.health, name='health'),
    # Frontend catch-all (SPA routing) - must be last
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)