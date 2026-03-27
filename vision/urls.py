"""
Vision app URL routing
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/models', views.api_models, name='api_models'),
    path('api/infer', views.api_infer, name='api_infer'),
    path('api/camera', views.api_camera, name='api_camera'),
    path('health', views.health, name='health'),
]