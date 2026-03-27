from django.apps import AppConfig


class VisionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'vision'

    def ready(self):
        """Initialize models on startup"""
        from . import models_handler
        models_handler.load_models()