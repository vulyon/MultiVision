"""
WSGI config for MultiVision project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'multivision.settings')
application = get_wsgi_application()