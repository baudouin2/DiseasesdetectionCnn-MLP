from django.urls import re_path
from .consumers import DiagnosticConsumer

websocket_urlpatterns = [
    re_path(r'ws/diagnostics/$', DiagnosticConsumer.as_asgi()),
]