from django.urls import path
from app.consumers import DiagnosticConsumer

websocket_urlpatterns = [
    path('ws/diagnostics/', DiagnosticConsumer.as_asgi()),
]