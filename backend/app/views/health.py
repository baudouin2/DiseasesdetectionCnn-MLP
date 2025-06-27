from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

@api_view(['GET'])
@permission_classes([AllowAny])  # <-- Rendez ce endpoint public
def health_check(request):
    return Response({"status": "ok", "message": "API is healthy"})
