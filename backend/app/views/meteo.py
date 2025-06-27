from django.db.models import Avg, Min, Max
from rest_framework import permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from app.models.meteo import MeteoData
from django.db import models
import logging
from rest_framework.permissions import AllowAny
from datetime import datetime, timedelta

logger = logging.getLogger("django")

class MeteoStatsView(APIView):
    """
    Retourne les statistiques météo pour une ville donnée (3 derniers mois).
    """
    permission_classes = [AllowAny]

    def get(self, request):
        city = request.GET.get('city')
        if not city:
            return Response({'error': 'Paramètre city requis.'}, status=400)
        try:
            from django.utils import timezone
            now = timezone.now()
            start_date = (now.replace(day=1) - timezone.timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0)
            qs = MeteoData.objects.filter(city=city, datetime__gte=start_date, datetime__lte=now)
            if not qs.exists():
                return Response({
                    "stats": {
                        "temperature": {"avg": None},
                        "humidity": {"avg": None},
                        "pressure": {"avg": None},
                        "wind_speed": {"avg": None},
                        "precipitation": {"avg": None},
                    },
                    "months": [],
                    "offline_mode": False,
                    "error": f"Aucune donnée météo trouvée pour {city}"
                }, status=200)
            # Agrégation compatible frontend (clé .avg)
            stats = {
                "temperature": {"avg": qs.aggregate(models.Avg('temperature'))['temperature__avg']},
                "humidity": {"avg": qs.aggregate(models.Avg('humidity'))['humidity__avg']},
                "pressure": {"avg": qs.aggregate(models.Avg('pressure'))['pressure__avg']},
                "wind_speed": {"avg": qs.aggregate(models.Avg('wind_speed'))['wind_speed__avg']},
                "precipitation": {"avg": qs.aggregate(models.Avg('precipitation'))['precipitation__avg']},
            }
            # Groupby manuel par mois
            months = {}
            for doc in qs:
                month = doc.datetime.strftime('%Y-%m')
                if month not in months:
                    months[month] = {
                        'month': month,
                        'temperature_sum': 0,
                        'humidity_sum': 0,
                        'pressure_sum': 0,
                        'wind_speed_sum': 0,
                        'precipitation_sum': 0,
                        'count': 0
                    }
                months[month]['temperature_sum'] += doc.temperature or 0
                months[month]['humidity_sum'] += doc.humidity or 0
                months[month]['pressure_sum'] += doc.pressure or 0
                months[month]['wind_speed_sum'] += doc.wind_speed or 0
                months[month]['precipitation_sum'] += doc.precipitation or 0
                months[month]['count'] += 1
            # Calcule les moyennes par mois
            months_list = []
            for m in sorted(months.values(), key=lambda x: x['month']):
                count = m['count']
                months_list.append({
                    'month': m['month'],
                    'temperature_avg': round(m['temperature_sum'] / count, 2) if count else None,
                    'humidity_avg': round(m['humidity_sum'] / count, 2) if count else None,
                    'pressure_avg': round(m['pressure_sum'] / count, 2) if count else None,
                    'wind_speed_avg': round(m['wind_speed_sum'] / count, 2) if count else None,
                    'precipitation_avg': round(m['precipitation_sum'] / count, 2) if count else None,
                })
            return Response({
                "stats": stats,
                "months": months_list,
                "offline_mode": False
            })
        except Exception as e:
            return Response({
                "stats": {
                    "temperature": {"avg": None},
                    "humidity": {"avg": None},
                    "pressure": {"avg": None},
                    "wind_speed": {"avg": None},
                    "precipitation": {"avg": None},
                },
                "months": [],
                "offline_mode": False,
                "error": f"Erreur serveur: {str(e)}"
            }, status=200)

class MeteoForecastView(APIView):
    permission_classes = [permissions.AllowAny]
    def get(self, request):
        city = request.GET.get('city')
        logger.info(f"[MeteoForecastView] GET for city: {city} by user: {getattr(request.user, 'id', None)}")
        if not city:
            logger.warning("[MeteoForecastView] No city provided.")
            return Response({'error': 'Localité requise'}, status=400)
        try:
            today = datetime.utcnow()
            future = today + timedelta(days=7)
            qs = MeteoData.objects.filter(city=city, datetime__gte=today, datetime__lte=future).order_by('datetime')
            logger.info(f"[MeteoForecastView] Forecast count: {qs.count()}")
            forecast = list(qs.values('datetime', 'temperature', 'pressure', 'humidity', 'wind_speed', 'precipitation', 'soil_type'))
            return Response({'forecast': forecast})
        except Exception as e:
            logger.exception(f"[MeteoForecastView] Error: {e}")
            return Response({'error': 'Erreur lors de la récupération des prévisions météo.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LocaliteListView(APIView):
    permission_classes = [permissions.AllowAny]
    def get(self, request):
        try:
            from django.conf import settings
            import pymongo

            # Récupère les variables d'environnement, gère le cas où user/password sont vides
            mongo_user = getattr(settings, "MONGO_USER", "") or None
            mongo_password = getattr(settings, "MONGO_PASSWORD", "") or None
            mongo_host = getattr(settings, "MONGO_HOST", "localhost")
            mongo_port = getattr(settings, "MONGO_PORT", 27017)
            mongo_db = getattr(settings, "MONGO_DATABASE", "tomatodb")

            if mongo_user and mongo_password:
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"
            else:
                mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db}"

            client = pymongo.MongoClient(mongo_uri)
            db = client[mongo_db]
            collection = db['donnees_meteo']
            localites = collection.distinct('city')
            localites = [str(loc) for loc in localites if loc]
            client.close()
            logger.info(f"[LocaliteListView] Found {len(localites)} localités.")
            return Response(localites)
        except Exception as e:
            logger.exception(f"[LocaliteListView] Error: {e}")
            return Response({'error': 'Erreur lors de la récupération des localités.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
