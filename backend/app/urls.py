from django.urls import path
from .views import farmer, admin
from .views import home
from .views import meteo
from .views import health  # Ajoutez cette ligne
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views.farmer import SimpleLoginView
from .views.admin import CollectMeteoDataView
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    # Home & Auth
    path('', home, name='home'),
    path('api/auth/login/', SimpleLoginView.as_view(), name='login'),
    path('api/auth/register/', farmer.RegisterView.as_view(), name='register'),

    # Diagnostic
    path('api/diagnose/', farmer.DiagnosisView.as_view(), name='diagnose'),
    path('api/diagnostics/upload/', farmer.DiagnosisView.as_view(), name='diagnostics-upload'),
    path('api/diagnostics/upload', farmer.DiagnosisView.as_view(), name='diagnostics-upload-no-slash'),
    path('api/diagnostics/history', farmer.FarmerHistoryView.as_view(), name='diagnostics-history'),

    # Admin - Dashboard & Users
    path('api/admin/dashboard/', admin.AdminDashboardView.as_view(), name='admin-dashboard'),
    path('api/admin/stats/', admin.AdminStatsView.as_view(), name='admin-stats'),
    path('api/admin/users/', admin.AdminUserListView.as_view(), name='admin-users'),
    path('api/admin/user/<str:user_id>/operations/', admin.AdminUserOperationsView.as_view(), name='admin-user-operations'),

    # Admin - Reviews & Validation
    path('api/admin/reviews/', admin.PendingReviewsListView.as_view(), name='admin-reviews'),
    path('api/admin/reviews/<str:pk>', admin.PendingReviewUpdateView.as_view(), name='admin-review-update'),
    path('api/admin/verify/<str:pk>/', admin.VerifyDiagnosticView.as_view(), name='verify-diagnostic'),

    # Admin - Collecte météo (sécurisé, admin only)
    # path('api/admin/collect-meteo/', CollectMeteoDataView.as_view(), name='admin-collect-meteo'),  # SUPPRIMÉ

    # Météo (public)
    path('api/meteo/localites/', meteo.LocaliteListView.as_view(), name='meteo-localites'),
    path('api/meteo/stats/', meteo.MeteoStatsView.as_view(), name='meteo-stats'),
    path('api/meteo/forecast/', meteo.MeteoForecastView.as_view(), name='meteo-forecast'),

    # Documentation automatique (Swagger/OpenAPI)
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),

    # Ajoutez la route santé backend :
    path('api/health/', health.health_check, name='health-check'),
]

# Sécurité et gestion des rôles :
# - Les endpoints /admin/* sont protégés par IsAdminUser dans les vues.
# - Les endpoints /diagnostics/* sont protégés par authentification ou permissions custom.
# - Les endpoints météo sont publics (peu sensibles).
# - Documentation accessible sur /api/docs/ (Swagger UI).

# ...JWT routes désactivées...