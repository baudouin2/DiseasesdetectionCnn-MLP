import logging
from rest_framework.permissions import IsAdminUser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from app.models import Diagnostic, User, PendingReview
from app.serializers import DiagnosticSerializer, DiagnosticVerificationSerializer, UserSerializer, PendingReviewSerializer
from django.utils import timezone
from django.db import models
import requests, gzip, json
from datetime import datetime
from django.conf import settings
from pymongo import MongoClient
from app.models.diagnostic import (
    extract_anomaly_features, CLASS_NAMES,
    CFG_PATH, FEUILLE_WEIGHTS, ANOMALIE_WEIGHTS, MLP_MODEL_PATH,
    PREPROCESSOR_PATH, TARGET_COLS_PATH, STATE_ENCODER_PATH
)
import joblib
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

logger = logging.getLogger("django")

SOIL_TYPES = {
    # ...existing code...
}

def get_soil_type(city_name):
    return SOIL_TYPES.get(city_name, "Inconnu")

class AdminDashboardView(APIView):
    """
    Admin dashboard: returns unverified diagnostics (paginated).
    """
    permission_classes = [IsAdminUser]

    def get(self, request):
        logger.info(f"[AdminDashboardView] GET by admin: {getattr(request.user, 'id', None)}")
        paginator = PageNumberPagination()
        paginator.page_size = 20
        unverified = Diagnostic.objects.filter(is_verified=False)
        result_page = paginator.paginate_queryset(unverified, request)
        serializer = DiagnosticSerializer(result_page, many=True)
        logger.info(f"[AdminDashboardView] Unverified diagnostics count: {unverified.count()}")
        return paginator.get_paginated_response(serializer.data)

class VerifyDiagnosticView(APIView):
    """
    Allows admin to verify a diagnostic.
    """
    permission_classes = [IsAdminUser]

    def patch(self, request, pk):
        logger.info(f"[VerifyDiagnosticView] PATCH for diagnostic {pk} by admin {getattr(request.user, 'id', None)}")
        try:
            diagnostic = Diagnostic.objects.get(pk=str(pk))
        except Diagnostic.DoesNotExist:
            logger.warning(f"[VerifyDiagnosticView] Diagnostic {pk} not found.")
            return Response({'error': 'Diagnostic non trouvé'}, status=status.HTTP_404_NOT_FOUND)
        # Correction : n'utilise pas user=None (clé étrangère obligatoire avec Djongo)
        user_obj = request.user if request.user and request.user.is_authenticated else None
        if user_obj is None:
            logger.error("[VerifyDiagnosticView] Impossible de sauvegarder la vérification : utilisateur admin non authentifié.")
            return Response({'error': "Utilisateur admin non authentifié. Veuillez vous connecter."}, status=status.HTTP_401_UNAUTHORIZED)
        serializer = DiagnosticVerificationSerializer(diagnostic, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            logger.info(f"[VerifyDiagnosticView] Diagnostic {pk} verified.")
            return Response(serializer.data)
        logger.warning(f"[VerifyDiagnosticView] Validation errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AdminStatsView(APIView):
    """
    Returns statistics for the admin dashboard.
    """
    permission_classes = [IsAdminUser]

    def get(self, request):
        logger.info(f"[AdminStatsView] GET by admin: {getattr(request.user, 'id', None)}")
        total_diagnostics = Diagnostic.objects.count()
        pending_reviews = Diagnostic.objects.filter(is_verified=False).count()
        users_last_week = User.objects.filter(date_joined__gte=(timezone.now() - timezone.timedelta(days=7))).count()
        # Example disease distribution
        disease_distribution = (
            Diagnostic.objects.values('disease')
            .order_by('disease')
            .annotate(count=models.Count('disease'))
        )
        # Map disease code to display name
        disease_map = dict(Diagnostic.Diseases.choices)
        disease_distribution = [
            {'disease': disease_map.get(item['disease'], item['disease']), 'count': item['count']}
            for item in disease_distribution
        ]
        logger.info(f"[AdminStatsView] Stats: total={total_diagnostics}, pending={pending_reviews}, users_last_week={users_last_week}")
        return Response({
            'total_diagnostics': total_diagnostics,
            'pending_reviews': pending_reviews,
            'users_last_week': users_last_week,
            'disease_distribution': disease_distribution,
        })

class AdminUserListView(APIView):
    """
    List all users (admin only).
    """
    permission_classes = [IsAdminUser]
    def get(self, request):
        logger.info(f"[AdminUserListView] GET by admin: {getattr(request.user, 'id', None)}")
        # Vérification explicite du token JWT pour éviter 401
        if not request.user or not request.user.is_authenticated or not getattr(request.user, 'is_admin', False):
            logger.warning("[AdminUserListView] Unauthorized access.")
            return Response({'detail': 'Authentication credentials were not provided.'}, status=401)
        users = User.objects.all()
        logger.info(f"[AdminUserListView] Users count: {users.count()}")
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

class AdminUserOperationsView(APIView):
    """
    List all diagnostics/operations made by a specific user (admin only).
    """
    permission_classes = [IsAdminUser]
    def get(self, request, user_id):
        logger.info(f"[AdminUserOperationsView] GET for user {user_id} by admin {getattr(request.user, 'id', None)}")
        if not request.user or not request.user.is_authenticated or not getattr(request.user, 'is_admin', False):
            logger.warning("[AdminUserOperationsView] Unauthorized access.")
            return Response({'detail': 'Authentication credentials were not provided.'}, status=401)
        diagnostics = Diagnostic.objects.filter(user_id=user_id).order_by('-created_at')
        logger.info(f"[AdminUserOperationsView] Diagnostics count for user {user_id}: {diagnostics.count()}")
        serializer = DiagnosticSerializer(diagnostics, many=True)
        return Response(serializer.data)

class PendingReviewsListView(APIView):
    """
    List all pending reviews for admin moderation.
    """
    permission_classes = [IsAdminUser]

    def get(self, request):
        logger.info(f"[PendingReviewsListView] GET by admin: {getattr(request.user, 'id', None)}")
        pending = PendingReview.objects.filter(status='pending').select_related('diagnostic', 'reviewed_by')
        logger.info(f"[PendingReviewsListView] Pending reviews count: {pending.count()}")
        serializer = PendingReviewSerializer(pending, many=True)
        return Response(serializer.data)

class PendingReviewUpdateView(APIView):
    """
    Approve or reject a pending review.
    """
    permission_classes = [IsAdminUser]

    def patch(self, request, pk):
        logger.info(f"[PendingReviewUpdateView] PATCH for review {pk} by admin {getattr(request.user, 'id', None)}")
        try:
            review = PendingReview.objects.get(pk=str(pk))
        except PendingReview.DoesNotExist:
            logger.warning(f"[PendingReviewUpdateView] Review {pk} not found.")
            return Response({'error': 'Revue non trouvée'}, status=status.HTTP_404_NOT_FOUND)
        # Correction : n'utilise pas user=None (clé étrangère obligatoire avec Djongo)
        user_obj = request.user if request.user and request.user.is_authenticated else None
        if user_obj is None:
            logger.error("[PendingReviewUpdateView] Impossible de sauvegarder la revue : utilisateur admin non authentifié.")
            return Response({'error': "Utilisateur admin non authentifié. Veuillez vous connecter."}, status=status.HTTP_401_UNAUTHORIZED)
        status_action = request.data.get('is_approved')
        notes = request.data.get('admin_notes', '')
        if status_action is True:
            review.status = 'approved'
            review.diagnostic.is_verified = True
            review.diagnostic.admin_comment = notes
            review.diagnostic.status = 'confirmed'
            review.diagnostic.save()
        else:
            review.status = 'rejected'
            review.diagnostic.admin_comment = notes
            review.diagnostic.save()
        review.save()
        return Response(PendingReviewSerializer(review).data)

class CollectMeteoDataView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        API_KEY = "e2cca7444aff3584afee6eae3c1c96af"
        mongo_uri = f"mongodb://{settings.MONGO_USER}:{settings.MONGO_PASSWORD}@{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
        client = MongoClient(mongo_uri)
        db = client[settings.MONGO_DATABASE]
        collection = db["donnees_meteo"]

        try:
            url = "http://bulk.openweathermap.org/sample/city.list.json.gz"
            resp = requests.get(url, stream=True)
            if resp.status_code != 200:
                return Response({"error": "Erreur lors de la récupération des villes."}, status=400)
            with gzip.GzipFile(fileobj=resp.raw) as gz:
                city_data = json.load(gz)
            cities = [city for city in city_data if city.get("country") == "CM"]
            count = 0
            for city in cities:
                city_id = city["id"]
                city_name = city["name"]
                url = f"http://api.openweathermap.org/data/2.5/forecast?id={city_id}&appid={API_KEY}&units=metric"
                r = requests.get(url)
                if r.status_code != 200:
                    continue
                data = r.json()
                for entry in data["list"]:
                    forecast_info = {
                        "city": city_name,
                        "id": city_id,
                        "datetime": datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S"),
                        "temperature": entry["main"]["temp"],
                        "pressure": entry["main"]["pressure"],
                        "humidity": entry["main"]["humidity"],
                        "wind_speed": entry["wind"]["speed"],
                        "precipitation": entry.get("rain", {}).get("3h", 0),
                        "soil_type": get_soil_type(city_name),
                    }
                    existing = collection.find_one({
                        "city": forecast_info["city"],
                        "datetime": forecast_info["datetime"]
                    })
                    if not existing:
                        collection.insert_one(forecast_info)
                        count += 1
            client.close()
            return Response({"success": True, "inserted": count})
        except Exception as e:
            return Response({"error": str(e)}, status=500)

def analyse_image_admin(image_path, meteo_stats, agro_data, image_name, logger=logger):
    # --- Pipeline identique à farmer.py ---
    predictor_feuille = load_predictor(CFG_PATH, FEUILLE_WEIGHTS, num_classes=1)
    predictor_anomalie = load_predictor(CFG_PATH, ANOMALIE_WEIGHTS, num_classes=4)
    mlp_model = tf.keras.models.load_model(MLP_MODEL_PATH)
    loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
    loaded_target_maladie_cols = joblib.load(TARGET_COLS_PATH)
    loaded_le_state = joblib.load(STATE_ENCODER_PATH)
    logger.info(f"[AdminAnalyse] ✅ Modèles et objets chargés avec succès.")
    logger.info(f"[AdminAnalyse] ➡️ Colonnes d'entrée attendues : {getattr(loaded_preprocessor, 'feature_names_in_', [])}")
    logger.info(f"[AdminAnalyse] ➡️ États possibles : {getattr(loaded_le_state, 'classes_', [])}")
    logger.info(f"[AdminAnalyse] ➡️ Colonnes de sortie (maladies) : {loaded_target_maladie_cols}")

    image_cv = cv2.imread(image_path)
    if image_cv is None:
        logger.error("[AdminAnalyse] ❌ Image introuvable.")
        raise Exception("Image introuvable")

    # 1. Détection des feuilles
    feuille_outputs = predictor_feuille(image_cv)
    feuilles = feuille_outputs["instances"]
    if not hasattr(feuilles, "pred_masks") or feuilles.pred_masks.shape[0] == 0:
        logger.warning("[AdminAnalyse] ⚠️ Aucune feuille détectée.")
        return [], "Inconnu", None, None

    mask_union = feuilles.pred_masks.sum(dim=0).clamp(max=1).cpu().numpy().astype(np.uint8)
    segmented_image = image_cv.copy()
    segmented_image[mask_union == 0] = 0
    surface_feuilles = int(mask_union.sum())

    # Sauvegarde l'image segmentée (feuilles)
    _, buffer_seg = cv2.imencode('.jpg', segmented_image)
    img_bytes_seg = buffer_seg.tobytes()
    img_name_seg = f"diagnostics/segmented_{os.path.basename(image_name)}"
    img_file_seg = ContentFile(img_bytes_seg, name=img_name_seg)
    segmented_path = default_storage.save(img_name_seg, img_file_seg)
    segmented_url = default_storage.url(segmented_path)
    logger.info(f"[AdminAnalyse] Image segmentée (feuilles) sauvegardée à : {segmented_url}")

    # 2. Détection des anomalies
    anomalie_outputs = predictor_anomalie(segmented_image)
    instances = anomalie_outputs["instances"]
    logger.info(f"[AdminAnalyse] Instances détectées : classes={getattr(instances, 'pred_classes', None)}, "
                f"boxes={getattr(getattr(instances, 'pred_boxes', None), 'tensor', None)}, "
                f"scores={getattr(instances, 'scores', None)}, "
                f"nb_masks={getattr(instances, 'pred_masks', np.array([])).shape[0]}")
    if not hasattr(instances, "pred_masks") or instances.pred_masks.shape[0] == 0:
        logger.warning("[AdminAnalyse] ⚠️ Aucune anomalie détectée.")

    features = extract_anomaly_features(instances, segmented_image, surface_feuilles)

    # 3. Génération de l'image annotée (anomalies sur feuilles)
    annotated_img = segmented_image.copy()
    anomalies_detected = False
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    pred_classes = getattr(instances, "pred_classes", np.zeros(0, dtype=int))
    masks = getattr(instances, "pred_masks", np.zeros((0, *segmented_image.shape[:2])))
    boxes = getattr(getattr(instances, "pred_boxes", None), "tensor", np.zeros((0, 4))).astype(int)
    if len(pred_classes) > 0:
        anomalies_detected = True
    for i, label_idx in enumerate(pred_classes):
        mask = masks[i].astype(np.uint8)
        color = palette[label_idx % len(palette)]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_img, contours, -1, color, 2)
        if boxes.shape[0] > i:
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
    if not anomalies_detected:
        annotated_img = segmented_image

    # 4. Prédiction des maladies (si des anomalies sont détectées)
    predicted_diseases = []
    if anomalies_detected:
        logger.info(f"[AdminAnalyse] ➡️ Prédiction des maladies avec le modèle MLP.")
        # Préparation des données pour le modèle MLP
        X_mlp = pd.DataFrame(columns=loaded_target_maladie_cols)
        for feature_set in features:
            feature_data = {col: 0 for col in loaded_target_maladie_cols}
            for key, value in feature_set.items():
                if key in feature_data:
                    feature_data[key] = value
            X_mlp = X_mlp.append(feature_data, ignore_index=True)
        logger.info(f"[AdminAnalyse] Données préparées pour le MLP : {X_mlp.head()}")
        mlp_predictions = mlp_model.predict(X_mlp)
        logger.info(f"[AdminAnalyse] Prédictions brutes du MLP : {mlp_predictions}")
        for pred in mlp_predictions:
            top_indices = pred.argsort()[-3:][::-1]  # 3 maladies les plus probables
            top_probs = pred[top_indices]
            predicted_diseases.append({
                "maladie_1": loaded_target_maladie_cols[top_indices[0]],
                "confiance_1": float(top_probs[0]),
                "maladie_2": loaded_target_maladie_cols[top_indices[1]],
                "confiance_2": float(top_probs[1]),
                "maladie_3": loaded_target_maladie_cols[top_indices[2]],
                "confiance_3": float(top_probs[2]),
            })
    else:
        logger.warning("[AdminAnalyse] Aucune anomalie détectée, pas de prédiction de maladie effectuée.")

    # Sauvegarde de l'image annotée
    _, buffer_annot = cv2.imencode('.jpg', annotated_img)
    img_bytes_annot = buffer_annot.tobytes()
    img_name_annot = f"diagnostics/annotated_{os.path.basename(image_name)}"
    img_file_annot = ContentFile(img_bytes_annot, name=img_name_annot)
    annotated_path = default_storage.save(img_name_annot, img_file_annot)
    annotated_url = default_storage.url(annotated_path)
    logger.info(f"[AdminAnalyse] Image annotée sauvegardée à : {annotated_url}")

    return predicted_diseases, segmented_url, annotated_url, features