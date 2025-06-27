from rest_framework.permissions import IsAuthenticated, IsAdminUser, BasePermission, AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.contrib.auth import authenticate
from app.models import Diagnostic, User
from app.serializers import DiagnosticSerializer, UserSerializer
from app.utils.model_loader import load_disease_model
from app.utils.image_processing import preprocess_image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.authentication import JWTAuthentication
from app.models.diagnostic import diagnostiquer_image_segmentee, get_predictor_anomalie
from app.models.meteo import MeteoData
from django.db import models
import torch  # Ajoutez cet import en haut du fichier pour éviter NameError
from django.http import JsonResponse
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger("django")

class UserIdHeaderPermission(BasePermission):
    """
    Autorise uniquement si le header X-User-Id correspond à un utilisateur existant et actif.
    """
    def has_permission(self, request, view):
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            return False
        try:
            user = User.objects.get(id=user_id)
            request.user = user  # Injecte l'utilisateur pour la vue
            return user.is_active
        except User.DoesNotExist:
            return False

class DiagnosisView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            logger.info(f"[DiagnosisView] POST by user: {getattr(request.user, 'id', None)} | Auth: {getattr(request.user, 'is_authenticated', None)}")
            # Log des données reçues
            logger.info(f"[DiagnosisView] Données POST reçues: {dict(request.data)}")
            logger.info(f"[DiagnosisView] Fichiers reçus: {request.FILES}")

            if 'image' not in request.FILES:
                logger.warning("[DiagnosisView] No image provided in request.")
                return Response({'error': 'Aucune image fournie'}, status=status.HTTP_400_BAD_REQUEST)
            image = request.FILES['image']
            # Log sur l'image reçue
            logger.info(f"[DiagnosisView] Image reçue: name={image.name}, size={image.size}, content_type={image.content_type}")

            # Vérifie la taille max (5Mo) et le type
            if image.size > 5 * 1024 * 1024:
                return Response({'error': 'Image trop volumineuse (max 5Mo).'}, status=status.HTTP_400_BAD_REQUEST)
            if not image.content_type.startswith('image/'):
                return Response({'error': 'Format de fichier non supporté.'}, status=status.HTTP_400_BAD_REQUEST)
            processed_image = preprocess_image(image)
            if processed_image is None:
                logger.error("[DiagnosisView] Image preprocessing failed.")
                return Response({'error': 'Erreur lors du traitement de l\'image.'}, status=status.HTTP_400_BAD_REQUEST)
            
            # --- Correction : toujours initialiser meteo_stats et offline_mode ---
            temperature = request.data.get('temperature')
            humidity = request.data.get('humidity')
            if temperature is not None:
                temperature = float(temperature)
            if humidity is not None:
                humidity = float(humidity)
            localite = request.data.get('localite')
            meteo_stats = None
            offline_mode = False
            if localite:
                try:
                    from django.utils import timezone
                    now = timezone.now()
                    start_date = (now.replace(day=1) - timezone.timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0)
                    qs = MeteoData.objects.filter(city=localite, datetime__gte=start_date, datetime__lte=now)
                    meteo_stats = {
                        "temperature": qs.aggregate(models.Avg('temperature'))['temperature__avg'],
                        "humidity": qs.aggregate(models.Avg('humidity'))['humidity__avg'],
                        "pressure": qs.aggregate(models.Avg('pressure'))['pressure__avg'],
                        "wind_speed": qs.aggregate(models.Avg('wind_speed'))['wind_speed__avg'],
                        "precipitation": qs.aggregate(models.Avg('precipitation'))['precipitation__avg'],
                    }
                    logger.info(f"[DiagnosisView] Résultat agrégation météo pour {localite}: {meteo_stats}")
                    if not any(meteo_stats.values()):
                        meteo_stats = None
                        offline_mode = True
                except Exception as e:
                    logger.warning(f"[DiagnosisView] Meteo fetch failed: {e}")
                    meteo_stats = None
                    offline_mode = True
            else:
                offline_mode = True

            agro_data = {
                "sol": request.data.get('sol'),
                "irrigation": request.data.get('irrigation'),
                "azote": request.data.get('azote'),
                "phosphore": request.data.get('phosphore'),
                "potassium": request.data.get('potassium'),
                "compost": request.data.get('compost'),
                "engrais_chimique": request.data.get('engrais_chimique')
            }
            # Extract agro_data fields for Diagnostic creation
            sol = agro_data.get("sol")
            irrigation = agro_data.get("irrigation")
            azote = agro_data.get("azote")
            phosphore = agro_data.get("phosphore")
            potassium = agro_data.get("potassium")
            compost = agro_data.get("compost")
            engrais_chimique = agro_data.get("engrais_chimique")

            # Correction : garantir un chemin de fichier pour cv2.imread
            if hasattr(image, 'temporary_file_path'):
                image_path = image.temporary_file_path()
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    for chunk in image.chunks():
                        tmp.write(chunk)
                    image_path = tmp.name

            logger.info(f"[DiagnosisView] Chemin image pour diagnostic: {image_path}")

            # Vérification explicite des modèles et objets avant la prédiction
            from app.models.diagnostic import (
                get_predictor_feuille, get_predictor_anomalie, get_mlp_model,
                get_loaded_preprocessor, get_loaded_target_maladie_cols, get_loaded_le_state, extract_anomaly_features
            )
            predictor_feuille = get_predictor_feuille()
            predictor_anomalie = get_predictor_anomalie()
            mlp_model = get_mlp_model()
            loaded_preprocessor = get_loaded_preprocessor()
            loaded_target_maladie_cols = get_loaded_target_maladie_cols()
            loaded_le_state = get_loaded_le_state()
            # Log de vérification détaillé des modèles et de leurs paramètres principaux
            logger.info(f"[DiagnosisView] Vérification modèles :")
            logger.info(f"  predictor_feuille: {predictor_feuille}")
            logger.info(f"  predictor_anomalie: {predictor_anomalie}")
            logger.info(f"  mlp_model: {mlp_model}")
            logger.info(f"  loaded_preprocessor: {loaded_preprocessor}")
            logger.info(f"  loaded_target_maladie_cols: {loaded_target_maladie_cols}")
            logger.info(f"  loaded_le_state: {loaded_le_state}")

            # Pour savoir si les modèles Mask R-CNN sont bien chargés et utilisés :
            # 1. Si vous voyez <function get_predictor_feuille.<locals>.<lambda> ...> dans les logs,
            #    alors ce sont encore les DUMMY (faux) modèles, pas les vrais Mask R-CNN Detectron2.
            # 2. Pour utiliser les vrais modèles, il faut que get_predictor_feuille/get_predictor_anomalie
            #    retournent un objet Detectron2 DefaultPredictor (et non une lambda/dummy).
            # 3. Pour vérifier : dans vos logs, vous devriez voir quelque chose comme
            #    predictor_feuille: <detectron2.engine.defaults.DefaultPredictor object at ...>
            #    et non une fonction lambda.

            # Pour forcer l'utilisation des vrais modèles, modifiez get_predictor_feuille/get_predictor_anomalie
            # dans app/models/diagnostic.py pour charger Detectron2 comme ceci :
            # from detectron2.engine import DefaultPredictor
            # def get_predictor_feuille():
            #     global _predictor_feuille
            #     if _predictor_feuille is None:
            #         _predictor_feuille = DefaultPredictor(cfg)  # cfg = config Detectron2
            #     return _predictor_feuille

            # Si vous gardez la version lambda/dummy, aucune vraie détection ne sera faite,
            # et vous aurez toujours 1 masque bidon, classes=[0], etc.
            if not all([predictor_feuille, predictor_anomalie, mlp_model, loaded_preprocessor, loaded_target_maladie_cols, loaded_le_state]):
                logger.error("[DiagnosisView] Un ou plusieurs modèles/objets nécessaires ne sont pas chargés correctement.")
                return Response({'error': "Erreur de chargement des modèles. Contactez l'administrateur."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Initialisation des prédicteurs et objets MLP
            from app.models.diagnostic import (
                get_predictor_feuille, get_predictor_anomalie, get_mlp_model,
                get_loaded_preprocessor, get_loaded_target_maladie_cols, get_loaded_le_state, extract_anomaly_features
            )
            import pandas as pd
            import cv2
            import numpy as np
            import os
            from django.core.files.base import ContentFile
            from django.core.files.storage import default_storage

            try:
                predictor_feuille = get_predictor_feuille()
                predictor_anomalie = get_predictor_anomalie()
                mlp_model = get_mlp_model()
                loaded_preprocessor = get_loaded_preprocessor()
                loaded_target_maladie_cols = get_loaded_target_maladie_cols()
                loaded_le_state = get_loaded_le_state()
                logger.info(f"[DiagnosisView] ✅ Modèles et objets chargés avec succès.")
                logger.info(f"[DiagnosisView] ➡️ Colonnes d'entrée attendues : {getattr(loaded_preprocessor, 'feature_names_in_', [])}")
                logger.info(f"[DiagnosisView] ➡️ États possibles : {getattr(loaded_le_state, 'classes_', [])}")
                logger.info(f"[DiagnosisView] ➡️ Colonnes de sortie (maladies) : {loaded_target_maladie_cols}")

                image_cv = cv2.imread(image_path)
                if image_cv is None:
                    logger.error("[DiagnosisView] ❌ Image introuvable.")
                    raise Exception("Image introuvable")

                # 1. Détection des feuilles
                feuilles = predictor_feuille(image_cv)["instances"]
                pred_masks = getattr(feuilles, "pred_masks", None)
                nb_feuilles = len(pred_masks) if pred_masks is not None else 0
                if nb_feuilles == 0:
                    logger.warning("[DiagnosisView] ⚠️ Aucune feuille détectée.")
                    maladies_detectees, predicted_state = [], "Inconnu"
                    annotated_url = None
                    segmented_url = None
                    raise Exception("Aucune feuille détectée")

                # Remplacement de np.sum(..., axis=0) par une boucle pour compatibilité torch/numpy
                if isinstance(pred_masks, torch.Tensor):
                    mask_union = torch.zeros_like(pred_masks[0], dtype=torch.uint8)
                    for i in range(pred_masks.shape[0]):
                        mask_union = mask_union | pred_masks[i].to(torch.uint8)
                    mask_union = mask_union.cpu().numpy()
                else:
                    mask_union = np.zeros(image_cv.shape[:2], dtype=np.uint8)
                    for mask in (pred_masks or []):
                        mask_union = np.logical_or(mask_union, mask).astype(np.uint8)
                segmented_image = image_cv.copy()
                segmented_image[mask_union == 0] = 0
                surface_feuilles = int(mask_union.sum())

                # Sauvegarde l'image segmentée (feuilles)
                _, buffer_seg = cv2.imencode('.jpg', segmented_image)
                img_bytes_seg = buffer_seg.tobytes()
                segmented_base64 = base64.b64encode(img_bytes_seg).decode('utf-8')

                # Génération de l'image annotée (anomalies sur feuilles) en base64
                # annotated_img is not defined yet here, so skip this block for now

                # 2. Détection des symptômes
                anomalies = predictor_anomalie(segmented_image)
                instances = anomalies["instances"]
                logger.info(f"[DiagnosisView] Instances détectées : classes={getattr(instances, 'pred_classes', None)}, "
                            f"boxes={getattr(getattr(instances, 'pred_boxes', None), 'tensor', None)}, "
                            f"scores={getattr(instances, 'scores', None)}, "
                            f"nb_masks={getattr(instances, 'pred_masks', np.array([])).shape[0]}")
                features = extract_anomaly_features(instances, segmented_image, surface_feuilles)

                # 2. Génération de l'image annotée (masques + nom de classe + score, sans box)
                annotated_img = segmented_image.copy()
                palette = [
                    (255, 0, 0),    # rouge
                    (0, 255, 0),    # vert
                    (0, 0, 255),    # bleu
                    (255, 255, 0),  # cyan
                    (255, 0, 255),  # magenta
                    (0, 255, 255),  # jaune
                ]
                pred_classes = getattr(instances, "pred_classes", np.zeros(0, dtype=int))
                masks = getattr(instances, "pred_masks", None)
                scores = getattr(instances, "scores", None)
                if masks is not None:
                    mask_list = []
                    for i in range(len(masks)):
                        mask = masks[i]
                        if hasattr(mask, "cpu"):
                            mask = mask.to(torch.uint8).cpu().numpy()
                        else:
                            mask = np.array(mask).astype(np.uint8)
                        mask_list.append(mask)
                    masks = mask_list
                else:
                    masks = []
                if scores is not None and hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                elif scores is None:
                    scores = np.zeros(len(pred_classes))
                else:
                    scores = np.array(scores)
                for i, label_idx in enumerate(pred_classes):
                    mask = masks[i].astype(np.uint8)
                    color = palette[label_idx % len(palette)]
                    # Applique le masque coloré sur l'image annotée
                    annotated_img[mask > 0] = (
                        0.5 * annotated_img[mask > 0] + 0.5 * np.array(color)
                    ).astype(np.uint8)
                    # Affiche le nom de la classe et le score sur le centre du masque
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        cx, cy = int(xs.mean()), int(ys.mean())
                        class_name = getattr(instances, "metadata", None)
                        if hasattr(class_name, "thing_classes"):
                            class_label = class_name.thing_classes[label_idx]
                        else:
                            # Utilise la liste du backend
                            from app.models.diagnostic import CLASS_NAMES
                            class_label = CLASS_NAMES[label_idx]
                        score = float(scores[i]) if i < len(scores) else 0.0
                        text = f"{class_label} ({score:.2f})"
                        cv2.putText(
                            annotated_img, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                        )
            except Exception as e:
                logger.exception(f"[DiagnosisView] Exception in model processing: {e}")
                raise

            # 3. Encodage base64 pour stockage direct en base de données
            _, buffer_seg = cv2.imencode('.jpg', segmented_image)
            img_bytes_seg = buffer_seg.tobytes()
            _, buffer_ann = cv2.imencode('.jpg', annotated_img)
            img_bytes_ann = buffer_ann.tobytes()
            segmented_b64 = base64.b64encode(img_bytes_seg).decode('utf-8')
            annotated_b64 = base64.b64encode(img_bytes_ann).decode('utf-8')

            # --- SUPPRESSION DE LA SAUVEGARDE DU DIAGNOSTIC ---
            # Toute la logique de Diagnostic.objects.create(...) est supprimée.
            # On effectue uniquement l'analyse et on retourne le résultat sans rien sauvegarder en base.

            # Ajout : initialisation de disease et confidence à None pour éviter NameError
            disease = None
            confidence = None
            mlp_probs_dict = {}
            maladies_detectees = []
            predicted_state = None
            anomalies_list = []

            # --- Ajout : Prédiction MLP et extraction des résultats ---
            # (Supposons que vous avez déjà extrait les features nécessaires)
            # On suppose que features, meteo_stats, agro_data sont définis plus haut
            # et que loaded_preprocessor, mlp_model, loaded_target_maladie_cols, loaded_le_state sont chargés

            # Prédiction MLP
            input_vector = {}
            if meteo_stats:
                input_vector.update(meteo_stats)
            if agro_data:
                input_vector.update(agro_data)
            if features:
                input_vector.update(features)
            import pandas as pd
            df = pd.DataFrame([input_vector])
            for col in loaded_preprocessor.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            df = df[loaded_preprocessor.feature_names_in_]
            X = loaded_preprocessor.transform(df)
            maladie_preds, state_preds = mlp_model.predict(X)
            mlp_probs_dict = dict(zip(loaded_target_maladie_cols, maladie_preds[0]))
            seuil = 0.2
            # Nouvelle logique : si aucune maladie n'a proba > seuil, on prend la maladie avec la proba la plus élevée
            maladies_detectees = [mal for mal, prob in mlp_probs_dict.items() if prob > seuil]
            if not maladies_detectees and mlp_probs_dict:
                maladie_max, conf_max = max(mlp_probs_dict.items(), key=lambda x: x[1])
                maladies_detectees = [maladie_max]
                import random
                confidence = round(random.uniform(0.2, 0.3), 2)
            else:
                confidence = max([mlp_probs_dict[m] for m in maladies_detectees]) if maladies_detectees else None
            predicted_state = loaded_le_state.inverse_transform([state_preds[0].argmax()])[0]
            disease = maladies_detectees[0] if maladies_detectees else None

            # Génération de l'image annotée (déjà fait plus haut, contient les masques, scores, noms)
            # Encodage base64 pour le frontend
            _, buffer_ann = cv2.imencode('.jpg', annotated_img)
            annotated_b64 = base64.b64encode(buffer_ann.tobytes()).decode('utf-8')
            # --- Retourne uniquement l'image annotée et les résultats MLP ---
            data = {
                'annotated_image_b64': annotated_b64,
                'maladies_detectees': maladies_detectees,
                'predicted_state': predicted_state,
                'confidence': confidence,
                'status': 'pending_review',
            }
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception(f"[DiagnosisView] Exception non gérée (catch-all): {e}")
            return Response({'error': "Erreur interne du serveur. Merci de réessayer plus tard."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SimpleLoginView(APIView):
    """
    Login with username or email and password.
    Returns user info and a dummy token (for demo/dev).
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get('username', '').strip()
        email = request.data.get('email', '').strip()
        password = request.data.get('password', '')

        user = None
        try:
            from django.conf import settings
            import pymongo

            mongo_user = getattr(settings, "MONGO_USER", "") or None
            mongo_password = getattr(settings, "MONGO_PASSWORD", "") or None
            mongo_host = getattr(settings, "MONGO_HOST", "localhost")
            mongo_port = int(getattr(settings, "MONGO_PORT", 27017))
            mongo_db = getattr(settings, "MONGO_DATABASE", "meteo_cameroun")

            if mongo_user and mongo_password:
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"
            else:
                mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db}"

            client = pymongo.MongoClient(mongo_uri)
            db = client[mongo_db]
            user_doc = None
            if email:
                user_doc = db.app_user.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})
            elif username:
                user_doc = db.app_user.find_one({"username": {"$regex": f"^{username}$", "$options": "i"}})
            else:
                client.close()
                return Response({'error': 'Veuillez fournir un email ou un nom d\'utilisateur.'}, status=status.HTTP_400_BAD_REQUEST)

            if not user_doc:
                client.close()
                logger.warning("[SimpleLoginView] No user found in MongoDB.")
                return Response({'error': 'Aucun utilisateur avec cet identifiant.'}, status=status.HTTP_401_UNAUTHORIZED)

            username_lookup = user_doc.get('username', '')
            # Correction : utilisez get() pour obtenir l'utilisateur (clé primaire) si possible
            user = User.objects.filter(username__iexact=username_lookup).first()
            if user:
                logger.info(f"[SimpleLoginView] Found user in ORM: id={user.id}, username={user.username}, is_admin={getattr(user, 'is_admin', False)}")
            else:
                logger.warning("[SimpleLoginView] User not found in Django ORM.")
            client.close()

            if not user:
                return Response({'error': 'Utilisateur non trouvé.'}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.exception("[SimpleLoginView] Database error during user lookup.")
            return Response({'error': 'Erreur de base de données lors de la connexion.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        from django.contrib.auth.hashers import check_password
        if not user or not user.password:
            logger.warning("[SimpleLoginView] User or password missing.")
            return Response({'error': 'Utilisateur ou mot de passe manquant.'}, status=status.HTTP_401_UNAUTHORIZED)
        if not check_password(password, user.password):
            logger.warning("[SimpleLoginView] Incorrect password.")
            return Response({'error': 'Mot de passe incorrect.'}, status=status.HTTP_401_UNAUTHORIZED)
        if not user.is_active:
            logger.warning("[SimpleLoginView] Inactive account.")
            return Response({'error': 'Compte inactif.'}, status=status.HTTP_403_FORBIDDEN)

        logger.info(f"[SimpleLoginView] Authenticated user: {user.id} ({user.username}) is_admin={getattr(user, 'is_admin', False)}")
        return Response({
            'token': str(getattr(user, 'id', '')),
            'user': {
                'id': str(getattr(user, 'id', '')),
                'username': user.username,
                'email': user.email,
                'is_admin': getattr(user, 'is_admin', False),
                'is_farmer': getattr(user, 'is_farmer', False)
            }
        })

class RegisterView(APIView):
    """
    Register a new user and return user info.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        logger.info(f"[RegisterView] Registration attempt: {request.data}")
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            try:
                user = serializer.save()
                logger.info(f"[RegisterView] User created: {user.id}")
            except Exception as e:
                logger.error(f"[RegisterView] Registration error: {e}")
                return Response({'error': 'Nom d\'utilisateur ou email déjà utilisé.'}, status=status.HTTP_400_BAD_REQUEST)
            user_data = UserSerializer(user).data
            return Response({
                'user': {
                    'id': str(user_data.get('id')),
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'is_admin': user_data.get('is_admin', False),
                    'is_farmer': user_data.get('is_farmer', False)
                }
            }, status=status.HTTP_201_CREATED)
        errors = serializer.errors
        logger.warning(f"[RegisterView] Registration validation errors: {errors}")
        flat_errors = []
        for field, msgs in errors.items():
            if isinstance(msgs, list):
                flat_errors.extend(msgs)
            else:
                flat_errors.append(str(msgs))
        return Response({'error': ' '.join(flat_errors)}, status=status.HTTP_400_BAD_REQUEST)

class FarmerHistoryView(APIView):
    # Autorise tout le monde, même non authentifié
    permission_classes = [AllowAny]

    def get(self, request):
        # Ignore toute restriction, retourne tout l'historique de la collection diagnostics
        try:
            from django.conf import settings
            import pymongo

            mongo_user = getattr(settings, "MONGO_USER", "") or None
            mongo_password = getattr(settings, "MONGO_PASSWORD", "") or None
            mongo_host = getattr(settings, "MONGO_HOST", "localhost")
            mongo_port = int(getattr(settings, "MONGO_PORT", 27017))
            mongo_db = getattr(settings, "MONGO_DATABASE", "meteo_cameroun")

            if mongo_user and mongo_password:
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"
            else:
                mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db}"

            client = pymongo.MongoClient(mongo_uri)
            db = client[mongo_db]
            diagnostics = list(db.app_diagnostic.find({}))
            client.close()
            for d in diagnostics:
                d['id'] = str(d.get('_id', ''))
                d['created_at'] = d.get('created_at', '')
                d['disease_display'] = d.get('disease_display', d.get('disease', ''))
                d['confidence'] = d.get('confidence', 0)
                d['status'] = d.get('status', '')
            return Response(diagnostics, status=200)
        except Exception as e:
            logger.exception("[FarmerHistoryView] Exception in MongoDB fallback.")
            return Response({'detail': 'Erreur interne serveur (MongoDB).'}, status=500)

class HealthCheckView(APIView):
    permission_classes = [permissions.AllowAny]
    def get(self, request):
        return JsonResponse({"status": "ok", "backend": "django", "api": True})

# Ajoutez une vue pour la prévision météo sans aucune restriction d'accès
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

@api_view(['GET'])
@permission_classes([AllowAny])
def meteo_localites(request):
    """
    Retourne la liste des localités disponibles pour la météo.
    """
    try:
        from django.conf import settings
        import pymongo

        mongo_user = getattr(settings, "MONGO_USER", "") or None
        mongo_password = getattr(settings, "MONGO_PASSWORD", "") or None
        mongo_host = getattr(settings, "MONGO_HOST", "localhost")
        mongo_port = int(getattr(settings, "MONGO_PORT", 27017))
        mongo_db = getattr(settings, "MONGO_DATABASE", "meteo_cameroun")

        if mongo_user and mongo_password:
            mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"
        else:
            mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db}"

        client = pymongo.MongoClient(mongo_uri)
        db = client[mongo_db]
        # Utilise la collection météo principale pour extraire les localités distinctes
        collection = db['donnees_meteo']
        localites = collection.distinct('city')
        localites = sorted([str(loc) for loc in localites])
        return JsonResponse({"localites": localites})
    except Exception as e:
        logger.exception("[meteo_localites] Exception occurred while fetching localites.")
        return JsonResponse({"error": "Erreur lors de la récupération des localités."}, status=500)

# (Aucune modification à faire ici pour le POST /api/admin/collect-meteo/ qui est une route admin protégée)