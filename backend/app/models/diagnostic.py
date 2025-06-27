# === Imports ===
import os
import logging
import cv2
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
import joblib
from django.conf import settings
from djongo import models
from app.utils.model_loader import (
    load_disease_model,
    load_preprocessor,
    load_target_maladie_cols,
    load_state_encoder
)
import json
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger("django")

# === Constantes et chemins des modèles ===

def get_static_modele_path(filename):
    """
    Retourne le chemin absolu du fichier dans static/modele/ à partir de BASE_DIR Django.
    """
    base_dir = getattr(settings, "BASE_DIR", None)
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "static", "modele", filename)
    logger.info(f"[get_static_modele_path] {filename} -> {path} ({'OK' if os.path.isfile(path) else 'NON'})")
    return path

CFG_PATH = get_static_modele_path("mask_rcnn_R_50_FPN_3x.yaml")
FEUILLE_WEIGHTS = get_static_modele_path("model_feuilles_final.pth")
ANOMALIE_WEIGHTS = get_static_modele_path("model_final.pth")
MLP_MODEL_PATH = get_static_modele_path("MLP_model.keras")
PREPROCESSOR_PATH = get_static_modele_path("preprocessor.joblib")
TARGET_COLS_PATH = get_static_modele_path("target_maladie_cols.joblib")
STATE_ENCODER_PATH = get_static_modele_path("label_encoder_state.joblib")
CLASS_NAMES = ['jaunissement', 'tache_brune', 'moisissure', 'mosaïque']

# === Chargement différé des modèles pour éviter les erreurs à l'import Django ===
_predictor_feuille = None
_predictor_anomalie = None

def get_predictor_feuille():
    global _predictor_feuille
    if _predictor_feuille is None:
        try:
            abs_cfg_path = CFG_PATH
            logger.info(f"[get_predictor_feuille] Chemin absolu config: {abs_cfg_path}")
            logger.info(f"[get_predictor_feuille] Fichier existe: {os.path.isfile(abs_cfg_path)}")
            try:
                logger.info(f"[get_predictor_feuille] Contenu du dossier: {os.listdir(os.path.dirname(abs_cfg_path))}")
            except Exception as e:
                logger.warning(f"[get_predictor_feuille] Impossible de lister le dossier: {e}")
            if not os.path.isfile(abs_cfg_path):
                logger.error(f"[get_predictor_feuille] Fichier config Detectron2 introuvable : {abs_cfg_path}")
                raise ImportError(
                    f"Le fichier de configuration Detectron2 est introuvable : {abs_cfg_path}\n"
                    "Vérifiez que le fichier existe bien à cet emplacement ou corrigez le chemin dans CFG_PATH."
                )
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            cfg = get_cfg()
            cfg.merge_from_file(abs_cfg_path)
            cfg.MODEL.WEIGHTS = FEUILLE_WEIGHTS
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            _predictor_feuille = DefaultPredictor(cfg)
        except ModuleNotFoundError as e:
            logger.error("[get_predictor_feuille] Detectron2 n'est pas installé ou mal importé : %s", e)
            raise ImportError(
                "Le module detectron2 n'est pas installé ou mal importé sur ce serveur.\n"
                "Vérifiez que le dossier 'backend/app/detectron2' contient bien un __init__.py et que le PYTHONPATH inclut 'backend/app'.\n"
                "Ou installez detectron2 globalement avec pip install detectron2.\n"
                "Voir : https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
            )
        except Exception as e:
            logger.error("[get_predictor_feuille] Erreur lors de l'initialisation du modèle Detectron2 : %s", e)
            raise ImportError(f"Erreur lors de l'initialisation du modèle Detectron2 : {e}")
    return _predictor_feuille

def get_predictor_anomalie():
    global _predictor_anomalie
    if _predictor_anomalie is None:
        try:
            abs_cfg_path = CFG_PATH
            logger.info(f"[get_predictor_anomalie] Chemin absolu config: {abs_cfg_path}")
            logger.info(f"[get_predictor_anomalie] Fichier existe: {os.path.isfile(abs_cfg_path)}")
            try:
                logger.info(f"[get_predictor_anomalie] Contenu du dossier: {os.listdir(os.path.dirname(abs_cfg_path))}")
            except Exception as e:
                logger.warning(f"[get_predictor_anomalie] Impossible de lister le dossier: {e}")
            if not os.path.isfile(abs_cfg_path):
                logger.error(f"[get_predictor_anomalie] Fichier config Detectron2 introuvable : {abs_cfg_path}")
                raise ImportError(
                    f"Le fichier de configuration Detectron2 est introuvable : {abs_cfg_path}\n"
                    "Vérifiez que le fichier existe bien à cet emplacement ou corrigez le chemin dans CFG_PATH."
                )
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            cfg = get_cfg()
            cfg.merge_from_file(abs_cfg_path)
            cfg.MODEL.WEIGHTS = ANOMALIE_WEIGHTS
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.66
            _predictor_anomalie = DefaultPredictor(cfg)
        except ModuleNotFoundError as e:
            logger.error("[get_predictor_anomalie] Detectron2 n'est pas installé : %s", e)
            raise ImportError(
                "Le module detectron2 n'est pas installé sur ce serveur. "
                "Pour activer la détection Mask R-CNN, installez detectron2 avec :\n"
                "pip install 'torch>=1.10' torchvision\n"
                "# puis suivez la doc officielle : https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
            )
        except Exception as e:
            logger.error("[get_predictor_anomalie] Erreur lors de l'initialisation du modèle Detectron2 : %s", e)
            raise ImportError(f"Erreur lors de l'initialisation du modèle Detectron2 : {e}")
    return _predictor_anomalie

def get_mlp_model():
    return load_disease_model()

def get_loaded_target_maladie_cols():
    # Utilise toujours le chemin absolu du vrai fichier target_maladie_cols.joblib
    path = TARGET_COLS_PATH
    if not os.path.isfile(path):
        logger.error(f"[get_loaded_target_maladie_cols] Fichier introuvable : {path}")
        raise FileNotFoundError(f"target_maladie_cols.joblib non trouvé à {path}")
    cols = joblib.load(path)
    logger.info(f"[get_loaded_target_maladie_cols] Colonnes cibles chargées : {cols}")
    return cols

def get_loaded_preprocessor():
    # Utilise toujours le chemin absolu du vrai fichier preprocessor.joblib
    path = PREPROCESSOR_PATH
    if not os.path.isfile(path):
        logger.error(f"[get_loaded_preprocessor] Fichier introuvable : {path}")
        raise FileNotFoundError(f"preprocessor.joblib non trouvé à {path}")
    preproc = joblib.load(path)
    logger.info(f"[get_loaded_preprocessor] Preprocessor chargé avec colonnes : {getattr(preproc, 'feature_names_in_', None)}")
    return preproc

def get_loaded_le_state():
    # Utilise toujours le chemin absolu du vrai fichier label_encoder_state.joblib
    path = STATE_ENCODER_PATH
    if not os.path.isfile(path):
        logger.error(f"[get_loaded_le_state] Fichier introuvable : {path}")
        raise FileNotFoundError(f"label_encoder_state.joblib non trouvé à {path}")
    le = joblib.load(path)
    # Filtrer la classe "mort" si elle existe
    if hasattr(le, "classes_"):
        le.classes_ = np.array([c for c in le.classes_ if c != "mort"])
    logger.info(f"[get_loaded_le_state] LabelEncoder state chargé : {getattr(le, 'classes_', None)}")
    return le

def build_mlp_vector_ordered(meteo_stats, agro_data, features):
    """
    Construit un vecteur de caractéristiques ordonné pour le MLP à partir des données météo, agro et features extraites.
    Les clés sont triées pour correspondre à l'ordre attendu par le préprocesseur.
    """
    vector = {}
    if meteo_stats:
        vector.update(meteo_stats)
    if agro_data:
        vector.update(agro_data)
    if features:
        vector.update(features)
    return vector

def extract_anomaly_features(instances, image, surface_feuilles):
    # Utilise torch/numpy selon le type de données
    masks = getattr(instances, "pred_masks", None)
    if masks is not None and hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    else:
        masks = np.zeros((0, *image.shape[:2]))
    boxes = getattr(getattr(instances, "pred_boxes", None), "tensor", None)
    if boxes is not None and hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy().astype(int)
    else:
        boxes = np.zeros((0, 4)).astype(int)
    pred_classes = getattr(instances, "pred_classes", None)
    if pred_classes is not None and hasattr(pred_classes, "cpu"):
        pred_classes = pred_classes.cpu().numpy()
    else:
        pred_classes = np.zeros(0, dtype=int)
    scores = getattr(instances, "scores", None)
    if scores is not None and hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.zeros(len(pred_classes))
    stats = defaultdict(lambda: defaultdict(list))
    count_by_label = defaultdict(int)
    surface_by_label = defaultdict(float)
    masks_by_label = defaultdict(list)
    bboxes_by_label = defaultdict(list)
    for i, label_idx in enumerate(pred_classes):
        # Correction : vérifiez que label_idx est bien dans les bornes de CLASS_NAMES
        if int(label_idx) < 0 or int(label_idx) >= len(CLASS_NAMES):
            logger.warning(f"[extract_anomaly_features] label_idx {label_idx} hors bornes pour CLASS_NAMES {CLASS_NAMES}")
            continue
        label = CLASS_NAMES[int(label_idx)]
        count_by_label[label] += 1
        x1, y1, x2, y2 = boxes[i]
        # Correction : vérifiez que les coordonnées sont valides
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = image[y1:y2, x1:x2]
        mask = masks[i][y1:y2, x1:x2]
        if mask.sum() == 0 or region.shape[0] < 2 or region.shape[1] < 2:
            continue
        stats[label]['score'].append(float(scores[i]))
        masks_by_label[label].append(mask.astype(np.uint8))
        bboxes_by_label[label].append((x1, y1, x2, y2))
        gray = rgb2gray(region)
        masked = (gray * mask * 255).astype(np.uint8)
        if np.any(masked):
            glcm = graycomatrix(masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
                stats[label][prop].append(graycoprops(glcm, prop)[0, 0])
    # Calcul surface par classe en tenant compte des superpositions
    for label in CLASS_NAMES:
        if masks_by_label[label]:
            mask_union = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask, (x1, y1, x2, y2) in zip(masks_by_label[label], bboxes_by_label[label]):
                h, w = y2 - y1, x2 - x1
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_union[y1:y2, x1:x2] = np.logical_or(mask_union[y1:y2, x1:x2], mask).astype(np.uint8)
            surface_by_label[label] = float(mask_union.sum())
        else:
            surface_by_label[label] = 0.0
    features = {}
    for label in CLASS_NAMES:
        features[f"nb_{label}"] = count_by_label[label]
        if surface_feuilles and surface_feuilles > 0:
            features[f"surf_{label}"] = float(surface_by_label[label]) * 100.0 / float(surface_feuilles)
        else:
            features[f"surf_{label}"] = 0.0
        for metric, prefix in [
            ('score', 'score_moy'),
            ('contrast', 'texture_contrast'), ('homogeneity', 'texture_homogeneity'),
            ('energy', 'texture_energy'), ('correlation', 'texture_correlation')
        ]:
            key = f"{prefix}_{label}"
            features[key] = float(np.mean(stats[label][metric])) if stats[label][metric] else 0.0
    return features

def diagnostiquer_image_segmentee(img_path, meteo_stats=None, agro_data=None):
    image = cv2.imread(img_path)
    if image is None:
        print("❌ Image introuvable.")
        return [], "Erreur"

    # 1. Détection des feuilles
    feuilles = get_predictor_feuille()(image)["instances"]
    nb_feuilles = len(getattr(feuilles, "pred_masks", []))
    if nb_feuilles == 0:
        print("⚠️ Aucune feuille détectée.")
        return [], "Inconnu"

    pred_masks = getattr(feuilles, "pred_masks", None)
    if isinstance(pred_masks, torch.Tensor):
        mask_union = torch.zeros_like(pred_masks[0], dtype=torch.uint8)
        for i in range(pred_masks.shape[0]):
            mask_union = mask_union | pred_masks[i].to(torch.uint8)
        mask_union = mask_union.cpu().numpy()
    else:
        mask_union = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in (pred_masks or []):
            mask_union = np.logical_or(mask_union, mask).astype(np.uint8)
    segmented_image = image.copy()
    segmented_image[mask_union == 0] = 0
    surface_feuilles = int(mask_union.sum())

    # 2. Détection des anomalies
    anomalies = get_predictor_anomalie()(segmented_image)

    # 3. Visualisation
    # METADATA_PATH = get_static_modele_path("metadata.json")
    # if os.path.exists(METADATA_PATH):
    #     with open(METADATA_PATH, "r") as f:
    #         metadata = json.load(f)
    # else:
    #     metadata = None

    # Utilisation d'un metadata vide si le fichier n'existe pas
    metadata = {}

    vis = Visualizer(image[:, :, ::-1], metadata=metadata)
    anomalies_img = vis.draw_instance_predictions(anomalies["instances"].to("cpu")).get_image()[:, :, ::-1]
    concat = np.hstack((image, segmented_image, anomalies_img))

    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image originale | Image segmentée | Anomalies détectées")
    plt.show()

    # 4. Features (surface par anomalie)
    features = extract_anomaly_features(anomalies["instances"], segmented_image, surface_feuilles)
    features["nb_feuilles"] = nb_feuilles  # Ajoute le nombre de feuilles détectées

    # Récupération des données météo/agro depuis la base de données ou les paramètres reçus
    full_vector = {}
    if meteo_stats:
        full_vector.update(meteo_stats)
    if agro_data:
        full_vector.update(agro_data)
    full_vector.update(features)
    df = pd.DataFrame([full_vector])

    # 5. Prétraitement
    loaded_preprocessor = get_loaded_preprocessor()
    for col in loaded_preprocessor.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[loaded_preprocessor.feature_names_in_]

    print("\n📄 Vecteur final passé au MLP :")
    print(df.head(1).T)

    # 6. Prédiction
    mlp_model = get_mlp_model()
    loaded_le_state = get_loaded_le_state()
    loaded_target_maladie_cols = get_loaded_target_maladie_cols()
    X = loaded_preprocessor.transform(df)
    maladie_preds, state_preds = mlp_model.predict(X)
    predicted_state = loaded_le_state.inverse_transform([np.argmax(state_preds[0])])[0]

    seuil = 0.5
    maladies_detectees = [loaded_target_maladie_cols[i] for i, val in enumerate(maladie_preds[0]) if val > seuil]

    # Utilisez la surface totale des anomalies pour la logique sain/bien_portant si besoin
    total_surface = 0.0
    for m in CLASS_NAMES:
        total_surface += features.get(f"surf_{m}", 0.0)
    if total_surface < 0.1:
        maladies_detectees = ["sain"]
        predicted_state = "bien_portant"

    print(f"\n🦠 Maladies détectées : {maladies_detectees}")
    print(f"📊 Stade de la plante : {predicted_state}")
    print("🔎 Probabilités MLP maladie :", dict(zip(loaded_target_maladie_cols, maladie_preds[0])))
    print("🗂️ Colonnes cibles :", loaded_target_maladie_cols)

    # Ajout : retourne aussi les features d'anomalies pour l'API (mask, name, score)
    anomalies = []
    instances = anomalies["instances"]
    masks = getattr(instances, "pred_masks", None)
    pred_classes = getattr(instances, "pred_classes", None)
    scores = getattr(instances, "scores", None)
    if masks is not None and pred_classes is not None and scores is not None:
        masks = masks.cpu().numpy() if hasattr(masks, "cpu") else masks
        pred_classes = pred_classes.cpu().numpy() if hasattr(pred_classes, "cpu") else pred_classes
        scores = scores.cpu().numpy() if hasattr(scores, "cpu") else scores
        for i in range(len(pred_classes)):
            label_idx = int(pred_classes[i])
            name = CLASS_NAMES[label_idx] if 0 <= label_idx < len(CLASS_NAMES) else str(label_idx)
            # Encode le mask en PNG base64 (pour affichage direct sur mobile)
            import base64
            import io
            from PIL import Image as PILImage
            mask_img = (masks[i] * 255).astype('uint8')
            pil_img = PILImage.fromarray(mask_img)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            mask_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            anomalies.append({
                "mask": mask_b64,
                "name": name,
                "score": float(scores[i])
            })

    # Retourne les résultats (maladies_detectees, predicted_state, anomalies, etc)
    return {
        "maladies_detectees": maladies_detectees,
        "predicted_state": predicted_state,
        "anomalies": anomalies,
    }

# CORRECTION : la récupération automatique des stats météo doit être AVANT la construction du vecteur full_vector
    if meteo_stats is None and agro_data and 'localite' in agro_data:
        from app.models.meteo import MeteoData
        from django.utils import timezone
        city = agro_data['localite']
        now = timezone.now()
        start_date = (now.replace(day=1) - timezone.timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0)
        qs = MeteoData.objects.filter(city=city, datetime__gte=start_date, datetime__lte=now)
        if qs.exists():
            meteo_stats = {
                "temperature": qs.aggregate(models.Avg('temperature'))['temperature__avg'],
                "humidity": qs.aggregate(models.Avg('humidity'))['humidity__avg'],
                "pressure": qs.aggregate(models.Avg('pressure'))['pressure__avg'],
                "wind_speed": qs.aggregate(models.Avg('wind_speed'))['wind_speed__avg'],
                "precipitation": qs.aggregate(models.Avg('precipitation'))['precipitation__avg'],
            }
        else:
            meteo_stats = {
                "temperature": None,
                "humidity": None,
                "pressure": None,
                "wind_speed": None,
                "precipitation": None,
            }
torch.set_num_threads(os.cpu_count() or 1)
# Pour forcer l'utilisation de tous les cœurs CPU pour les opérations torch (si pas sur GPU)

# 3. Pour TensorFlow :
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 1)
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count() or 1)

# 4. Pour OpenCV :
cv2.setNumThreads(os.cpu_count() or 1)

# 5. Pour joblib (scikit-learn, pandas parallel) : utilisez n_jobs=-1 dans les fonctions qui le supportent.

# Note : Ces réglages sont à placer au début de votre script (après les imports).

# === Modèle principal Diagnostic ===
class Diagnostic(models.Model):
    class Diseases(models.TextChoices):
        EARLY_BLIGHT = 'EB', 'Brûlure précoce'
        LATE_BLIGHT = 'LB', 'Brûlure tardive'
        SEPTORIA = 'SP', 'Septoriose'
        HEALTHY = 'HE', 'Saine'
        # Ajouter d'autres maladies si besoin

    image = models.ImageField(upload_to='diagnostics/')
    disease = models.CharField(max_length=2, choices=Diseases.choices)
    confidence = models.FloatField()
    temperature = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,  # Autorise la valeur NULL pour éviter les erreurs si l'utilisateur est supprimé
        blank=True
    )
    is_verified = models.BooleanField(default=False)
    admin_comment = models.TextField(blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending_review', 'En revue'),
            ('confirmed', 'Confirmé'),
        ],
        default='pending_review'
    )
    annotated_image = models.ImageField(upload_to='diagnostics/annotated/', null=True, blank=True)
    # Champs agronomiques
    sol = models.CharField(max_length=32, blank=True, null=True)
    irrigation = models.CharField(max_length=32, blank=True, null=True)
    azote = models.FloatField(blank=True, null=True)
    phosphore = models.FloatField(blank=True, null=True)
    potassium = models.FloatField(blank=True, null=True)
    compost = models.FloatField(blank=True, null=True)
    engrais_chimique = models.FloatField(blank=True, null=True)
    # Ajoutez ces champs pour stocker les images en base64
    segmented_image_b64 = models.TextField(null=True, blank=True)
    annotated_image_b64 = models.TextField(null=True, blank=True)

    @property
    def is_validated(self):
        return self.is_verified or self.status == 'confirmed'

    @property
    def is_pending(self):
        return not self.is_verified and self.status == 'pending_review'

    def __str__(self):
        return f"{self.get_disease_display()} ({self.confidence}%)"

    @property
    def disease_display(self):
        return self.get_disease_display()

# === Modèle PendingReview pour la modération admin ===
class PendingReview(models.Model):
    diagnostic = models.ForeignKey('Diagnostic', on_delete=models.CASCADE)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'En attente'),
        ('approved', 'Approuvé'),
        ('rejected', 'Rejeté')
    ])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review for {self.diagnostic} by {self.reviewed_by}"

    def save(self, *args, **kwargs):
        logger.debug(f"[PendingReview.save] Saving review for diagnostic {self.diagnostic_id} by {self.reviewed_by_id}")
        super().save(*args, **kwargs)
