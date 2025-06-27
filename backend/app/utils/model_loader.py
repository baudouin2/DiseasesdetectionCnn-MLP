import joblib
import logging
import os
from django.conf import settings

def get_abs_path(rel_path):
    # Always resolve from Django BASE_DIR
    return os.path.normpath(os.path.join(settings.BASE_DIR, '..', rel_path))

CFG_PATH = get_abs_path("backend/static/modele/mask_rcnn_R_50_FPN_3x.yaml")
FEUILLE_WEIGHTS = get_abs_path("backend/static/modele/model_feuilles_final.pth")
ANOMALIE_WEIGHTS = get_abs_path("backend/static/modele/model_final.pth")
MLP_MODEL_PATH = get_abs_path("backend/static/modele/MLP_model.keras")
PREPROCESSOR_PATH = get_abs_path("backend/static/modele/preprocessor.joblib")
TARGET_COLS_PATH = get_abs_path("backend/static/modele/target_maladie_cols.joblib")
STATE_ENCODER_PATH = get_abs_path("backend/static/modele/label_encoder_state.joblib")

logger = logging.getLogger("django")

def get_model_path():
    return MLP_MODEL_PATH

def load_disease_model():
    import tensorflow as tf
    model_path = get_model_path()
    logger.info(f"[model_loader] Tentative de chargement du modèle à : {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"[model_loader] Fichier modèle introuvable à : {model_path}")
        raise FileNotFoundError(
            f"File not found: filepath={model_path}. "
            "Please ensure the file is an accessible `.keras` zip file."
        )
    return tf.keras.models.load_model(model_path)

def load_preprocessor():
    """
    Charge et retourne le préprocesseur (scaler/encoder) utilisé pour la préparation des features.
    """
    try:
        if not os.path.exists(PREPROCESSOR_PATH):
            logger.error(f"[model_loader] Fichier preprocessor introuvable à : {PREPROCESSOR_PATH}")
            raise FileNotFoundError(f"File not found: {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("[model_loader] Preprocessor chargé avec succès.")
        return preprocessor
    except Exception as e:
        logger.exception(f"[model_loader] Erreur lors du chargement du preprocessor: {e}")
        raise

def load_target_maladie_cols():
    """
    Charge et retourne la liste des colonnes cibles (maladies) pour la sortie du modèle.
    """
    try:
        if not os.path.exists(TARGET_COLS_PATH):
            logger.error(f"[model_loader] Fichier target_maladie_cols introuvable à : {TARGET_COLS_PATH}")
            raise FileNotFoundError(f"File not found: {TARGET_COLS_PATH}")
        cols = joblib.load(TARGET_COLS_PATH)
        logger.info("[model_loader] Colonnes cibles maladies chargées.")
        return cols
    except Exception as e:
        logger.exception(f"[model_loader] Erreur lors du chargement des colonnes cibles: {e}")
        raise

def load_state_encoder():
    """
    Charge et retourne l'encodeur d'état (LabelEncoder) pour la sortie du modèle.
    """
    try:
        if not os.path.exists(STATE_ENCODER_PATH):
            logger.error(f"[model_loader] Fichier state_encoder introuvable à : {STATE_ENCODER_PATH}")
            raise FileNotFoundError(f"File not found: {STATE_ENCODER_PATH}")
        encoder = joblib.load(STATE_ENCODER_PATH)
        logger.info("[model_loader] Encodeur d'état chargé.")
        return encoder
    except Exception as e:
        logger.exception(f"[model_loader] Erreur lors du chargement de l'encodeur d'état: {e}")
        raise

def load_all_model_objects():
    """
    Charge tous les objets nécessaires pour la prédiction (modèle, preprocessor, colonnes, encodeur).
    Retourne un dict.
    """
    return {
        "model": load_disease_model(),
        "preprocessor": load_preprocessor(),
        "target_cols": load_target_maladie_cols(),
        "state_encoder": load_state_encoder()
    }


