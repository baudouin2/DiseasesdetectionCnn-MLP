import cv2
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile
import logging

logger = logging.getLogger("django")

def preprocess_image(image_file):
    """
    Convertit une image uploadée (InMemoryUploadedFile ou chemin) en tableau numpy prétraité pour l'inférence modèle.
    Retourne None en cas d'échec.
    """
    try:
        # Lecture de l'image selon le type d'entrée
        if isinstance(image_file, InMemoryUploadedFile):
            image_bytes = image_file.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(getattr(image_file, 'path', str(image_file)))
        if img is None:
            logger.error("[preprocess_image] cv2.imread returned None.")
            return None
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.exception(f"[preprocess_image] Image preprocessing failed: {e}")
        return None