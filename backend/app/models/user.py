from djongo import models
from django.contrib.auth.models import AbstractUser
import logging

logger = logging.getLogger("django")

class User(AbstractUser):
    is_farmer = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Utilisateur"
        verbose_name_plural = "Utilisateurs"

    def __str__(self):
        return self.username

    # Ne surchargez pas save, ne faites aucune mise à jour automatique lors de l'authentification ou du login.
    # Laissez Django/Djongo gérer la persistance normalement.