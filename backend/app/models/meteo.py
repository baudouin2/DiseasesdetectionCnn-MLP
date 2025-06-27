from djongo import models
from django.utils import timezone

class MeteoData(models.Model):
    city = models.CharField(max_length=100, db_index=True)
    city_id = models.IntegerField(db_index=True)
    datetime = models.DateTimeField(db_index=True)
    temperature = models.FloatField()
    pressure = models.FloatField()
    humidity = models.FloatField()
    wind_speed = models.FloatField()
    precipitation = models.FloatField()
    soil_type = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        db_table = 'donnees_meteo'
        verbose_name = "Donnée météo"
        verbose_name_plural = "Données météo"
        indexes = [
            models.Index(fields=['city']),
            models.Index(fields=['datetime']),
        ]

    def save(self, *args, **kwargs):
        # Force timezone-aware datetime (évite le warning Django)
        if self.datetime and timezone.is_naive(self.datetime):
            self.datetime = timezone.make_aware(self.datetime, timezone.get_default_timezone())
        super().save(*args, **kwargs)

    @classmethod
    def ensure_timezone_aware(cls, dt):
        # Utilitaire pour rendre un datetime timezone-aware
        if dt and timezone.is_naive(dt):
            return timezone.make_aware(dt, timezone.get_default_timezone())
        return dt

    def __str__(self):
        return f"{self.city} - {self.datetime.date()}"
