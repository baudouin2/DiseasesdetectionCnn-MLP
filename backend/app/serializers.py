from rest_framework import serializers
from .models import Diagnostic

class DiagnosticSerializer(serializers.ModelSerializer):
    segmented_image_b64 = serializers.CharField(required=False, allow_blank=True)
    annotated_image_b64 = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = Diagnostic
        fields = [
            'id',
            'patient_name',
            'doctor_name',
            'diagnosis',
            'segmented_image_b64',
            'annotated_image_b64',
            'created_at',
            'updated_at',
        ]