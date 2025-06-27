from rest_framework import serializers
from app.models import Diagnostic, PendingReview
import logging

logger = logging.getLogger("django")

class DiagnosticSerializer(serializers.ModelSerializer):
    id = serializers.SerializerMethodField()
    user = serializers.SerializerMethodField()
    disease_display = serializers.CharField(source='get_disease_display', read_only=True)
    status = serializers.CharField(read_only=True)

    class Meta:
        model = Diagnostic
        fields = '__all__'
        read_only_fields = ['user', 'created_at', 'is_verified', 'status']

    def get_id(self, obj):
        return str(getattr(obj, 'id', ''))

    def get_user(self, obj):
        # Si user est un ObjectId ou un objet, on retourne str
        user = getattr(obj, 'user', None)
        return str(user) if user is not None else None

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        # Cast ObjectId fields to str for MongoDB compatibility
        for field in ['id', 'user']:
            if field in ret and ret[field] is not None:
                ret[field] = str(ret[field])
        # Correction : cast aussi tous les champs ForeignKey (ex: user) et tout champ non natif
        for field, value in ret.items():
            # Si la valeur ressemble à un ObjectId (24 caractères hex) ou n'est pas un str/int/float/bool, cast en str
            if (hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool))) or (
                isinstance(value, str) and len(value) == 24 and all(c in '0123456789abcdef' for c in value.lower())
            ):
                try:
                    ret[field] = str(value)
                except Exception:
                    pass
        # Ajoute les champs calculés pour la cohérence frontend/backend
        ret['disease_display'] = getattr(instance, 'disease_display', ret.get('disease_display'))
        ret['status'] = getattr(instance, 'status', ret.get('status'))
        logger.debug(f"[DiagnosticSerializer] to_representation: {ret}")
        return ret

class DiagnosticVerificationSerializer(serializers.ModelSerializer):
    """
    Serializer for verifying diagnostics.
    """
    class Meta:
        model = Diagnostic
        fields = ['is_verified', 'admin_comment']

class PendingReviewSerializer(serializers.ModelSerializer):
    """
    Serializer for PendingReview model.
    """
    class Meta:
        model = PendingReview
        fields = '__all__'

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        # Cast ObjectId fields to str for MongoDB compatibility
        for field in ['id', 'diagnostic', 'reviewed_by']:
            if field in ret and ret[field] is not None:
                ret[field] = str(ret[field])
        logger.debug(f"[PendingReviewSerializer] to_representation: {ret}")
        return ret