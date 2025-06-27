from rest_framework import serializers
from app.models import User
from django.contrib.auth.password_validation import validate_password
import logging

logger = logging.getLogger("django")

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'is_farmer', 'is_admin', 'password']
        extra_kwargs = {'password': {'write_only': True}}
        # Désactive les validators uniques automatiques pour éviter .exists()
        validators = []

    def create(self, validated_data):
        """
        Crée un nouvel utilisateur avec gestion unique des rôles et vérification unicité.
        Utilise une vérification manuelle pour éviter les problèmes de connexion MongoClient/Djongo.
        """
        is_admin = validated_data.pop('is_admin', False)
        username = validated_data['username']
        email = validated_data.get('email', '')

        # Vérification manuelle d'unicité (évite .exists()/.unique validator)
        if list(User.objects.filter(username__iexact=username)[:1]):
            logger.warning(f"[UserSerializer] Username already used: {username}")
            raise serializers.ValidationError({'username': "Ce nom d'utilisateur est déjà utilisé."})
        if email and list(User.objects.filter(email__iexact=email)[:1]):
            logger.warning(f"[UserSerializer] Email already used: {email}")
            raise serializers.ValidationError({'email': "Cet email est déjà utilisé."})

        user = User.objects.create_user(
            username=username,
            email=email,
            password=validated_data['password']
        )
        user.is_admin = is_admin
        user.is_farmer = not is_admin
        user.save()
        logger.info(f"[UserSerializer] User created: {user.id}")
        return user

    def update(self, instance, validated_data):
        """
        Mise à jour utilisateur sans permettre la modification des rôles via l'API.
        """
        validated_data.pop('is_admin', None)
        validated_data.pop('is_farmer', None)
        logger.info(f"[UserSerializer] User update: {instance.id}")
        return super().update(instance, validated_data)

    def to_representation(self, instance):
        """
        Nettoie la sortie (pas de password, id en str).
        """
        ret = super().to_representation(instance)
        ret.pop('password', None)
        if 'id' in ret and ret['id'] is not None:
            ret['id'] = str(ret['id'])
        return ret