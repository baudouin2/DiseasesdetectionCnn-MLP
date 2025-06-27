from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from app.models import User, Diagnostic, PendingReview
from app.serializers.diagnostic import DiagnosticSerializer
from django.http import HttpResponse
import json

class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_farmer', 'is_admin')
    search_fields = ('username', 'email')
    ordering = ('username',)
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal Info', {'fields': ('email',)}),
        ('Permissions', {'fields': ('is_farmer', 'is_admin', 'is_active', 'is_staff', 'is_superuser')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2', 'is_farmer', 'is_admin'),
        }),
    )
    # list_filter intentionally omitted for Djongo compatibility

@admin.register(Diagnostic)
class DiagnosticAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'disease', 'confidence', 'created_at')
    list_filter = ('disease', 'created_at')
    search_fields = ('user__username', 'disease')
    readonly_fields = ('confidence', 'created_at')
    ordering = ('-created_at',)
    actions = ['export_as_json']

    def export_as_json(self, request, queryset):
        serializer = DiagnosticSerializer(queryset, many=True)
        response = HttpResponse(
            json.dumps(serializer.data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        response['Content-Disposition'] = 'attachment; filename=diagnostics.json'
        return response
    export_as_json.short_description = "Exporter la sélection en JSON"

@admin.register(PendingReview)
class PendingReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'diagnostic', 'status', 'reviewed_by', 'created_at')
    list_filter = ('status',)
    raw_id_fields = ('diagnostic', 'reviewed_by')
    ordering = ('-created_at',)

admin.site.register(User, CustomUserAdmin)
# Register your models here.
