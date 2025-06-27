import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from app.models import Diagnostic
from app.serializers.diagnostic import DiagnosticSerializer

class DiagnosticConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        # For MongoDB/Djongo, user id may be ObjectId or str, so cast to str for group name
        self.room_group_name = f"user_{str(self.user.id)}_diagnostics"
        
        # Rejoindre le groupe
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()

    async def disconnect(self, close_code):
        # Quitter le groupe
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        action = text_data_json.get('action')
        
        if action == 'subscribe':
            # Envoyer les diagnostics récents
            diagnostics = await self.get_recent_diagnostics()
            await self.send(text_data=json.dumps({
                'type': 'initial_data',
                'data': diagnostics
            }))

    async def diagnostic_update(self, event):
        # Envoyer les mises à jour aux clients
        await self.send(text_data=json.dumps(event))

    @database_sync_to_async
    def get_recent_diagnostics(self):
        # For MongoDB/Djongo, always cast user id to str for filtering if needed
        diagnostics = Diagnostic.objects.filter(user=self.user).order_by('-created_at')[:10]
        serializer = DiagnosticSerializer(diagnostics, many=True)
        return serializer.data