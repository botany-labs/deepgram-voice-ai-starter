import base64
import json
import time

from fastapi import WebSocket


class ClientConnection:
    def __init__(
        self,
        client_id: str,
        websocket: WebSocket,
        system_prompt: str,
    ):
        self.client_id = client_id
        self.websocket = websocket
        self.num_voice_inputs = 0
        self.system_prompt = system_prompt
        self.messages = []

    async def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})
        await self.websocket.send_text(
            json.dumps({"type": "logs", "message": f"User: {message}"})
        )

    async def add_system_message(self, message: str):
        self.messages.append({"role": "system", "content": message})
        await self.websocket.send_text(
            json.dumps({"type": "logs", "message": f"System: {message}"})
        )

    async def generic_log(self, message: str):
        current_time = (
            time.strftime("%H:%M:%S", time.localtime())
            + f".{int(time.time() % 1 * 1000):03d}"
        )
        await self.websocket.send_text(
            json.dumps({"type": "logs", "message": f"===== {message}"})
        )

    async def send_audio(self, chunk: bytes):
        b64_chunk = base64.b64encode(chunk).decode("utf-8")
        await self.websocket.send_text(
            json.dumps({"type": "audio", "b64buffer": b64_chunk})
        )

    def dump(self):
        print(self.messages)

    def get_messages(self):
        return [{"role": "system", "content": self.system_prompt}] + self.messages
