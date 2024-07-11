import base64
import json
import time
from typing import Any, Dict, Generator, List, Optional, Union

from websockets.sync.client import connect


class ElevenWebSocket:
    """A synchronous websocket for eleven labs, based on the Cartesia websocket client"""

    def __init__(
        self,
        ws_url: str,
        api_key: str,
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        self.websocket = None

    def connect(self):
        """This method connects to the WebSocket if it is not already connected."""
        if self.websocket is None or self._is_websocket_closed():
            self.websocket = connect(self.ws_url)
            bos_message = {
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self.api_key,
            }
            self.websocket.send(json.dumps(bos_message))

    def _is_websocket_closed(self):
        return self.websocket.socket.fileno() == -1

    def close(self):
        """This method closes the WebSocket connection. *Highly* recommended to call this method when done using the WebSocket."""
        if self.websocket is not None and not self._is_websocket_closed():
            self.websocket.close()

    def _convert_response(self, response: Dict[str, any]) -> Dict[str, Any]:
        return {
            "audio": base64.b64decode(response["audio"]),
        }

    def send(
        self,
        text: str,
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """Send a request to the WebSocket to generate audio."""
        request_body = {"text": text, "try_trigger_generation": True}
        return self._websocket_generator(request_body)

    def _websocket_generator(self, request_body: Dict[str, Any]):
        self.websocket.send(json.dumps(request_body))
        # Send EOS message with an empty string instead of a single space
        eos_message = {"text": ""}
        self.websocket.send(json.dumps(eos_message))

        try:
            while True:
                response = json.loads(self.websocket.recv())
                if "error" in response:
                    raise RuntimeError(f"Error generating audio:\n{response['error']}")
                if "isFinal" in response and response["isFinal"]:
                    self.close()
                    self.connect()
                    break
                yield self._convert_response(response=response)
        except Exception as e:
            # Close the websocket connection if an error occurs.
            if self.websocket and not self._is_websocket_closed():
                self.websocket.close()
            raise RuntimeError(f"Failed to generate audio. {response}") from e
