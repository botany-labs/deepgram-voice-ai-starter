import os
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import requests
from cartesia import Cartesia
from dotenv import load_dotenv
from elevenlabs import stream
from pyht import Client, Format, TTSOptions
from websockets.sync.client import connect

from internals.eleven_labs_websocket import ElevenWebSocket

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
deepgram_key = os.getenv("DEEPGRAM_API_KEY")
cartesia_key = os.getenv("CARTESIA_API_KEY")

pyht_user = os.getenv("PYHT_USER")
pyht_api_key = os.getenv("PYHT_API_KEY")

elevenlabs_key = os.getenv("ELEVEN_API_KEY")


class TextToSpeech:
    def stream_to_stream(
        self, text_stream: Generator[str, None, None], yield_original_text: bool = False
    ) -> Generator[bytes, None, None]:
        for text in text_stream:
            print(f"TextToSpeech: {text}")
            if text == "":
                continue

            for audio_chunk in self._stream_speech(text):
                yield audio_chunk
            if yield_original_text:
                yield text

    def full_to_stream(
        self, text: str, yield_original_text: bool = False
    ) -> Generator[bytes, None, None]:
        for audio_chunk in self._stream_speech(text):
            yield audio_chunk
        if yield_original_text:
            yield text

    def full_to_full(self, text: str) -> bytes:
        buffer = b""
        for chunk in self._stream_speech(text):
            buffer += chunk
        return buffer

    @abstractmethod
    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        """
        asks the user_message to the LLM and calls the audio_callback with bytes that represent sound
        """
        pass


class TextToSpeechFactory:
    def get(self, model):
        if model == "openai":
            return OpenAITextToSpeech()
        if model == "deepgram":
            return DeepgramTextToSpeech()
        if model == "cartesia":
            return CartesiaTextToSpeech()
        if model == "playht":
            return PlayHTTextToSpeech()
        if model == "eleven_labs":
            return ElevenLabsTextToSpeech()
        else:
            raise ValueError(f"Model {model} not supported")


class OpenAITextToSpeech(TextToSpeech):
    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {openai_key}",  # Replace with your API key
        }

        data = {
            "model": "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "pcm",
        }

        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                buffer = b""
                for chunk in response.iter_content(chunk_size=512):
                    buffer += chunk
                    while len(buffer) >= 512:
                        yield buffer[:512]
                        buffer = buffer[512:]
                if buffer:
                    yield buffer
            else:
                print(f"Error: {response.status_code} - {response.text}")


class DeepgramTextToSpeech(TextToSpeech):
    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&container=none&encoding=linear16"

        payload = {"text": text}

        headers = {
            "Authorization": f"Token {deepgram_key}",
            "Content-Type": "application/json",
        }

        with requests.post(
            DEEPGRAM_URL, headers=headers, json=payload, stream=True
        ) as response:
            if response.status_code == 200:
                buffer = b""
                for chunk in response.iter_content(chunk_size=512):
                    buffer += chunk
                    while len(buffer) >= 512:
                        yield buffer[:512]
                        buffer = buffer[512:]
                if buffer:
                    yield buffer
            else:
                print(f"Error: {response.status_code} - {response.text}")


class CartesiaTextToSpeech(TextToSpeech):
    def __init__(self):
        self.cartesia_client = Cartesia(api_key=cartesia_key)
        self.ws = self.cartesia_client.tts.websocket()  # 200ms
        self.voice = self.cartesia_client.voices.get(
            id="a0e99841-438c-4a64-b679-ae501e7d6091"
        )

    def __del__(self):
        self.ws.close()

    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        model_id = "sonic-english"

        output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        }

        # Generate and stream audio using the websocket
        for output in self.ws.send(
            model_id=model_id,
            transcript=text,
            voice_embedding=self.voice["embedding"],
            stream=True,
            output_format=output_format,
        ):
            yield output["audio"]


class PlayHTTextToSpeech(TextToSpeech):
    def __init__(self):
        self.play_client = Client(
            pyht_user,
            pyht_api_key,
        )

    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        options = TTSOptions(
            voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
            format=Format.FORMAT_WAV,
        )
        first_chunk = True
        for chunk in self.play_client.tts(
            text=text, voice_engine="PlayHT2.0-turbo", options=options
        ):
            if first_chunk:
                chunk = chunk[100:]
                first_chunk = False
            yield chunk


class ElevenLabsTextToSpeech(TextToSpeech):
    def __init__(self):
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        self.ws = ElevenWebSocket(
            ws_url=f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2&output_format=pcm_24000&optimize_streaming_latency=4",
            api_key=elevenlabs_key,
        )
        self.ws.connect()

    def __del__(self):
        self.ws.close()

    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        # Generate and stream audio using the websocket
        for output in self.ws.send(
            text=text,
        ):
            yield output["audio"]

    # def __init__(self):
    #     self.client = ElevenLabs(api_key=elevenlabs_key)

    # def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
    #     audio_stream = self.client.generate(
    #         text=text,
    #         output_format="pcm_24000",
    #         stream=True,
    #         optimize_streaming_latency=4,
    #     )
    #     for chunk in audio_stream:
    #         yield chunk
