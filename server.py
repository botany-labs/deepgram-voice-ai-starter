import os
import time
import uuid

from deepgram import (
    AsyncLiveClient,
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveResultResponse,
    LiveTranscriptionEvents,
)
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocketState

from internals.client_connection import ClientConnection
from internals.llm import LLMFactory
from internals.text_to_speech import TextToSpeechFactory

app = FastAPI()

deepgram = DeepgramClient(
    os.getenv("DEEPGRAM_API_KEY"),
    DeepgramClientOptions(options={"keepalive": "true"}),
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


final_phrases = []

PROMPT = """
You're answering the phones at Zoob Zib, a thai restaurant in hells kitchen new york.
The menu has 2 items, pad thai and mango sticky rice.
Speak conversationally, your words will be spoken out loud.
Give 1 or 2 sentences at a time, and then wait for the user to respond
"""

INITIAL_MSG = "Thanks for calling zoob zib, how can I help you?"


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = str(uuid.uuid4())
    client = ClientConnection(client_id, websocket, PROMPT)

    # Edit these to change models
    llm = LLMFactory().get("groq", "llama3-70b-8192")
    tts = TextToSpeechFactory().get("playht")

    try:
        dg_connection: AsyncLiveClient = deepgram.listen.asynclive.v("1")

        async def on_message(self, result: LiveResultResponse, **kwargs):
            global final_phrases
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            if result.is_final:
                final_phrases.append(sentence)
                if result.speech_final:
                    complete_sentence = " ".join(final_phrases)
                    print(f"User: {complete_sentence}")
                    final_phrases = []

                    start_time = time.time()
                    end_time = None

                    await client.add_user_message(
                        complete_sentence,
                    )
                    llm_response = llm.stream_text(
                        client.get_messages(),
                    )
                    for chunk in tts.stream_to_stream(
                        llm_response, yield_original_text=True
                    ):
                        if isinstance(chunk, str):
                            await client.add_system_message(chunk)
                        else:
                            if end_time is None:
                                end_time = time.time()
                            await client.send_audio(chunk)

                    if end_time is not None:
                        await client.generic_log(
                            f"LLM + TTS: {end_time - start_time:.3f}s"
                        )

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        for chunk in tts.full_to_stream(INITIAL_MSG, yield_original_text=True):
            if isinstance(chunk, str):
                await client.add_system_message(chunk)
            else:
                await client.send_audio(chunk)

        await dg_connection.start(
            LiveOptions(
                punctuate=True, interim_results=False, language="en-US", endpointing=100
            )
        )
        while True:
            data = await websocket.receive_bytes()
            await dg_connection.send(data)
    except Exception as e:
        raise Exception(f"Could not process audio: {e}")
    finally:
        await dg_connection.finish()
        if websocket.state == WebSocketState.CONNECTED:
            await websocket.close()
