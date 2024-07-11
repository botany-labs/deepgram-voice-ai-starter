# Deepgram Voice AI Starter

Voice AI starter with:
STT: Deepgram
LLM: Groq, OpenAI
TTS: Cartesia, PlayHT, OpenAI, ElevenLabs

## Installation

```
pipenv install
```

## Running

```
cp sample.env .env
```

Setup relevant .env variables

Start command:

```
pipenv run uvicorn server:app --reload
```
