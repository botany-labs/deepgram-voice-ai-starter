import os
from abc import ABC, abstractmethod
from typing import Generator, Iterable, List

from groq import Groq
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

openai_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


class LLM(ABC):
    def full_text(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> str:
        """
        Takes a list of messages and returns the full response from the LLM by aggregating chunks from _stream_text.
        """
        response = ""
        for chunk in self._stream_text(messages):
            response += chunk
        return response

    # def streamText(
    #     self, messages: Iterable[ChatCompletionMessageParam]
    # ) -> Generator[str, None, None]:
    #     """
    #     Takes a list of messages and returns a generator that yields the response from the LLM in chunks.
    #     """
    #     return self._stream_text(messages)

    def _stream_as_sentences(
        self, stream: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """
        Takes a generator that yields the response from the LLM in chunks and returns a generator that yields the response from the LLM in sentences.
        """
        curr_sentence = ""
        for chunk in stream:
            curr_sentence += chunk
            if "." in curr_sentence or "!" in curr_sentence or "?" in curr_sentence:
                yield curr_sentence
                curr_sentence = ""
        yield curr_sentence

    def stream_text(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> Generator[str, None, None]:
        """
        Takes a list of messages and returns a generator that yields the response from the LLM in sentences.
        """
        received_first_chunk = False
        for sentence in self._stream_as_sentences(self._stream_text(messages)):
            if not received_first_chunk:
                received_first_chunk = True
            yield sentence

    @abstractmethod
    def _stream_text(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        """
        Takes a list of messages and returns a generator that yields the response from the LLM in chunks.
        """
        pass


class LLMFactory:
    def get(self, provider: str, model: str):
        if provider == "openai":
            return OpenAILLM(model)
        elif provider == "groq":
            return GroqLLM(model)
        else:
            raise ValueError(f"Provider {provider} not supported")


class OpenAILLM(LLM):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=openai_key)

    def _stream_text(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, temperature=0
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""


class GroqLLM(LLM):
    def __init__(self, model: str):
        self.model = model
        self.client = Groq(api_key=groq_api_key)

    def _stream_text(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, temperature=0
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""
