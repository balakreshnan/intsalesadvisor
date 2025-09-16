from __future__ import annotations

import os
import uuid
import json
import asyncio
import base64
import logging
import threading
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from typing import Dict, Union, Literal, Optional, Set, Callable, Awaitable
from typing_extensions import AsyncIterator, TypedDict, Required
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.client import ClientConnection as AsyncWebsocket
from websockets.asyncio.client import HeadersLike
from websockets.typing import Data
from websockets.exceptions import WebSocketException
from azure.identity import DefaultAzureCredential
from azure.core.credentials_async import AsyncTokenCredential

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
AUDIO_SAMPLE_RATE = 24000

AudioTimestampTypes = Literal["word"]
load_dotenv()

class AzureDeepNoiseSuppression(TypedDict, total=False):
    type: Literal["azure_deep_noise_suppression"]

class ServerEchoCancellation(TypedDict, total=False):
    type: Literal["server_echo_cancellation"]

class AzureSemanticVAD(TypedDict, total=False):
    type: Literal["azure_semantic_vad"]
    threshold: float
    silence_duration_ms: int
    prefix_padding_ms: int

class Animation(TypedDict, total=False):
    outputs: Set[Literal["blendshapes", "viseme_id", "emotion"]]

class Session(TypedDict, total=False):
    voice: Dict[str, Union[str, float]]
    turn_detection: Union[AzureSemanticVAD]
    input_audio_noise_reduction: AzureDeepNoiseSuppression
    input_audio_echo_cancellation: ServerEchoCancellation
    animation: Animation
    output_audio_timestamp_types: Set[AudioTimestampTypes]
    input_audio_sampling_rate: int

class SessionUpdateEventParam(TypedDict, total=False):
    type: Literal["session.update"]
    session: Required[Session]
    event_id: str

class AsyncVoiceLiveSessionResource:
    def __init__(self, connection: AsyncVoiceLiveConnection) -> None:
        self._connection = connection

    async def update(self, *, session: Session, event_id: str | None = None) -> None:
        param: SessionUpdateEventParam = {"type": "session.update", "session": session, "event_id": event_id}
        data = json.dumps(param)
        await self._connection.send_with_retry(data)

class AsyncVoiceLiveConnection:
    session: AsyncVoiceLiveSessionResource
    _connection: AsyncWebsocket

    def __init__(self, url: str, additional_headers: HeadersLike) -> None:
        self._url = url
        self._additional_headers = additional_headers
        self._connection = None
        self.session = AsyncVoiceLiveSessionResource(self)

    async def __aenter__(self) -> AsyncVoiceLiveConnection:
        try:
            self._connection = await ws_connect(self._url, additional_headers=self._additional_headers, ping_interval=20)
        except WebSocketException as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None

    enter = __aenter__
    close = __aexit__

    async def __aiter__(self) -> AsyncIterator[Data]:
        async for data in self._connection:
            yield data

    async def recv(self) -> Data:
        return await self._connection.recv()

    async def recv_bytes(self) -> bytes:
        return await self._connection.recv()

    async def send(self, message: Data) -> None:
        await self._connection.send(message)

    async def send_with_retry(self, message: Data, max_retries: int = 3) -> None:
        for attempt in range(max_retries):
            try:
                await self._connection.send(message)
                return
            except (ConnectionResetError, WebSocketException) as e:
                logger.warning(f"Send attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

class AsyncAzureVoiceLive:
    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = "2025-05-01-preview",
        api_key: str | None = None,
        azure_ad_token_credential: AsyncTokenCredential | None = None,
    ) -> None:
        if azure_endpoint is None:
            azure_endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")

        if azure_endpoint is None:
            raise ValueError("AZURE_VOICE_LIVE_ENDPOINT must be set.")

        if api_key is None and azure_ad_token_credential is None:
            api_key = os.environ.get("AZURE_VOICE_LIVE_API_KEY")

        if api_key is None and azure_ad_token_credential is None:
            raise ValueError("Missing credentials: Provide 'api_key' or 'azure_ad_token_credential'.")

        if api_key and azure_ad_token_credential:
            raise ValueError("Provide only one of 'api_key' or 'azure_ad_token_credential'.")

        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._azure_ad_token_credential = azure_ad_token_credential
        self._connection = None
        self._token = None

    async def get_token_async(self) -> str:
        if self._azure_ad_token_credential:
            scopes = "https://ai.azure.com/.default"
            token = await self._azure_ad_token_credential.get_token(scopes)
            return token.token
        return None

    def connect(self, model: str) -> AsyncVoiceLiveConnection:
        if self._connection is not None:
            raise ValueError("Already connected.")
        if not model:
            raise ValueError("Model name is required.")
        if not isinstance(model, str):
            raise TypeError(f"'model' must be str, got {type(model).__name__}.")

        url = f"{self._azure_endpoint.rstrip('/')}/voice-agent/realtime?api-version={self._api_version}&model={model}"
        # For agent-based pipeline: url += f"&agent_id={os.environ.get('AGENT_ID')}&project_id={os.environ.get('PROJECT_ID')}"
        url = url.replace("https://", "wss://")

        if self._api_key:
            auth_header = {"api-key": self._api_key}
        else:
            self._token = asyncio.run(self.get_token_async())
            auth_header = {"Authorization": f"Bearer {self._token}"}

        request_id = uuid.uuid4()
        headers = {"x-ms-client-request-id": str(request_id), **auth_header}

        self._connection = AsyncVoiceLiveConnection(url, additional_headers=headers)
        return self._connection

class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            blocksize=1200,
        )
        self.playing = False

    def callback(self, outdata, frames, time, status):
        with self.lock:
            data = np.empty(0, dtype=np.int16)
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))
        outdata[:] = data.reshape(-1, 1)

    def add_data(self, data: bytes):
        with self.lock:
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()

    def stop(self):
        with self.lock:
            self.queue = []
        self.playing = False
        self.stream.stop()

    def terminate(self):
        self.stream.close()

async def listen_and_send_audio(connection: AsyncVoiceLiveConnection) -> None:
    logger.info("Starting audio stream ...")
    stream = sd.InputStream(channels=1, samplerate=AUDIO_SAMPLE_RATE, dtype="int16")
    try:
        stream.start()
        read_size = int(AUDIO_SAMPLE_RATE * 0.02)
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0.01)
                continue
            data, _ = stream.read(read_size)
            audio_energy = np.abs(data).mean()
            # logger.debug(f"Audio energy: {audio_energy:.2f}")
            if audio_energy < 100:  # Lowered threshold for quieter mics
                continue
            audio = base64.b64encode(data).decode("utf-8")
            param = {"type": "input_audio_buffer.append", "audio": audio, "event_id": ""}
            data_json = json.dumps(param)
            try:
                await connection.send_with_retry(data_json)
            except (ConnectionResetError, WebSocketException) as e:
                logger.error(f"WebSocket error while sending audio: {e}")
                break
    except Exception as e:
        logger.debug(f"Audio stream interrupted: {e}")
    finally:
        stream.stop()
        stream.close()
        logger.info("Audio stream closed.")

async def receive_audio_and_playback(connection: AsyncVoiceLiveConnection) -> None:
    last_audio_item_id = None
    audio_player = AudioPlayerAsync()
    logger.info("Starting audio playback ...")
    turn_count = 0
    max_turns = 10
    try:
        async for raw_event in connection:
            try:
                event = json.loads(raw_event)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode event: {e}")
                continue

            logger.debug(f"Received event type: {event.get('type')}")

            if event.get("type") == "response.audio.delta":
                if event.get("item_id") != last_audio_item_id:
                    last_audio_item_id = event.get("item_id")
                    logger.debug(f"New audio item: {last_audio_item_id}")

                bytes_data = base64.b64decode(event.get("delta", ""))
                audio_player.add_data(bytes_data)

                animation = event.get("animation")
                if animation:
                    logger.info(f"Avatar animation data (turn {turn_count}): {animation}")
                    # Extend: Send to renderer (e.g., Three.js or WebRTC for TTS Avatar)

            elif event.get("type") == "response.audio_transcript.delta":
                logger.debug(f"Transcript delta: {event.get('transcript_delta', '')}")

            elif event.get("type") == "error":
                logger.error(f"Server error: {event}")
                if event.get("error", {}).get("code") in ["connection_failed", "authentication_failed"]:
                    break
            elif event.get("type") == "response.done":
                logger.info(f"Response done received for turn {turn_count}")
                turn_count += 1
                if turn_count >= max_turns:
                    logger.info("Max turns reached, exiting.")
                    break
            else:
                logger.debug(f"Unhandled event: {event}")

    except (ConnectionResetError, WebSocketException) as e:
        logger.error(f"WebSocket error in audio playback: {e}")
    except Exception as e:
        logger.error(f"Error in audio playback: {e}")
    finally:
        audio_player.terminate()
        logger.info("Playback done.")

async def read_keyboard_and_quit() -> None:
    print("Press 'q' and Enter to quit the chat.")
    while True:
        user_input = await asyncio.to_thread(input)
        if user_input.strip().lower() == 'q':
            print("Quitting the chat...")
            break

async def main() -> None:
    endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
    deployment = os.environ.get("AZURE_VOICE_LIVE_DEPLOYMENT", "gpt-4o-realtime-preview")
    api_key = os.environ.get("AZURE_VOICE_LIVE_API_KEY")

    if not endpoint:
        raise ValueError("AZURE_VOICE_LIVE_ENDPOINT must be set.")

    client = AsyncAzureVoiceLive(azure_endpoint=endpoint, api_key=api_key)
    async with client.connect(model=deployment) as connection:
        await connection.session.update(
            session={
                "input_audio_sampling_rate": AUDIO_SAMPLE_RATE,
                "turn_detection": {
                    "type": "azure_semantic_vad",
                    "threshold": 0.25,
                    "prefix_padding_ms": 100,
                    "silence_duration_ms": 1200,
                },
                "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
                "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
                "voice": {
                    "name": "en-US-Ava:DragonHDLatestNeural",
                    "type": "azure-standard",
                    "temperature": 0.8,
                },
                "animation": {
                    "outputs": ["viseme_id", "blendshapes", "emotion"]
                },
            }
        )

        # Add initial conversation item to start the session
        initial_event_id = str(uuid.uuid4())
        initial_param = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant. Respond concisely to user queries."}
                ]
            },
            "event_id": initial_event_id
        }
        await connection.send_with_retry(json.dumps(initial_param))
        logger.info(f"Sent initial conversation item with event_id: {initial_event_id}")

        send_task = asyncio.create_task(listen_and_send_audio(connection))
        receive_task = asyncio.create_task(receive_audio_and_playback(connection))
        keyboard_task = asyncio.create_task(read_keyboard_and_quit())

        print("Starting the chat ... (Speak now!)")
        await asyncio.wait([send_task, receive_task, keyboard_task], return_when=asyncio.FIRST_COMPLETED)

        send_task.cancel()
        receive_task.cancel()
    print("Chat done.")

if __name__ == "__main__":
    try:
        load_dotenv()
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")