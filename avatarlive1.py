import os
import asyncio
import json
import base64
import time
import traceback
from typing import Optional

import websockets
from dotenv import load_dotenv

load_dotenv()

AZURE_RESOURCE = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT", "your-realtime-deployment")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

PING_INTERVAL_SEC = 20
RESPONSE_TIMEOUT_SEC = 45

def build_uri():
    return (
        f"wss://{AZURE_RESOURCE}.openai.azure.com/openai/realtime"
        f"?api-version={AZURE_API_VERSION}&model={AZURE_DEPLOYMENT}"
    )

async def send_json(ws, obj):
    await ws.send(json.dumps(obj))

async def heartbeat(ws, shutdown_event: asyncio.Event):
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(PING_INTERVAL_SEC)
            # Some servers accept a lightweight ping event; if not, use ws.ping()
            try:
                await ws.ping()
            except Exception:
                break
    finally:
        # Returning ends the task
        pass

async def receive_loop(ws, shutdown_event: asyncio.Event):
    try:
        while not shutdown_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=RESPONSE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                print("[WARN] Receive timeout; signaling shutdown.")
                shutdown_event.set()
                break
            except websockets.ConnectionClosedOK:
                print("[INFO] Connection closed cleanly by server.")
                shutdown_event.set()
                break
            except websockets.ConnectionClosedError as e:
                print(f"[ERROR] Connection closed with error: {e}")
                shutdown_event.set()
                break

            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                print("[WARN] Non-JSON frame received (possibly binary/audio?)")
                continue

            etype = evt.get("type")
            if etype in ("response.output_text.delta", "response.text.delta"):
                print(evt.get("delta",""), end="", flush=True)
            elif etype in ("response.output_audio.delta", "response.audio.delta"):
                # handle base64 PCM
                b64 = evt.get("delta")
                if b64:
                    audio_bytes = base64.b64decode(b64)
                    # TODO: enqueue to playback
            elif etype == "response.done":
                print("\n[INFO] Response finished.")
            elif etype == "error":
                print("[SERVER ERROR]", evt)
                shutdown_event.set()
                break
            # else: ignore or log debug
    finally:
        shutdown_event.set()

async def microphone_loop(ws, shutdown_event: asyncio.Event):
    # Pseudocode placeholder for capturing mic and sending audio frames
    # Ensure you gracefully exit on shutdown_event
    try:
        while not shutdown_event.is_set():
            # mic_data = capture_chunk()
            # b64 = base64.b64encode(mic_data).decode('ascii')
            # await send_json(ws, {
            #     "type": "input_audio_buffer.append",
            #     "audio": b64
            # })
            await asyncio.sleep(0.05)
    except websockets.ConnectionClosed:
        pass
    finally:
        # flush signal if required
        # await send_json(ws, {"type": "input_audio_buffer.commit"})
        pass

async def main():
    if not AZURE_RESOURCE or not AZURE_KEY:
        print("[ERROR] Missing AZURE_OPENAI_RESOURCE or AZURE_OPENAI_KEY.")
        return

    uri = build_uri()
    headers = [
        ("api-key", AZURE_KEY),
        ("Sec-WebSocket-Protocol", "realtime"),
    ]
    print(f"[INFO] Connecting: {uri}")

    async with websockets.connect(uri, extra_headers=headers, max_size=None, ping_interval=None) as ws:
        shutdown_event = asyncio.Event()

        # Session configuration (voice options may vary; adjust fields)
        await send_json(ws, {
            "type": "session.update",
            "session": {
                "output_modalities": ["text", "audio"],
                "audio_format": "pcm16",
                # "voice": "alloy"  # if required by your service
            }
        })

        # Create tasks
        recv_task = asyncio.create_task(receive_loop(ws, shutdown_event))
        hb_task = asyncio.create_task(heartbeat(ws, shutdown_event))
        # mic_task = asyncio.create_task(microphone_loop(ws, shutdown_event))  # enable if needed

        # Simple interactive loop
        try:
            while not shutdown_event.is_set():
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input("You: "))
                if user_input.strip().lower() == "q":
                    print("[INFO] Quitting...")
                    shutdown_event.set()
                    break

                # Send a message
                await send_json(ws, {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_input}],
                    }
                })
                await send_json(ws, {"type": "response.create"})

        finally:
            shutdown_event.set()
            # Cancel tasks gracefully
            for task in (recv_task, hb_task):
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(recv_task, hb_task, return_exceptions=True)
            except Exception:
                pass
            # Close handshake
            try:
                await ws.close(code=1000, reason="normal")
            except Exception:
                pass
            print("[INFO] Closed websocket.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception:
        traceback.print_exc()