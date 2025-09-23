import os
import base64
import asyncio
import sys
from openai import AsyncAzureOpenAI
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

def handle_response_done(event):
    response = event.response
    print("=== ✅ Response Done ===")
    print(f"Response ID: {response.id}")
    print(f"Conversation ID: {response.conversation_id}")
    print(f"Status: {response.status}")
    print(f"Modalities: {response.modalities}")
    print(f"Voice: {response.voice}")

    for item in response.output:
        if hasattr(item, "content"):
            for c in item.content:
                if c.type == "audio" and c.transcript:
                    print("\n🗣️ Final Transcript:")
                    print(c.transcript)

    usage = response.usage
    print("\n=== 📊 Token Usage ===")
    print(f" Input tokens: {usage.input_tokens}")
    print(f" Output tokens: {usage.output_tokens}")
    print(f" Total tokens: {usage.total_tokens}")
    print(f" Audio output tokens: {usage.output_token_details.audio_tokens}")
    print(f" Text output tokens: {usage.output_token_details.text_tokens}")

# Add this function before main()
def get_dynamic_context(topic: str) -> str:
    # Simulate fetching from a DB/file; replace with real logic
    if topic == "earbuds":
        return "You are advising on earbuds. Key facts: Model XYZ, $99, Bluetooth 5.0, sweat-resistant."
    return "General sales advisor context: Be friendly and informative."

async def main() -> None:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-10-01-preview",
    )

    dynamic_instructions = get_dynamic_context("earbuds")  # Call your functio

    # Open a sounddevice output stream (24kHz mono)
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
    stream.start()

    async with client.beta.realtime.connect(
        model="gpt-realtime",
    ) as connection:
        await connection.session.update(session={
            "output_modalities": ["text", "audio"],
            "instructions": dynamic_instructions,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            }
        })

        def audio_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            # Convert float32 mono input to int16 PCM bytes
            audio_data = (indata[:, 0] * 32767).astype(np.int16)
            audio_bytes = audio_data.tobytes()
            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            # Send delta asynchronously from the callback thread
            future = asyncio.run_coroutine_threadsafe(
                connection.input_audio_buffer.append(audio=b64_audio),
                loop
            )
            # Optionally wait for completion: future.result()

        loop = asyncio.get_running_loop()

        # Start input stream from microphone (24kHz mono, 20ms chunks)
        input_stream = sd.InputStream(
            samplerate=24000,
            channels=1,
            dtype='float32',
            blocksize=480,  # 24000 / 50 = 480 samples for 20ms
            callback=audio_callback
        )
        input_stream.start()

        print("🔴 Listening for audio input... Speak now! (Ctrl+C to quit)")

        try:
            async for event in connection:
                if event.type == "response.audio.delta":
                    audio_data = base64.b64decode(event.delta)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    stream.write(audio_np)  # 🔊 play immediately

                elif event.type == "response.audio_transcript.delta":
                    print(event.delta, flush=True, end="")

                elif event.type == "response.audio_transcript.done":
                    print()

                elif event.type == "response.done":
                    handle_response_done(event)
                    print("✅ Ready for next input\n")
                    # Continue listening (no break)

        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            input_stream.stop()
            input_stream.close()

    stream.stop()
    stream.close()


if __name__ == "__main__":
    asyncio.run(main())