import os
import base64
import asyncio
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


async def main() -> None:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-10-01-preview",
    )

    # Open a sounddevice output stream (24kHz mono)
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
    stream.start()

    async with client.beta.realtime.connect(
        model="gpt-realtime",
    ) as connection:
        await connection.session.update(session={"output_modalities": ["text", "audio"]})

        while True:
            user_input = input("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await connection.response.create()

            async for event in connection:
                if event.type == "response.audio.delta":
                    audio_data = base64.b64decode(event.delta)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    stream.write(audio_np)  # 🔊 play immediately

                elif event.type == "response.audio_transcript.delta":
                    print(event.delta, flush=True, end="")

                elif event.type == "response.audio_transcript.done":
                    # for c in event.item.content:
                    #     if c.type == "audio" and c.transcript:
                    #         print(f"\n[Final Transcript] {c.transcript}")
                    print()

                elif event.type == "response.done":
                    # handle_response_done(event)
                    print("✅ Response complete\n")
                    break

    stream.stop()
    stream.close()


if __name__ == "__main__":
    asyncio.run(main())