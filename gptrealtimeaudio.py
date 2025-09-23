import os
import base64
import asyncio
import sys
import json
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


# Define a local function that takes a text string (query) and returns some content
def get_local_context(query: str) -> str:
    """
    Local function to provide context based on a text query.
    This is a simple example; expand it as needed (e.g., check against a dict, file, or compute dynamically).
    """
    context_map = {
        "earbuds": "Wireless Earbuds: Price $99, Features: Noise-cancelling, 20-hour battery, Bluetooth 5.0. Great for workouts!",
        "laptop": "Gaming Laptop: Intel i7, 16GB RAM, RTX 3060 GPU, 512GB SSD. Ideal for gaming and productivity.",
        "phone": "Smartphone: 6.1-inch OLED, 128GB storage, Triple camera. Runs latest OS with excellent battery life.",
        "default": "General knowledge: We're a tech store specializing in gadgets. Ask about products for more details!"
    }
    
    # Simple keyword matching; in production, use regex, fuzzy search, etc.
    query_lower = query.lower()
    for key in context_map:
        if key in query_lower:
            return context_map[key]
    return context_map["default"]


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
        # Define the tool schema for the local function
        get_context_tool = {
            "type": "function",
            "name": "get_local_context",
            "description": "Call this to retrieve relevant context or knowledge from the local store based on a query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text query to fetch context for (e.g., 'earbuds features')."
                    }
                },
                "required": ["query"]
            }
        }

        await connection.session.update(session={
            "output_modalities": ["text", "audio"],
            "instructions": "You are a helpful sales advisor for tech gadgets. Use the get_local_context tool to fetch specific product details when a user asks about items like earbuds, laptops, or phones. Incorporate the returned info into your response conversationally.",
            "tools": [get_context_tool],  # Enable the tool
            "tool_choice": "auto",  # Model decides when to call; set to {"type": "function", "function": {"name": "get_local_context"}} to force
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

        print("🔴 Listening for audio input... Speak now! (e.g., 'Tell me about earbuds') (Ctrl+C to quit)")

        pending_tool_calls = {}  # Track ongoing tool calls (deltas)

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

                elif event.type == "response.function_call_arguments.delta":
                    # Accumulate function call arguments (sent in deltas)
                    call_id = event.call_id
                    if call_id not in pending_tool_calls:
                        pending_tool_calls[call_id] = {
                            "name": event.name,
                            "arguments": ""
                        }
                    pending_tool_calls[call_id]["arguments"] += event.delta

                elif event.type == "response.function_call.done":
                    # Tool call is complete; execute and return
                    call_id = event.call_id
                    tool_call = pending_tool_calls.pop(call_id, None)
                    if tool_call:
                        try:
                            args = json.loads(tool_call["arguments"])
                            query = args.get("query", "")
                            if tool_call["name"] == "get_local_context":
                                result = get_local_context(query)
                                # Send back the result as a function return item
                                await connection.conversation.item.create(
                                    item={
                                        "type": "function_call_return",
                                        "call_id": call_id,
                                        "result": result
                                    }
                                )
                        except json.JSONDecodeError:
                            print(f"Error parsing tool args for call {call_id}", file=sys.stderr)
                        # Trigger the model to continue generating response with tool result
                        await connection.response.create()

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