import os
import base64
import asyncio
from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def main() -> None:
    """
    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """

    credential = DefaultAzureCredential()
    token_provider=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2025-08-28",
    )
    async with client.beta.realtime.connect(
        model="gpt-realtime",  # name of your deployment
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
                if event.type == "response.output_text.delta":
                    print(event.delta, flush=True, end="")
                elif event.type == "response.output_audio.delta":

                    audio_data = base64.b64decode(event.delta)
                    print(f"Received {len(audio_data)} bytes of audio data.")
                elif event.type == "response.output_audio_transcript.delta":
                    print(f"Received text delta: {event.delta}")
                elif event.type == "response.output_text.done":
                    print()
                elif event.type == "response.done":
                    break

    await credential.close()

asyncio.run(main())