import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List

# Load environment variables to get the API key
load_dotenv()

# Initialize the ASYNC OpenAI client once when the module is imported
# This is efficient and best practice for FastAPI.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts using OpenAI's async API.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    if not texts:
        return []

    try:
        # The API is optimized for batching. This is much faster.
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        # Extract the embedding vectors from the response
        return [item.embedding for item in response.data]

    except Exception as e:
        print(f"An error occurred while generating embeddings in batch: {e}")
        # In a real app, you might want more sophisticated error handling
        return [[] for _ in texts]  # Return empty lists on failure