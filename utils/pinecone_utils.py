import uuid
from typing import List, Dict, Any
from pinecone import Pinecone as PineconeSync  # Import sync version for utility tasks if needed
from pinecone.grpc import PineconeGRPC as PineconeAsync  # Main async client

# Use a relative import to get our new async embedding function
from .embeddings import get_openai_embeddings_batch


async def upsert_documents_to_pinecone(
        pc_async_client: PineconeAsync,
        pinecone_index_name: str,
        chunks: List[str],
        captions: List[Dict[str, Any]]  # Expecting [{'text': caption, 'page': num}]
):
    """
    Embeds and upserts document chunks and image captions to Pinecone asynchronously.

    Args:
        pc_async_client: An initialized PineconeGRPC (async) client instance.
        pinecone_index_name: The name of the target Pinecone index.
        chunks: A list of text chunks from the document.
        captions: A list of dictionaries, each with caption text and page number.
    """
    print("Starting document and caption upsert process...")

    # 1. Combine all text content for efficient batch embedding
    texts_to_embed = chunks + [caption for caption in captions]

    if not texts_to_embed:
        print("No text or captions to upsert.")
        return

    # 2. Get all embeddings in a single, fast API call
    print(f"Generating embeddings for {len(texts_to_embed)} items in a single batch...")
    all_embeddings = await get_openai_embeddings_batch(texts_to_embed)

    # 3. Prepare vectors in the new SDK format: a list of dictionaries
    vectors_to_upsert = []

    # Process text chunks
    for i, chunk in enumerate(chunks):
        if all_embeddings[i]:  # Check if embedding was successful
            vectors_to_upsert.append({
                "id": f"chunk_{uuid.uuid4()}",
                "values": all_embeddings[i],
                "metadata": {"type": "text", "text": chunk}
            })

    # Process image captions
    chunk_count = len(chunks)
    print(captions)
    for i, caption_data in enumerate(captions):
        print(caption_data)
        embedding_index = chunk_count + i
        if all_embeddings[embedding_index]:  # Check embedding
            vectors_to_upsert.append({
                "id": f"caption_{uuid.uuid4()}",
                "values": all_embeddings[embedding_index],
                "metadata": {
                    "type": "image_caption",
                    "text": caption_data
                }
            })
    # 4. Connect to the index and upsert asynchronously
    if not vectors_to_upsert:
        print("No valid vectors were generated to upsert.")
        return
    try:
        print(f"Connecting to index '{pinecone_index_name}' and upserting {len(vectors_to_upsert)} vectors...")
        index = pc_async_client.Index(pinecone_index_name)
        # upsert the vectors to Pinecone
        index.upsert(vectors=vectors_to_upsert, batch_size=100)

        print("Successfully upserted vectors to Pinecone.")
    except Exception as e:
        print(f"An error occurred during Pinecone upsert: {e}")