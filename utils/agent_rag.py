import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from pinecone import Pinecone
from dotenv import load_dotenv

from .embeddings import get_openai_embeddings_batch

load_dotenv()

# --- Clients ---
LLM_MODEL = "gpt-4o"
llm = ChatOpenAI(model=LLM_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
serper_search = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# --- Tools ---
async def pinecone_search(query: str, index) -> List[Dict[str, Any]]:
    print(f"[RAG] pinecone search → {query}")

    q_emb = (await get_openai_embeddings_batch([query]))[0]

    res = index.query(
        vector=q_emb,
        top_k=5,
        include_metadata=True
    )

    matches = res.get("matches", []) or []
    return [m.get("metadata", {}) for m in matches]


def web_search(query: str) -> str:
    print(f"[RAG] web search → {query}")
    try:
        r = serper_search.results(query)
        organic = r.get("organic", [])
        return "\n".join(f"- {o.get('snippet','')}" for o in organic[:4]) or "no hits"
    except Exception as e:
        print("serper error:", e)
        return "search error"


# --- Router ---
async def router(query: str) -> str:
    prompt = f"""
You are a tool router. 
Decide which source to use for this query.

If the question likely refers to the uploaded document or a report, choose:
"pinecone_search"

If it is general world knowledge, choose:
"web_search"

Only reply with one of the two tokens.

Query: {query}
"""
    r = await llm.ainvoke([{"role": "user", "content": prompt}])
    return r.content.strip().lower()


# --- Answer Synthesizer ---
async def answer_synthesizer(query: str, context: str) -> str:
    prompt = f"""
You are a helpful AI assistant. Use the context below to answer the question.
If the answer is not present in the context, say: "I cannot find it in the sources."

Context:
{context}

Question: {query}
"""
    r = await llm.ainvoke([{"role": "user", "content": prompt}])
    return r.content


# --- Main RAG Agent ---
async def run_rag_agent(query: str, pinecone_index) -> Dict[str, Any]:
    tool = await router(query)
    citations = []

    if "pinecone" in tool:
        ctxs = await pinecone_search(query, pinecone_index)

        # Combine text chunks + track citations
        context = ""
        seen_pages = set()

        for c in ctxs:
            page_num = c.get("page_number")
            page_img = c.get("page_img_path")
            text = c.get("text", "")
            cap = c.get("captions", [])

            context += f"\n\n[Page {page_num}] {text}"
            if cap:
                context += f"\nImage captions: {'; '.join(cap)}"

            # Save citation
            if page_num and page_num not in seen_pages:
                citations.append({
                    "page": page_num,
                    "image": page_img
                })
                seen_pages.add(page_num)

    elif "web" in tool:
        context = web_search(query)

    else:
        context = "router_error"

    answer = await answer_synthesizer(query, context)

    return {
        "answer": answer,
        "citations": citations
    }
