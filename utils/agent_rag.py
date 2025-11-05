import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

from .embeddings import get_openai_embeddings_batch

# --- 1. Clients ---
LLM_MODEL = "gpt-4o"
llm = ChatOpenAI(model=LLM_MODEL, api_key=os.getenv("OPENAI_API_KEY"))

serper_search = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# note: you need to pass this to run_rag_agent argument
# index = pc.Index("rag-agent")


# --- 2. Tools ---

async def pinecone_search(query: str, index) -> List[Dict[str, Any]]:
    print(f"[RAG] pinecone search → {query}")

    q_emb = (await get_openai_embeddings_batch([query]))[0]

    res = index.query(
        vector=q_emb,
        top_k=4,
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


# --- 3. Router / Synth ---

async def router(query: str) -> str:
    prompt = f"""
You are a router. choose one token only:
pinecone_search  → if query is about the uploaded doc
web_search       → if query is general world knowledge

query: {query}
"""

    r = await llm.ainvoke([{"role":"user","content":prompt}])
    return r.content.strip()


async def answer_synthesizer(query: str, context: str) -> str:
    prompt = f"""
Answer strictly using ONLY this context:
{context}

If the answer is not inside context say:
"I cannot find it in the sources."

Query: {query}
"""

    r = await llm.ainvoke([{"role":"user","content":prompt}])
    return r.content


# --- 4. main ---

async def run_rag_agent(query: str, pinecone_index) -> Dict[str, Any]:
    tool = await router(query)

    if tool == "pinecone_search":
        ctxs = await pinecone_search(query, pinecone_index)
        context = "\n".join(c.get("text","") for c in ctxs)

    elif tool == "web_search":
        context = web_search(query)

    else:
        context = "router error"

    answer = await answer_synthesizer(query, context)
    return {"answer": answer}
