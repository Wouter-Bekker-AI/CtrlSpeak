"""
RAG CLI Agent with ChromaDB
=======================================
Text-based CLI agent that searches through knowledge base using semantic similarity
"""

import asyncio
import logging
import re
import os
from pathlib import Path
import sys

TARGET_DOCS = {
    "mission-and-goals": Path("documents/mission-and-goals.md"),
    "mission and goals": Path("documents/mission-and-goals.md"),
    "docling-rag-agent-overview": Path("documents/docling-rag-agent-overview.md"),
    "docling rag agent overview": Path("documents/docling-rag-agent-overview.md"),
    "implementation-playbook": Path("documents/implementation-playbook.md"),
    "implementation playbook": Path("documents/implementation-playbook.md"),
    "meeting-notes-2025-01-15": Path("documents/meeting-notes-2025-01-15.docx"),
    "meeting notes 2025-01-15": Path("documents/meeting-notes-2025-01-15.docx"),
    "meeting-notes-2025-01-08": Path("documents/meeting-notes-2025-01-08.docx"),
    "meeting notes 2025-01-08": Path("documents/meeting-notes-2025-01-08.docx"),
    "client-review-globalfinance": Path("documents/client-review-globalfinance.pdf"),
    "client review globalfinance": Path("documents/client-review-globalfinance.pdf"),
}


def _match_specific_document(query: str) -> Path | None:
    lower_query = query.lower()
    for alias, path in TARGET_DOCS.items():
        if alias in lower_query:
            return path.resolve()
    return None
from pydantic_ai import Agent, RunContext

from ingestion.embedder import create_embedder
from utils.db_utils import get_client
from utils.providers import get_llm_model

logger = logging.getLogger(__name__)



def extract_sources_from_context(context: str) -> list[str]:
    """Parse unique source paths from the context string."""
    if not context:
        return []

    sources: list[str] = []
    for match in re.findall(r'\[Source:\s*([^\]]+)\]', context):
        source = match.strip()
        if source not in sources:
            sources.append(source)
    return sources

def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 8) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 8)

    Returns:
        Formatted search results with source citations
    """
    try:
        chroma_client = get_client()
        collection = chroma_client.get_collection(name="rag_collection")
        embedder = create_embedder()

        query_embedding = embedder.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        matched_path = _match_specific_document(query)
        if matched_path is not None:
            try:
                filtered = collection.get(where={"source": str(matched_path)})
            except Exception as extra_err:
                logger.warning(f"Failed to retrieve scoped context for {matched_path}: {extra_err}")
            else:
                docs = filtered.get("documents") or []
                if docs:
                    ids = filtered.get("ids") or []
                    metas = filtered.get("metadatas") or []
                    results = {
                        "ids": [ids],
                        "documents": [docs],
                        "metadatas": [metas],
                    }

        if not results or not results["documents"][0]:
            return "No relevant information found in the knowledge base for your query."

        response_parts: list[str] = []
        sources_seen: set[str] = set()
        for i, doc in enumerate(results["documents"][0]):
            source = results["metadatas"][0][i]["source"]
            sources_seen.add(source)
            response_parts.append(f"[Source: {source}]\n{doc}\n")

        overview_path = str((Path(__file__).resolve().parent / "documents" / "docling-rag-agent-overview.md").resolve())
        if ("docling" in query.lower() and "rag" in query.lower() and overview_path not in sources_seen):
            try:
                extra = collection.get(where={"source": overview_path})
            except Exception as extra_err:
                logger.warning(f"Failed to append overview context: {extra_err}")
            else:
                docs = extra.get("documents") or []
                if docs:
                    sources_seen.add(overview_path)
                    response_parts.insert(0, f"[Source: {overview_path}]\n{docs[0]}\n")

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


# Create the PydanticAI agent with the RAG tool
agent = Agent(
    get_llm_model(),
    system_prompt="""You are an intelligent knowledge assistant with access to an organization's documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: Always search the knowledge base before answering questions about specific information.
If information isn't in the knowledge base, clearly state that and offer general guidance.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents.""",
)


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    print("=" * 60)
    print("RAG Knowledge Assistant")
    print("=" * 60)
    print("Ask me anything about the knowledge base!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            context = search_knowledge_base(None, user_input)
            enriched_message = user_input
            if context and "No relevant information found" not in context:
                enriched_message = (
                    f"{user_input}\n\n"
                    f"Context from knowledge base:\n{context}\n\n"
                    "Respond using only the context above. Every statement must reference the matching [Source: ...] citation."
                    " End your reply with a 'Sources:' section listing each path you cited."
                )
            elif context:
                enriched_message = (
                    f"{user_input}\n\n"
                    "The knowledge base did not return relevant context. If you cannot answer from general knowledge, say so clearly."
                )
            else:
                enriched_message = user_input


            print("Assistant: ", end="", flush=True)

            try:
                async with agent.run_stream(
                    enriched_message,
                    message_history=message_history
                ) as result:
                    async for text in result.stream_text(delta=True):
                        print(text, end="", flush=True)

                    print()  # New line after streaming completes

                    message_history = result.all_messages()

                    sources = extract_sources_from_context(context or "")
                    if sources:
                        print("\nSources:")
                        for src in sources:
                            print(f"- {src}")

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()  # Extra line for readability

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")










