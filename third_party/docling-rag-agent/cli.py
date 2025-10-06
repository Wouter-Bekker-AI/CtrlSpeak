"""
Command Line Interface for Docling RAG Agent.

Enhanced CLI with colors, formatting, and improved user experience.
"""

import asyncio
import argparse
import logging
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from pydantic_ai import Agent, RunContext

from ingestion.embedder import create_embedder
from utils.db_utils import get_client
from utils.providers import get_llm_model

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
logger = logging.getLogger(__name__)

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

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
                metas = extra.get("metadatas") or []
                if docs:
                    sources_seen.add(overview_path)
                    response_parts.insert(0, f"[Source: {overview_path}]\n{docs[0]}\n")

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


SYSTEM_PROMPT = """You are an intelligent knowledge assistant with access to an organization's documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: Always search the knowledge base before answering questions about specific information.
If information isn't in the knowledge base, clearly state that and offer general guidance.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents."""

agent = Agent(
    get_llm_model(),
    system_prompt=SYSTEM_PROMPT,
)


class RAGAgentCLI:
    """Enhanced CLI for interacting with the RAG Agent."""

    def __init__(self):
        """Initialize CLI."""
        self.message_history = []

    def print_banner(self):
        """Print welcome banner."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
        print("ðŸ¤– Docling RAG Knowledge Assistant")
        print("=" * 60)
        print(f"{Colors.WHITE}AI-powered document search with streaming responses")
        print(f"Type 'exit', 'quit', or Ctrl+C to exit")
        print(f"Type 'help' for commands")
        print("=" * 60 + f"{Colors.END}\n")

    def print_help(self):
        """Print help information."""
        help_text = f"""
{Colors.BOLD}Available Commands:{Colors.END}
  {Colors.GREEN}help{Colors.END}           - Show this help message
  {Colors.GREEN}clear{Colors.END}          - Clear conversation history
  {Colors.GREEN}stats{Colors.END}          - Show conversation statistics
  {Colors.GREEN}exit/quit{Colors.END}      - Exit the CLI

{Colors.BOLD}Usage:{Colors.END}
  Simply type your question and press Enter to chat with the agent.
  The agent will search the knowledge base and provide answers with source citations.

{Colors.BOLD}Features:{Colors.END}
  â€¢ Semantic search through embedded documents
  â€¢ Streaming responses in real-time
  â€¢ Conversation history maintained across turns
  â€¢ Source citations for all information

{Colors.BOLD}Examples:{Colors.END}
  - "What are the main topics in the knowledge base?"
  - "Tell me about [specific topic from your documents]"
  - "Summarize information about [subject]"
"""
        print(help_text)

    def print_stats(self):
        """Print conversation statistics."""
        message_count = len(self.message_history)
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}ðŸ“Š Session Statistics:{Colors.END}")
        print(f"  Messages in history: {message_count}")
        print(f"  Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.BLUE}{'â”€' * 60}{Colors.END}\n")

    def check_database(self) -> bool:
        """Check database connection."""
        try:
            chroma_client = get_client()
            collections = chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            if "rag_collection" in collection_names:
                collection = chroma_client.get_collection(name="rag_collection")
                count = collection.count()
                print(f"{Colors.GREEN}âœ“ Knowledge base ready: {count} chunks{Colors.END}")
            else:
                print(f"{Colors.YELLOW}âœ— Knowledge base not found. Please ingest documents first.{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}âœ— Database connection failed: {e}{Colors.END}")
            return False

    def extract_tool_calls(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool call information from messages."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        tools_used = []
        for msg in messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tools_used.append({
                            'tool_name': part.tool_name,
                            'args': part.args,
                            'tool_call_id': part.tool_call_id
                        })
        return tools_used

    def format_tools_used(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools used for display."""
        if not tools:
            return ""

        formatted = f"\n{Colors.MAGENTA}{Colors.BOLD}ðŸ›  Tools Used:{Colors.END}\n"
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('tool_name', 'unknown')
            args = tool.get('args', {})

            formatted += f"  {Colors.CYAN}{i}. {tool_name}{Colors.END}"

            if args and isinstance(args, dict):
                key_args = []
                if 'query' in args:
                    query_preview = str(args['query'])[:50] + '...' if len(str(args['query'])) > 50 else str(args['query'])
                    key_args.append(f"query='{query_preview}'")
                if 'limit' in args:
                    key_args.append(f"limit={args['limit']}")

                if key_args:
                    formatted += f" ({ ', '.join(key_args) })"

            formatted += "\n"

        return formatted



    @staticmethod
    def extract_sources_from_context_text(context: str) -> list[str]:
        """Return unique source paths parsed from context string."""
        if not context:
            return []

        sources: list[str] = []
        for match in re.findall(r'\[Source:\s*([^\]]+)\]', context):
            source = match.strip()
            if source not in sources:
                sources.append(source)
        return sources


    async def stream_chat(self, message: str) -> None:
        try:
            context = search_knowledge_base(None, message)
            enriched_message = message
            if context and "No relevant information found" not in context:
                enriched_message = (
                    f"{message}\n\n"
                    f"Context from knowledge base:\n{context}\n\n"
                    "Respond using only the context above. Every statement must reference the matching [Source: ...] citation."
                    " End your reply with a 'Sources:' section listing each path you cited."
                )
            elif context:
                enriched_message = (
                    f"{message}\n\n"
                    "The knowledge base did not return relevant context. If you cannot answer from general knowledge, say so clearly."
                )
            else:
                enriched_message = message


            print(f"\n{Colors.BOLD}Assistant:{Colors.END} ", end="", flush=True)

            async with agent.run_stream(
                enriched_message,
                message_history=self.message_history
            ) as result:
                async for text in result.stream_text(delta=True):
                    print(text, end="", flush=True)

                print()

                self.message_history = result.all_messages()

            sources = self.extract_sources_from_context_text(context or "")
            if sources:
                print("\nSources:")
                for src in sources:
                    print(f"- {src}")

            print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")

        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}")
            logger.error(f"Chat error: {e}", exc_info=True)

    async def run(self):
        """Run the CLI main loop."""
        self.print_banner()

        if not self.check_database():
            return

        print(f"{Colors.GREEN}Ready to chat! Ask me anything about the knowledge base.{Colors.END}\n")

        try:
            while True:
                try:
                    user_input = input(f"{Colors.BOLD}You: {Colors.END}").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print(f"{Colors.CYAN}ðŸ‘‹ Thank you for using the knowledge assistant. Goodbye!{Colors.END}")
                        break
                    elif user_input.lower() == 'help':
                        self.print_help()
                        continue
                    elif user_input.lower() == 'clear':
                        self.message_history = []
                        print(f"{Colors.GREEN}âœ“ Conversation history cleared{Colors.END}")
                        continue
                    elif user_input.lower() == 'stats':
                        self.print_stats()
                        continue

                    await self.stream_chat(user_input)

                except KeyboardInterrupt:
                    print(f"\n{Colors.CYAN}ðŸ‘‹ Goodbye!{Colors.END}")
                    break
                except EOFError:
                    print(f"\n{Colors.CYAN}ðŸ‘‹ Goodbye!{Colors.END}")
                    break

        except Exception as e:
            print(f"{Colors.RED}âœ— CLI error: {e}{Colors.END}")
            logger.error(f"CLI error: {e}", exc_info=True)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI for Docling RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--model',
        default=None,
        help='Override LLM model (e.g., gemma3:1b)'
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.verbose:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)

    if args.model:
        model_name = args.model
    else:
        model_name = "gemma3:1b"

    global agent
    agent = Agent(
        get_llm_model(model_name),
        system_prompt=SYSTEM_PROMPT,
    )

    cli = RAGAgentCLI()

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}ðŸ‘‹ Goodbye!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}âœ— CLI startup error: {e}{Colors.END}")
        logger.error(f"Startup error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
























