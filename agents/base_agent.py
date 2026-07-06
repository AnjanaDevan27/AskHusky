"""
Base Agent — AskHusky
Shared base class for all specialized agents.
Each agent inherits this and overrides the system prompt.
"""

import os
import logging
from anthropic import Anthropic
from dotenv import load_dotenv
from rag.retriever import retrieve, format_context
from rag.safety import safe_response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = Anthropic()


class BaseAgent:
    """
    Base class for all AskHusky specialized agents.

    Each subclass must define:
        - name: str — agent identifier
        - system_prompt: str — domain-specific instructions
    """

    name: str = "base"
    system_prompt: str = ""

    def answer(self, query: str) -> str:
        """
        Retrieve relevant OGS context and generate a grounded answer.

        Flow:
            1. Retrieve top 5 OGS chunks from Pinecone
            2. Format as context string
            3. Call Claude with context + query
            4. Wrap response in safety layer
        """
        logger.info(f"[{self.name}] Processing: '{query[:80]}'")

        # Step 1 — Retrieve context
        chunks  = retrieve(query)
        context = format_context(chunks)

        # Step 2 — Build prompt
        user_message = f"""Use the following OGS content to answer the student's question.
Only use information from the provided context.
If the context does not contain enough information, say so clearly.

--- OGS CONTEXT ---
{context}
--- END CONTEXT ---

Student question: {query}"""

        # Step 3 — Generate answer
        response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        system=self.system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

        answer = response.content[0].text.strip()
        logger.info(f"[{self.name}] Answer generated ({len(answer)} chars)")

        # Step 4 — Apply safety layer
        return safe_response(query, answer)