"""
Base Agent — AskHusky
Shared base class for all specialized agents.
Each agent inherits this and overrides domain_prompt.
Includes conversation memory for context within a session.
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

# -- Shared Husky Persona ------------------------------------------------------

HUSKY_PERSONA = """
You are Husky, AskHusky's AI advisor for Northeastern University F-1 international students.

Your personality:
- Professional and conversational — students should feel comfortable, not like they are talking to a robot
- Warm and reassuring — many students asking these questions are anxious about their visa status
- Direct and clear — get to the point, avoid unnecessary filler phrases
- Honest — if you do not have the answer, say so clearly and direct the student to OGS

Response style:
- Never start with "Based on the provided OGS context" or similar robotic phrases
- Never start with "Great question!" or "Good question!" — get straight to the answer
- Keep responses concise and conversational — this is a voice interface, not an essay
- Use plain language first, then add detail if needed
- Speak in complete sentences suitable for text-to-speech

When you do not have enough information to answer:
- Acknowledge the question warmly
- Be honest that you do not have that specific information
- Direct the student to OGS: call 617-373-2310, Monday to Friday 8:30am to 4:30pm ET
- For urgent situations outside OGS hours, mention GSOC is available 24/7

When answering follow-up questions:
- Use the conversation history to maintain context
- If a student asks you to repeat something, repeat it clearly and concisely
- If a student references something from earlier in the conversation, acknowledge it naturally

Only answer based on the provided OGS context. Never make up immigration rules or timelines.
""".strip()


class BaseAgent:
    """
    Base class for all AskHusky specialized agents.

    Each subclass must define:
        - name: str — agent identifier used in logs and routing
        - domain_prompt: str — domain-specific instructions appended to Husky persona

    Interview talking point:
    "I used a shared base agent with a common Husky persona and a domain_prompt
    override pattern. This keeps personality consistent across all 7 agents while
    letting each specialize in its area. The answer() method retrieves OGS context
    from Pinecone, passes conversation history for session memory, and wraps
    the response in a safety layer that adds the OGS disclaimer."
    """

    name: str = "base"
    domain_prompt: str = ""

    @property
    def system_prompt(self) -> str:
        return f"{HUSKY_PERSONA}\n\n{self.domain_prompt}"

    def answer(self, query: str, history: list[dict] | None = None) -> str:
        """
        Retrieve relevant OGS context and generate a grounded answer.
        Accepts optional conversation history for session context.

        Flow:
            1. Retrieve top 8 OGS chunks from Pinecone
            2. Format as context string
            3. Build message list with history + current query
            4. Call Claude Haiku with system prompt + messages
            5. Wrap response in safety layer (disclaimer + GSOC routing)
        """
        logger.info(f"[{self.name}] Processing: '{query[:80]}'")

        # Step 1 — Retrieve context from Pinecone
        chunks  = retrieve(query)
        context = format_context(chunks)

        # Step 2 — Build messages with conversation history
        messages = []

        if history:
            messages.extend(history)

        # Current user message with OGS context injected
        user_message = f"""Use the following OGS content to answer the student's question.
Only use information from the provided context.
If the context does not contain the answer, say so honestly and direct the student to OGS at 617-373-2310.

--- OGS CONTEXT ---
{context}
--- END CONTEXT ---

Student question: {query}"""

        messages.append({"role": "user", "content": user_message})

        # Step 3 — Generate answer
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=self.system_prompt,
            messages=messages
        )

        answer = response.content[0].text.strip()
        logger.info(f"[{self.name}] Answer generated ({len(answer)} chars)")

        # Step 4 — Apply safety layer
        return safe_response(query, answer)