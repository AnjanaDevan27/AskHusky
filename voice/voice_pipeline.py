"""
Voice Pipeline — AskHusky
Pipecat pipeline wiring Deepgram Flux STT + Claude agents + ElevenLabs TTS.
Handles real-time voice conversation with built-in turn detection.
"""

import os
import re
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in path for agent imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -- Text Processing -----------------------------------------------------------

def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting before sending to TTS.
    ElevenLabs will speak markdown syntax literally otherwise.
    """
    text = re.sub(r'\*\*|__', '', text)       # bold
    text = re.sub(r'\*|_', '', text)           # italic
    text = re.sub(r'#{1,6}\s', '', text)       # headers
    text = re.sub(r'[^\x00-\x7F]', '', text)  # emojis and non-ASCII
    text = re.sub(r'\s+', ' ', text)           # collapse extra whitespace
    return text.strip()


def flatten_for_tts(text: str) -> str:
    """
    Flatten response into a single continuous block for TTS.
    Reduces the number of simultaneous ElevenLabs WebSocket contexts
    opened by Pipecat, which splits on punctuation and newlines.
    """
    text = text.replace('\n\n', '. ')
    text = text.replace('\n', ' ')
    text = text.replace('- ', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -- Self-transcription filter -------------------------------------------------

IGNORED_PHRASES = [
    "hello", "hi there", "welcome", "i'm here to help",
    "i'm ready", "how can i", "what can i",
    "based on the provided", "i notice your",
    "this sounds like", "please contact",
    "i don't have enough", "i'm ready to help",
    "to book an appointment", "it looks like your",
    "the context", "the provided context",
    "could you please", "i understand you",
    "you're right", "i don't have a previous",
    "mm", "uh", "um", "husky", "as husky",
    "great question", "good question",
    "i'm not sure", "here's how", "here's what",
    "just to recap", "of course", "sure!",
    "i have no more", "okay. uh,",
]


# -- Agent Processor -----------------------------------------------------------

class AskHuskyProcessor(FrameProcessor):
    """
    Bridges Pipecat's frame pipeline to our LangGraph agent system.
    Receives transcribed text from Deepgram Flux, routes through the
    Orchestrator, and returns the agent's response as text for TTS.

    Includes in-session conversation history so Husky can answer
    follow-up questions and repeat information from earlier in the chat.
    History resets when the pipeline stops.

    Interview talking point:
    "I built a custom Pipecat FrameProcessor that bridges streaming STT
    to a 7-agent LangGraph system. It maintains session-level conversation
    history so students can ask follow-up questions like 'can you repeat
    that number?' and Husky has full context to answer correctly."
    """

    def __init__(self):
        super().__init__()
        from agents.orchestrator import classify_intent
        from agents.cpt import CPTAgent
        from agents.opt import OPTAgent
        from agents.travel import TravelAgent
        from agents.visa_status import VisaStatusAgent
        from agents.coop import CoopAgent
        from agents.urgency import UrgencyAgent
        from agents.appointment import AppointmentAgent

        self._classify = classify_intent
        self._agents = {
            "cpt":         CPTAgent(),
            "opt":         OPTAgent(),
            "travel":      TravelAgent(),
            "visa_status": VisaStatusAgent(),
            "coop":        CoopAgent(),
            "urgency":     UrgencyAgent(),
            "appointment": AppointmentAgent(),
        }

        # In-session conversation history for context-aware responses.
        # Allows Husky to answer follow-ups like "can you repeat that number?"
        # Resets when the pipeline stops.
        self._conversation_history = []

        logger.info("AskHuskyProcessor initialized with all 7 agents")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import TranscriptionFrame, TextFrame

        if isinstance(frame, TranscriptionFrame):
            query = frame.text.strip()

            # Ignore empty or very short transcriptions from ambient noise
            if not query or len(query) < 15:
                return

            # Ignore TTS audio bleeding into the mic
            if any(query.lower().startswith(p) for p in IGNORED_PHRASES):
                logger.info(f"Ignoring self-transcription: '{query[:50]}'")
                return

            logger.info(f"Received query: '{query}'")

            # Route to correct agent via Orchestrator
            agent_name = self._classify(query)
            agent      = self._agents[agent_name]
            logger.info(f"Routing to: {agent_name}")

            # Get response from agent with conversation history for context
            response = agent.answer(query, history=self._conversation_history)

            # Strip disclaimer and markdown for TTS
            # Disclaimer is shown as text overlay in the UI
            response = strip_markdown(response.split("---")[0].strip())

            if not response:
                return

            # Flatten into single block to reduce concurrent TTS contexts
            response_flat = flatten_for_tts(response)

            # Update conversation history for next turn
            self._conversation_history.append({
                "role":    "user",
                "content": query
            })
            self._conversation_history.append({
                "role":    "assistant",
                "content": response
            })

            # Keep last 10 exchanges (20 messages) to stay within token limits
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

            logger.info(f"Sending to TTS ({len(response_flat)} chars)")
            await self.push_frame(TextFrame(response_flat))

        else:
            await self.push_frame(frame, direction)


# -- Pipeline ------------------------------------------------------------------

async def run_voice_pipeline():
    """
    Run the AskHusky voice pipeline locally.

    Pipeline:
        Mic -> Deepgram Flux STT -> AskHusky agents -> ElevenLabs TTS -> Speakers

    Key design decisions:
    - Warmup: embedding model and Pinecone pre-loaded before pipeline starts
      to eliminate 2-3 second first-query latency.
    - No Silero VAD: Deepgram Flux handles turn detection natively.
      Running both simultaneously causes conflicts and premature interruptions.
    - allow_interruptions=False: Agent speaks fully before listening again.
      Prevents TTS audio bleeding into mic from triggering false interruptions.
    - flatten_for_tts: Reduces concurrent ElevenLabs WebSocket contexts.
    - Conversation history: Passed to agents so Husky can answer follow-ups.
    - ElevenLabs Starter plan: Required for reliable audio — free tier caps
      at 5 concurrent WebSocket contexts which Pipecat exceeds on longer answers.
    """

    logger.info("Initializing AskHusky voice pipeline...")

    # Pre-load embedding model and Pinecone connection before pipeline starts.
    # Without this, first query takes 2-3 extra seconds while model loads.
    logger.info("Warming up retriever...")
    from rag.retriever import warmup
    warmup()
    logger.info("Retriever warmed up")

    # Transport — local audio, no VAD (Flux handles turn detection)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # STT — Deepgram Flux with conservative thresholds.
    # eot_timeout_ms=1500 is tighter now that interruptions are off —
    # the pipeline won't re-listen until TTS finishes anyway.
    stt = DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramFluxSTTService.Settings(
            model="flux-general-en",
            eager_eot_threshold=0.7,
            eot_threshold=0.9,
            eot_timeout_ms=1500,
            keyterm=[
                "CPT", "OPT", "SEVIS", "F-1", "I-20",
                "EAD", "STEM", "OGS", "co-op", "Northeastern",
                "curricular", "optional", "practical", "training",
                "Husky", "AskHusky",
            ],
        ),
    )

    # Agent processor — bridges STT transcriptions to LangGraph agents
    agent_processor = AskHuskyProcessor()

    # TTS — ElevenLabs streaming WebSocket, lowest latency model.
    # Requires Starter plan ($5/month) for reliable audio on longer responses.
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model="eleven_flash_v2_5",
            stability=0.5,
            similarity_boost=0.8,
        ),
    )

    # Wire the pipeline
    pipeline = Pipeline([
        transport.input(),   # mic audio in
        stt,                 # Deepgram Flux: audio -> text + turn detection
        agent_processor,     # AskHusky agents: text -> response text
        tts,                 # ElevenLabs: response text -> audio
        transport.output(),  # speakers audio out
    ])

    logger.info("AskHusky voice pipeline ready. Speak your question.")
    logger.info("Tip: use headphones to prevent mic/speaker feedback.")

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(allow_interruptions=False)
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_voice_pipeline())