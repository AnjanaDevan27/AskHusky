"""
Voice Pipeline — AskHusky
Pipecat pipeline wiring Deepgram Flux STT + Claude agents + ElevenLabs TTS.
Handles real-time voice conversation with built-in turn detection.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.base_transport import TransportParams

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -- Agent Processor -----------------------------------------------------------

class AskHuskyProcessor(FrameProcessor):
    """
    Bridges Pipecat's frame pipeline to our LangGraph agent system.
    Receives transcribed text from Deepgram, routes through the
    Orchestrator, and returns the agent's response as text for TTS.
    """

    def __init__(self):
        super().__init__()
        # Import here to avoid circular imports
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
            "cpt":          CPTAgent(),
            "opt":          OPTAgent(),
            "travel":       TravelAgent(),
            "visa_status":  VisaStatusAgent(),
            "coop":         CoopAgent(),
            "urgency":      UrgencyAgent(),
            "appointment":  AppointmentAgent(),
        }
        logger.info("AskHuskyProcessor initialized with all 7 agents")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import TranscriptionFrame, TextFrame

        if isinstance(frame, TranscriptionFrame):
            query = frame.text.strip()
            if not query:
                return

            logger.info(f"Received query: '{query}'")

            # Route to correct agent
            agent_name = self._classify(query)
            agent      = self._agents[agent_name]

            logger.info(f"Routing to: {agent_name}")

            # Get response from agent (includes safety layer)
            response = agent.answer(query)

            # Strip disclaimer for TTS — students hear the answer,
            # disclaimer is shown in the UI text overlay instead
            response = response.split("---")[0].strip()

            logger.info(f"Sending response to TTS ({len(response)} chars)")

            # Push response as text for ElevenLabs to speak
            await self.push_frame(TextFrame(response))
        else:
            await self.push_frame(frame, direction)


# -- Pipeline ------------------------------------------------------------------

async def run_voice_pipeline():
    """
    Run the AskHusky voice pipeline locally.
    Deepgram Flux STT -> AskHusky agents -> ElevenLabs TTS
    """

    logger.info("Initializing AskHusky voice pipeline...")

    # Transport — local audio (mic in, speakers out)
    transport = LocalAudioTransport(
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # STT — Deepgram Flux with built-in turn detection
    stt = DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramFluxSTTService.Settings(
            model="flux-general-en",
            eager_eot_threshold=0.5,   # faster response triggering
            eot_threshold=0.8,
            keyterm=[                   # boost immigration-specific terms
                "CPT", "OPT", "SEVIS", "F-1", "I-20",
                "EAD", "STEM", "OGS", "co-op", "Northeastern"
            ],
        ),
    )

    # Agent processor — bridges STT to our LangGraph agents
    agent_processor = AskHuskyProcessor()

    # TTS — ElevenLabs streaming WebSocket
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model="eleven_flash_v2_5",  # lowest latency ElevenLabs model
            stability=0.5,
            similarity_boost=0.8,
        ),
    )

    # Wire the pipeline
    pipeline = Pipeline([
        transport.input(),    # mic audio in
        stt,                  # Deepgram Flux: audio -> text + turn detection
        agent_processor,      # AskHusky agents: text -> response text
        tts,                  # ElevenLabs: response text -> audio
        transport.output(),   # speakers audio out
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True)
    )

    logger.info("AskHusky voice pipeline ready. Speak your question.")

    runner = PipelineRunner()
    await runner.run(task)


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_voice_pipeline())