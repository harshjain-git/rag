# agent/tools/

"""Tool adapter package.

Contains adapter classes that wrap agent capabilities into the ``core.ports.Tool``
protocol so they can be registered in the ``ToolRegistry`` and invoked generically
by the ``AgentPipeline``.
"""

from .orchestrator_tool import OrchestratorTool
from .retrieval_adapter import RetrievalTool
from .generation_tool import GenerationTool
from .verification_tool import VerificationTool
from .query_rewriter_adapter import QueryRewriterTool
from .question_generator_adapter import QuestionGeneratorTool

__all__ = [
    "OrchestratorTool",
    "RetrievalTool",
    "GenerationTool",
    "VerificationTool",
    "QueryRewriterTool",
    "QuestionGeneratorTool",
]
