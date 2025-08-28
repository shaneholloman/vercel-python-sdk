from .generate_text import generate_text, stream_text, StreamTextResult
from .providers.language_model import LanguageModel
from typing import List, Callable, Optional
from .tool import Tool
from .types import OnStepFinishResult


def print_step(step_info: OnStepFinishResult) -> None:
    print("🤖 Agent step info:")
    print(f"Step type: {step_info.step_type}")
    print(f"Finish reason: {step_info.finish_reason}")
    print(f"Tool calls: {step_info.tool_calls}")
    print(f"Tool results: {step_info.tool_results}")
    print(f"Output text: {step_info.text}")
    print(f"Response metadata: {step_info.response}")


class Agent:
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        system: str = "",
        tools: Optional[List[Tool]] = None,
        on_step: Optional[Callable[[OnStepFinishResult], None]] = None,
        max_steps: int = 100,
    ):
        self.name = name
        self.model = model
        self.system = system
        self.tools = tools or []
        self.on_step = on_step
        self.max_steps = max_steps

    def run(self, user_input: str) -> str:
        response = generate_text(
            model=self.model,
            system=self.system,
            prompt=user_input,
            tools=self.tools,
            on_step=self.on_step,
            max_steps=self.max_steps,
        )
        return response.text

    def stream(self, user_input: str) -> StreamTextResult:
        response = stream_text(
            model=self.model,
            system=self.system,
            prompt=user_input,
            tools=self.tools,
            on_step=self.on_step,
            max_steps=self.max_steps,
        )
        return response
