from __future__ import annotations

from typing import Optional, Union, Literal, Any

from pydantic import BaseModel, ConfigDict


class _UIBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)


class UIStreamStartPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["start"] = "start"
    message_id: Optional[str] = None


class UITextStartPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["text-start"] = "text-start"
    id: str


class UITextDeltaPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class UITextEndPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["text-end"] = "text-end"
    id: str


class UIFinishMessagePart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["finish"] = "finish"


class UIErrorPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["error"] = "error"
    error_text: str


# Union must be declared after all classes are defined


# ---------------------------------------------------------------------------
# Tool-related stream parts
# ---------------------------------------------------------------------------


class StartStepPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["start-step"] = "start-step"


class FinishStepPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["finish-step"] = "finish-step"


class ToolInputStartPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["tool-input-start"] = "tool-input-start"
    tool_call_id: str
    tool_name: str


class ToolInputDeltaPart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["tool-input-delta"] = "tool-input-delta"
    tool_call_id: str
    input_text_delta: str


class ToolInputAvailablePart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["tool-input-available"] = "tool-input-available"
    tool_call_id: str
    tool_name: str
    input: Any


class ToolOutputAvailablePart(_UIBaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["tool-output-available"] = "tool-output-available"
    tool_call_id: str
    output: Any


AnyUIStreamPart = Union[
    UIStreamStartPart,
    UITextStartPart,
    UITextDeltaPart,
    UITextEndPart,
    UIFinishMessagePart,
    UIErrorPart,
    # Tool/step parts
    StartStepPart,
    FinishStepPart,
    ToolInputStartPart,
    ToolInputDeltaPart,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
]
