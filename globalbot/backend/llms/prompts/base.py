from __future__ import annotations

import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from globalbot.backend.base import AnyMessage, HumanMessage, SystemMessage


class BasePrompt(BaseModel):
    template: str = ""

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def __call__(self, **kwargs: Any) -> str:
        return self.format(**kwargs)


class PromptTemplate(BasePrompt):
    input_variables: List[str] = Field(default_factory=list)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        variables = list(set(re.findall(r"\{(\w+)\}", template)))
        return cls(template=template, input_variables=variables)

    def format(self, **kwargs: Any) -> str:
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)


class ChatPromptTemplate(BaseModel):
    system: Optional[str] = None
    human: str = "{input}"
    input_variables: List[str] = Field(default_factory=list)

    @classmethod
    def from_messages(cls, system: Optional[str], human: str) -> "ChatPromptTemplate":
        variables = list(set(re.findall(r"\{(\w+)\}", (system or "") + human)))
        return cls(system=system, human=human, input_variables=variables)

    def format_messages(self, **kwargs: Any) -> List[AnyMessage]:
        messages: List[AnyMessage] = []
        if self.system:
            messages.append(SystemMessage(content=self.system.format(**kwargs)))
        messages.append(HumanMessage(content=self.human.format(**kwargs)))
        return messages

    def __call__(self, **kwargs: Any) -> List[AnyMessage]:
        return self.format_messages(**kwargs)
