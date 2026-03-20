from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

from globalbot.backend.utils import LoggingMixin


class BaseComponent(LoggingMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: Optional[str] = None

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        yield self.run(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(*args, **kwargs))

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        result = await self.arun(*args, **kwargs)
        yield result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def pipe(self, other: "BaseComponent") -> "Pipeline":
        return Pipeline(steps=[self, other])

    def __or__(self, other: "BaseComponent") -> "Pipeline":
        return self.pipe(other)

    @classmethod
    def component_name(cls) -> str:
        return cls.__name__

    def describe(self) -> dict:
        return {
            "component": self.component_name(),
            "name": self.name,
            "config": self.model_dump(exclude={"name"}),
        }

    def as_langchain_runnable(self):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.run)


class Pipeline(BaseComponent):
    steps: list = Field(default_factory=list)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.log.info("pipeline.run.start", n_steps=len(self.steps))
        result = self.steps[0].run(*args, **kwargs)
        for i, step in enumerate(self.steps[1:], 2):
            self.log.debug("pipeline.step", step=i, component=type(step).__name__)
            result = step.run(result)
        self.log.info("pipeline.run.end", n_steps=len(self.steps))
        return result

    def pipe(self, other: BaseComponent) -> "Pipeline":
        return Pipeline(steps=self.steps + [other])
