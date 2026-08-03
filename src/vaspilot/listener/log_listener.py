# from crewai.utilities.events import (
#     CrewKickoffStartedEvent,
#     CrewKickoffCompletedEvent,
#     AgentExecutionCompletedEvent,
#     AgentExecutionStartedEvent,
#     ToolUsageStartedEvent,
#     ToolUsageFinishedEvent,
#     ToolUsageErrorEvent,
#     TaskEvaluationEvent,
# )

from crewai.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
    TaskEvaluationEvent,
    BaseEventListener,
)

from crewai.events.event_bus import CrewAIEventsBus

from typing import Dict, Any

# from crewai.utilities.events.base_event_listener import BaseEventListener
# from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    @abstractmethod
    def agent_message(self, agent_role, message):
        pass

    @abstractmethod
    def system_message(self, message):
        pass

    @abstractmethod
    def system_log(self, message):
        pass

    @abstractmethod
    def tool_log(self, tool_name, message, input:bool=True):
        pass

class LogListener(BaseEventListener):
    def __init__(self, logger: BaseLogger, crew_fingerprint:str , exclude_agents: list = [], exclude_tools: list = ["Delegate work to coworker"]):
        super().__init__()
        self.logger = logger
        self.exclude_agents = exclude_agents
        self.exclude_tools = exclude_tools
        self.crew_fingerprint = crew_fingerprint

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            if event.crew is not None:
                if event.crew.fingerprint.uuid_str == self.crew_fingerprint:
                    self.logger.system_log(f"Crew '{self.crew_fingerprint}' has started execution!")
            print(event.source_fingerprint, self.crew_fingerprint)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            if event.crew is not None:
                if event.crew.fingerprint.uuid_str == self.crew_fingerprint:
                    self.logger.system_log(f"Crew '{self.crew_fingerprint}' has finished execution!")

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            if source.crew is not None:
                if source.crew.fingerprint.uuid_str == self.crew_fingerprint:
                    self.logger.system_log(f"Agent '{event.agent.role}' started task")
                    if not event.agent.role in self.exclude_agents:
                        self.logger.system_message(f"'{event.agent.role}' recieved task:\n {event.task_prompt}")
                        self.logger.agent_message(event.agent.role, f"Input:\n {event.task_prompt}")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            if source.crew is not None:
                if source.crew.fingerprint.uuid_str == self.crew_fingerprint:
                    self.logger.system_log(f"Agent '{event.agent.role}' completed task")
                    self.logger.agent_message(event.agent.role, f"Output:\n {event.output}")

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            if source.agent is not None:
                if source.agent.crew is not None:
                    if source.agent.crew.fingerprint.uuid_str == self.crew_fingerprint:
                        self.logger.system_log(f"Tool '{event.tool_name}' started")
                        if not event.tool_name in self.exclude_tools:
                            self.logger.tool_log(event.tool_name, event.tool_args, input=True)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            if source.agent is not None:
                if source.agent.crew is not None:
                    if source.agent.crew.fingerprint.uuid_str == self.crew_fingerprint:
                        self.logger.system_log(f"Tool '{event.tool_name}' finished")
                        if not event.tool_name in self.exclude_tools:
                            self.logger.tool_log(event.tool_name, event.output, input=False)