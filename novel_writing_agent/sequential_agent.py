"""Sequential agent orchestrator for novel writing workflow."""

from typing import Dict, Any, List
from dataclasses import dataclass
from .agent import Agent
from .orchestrator import Orchestrator

@dataclass
class WorkflowStep:
    """Represents a step in the sequential workflow."""
    agent: Agent
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]

class SequentialAgent(Agent):
    """Main orchestrator that executes agents in sequence with context passing."""
    
    def __init__(self, workflow_config: Dict[str, Any]):
        super().__init__(
            name="sequential_agent",
            description="Orchestrates multi-agent novel writing workflow"
        )
        self.workflow = self._build_workflow(workflow_config)
        self.orchestrator = Orchestrator()
        
    def _build_workflow(self, config: Dict[str, Any]) -> List[WorkflowStep]:
        """Build workflow steps from configuration."""
        steps = []
        for step_config in config["steps"]:
            agent = self._init_agent(step_config["agent_type"])
            steps.append(WorkflowStep(
                agent=agent,
                input_mapping=step_config.get("input_mapping", {}),
                output_mapping=step_config.get("output_mapping", {})
            ))
        return steps
        
    def _init_agent(self, agent_type: str) -> Agent:
        """Initialize agent instance based on type."""
        # This would be replaced with proper dependency injection
        if agent_type == "data_processor":
            from .data_processor import DataProcessor
            return DataProcessor()
        elif agent_type == "analyzer":
            from .analyzer import Analyzer
            return Analyzer()
        elif agent_type == "report_generator":
            from .report_generator import ReportGenerator
            return ReportGenerator()
        elif agent_type == "writer":
            from .writer import Writer
            return Writer(self.model)
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with input data.
        
        Args:
            input_data: Raw input data for the workflow
            
        Returns:
            Final output from the workflow execution
        """
        context = {"input": input_data}
        
        for step in self.workflow:
            # Prepare step input from context using mapping
            step_input = {
                target: context[source]
                for target, source in step.input_mapping.items()
                if source in context
            }
            
            # Execute step
            result = step.agent.process(step_input)
            
            # Update context with outputs using mapping
            for target, source in step.output_mapping.items():
                if source in result:
                    context[target] = result[source]
                    
        return context.get("final_output", {})