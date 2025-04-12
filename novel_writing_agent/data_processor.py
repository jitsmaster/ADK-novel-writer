"""Data processing agent for novel writing workflow."""

from typing import Dict, Any
from pydantic import BaseModel
from .agent import Agent
from .orchestrator import StoryInput

class DataProcessor(Agent):
    """Processes raw input data into standardized format."""
    
    def __init__(self):
        super().__init__(
            name="data_processor",
            description="Validates and processes raw story input data"
        )
        self.validator = StoryInput
        
    def process(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and standardize input data.
        
        Args:
            raw_input: Raw story input data
            
        Returns:
            Standardized processed data
        """
        try:
            validated = self.validator(**raw_input)
            return {
                "plot": validated.plot,
                "characters": [char.dict() for char in validated.characters],
                "style": validated.style_preferences.dict(),
                "target_length": validated.target_length
            }
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            raise ValueError(f"Invalid input data: {str(e)}")