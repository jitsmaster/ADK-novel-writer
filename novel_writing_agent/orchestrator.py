"""Orchestrator module for novel writing agent.

Handles data validation, prompt generation and output standardization.
"""
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CharacterProfile(BaseModel):
    """Data model for character profile."""
    name: str
    archetype: str
    traits: List[str]
    flaws: List[str]
    background: str
    motivation: str

class StylePreferences(BaseModel):
    """Data model for writing style preferences."""
    tone: str = "neutral"
    pacing: str = "medium"
    point_of_view: str = "third_person"
    dialogue_ratio: float = 0.3

    @validator('tone')
    def validate_tone(cls, v):
        valid_tones = ["formal", "casual", "poetic", "dramatic", "neutral"]
        if v not in valid_tones:
            raise ValueError(f"Invalid tone. Must be one of: {valid_tones}")
        return v

class StoryInput(BaseModel):
    """Data model for story input validation."""
    plot: str
    characters: List[CharacterProfile]
    style_preferences: StylePreferences
    target_length: Optional[int] = 1000

    @validator('target_length')
    def validate_length(cls, v):
        if v is not None and v < 100:
            raise ValueError("Target length must be at least 100 words")
        return v

class PromptGenerator:
    """Generates standardized prompts for the writing engine."""
    
    def __init__(self):
        self.templates = {
            'basic': (
                "Write a chapter based on this plot: {plot}\n\n"
                "Characters:\n{characters}\n\n"
                "Style guidelines: {style}"
            ),
            'advanced': (
                "Compose a {style[tone]}-toned chapter of approximately {target_length} words.\n"
                "Plot Summary: {plot}\n\n"
                "Character Details:\n{character_details}\n\n"
                "Additional Requirements:\n"
                "- Pacing: {style[pacing]}\n"
                "- Point of View: {style[point_of_view]}\n"
                "- Dialogue Ratio: ~{style[dialogue_ratio]*100}%"
            )
        }

    def generate(self, story_data: Dict, strategy: str = 'advanced') -> Dict:
        """Generate standardized writing prompt.
        
        Args:
            story_data: Validated story input data
            strategy: Prompt generation strategy ('basic' or 'advanced')
            
        Returns:
            Standardized prompt in JSON format
        """
        try:
            # Convert characters to formatted string
            char_details = "\n".join(
                f"- {char.name} ({char.archetype}): {', '.join(char.traits)}. "
                f"Motivation: {char.motivation}"
                for char in story_data.characters
            )

            prompt_template = self.templates[strategy]
            
            prompt_text = prompt_template.format(
                plot=story_data.plot,
                characters=char_details,
                style=story_data.style_preferences.dict(),
                target_length=story_data.target_length,
                character_details=char_details
            )

            return {
                "task_type": "chapter_generation",
                "context": {
                    "plot": story_data.plot,
                    "characters": [char.dict() for char in story_data.characters]
                },
                "style": story_data.style_preferences.dict(),
                "requirements": {
                    "length": story_data.target_length,
                    "focus_points": []  # Can be populated from analysis
                },
                "raw_prompt": prompt_text
            }
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            raise

class Orchestrator:
    """Main orchestrator class handling end-to-end processing."""
    
    def __init__(self, tool_manager=None):
        self.input_processor = StoryInput
        self.prompt_engine = PromptGenerator()
        self.tool_manager = tool_manager
        
    def generate_first_chapter(self) -> Dict:
        """
        Generate a complete first chapter by:
        1. Developing 3 characters (hero, mentor, villain)
        2. Generating a fantasy plot idea
        3. Formatting as StoryInput
        4. Generating the chapter content
        
        Returns:
            Dictionary with chapter content and metrics
        """
        if not self.tool_manager:
            raise ValueError("Tool manager must be provided to generate a chapter")
            
        # Step 1: Develop characters
        hero = self.tool_manager.develop_character("hero")
        if hero["status"] != "success":
            raise ValueError(f"Failed to develop hero character: {hero.get('error_message', '')}")
            
        mentor = self.tool_manager.develop_character("mentor")
        if mentor["status"] != "success":
            raise ValueError(f"Failed to develop mentor character: {mentor.get('error_message', '')}")
            
        villain = self.tool_manager.develop_character("villain")
        if villain["status"] != "success":
            raise ValueError(f"Failed to develop villain character: {villain.get('error_message', '')}")
        
        # Step 2: Generate plot
        plot_data = self.tool_manager.generate_plot_idea("fantasy")
        if plot_data["status"] != "success":
            raise ValueError(f"Failed to generate plot: {plot_data.get('error_message', '')}")
        
        # Step 3: Create StoryInput
        story_input = StoryInput(
            plot=plot_data["idea"],  # Fixed: using "idea" instead of "plot"
            characters=[
                CharacterProfile(**hero["character"]),  # Fixed: accessing the character data
                CharacterProfile(**mentor["character"]),
                CharacterProfile(**villain["character"])
            ],
            style_preferences=StylePreferences(),
            target_length=2000
        )
        
        # Step 4: Generate chapter using the tool manager
        result = self.tool_manager.generate_chapter(story_input.dict())
        if result["status"] != "success":
            raise ValueError(f"Failed to generate chapter: {result.get('error_message', '')}")
            
        return result["chapter"]
    def process(self, raw_input: Union[Dict, str]) -> Dict:
        """Process raw story input into standardized prompt.
        
        Args:
            raw_input: Either JSON string or dict with story data
            
        Returns:
            Standardized prompt for writing engine
        """
        try:
            # Parse input if it's JSON string
            if isinstance(raw_input, str):
                input_data = json.loads(raw_input)
            else:
                input_data = raw_input
            
            # Validate input
            validated = self.input_processor(**input_data)
            logger.info("Input validation successful")
            
            # Generate standardized prompt
            prompt = self.prompt_engine.generate(validated)
            logger.info(f"Generated prompt for {len(validated.characters)} characters")
            
            return prompt
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {str(e)}")
            raise ValueError("Invalid JSON format") from e
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise