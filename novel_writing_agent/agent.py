import json
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from functools import wraps
import inspect
import logging
from collections import UserList
from typing import Dict, Any, Optional
from .orchestrator import Orchestrator
from .writer import Writer

class ToolManager(UserList):
    def __init__(self, tools):
        super().__init__(tools)
        self._tools_map = {tool.name: tool for tool in tools}
        
    def __getattr__(self, name):
        if name in self._tools_map:
            # Return the actual function reference
            return self._tools_map[name]
        raise AttributeError(f"No tool named '{name}'")

# Tool registry list
tool_registry = []

# Tool decorator to define metadata and register tools
def tool(description=None):
    """Decorator for registering tool functions with metadata.
    
    Args:
        description (str, optional): Description of what the tool does.
            If None, the function's docstring will be used.
    
    Returns:
        function: Decorated function with metadata attached.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Extract function metadata
        func_name = func.__name__
        func_doc = func.__doc__ or ""
        func_signature = inspect.signature(func)
        
        # Use provided description or function docstring
        wrapper.description = description or func_doc.strip()
        wrapper.name = func_name
        wrapper.parameters = {
            name: {
                "type": param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any",
                "description": "",  # Could be enhanced with param docs extraction
                "required": param.default == inspect.Parameter.empty
            }
            for name, param in func_signature.parameters.items()
        }
        
        # Register the tool
        tool_registry.append(wrapper)
        return wrapper
    return decorator

# Define functions for the novel writing assistant tools with decorator
@tool(description="Generates a plot idea for a specified genre.")
def generate_plot_idea(genre: str, description: Optional[str] = None) -> dict:
    """Generates a plot idea for a specified genre using LLM.
    
    Args:
        genre: The genre of the plot idea (e.g. fantasy, sci-fi)
        description: Optional additional description to guide the plot generation
    
    Args:
        genre (str): The genre for which to generate a plot idea.
        
    Returns:
        dict: status and result or error msg.
    """
    try:
        response = root_agent.model.complete(
            prompt=f"Generate an original and creative plot idea for a {genre} story. "
                   "The idea should be 1-2 sentences long and contain an interesting conflict or twist.",
            temperature=0.8
        )
        return {
            "status": "success",
            "idea": response.choices[0].text.strip()
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to generate plot idea: {str(e)}"
        }

@tool(description="Develops a character based on a specified archetype.")
def develop_character(archetype: str) -> dict:
    """Develops a character based on a specified archetype using LLM.
    
    Args:
        archetype (str): The character archetype to develop.
        
    Returns:
        dict: status and character details or error msg.
    """
    try:
        response = root_agent.model.complete(
            prompt=f"Develop a detailed character profile for a {archetype} archetype. "
                   "Include traits, flaws, background and motivation in JSON format.",
            temperature=0.7
        )
        return {
            "status": "success",
            "character": json.loads(response.choices[0].text.strip())
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to develop character: {str(e)}"
        }

@tool(description="Provides a writing prompt with suggested word count.")
def writing_prompt(word_count: int) -> dict:
    """Provides a writing prompt with suggested word count using LLM.
    
    Args:
        word_count (int): The target word count for the prompt.
        
    Returns:
        dict: status and prompt or error msg.
    """
    try:
        response = root_agent.model.complete(
            prompt=f"Generate a creative writing prompt suitable for a story of about {word_count} words. "
                   "The prompt should include an interesting premise and potential conflict.",
            temperature=0.7
        )
        return {
            "status": "success",
            "prompt": response.choices[0].text.strip(),
            "target_word_count": word_count
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to generate writing prompt: {str(e)}"
        }

# Initialize core components
orchestrator = Orchestrator()
writer = Writer(model=LiteLlm(
    model="ollama/gemma3:4b",
    api_base="http://localhost:11434",
    timeout=60,
    temperature=0.7
))

@tool(description="Generate a complete chapter from structured story data.")
def generate_chapter(story_data: Dict[str, Any]) -> Dict:
    """Generate a complete chapter using orchestrator and writer pipeline.
    
    Args:
        story_data: Dictionary containing plot, characters and style preferences
        
    Returns:
        Dictionary with chapter content and metrics
    """
    try:
        # Process through orchestrator - ensure orchestrator has the tool_manager
        if not orchestrator.tool_manager:
            orchestrator.tool_manager = root_agent.tools
            
        # Process data through orchestrator
        prompt = orchestrator.process(story_data)
        
        # Generate chapter content through writer
        result = writer.process(prompt)
        
        return {
            "status": "success",
            "chapter": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

@tool(description="Analyze text for style consistency with detailed parameters.")
def analyze_style(
    text: str,
    tone: str = "neutral",
    contraction_score: Optional[float] = None,
    sentence_length: Optional[int] = None,
    metaphor_density: Optional[float] = None,
    alliteration_score: Optional[float] = None
) -> Dict:
    """Analyze text against specified style parameters.
    
    Args:
        text: Content to analyze
        tone: Writing tone (formal/casual/poetic)
        contraction_score: Target contraction usage ratio (0-1)
        sentence_length: Target average sentence length
        metaphor_density: Target metaphor density (0-1)
        alliteration_score: Target alliteration score (0-1)
        
    Returns:
        Dictionary with style analysis metrics including:
        - status: success/error
        - metrics: Dictionary with style scores
    """
    analyzer = StyleAnalyzer()
    
    # Handle legacy style_preferences if provided
    if style_preferences is not None:
        metrics = analyzer.analyze(text, style_preferences)
    else:
        metrics = analyzer.analyze_style_v2(
            text,
            tone=tone,
            contraction_score=contraction_score,
            sentence_length=sentence_length,
            metaphor_density=metaphor_density,
            alliteration_score=alliteration_score
        )
    
    return {
        "status": "success",
        "metrics": metrics
    }

# Create the agent with enhanced capabilities
root_agent = Agent(
    name="novel_writing_assistant",
    model=LiteLlm(
        model="ollama/gemma3:4b",
        api_base="http://localhost:11434",
        timeout=60,
        temperature=0.7
    ),
    description="Advanced agent for novel writing with full chapter generation capabilities.",
    instruction="I can help develop your novel by generating plot ideas, developing characters, "
               "providing writing prompts, and generating complete chapters. "
               "I'm running locally via Ollama.",
    tools=ToolManager(tool_registry),
)