from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Define functions for the novel writing assistant tools
def generate_plot_idea(genre: str) -> dict:
	"""Generates a plot idea for a specified genre.
	
	Args:
		genre (str): The genre for which to generate a plot idea.
		
	Returns:
		dict: status and result or error msg.
	"""
	if genre.lower() == "mystery":
		return {
			"status": "success",
			"idea": "A respected detective finds evidence that their mentor may have framed an innocent person 20 years ago, forcing them to choose between loyalty and justice."
		}
	elif genre.lower() == "fantasy":
		return {
			"status": "success",
			"idea": "In a world where magic is tied to memories, a scholar discovers a forgotten spell that could save their dying city, but casting it requires sacrificing their most precious memory."
		}
	elif genre.lower() == "romance":
		return {
			"status": "success",
			"idea": "Two rival food critics who've spent years writing scathing reviews about each other's work are forced to collaborate on a cookbook, discovering they have more in common than they thought."
		}
	else:
		return {
			"status": "success",
			"idea": f"An ordinary person discovers they have an extraordinary ability that makes them both valuable and hunted in the {genre} world."
		}

def develop_character(archetype: str) -> dict:
	"""Develops a character based on a specified archetype.
	
	Args:
		archetype (str): The character archetype to develop.
		
	Returns:
		dict: status and character details or error msg.
	"""
	archetypes = {
		"hero": {
			"traits": "Brave, selfless, determined",
			"flaws": "Overconfident, stubborn, self-sacrificing to a fault",
			"background": "Ordinary upbringing with a defining incident that revealed their exceptional nature",
			"motivation": "To protect others and prove their worth"
		},
		"mentor": {
			"traits": "Wise, patient, experienced",
			"flaws": "Secretive, manipulative, haunted by past failures",
			"background": "Has lived through similar challenges as the protagonist, but failed in some crucial way",
			"motivation": "To guide others to succeed where they once failed"
		},
		"villain": {
			"traits": "Intelligent, driven, charismatic",
			"flaws": "Arrogant, ruthless, unable to see other perspectives",
			"background": "Once idealistic but twisted by trauma or betrayal",
			"motivation": "To reshape the world according to their vision, or avenge perceived wrongs"
		}
	}
	
	if archetype.lower() in archetypes:
		return {
			"status": "success",
			"character": archetypes[archetype.lower()]
		}
	else:
		return {
			"status": "error",
			"error_message": f"Sorry, the archetype '{archetype}' is not available."
		}

def writing_prompt(word_count: int) -> dict:
	"""Provides a writing prompt with suggested word count.
	
	Args:
		word_count (int): The target word count for the prompt.
		
	Returns:
		dict: status and prompt or error msg.
	"""
	prompts = {
		"short": "Write a scene where a character discovers a hidden letter that changes everything they believed about their family.",
		"medium": "Write a story about someone who wakes up with the ability to hear others' thoughts, but only when they're thinking about secrets.",
		"long": "Write a story that begins with a character receiving an unexpected inheritance that comes with unusual conditions, and ends with them making a difficult choice between what they want and what they need."
	}
	
	if word_count <= 500:
		prompt_type = "short"
	elif word_count <= 2000:
		prompt_type = "medium"
	else:
		prompt_type = "long"
	
	return {
		"status": "success",
		"prompt": prompts[prompt_type],
		"target_word_count": word_count
	}

# Create the agent with Ollama integration
# LiteLLM knows how to connect to a local Ollama server by default
root_agent = Agent(
	name="novel_writing_assistant",
	model=LiteLlm(model="ollama/gemma3:4b"), # Standard LiteLLM format for Ollama
	description="Agent to help with novel writing and creative storytelling.",
	instruction="I can help you develop your novel by generating plot ideas, developing characters, and providing writing prompts. I'm running locally via Ollama.",
	tools=[generate_plot_idea, develop_character, writing_prompt],
)