from novel_writing_agent.agent import root_agent

def main():
	print("Testing Novel Writing Assistant Agent...")
	
	# Test the plot idea generator
	print("\n1. Testing Plot Idea Generator:")
	result = root_agent.tools.generate_plot_idea(genre="mystery")
	print(f"Plot Idea: {result['idea']}")
	
	# Test the character development tool
	print("\n2. Testing Character Development:")
	result = root_agent.tools.develop_character(archetype="hero")
	print(f"Character Traits: {result['character']['traits']}")
	print(f"Character Flaws: {result['character']['flaws']}")
	
	# Test the writing prompt tool
	print("\n3. Testing Writing Prompt:")
	result = root_agent.tools.writing_prompt(word_count=1000)
	print(f"Prompt: {result['prompt']}")
	print(f"Target Word Count: {result['target_word_count']}")
	
	print("\nAll tests completed successfully!")

if __name__ == "__main__":
	main()