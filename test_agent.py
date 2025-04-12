from novel_writing_agent.orchestrator import Orchestrator
from novel_writing_agent.agent import ToolManager, root_agent

def test_generate_first_chapter():
    """Test the first chapter generation workflow."""
    tool_manager = ToolManager([
        root_agent.generate_plot_idea,
        root_agent.develop_character,
        root_agent.generate_chapter
    ])
    orchestrator = Orchestrator(tool_manager)
    
    result = orchestrator.generate_first_chapter()
    print("Generated chapter:", result)

if __name__ == "__main__":
    test_generate_first_chapter()
"""Tests for novel writing agent components."""
import pytest
import time
from unittest.mock import MagicMock, patch
from novel_writing_agent.orchestrator import (
    Orchestrator,
    StoryInput,
    CharacterProfile,
    StylePreferences
)
from novel_writing_agent.writer import Writer, StyleAnalyzer
from novel_writing_agent.agent import analyze_style

class TestOrchestrator:
    """Tests for orchestrator module."""
    
    @pytest.fixture
    def sample_input(self):
        return {
            "plot": "A hero's journey",
            "characters": [{
                "name": "Hero",
                "archetype": "protagonist",
                "traits": ["brave", "determined"],
                "flaws": ["impulsive"],
                "background": "Farm boy",
                "motivation": "Save the kingdom"
            }],
            "style_preferences": {
                "tone": "dramatic",
                "pacing": "medium",
                "point_of_view": "third_person",
                "dialogue_ratio": 0.3
            },
            "target_length": 1000
        }
    
    def test_input_validation(self, sample_input):
        """Test story input validation."""
        # Valid input
        validated = StoryInput(**sample_input)
        assert validated.plot == "A hero's journey"
        assert len(validated.characters) == 1
        
        # Invalid tone
        with pytest.raises(ValueError):
            invalid = sample_input.copy()
            invalid["style_preferences"]["tone"] = "invalid"
            StoryInput(**invalid)
            
        # Short length
        with pytest.raises(ValueError):
            invalid = sample_input.copy()
            invalid["target_length"] = 50
            StoryInput(**invalid)
    
    def test_prompt_generation(self, sample_input):
        """Test prompt generation."""
        orchestrator = Orchestrator()
        prompt = orchestrator.process(sample_input)
        
        assert "task_type" in prompt
        assert "raw_prompt" in prompt
        assert "Hero" in prompt["raw_prompt"]
        
        # Test JSON input
        json_input = json.dumps(sample_input)
        prompt_json = orchestrator.process(json_input)
        assert prompt_json["task_type"] == "chapter_generation"

class TestWriter:
    """Tests for writer module."""
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.complete.return_value.choices = [MagicMock(text="Sample chapter content")]
        return model
    
    @pytest.fixture
    def sample_prompt(self):
        return {
            "task_type": "chapter_generation",
            "context": {
                "plot": "A hero's journey",
                "characters": [{
                    "name": "Hero",
                    "archetype": "protagonist"
                }]
            },
            "style": {
                "tone": "dramatic",
                "pacing": "medium"
            },
            "requirements": {
                "length": 1000
            },
            "raw_prompt": "Write a chapter about a hero"
        }
    
    def test_style_analysis(self):
        """Test style analyzer metrics."""
        analyzer = StyleAnalyzer()
        text = "This is a test. It's working!"
        metrics = analyzer.analyze(text, {"tone": "casual"})
        
        assert metrics["style_score"] > 0
        assert metrics["contraction_score"] > 0
        assert metrics["sentence_length"] == 4
        
        # Test empty text
        metrics = analyzer.analyze("", {"tone": "formal"})
        assert metrics["sentence_length"] == 0
    
    def test_content_generation(self, mock_model, sample_prompt):
        """Test chapter generation."""
        writer = Writer(mock_model)
        output = writer.process(sample_prompt)
        
        assert output["content"] == "Sample chapter content"
        assert output["metrics"]["word_count"] > 0
        assert output["status"] == "success"
        
        # Test error handling
        mock_model.complete.side_effect = Exception("API error")
        output = writer.process(sample_prompt)
        assert output["status"].startswith("error")

class TestAgentTools:
    """Tests for agent tool functions."""
    
    @pytest.fixture
    def sample_text(self):
        return "The quick brown fox jumps over the lazy dog. It's a classic example."
    
    @pytest.fixture
    def mock_analyzer(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = {
            "style_score": 0.85,
            "contraction_score": 0.5,
            "sentence_length": 10
        }
        analyzer.analyze_style_v2.return_value = {
            "style_score": 0.9,
            "contraction_score": 0.5,
            "sentence_length": 10,
            "metaphor_density": 0.2,
            "alliteration_score": 0.3
        }
        return analyzer
    
    def test_analyze_style_new_format(self, sample_text, mock_analyzer):
        """Test analyze_style with new parameter format."""
        with patch('novel_writing_agent.agent.StyleAnalyzer', return_value=mock_analyzer):
            result = analyze_style(
                text=sample_text,
                tone="formal",
                contraction_score=0.5,
                sentence_length=15,
                metaphor_density=0.2,
                alliteration_score=0.3
            )
            
            assert result["status"] == "success"
            assert result["metrics"]["style_score"] == 0.9
            mock_analyzer.analyze_style_v2.assert_called_once()
    
    def test_analyze_style_legacy_format(self, sample_text, mock_analyzer):
        """Test backward compatibility with legacy style_preferences."""
        with patch('novel_writing_agent.agent.StyleAnalyzer', return_value=mock_analyzer):
            result = analyze_style(
                text=sample_text,
                style_preferences={
                    "tone": "formal",
                    "pacing": "medium"
                }
            )
            
            assert result["status"] == "success"
            assert result["metrics"]["style_score"] == 0.85
            mock_analyzer.analyze.assert_called_once()
    
    def test_analyze_style_parameter_validation(self, sample_text):
        """Test parameter validation and error handling."""
        # Invalid contraction score
        with pytest.raises(ValueError):
            analyze_style(sample_text, contraction_score=1.5)
            
        # Invalid tone
        with pytest.raises(ValueError):
            analyze_style(sample_text, tone="invalid_tone")
            
        # Empty text
        result = analyze_style("", tone="formal")
        assert result["metrics"]["sentence_length"] == 0
    
    @pytest.mark.benchmark
    def test_analyze_style_performance(self, sample_text, benchmark):
        """Performance test comparing old and new style analysis."""
        # Test new format performance
        benchmark(lambda: analyze_style(
            sample_text,
            tone="formal",
            contraction_score=0.5,
            sentence_length=15
        ))
        
        # Test legacy format performance
        benchmark(lambda: analyze_style(
            sample_text,
            style_preferences={"tone": "formal"}
        ))

class TestIntegration:
    """Integration tests between orchestrator and writer."""
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.complete.return_value.choices = [MagicMock(text="Integrated chapter")]
        return model
    
    def test_end_to_end(self, mock_model):
        """Test full orchestration to writing pipeline."""
        from novel_writing_agent.orchestrator import Orchestrator
        from novel_writing_agent.writer import Writer
        
        # Setup
        orchestrator = Orchestrator()
        writer = Writer(mock_model)
        
        # Input data
        input_data = {
            "plot": "Integration test plot",
            "characters": [{
                "name": "Test",
                "archetype": "test",
                "traits": ["test"],
                "flaws": ["test"],
                "background": "test",
                "motivation": "test"
            }],
            "style_preferences": {
                "tone": "formal",
                "pacing": "slow",
                "point_of_view": "third_person",
                "dialogue_ratio": 0.2
            }
        }
        
        # Process
        prompt = orchestrator.process(input_data)
        output = writer.process(prompt)
        
        # Verify
        assert "Integrated chapter" in output["content"]
        assert output["metrics"]["word_count"] > 0
