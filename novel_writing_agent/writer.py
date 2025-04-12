"""Writer tool agent for novel generation.
Implements Agent interface while preserving existing functionality.

Handles prompt parsing, content generation and output formatting.
"""
import re
import logging
from typing import Dict, List, Optional, Any, ClassVar
from dataclasses import dataclass
from .agent import Agent
from pydantic import BaseModel, ValidationError

# Configure logging (consistent with orchestrator)
logger = logging.getLogger(__name__)

class WritingMetrics(BaseModel):
    """Metrics for generated content."""
    word_count: int
    style_score: float
    coherence: float
    readability: float

class ChapterOutput(BaseModel):
    """Formatted chapter output."""
    content: str
    metrics: WritingMetrics
    status: str = "success"
    warnings: List[str] = []

class StyleAnalyzer:
    """Analyzes and enforces style parameters."""
    
    def __init__(self):
        self.style_rules = {
            'formal': {
                'contraction_score': 0.1,
                'sentence_length': 25
            },
            'casual': {
                'contraction_score': 0.8,
                'sentence_length': 15
            },
            'poetic': {
                'metaphor_density': 0.3,
                'alliteration_score': 0.2
            }
        }
    
    def analyze_style_v2(
        self,
        text: str,
        tone: str = "neutral",
        contraction_score: Optional[float] = None,
        sentence_length: Optional[int] = None,
        metaphor_density: Optional[float] = None,
        alliteration_score: Optional[float] = None
    ) -> Dict[str, float]:
        """Analyze text against style guidelines with explicit parameters.
        
        Args:
            text: Content to analyze
            tone: Writing tone (formal/casual/poetic)
            contraction_score: Target contraction usage ratio (0-1)
            sentence_length: Target average sentence length
            metaphor_density: Target metaphor density (0-1)
            alliteration_score: Target alliteration score (0-1)
            
        Returns:
            Dictionary with style metrics including:
            - style_score: Overall style compliance (0-1)
            - contraction_score: Actual contraction usage
            - sentence_length: Actual average sentence length
            - word_length: Average word length
        """
        # Calculate base metrics
        metrics = {
            'style_score': 0.9,  # Base score
            'contraction_score': self._calculate_contraction_score(text),
            'sentence_length': self._average_sentence_length(text),
            'word_length': self._average_word_length(text)
        }
        
        # Apply style-specific adjustments
        if tone in self.style_rules:
            for k, v in self.style_rules[tone].items():
                metrics['style_score'] -= abs(metrics.get(k, 0) - v) * 0.1
                
        # Apply explicit parameter overrides
        if contraction_score is not None:
            metrics['style_score'] -= abs(metrics['contraction_score'] - contraction_score) * 0.1
            
        if sentence_length is not None:
            metrics['style_score'] -= abs(metrics['sentence_length'] - sentence_length) * 0.05
            
        if metaphor_density is not None:
            metrics['style_score'] -= abs(metrics.get('metaphor_density', 0) - metaphor_density) * 0.15
            
        if alliteration_score is not None:
            metrics['style_score'] -= abs(metrics.get('alliteration_score', 0) - alliteration_score) * 0.15
            
        metrics['style_score'] = max(0.1, min(1.0, metrics['style_score']))
        return metrics
        
    def analyze(self, text: str, style: Dict) -> Dict:
        """Legacy analyze method for backward compatibility."""
        return self.analyze_style_v2(
            text,
            tone=style.get('tone', 'neutral'),
            contraction_score=style.get('contraction_score'),
            sentence_length=style.get('sentence_length'),
            metaphor_density=style.get('metaphor_density'),
            alliteration_score=style.get('alliteration_score')
        )
    
    def _calculate_contraction_score(self, text: str) -> float:
        contractions = len(re.findall(r"\b\w+'[\w]+\b", text))
        total_words = len(text.split())
        return contractions / max(1, total_words)
    
    def _average_sentence_length(self, text: str) -> float:
        sentences = re.split(r'[.!?]', text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    
    def _average_word_length(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        return sum(len(w) for w in words) / len(words)

class WritingEngine:
    """Core writing generation engine."""
    
    def __init__(self):
        self.style_analyzer = StyleAnalyzer()
        self.min_length = 100
        self.max_length = 5000
    
    def generate(self, prompt: Dict, model: Any) -> ChapterOutput:
        """Generate chapter content from standardized prompt.
        
        Args:
            prompt: Standardized prompt from orchestrator
            model: LLM model instance for generation
            
        Returns:
            Formatted chapter output with metrics
        """
        try:
            # Validate prompt structure
            if not all(k in prompt for k in ['context', 'style', 'requirements']):
                raise ValueError("Invalid prompt format")
                
            # Generate initial draft
            raw_prompt = prompt.get('raw_prompt',
                f"Write a chapter with this plot: {prompt['context']['plot']}")
            
            # Handle different model interfaces
            if hasattr(model, 'complete'):
                # LiteLlm style interface
                response = model.complete(
                    prompt=raw_prompt,
                    temperature=0.7,
                    max_tokens=prompt['requirements'].get('length', 1000) * 1.2
                )
                content = response.choices[0].text.strip()
            elif hasattr(model, 'generate'):
                # Generic LLM interface
                content = model.generate(
                    prompt=raw_prompt,
                    max_length=prompt['requirements'].get('length', 1000) * 1.2
                )
            else:
                raise ValueError("Model must implement either complete() or generate() method")
            
            # Analyze and refine
            metrics = self.style_analyzer.analyze(content, prompt['style'])
            word_count = len(content.split())
            
            # Check length compliance
            warnings = []
            target_length = prompt['requirements'].get('length')
            if target_length and abs(word_count - target_length) > 0.2 * target_length:
                warnings.append(f"Word count deviation: {word_count} vs target {target_length}")
            
            logger.info(f"Generated chapter with {word_count} words (style score: {metrics['style_score']:.2f})")
            
            return ChapterOutput(
                content=content,
                metrics=WritingMetrics(
                    word_count=word_count,
                    style_score=metrics['style_score'],
                    coherence=0.9,  # Placeholder for actual analysis
                    readability=0.85  # Placeholder
                ),
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}", exc_info=True)
            return ChapterOutput(
                content="",
                metrics=WritingMetrics(
                    word_count=0,
                    style_score=0,
                    coherence=0,
                    readability=0
                ),
                status=f"error: {str(e)}"
            )

class Writer(Agent):
    """Tool agent that generates novel content while preserving existing functionality.
    
    Attributes:
        engine: Shared writing engine instance (class attribute)
        model: LLM model instance for text generation (instance attribute)
    """
    
    # Shared writing engine instance
    engine: ClassVar[WritingEngine] = WritingEngine()
    
    def __init__(self, model: Any):
        """Initialize Writer tool agent.
        
        Args:
            model: LLM model instance for text generation
        """
        super().__init__(
            name="writer",
            description="Generates novel content using LLM model"
        )
        self.model = model
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data (implements Agent interface).
        
        Args:
            input_data: Standardized input from previous agent
            
        Returns:
            Generated chapter content with metrics
        """
        try:
            output = self.engine.generate(input_data, self.model)
            logger.info(f"Generated chapter with {output.metrics.word_count} words")
            return {
                "content": output.content,
                "metrics": output.metrics.dict(),
                "status": output.status,
                "warnings": output.warnings
            }
        except Exception as e:
            logger.error(f"Writing process failed: {str(e)}")
            return {
                "content": "",
                "metrics": {
                    "word_count": 0,
                    "style_score": 0,
                    "coherence": 0,
                    "readability": 0
                },
                "status": f"error: {str(e)}",
                "warnings": []
            }