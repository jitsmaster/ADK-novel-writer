"""Analysis agent for novel writing workflow."""

from typing import Dict, Any
from .agent import Agent
from .writer import StyleAnalyzer

class Analyzer(Agent):
    """Analyzes processed story data for consistency and quality."""
    
    def __init__(self):
        super().__init__(
            name="analyzer",
            description="Analyzes story data for style consistency and coherence"
        )
        self.style_analyzer = StyleAnalyzer()
        
    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze processed story data.
        
        Args:
            processed_data: Standardized data from DataProcessor
            
        Returns:
            Analysis results with metrics and recommendations
        """
        try:
            content = processed_data.get("plot", "") + " " + \
                     " ".join(c.get("background", "") for c in processed_data.get("characters", []))
            
            metrics = self.style_analyzer.analyze(content, processed_data["style"])
            coherence = self._calculate_coherence(processed_data)
            
            return {
                "style_analysis": metrics,
                "coherence_score": coherence,
                "recommendations": self._generate_recommendations(metrics, coherence)
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise ValueError(f"Analysis error: {str(e)}")
            
    def _calculate_coherence(self, data: Dict[str, Any]) -> float:
        """Calculate story coherence score (0-1)."""
        # Placeholder implementation - should analyze character/plot consistency
        return 0.85
        
    def _generate_recommendations(self, metrics: Dict[str, Any], coherence: float) -> Dict[str, str]:
        """Generate improvement recommendations based on analysis."""
        recs = {}
        if metrics.get("style_score", 0) < 0.7:
            recs["style"] = "Consider adjusting tone or pacing for better style consistency"
        if coherence < 0.8:
            recs["coherence"] = "Review character motivations and plot alignment"
        return recs