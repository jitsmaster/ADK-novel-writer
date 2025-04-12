"""Report generation agent for novel writing workflow."""

from typing import Dict, Any
from .agent import Agent

class ReportGenerator(Agent):
    """Generates structured reports from analysis results."""
    
    def __init__(self):
        super().__init__(
            name="report_generator",
            description="Generates structured reports from analysis data"
        )
        
    def generate(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured report from analysis.
        
        Args:
            analysis_result: Analysis data from Analyzer
            
        Returns:
            Structured report with summary and recommendations
        """
        try:
            return {
                "summary": self._generate_summary(analysis_result),
                "recommendations": analysis_result.get("recommendations", {}),
                "metrics": {
                    "style_score": analysis_result["style_analysis"].get("style_score", 0),
                    "coherence_score": analysis_result.get("coherence_score", 0)
                }
            }
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise ValueError(f"Report error: {str(e)}")
            
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary of analysis."""
        style_score = analysis["style_analysis"].get("style_score", 0)
        coherence = analysis.get("coherence_score", 0)
        
        summary = []
        summary.append(f"Style Analysis: {style_score:.1%} compliance")
        summary.append(f"Story Coherence: {coherence:.1%} score")
        
        if style_score < 0.7:
            summary.append("⚠️ Style consistency needs improvement")
        if coherence < 0.8:
            summary.append("⚠️ Story coherence could be enhanced")
            
        return "\n".join(summary)