import json
import os
import streamlit as st
from statistical_analyzer import StatisticalAnalyzer
from apy_data_loader import APYDataLoader

class AIAnalyzer:
    """Statistical analysis using real APY data for market insights and forecasting"""
    
    def __init__(self, apy_loader=None):
        self.apy_loader = apy_loader or APYDataLoader()
        self.statistical_analyzer = StatisticalAnalyzer(self.apy_loader)
        if apy_loader is None:
            st.success("Loaded real agricultural data for statistical analysis")
    
    def analyze_market_trends(self, market_data):
        """Analyze market trends using statistical methods"""
        return self.statistical_analyzer.analyze_market_trends(market_data)
    
    def analyze_forecast(self, forecast_data):
        """Analyze forecast results using statistical methods"""
        return self.statistical_analyzer.analyze_forecast(forecast_data)
    
    def generate_crop_recommendations(self, comparison_data):
        """Generate crop investment/planting recommendations based on statistical analysis"""
        return self.statistical_analyzer.generate_crop_recommendations(comparison_data)
    
    def analyze_seasonal_patterns(self, seasonal_data):
        """Analyze seasonal patterns using statistical methods"""
        return self.statistical_analyzer.analyze_seasonal_patterns(seasonal_data)
    
    def _prepare_market_summary(self, market_data):
        """Prepare market data summary for AI analysis"""
        try:
            if market_data is None or market_data.empty:
                return "No market data available"
            
            summary = {
                "current_price": market_data['price'].iloc[-1] if 'price' in market_data.columns else 0,
                "price_change_7d": market_data['price_change_7d'].iloc[-1] if 'price_change_7d' in market_data.columns else 0,
                "volatility": market_data['volatility'].mean() if 'volatility' in market_data.columns else 0,
                "moving_avg_7": market_data['ma_7'].iloc[-1] if 'ma_7' in market_data.columns else 0,
                "moving_avg_30": market_data['ma_30'].iloc[-1] if 'ma_30' in market_data.columns else 0,
                "seasonal_factor": market_data['seasonal_factor'].iloc[-1] if 'seasonal_factor' in market_data.columns else 1,
                "data_points": len(market_data),
                "date_range": f"{market_data['date'].min()} to {market_data['date'].max()}" if 'date' in market_data.columns else "Unknown"
            }
            
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            return f"Error preparing market summary: {str(e)}"
    
    def _prepare_forecast_summary(self, forecast_data):
        """Prepare forecast data summary for AI analysis"""
        try:
            summary = {
                "forecast_periods": forecast_data.get('weeks', 0),
                "peak_demand_week": forecast_data.get('peak_week', 0),
                "average_change": forecast_data.get('avg_change', 0),
                "confidence_score": forecast_data.get('confidence', 0),
                "trend_direction": forecast_data.get('trend', 'unknown'),
                "key_factors": forecast_data.get('factors', []),
                "forecast_values": forecast_data.get('values', [])
            }
            
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            return f"Error preparing forecast summary: {str(e)}"
    
    def _prepare_comparison_summary(self, comparison_data):
        """Prepare comparison data summary for AI analysis"""
        try:
            summary = {}
            for crop, data in comparison_data.items():
                summary[crop] = {
                    "score": data.get('score', 0),
                    "trend": data.get('trend', 'neutral'),
                    "volatility": data.get('volatility', 0),
                    "growth_potential": data.get('growth_potential', 0),
                    "market_share": data.get('market_share', 0)
                }
            
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            return f"Error preparing comparison summary: {str(e)}"
    
    def _prepare_seasonal_summary(self, seasonal_data):
        """Prepare seasonal data summary for AI analysis"""
        try:
            if not seasonal_data:
                return "No seasonal data available"
            
            summary = {
                "monthly_patterns": seasonal_data.get('monthly', {}).to_dict() if hasattr(seasonal_data.get('monthly', {}), 'to_dict') else {},
                "quarterly_patterns": seasonal_data.get('quarterly', {}).to_dict() if hasattr(seasonal_data.get('quarterly', {}), 'to_dict') else {},
                "peak_periods": len(seasonal_data.get('peaks', [])),
                "trough_periods": len(seasonal_data.get('troughs', []))
            }
            
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            return f"Error preparing seasonal summary: {str(e)}"
    
    def _format_recommendations(self, recommendations_json):
        """Format AI recommendations for display"""
        try:
            formatted = "## AI Strategic Recommendations\n\n"
            
            # Top recommendations
            if "top_recommendations" in recommendations_json:
                formatted += "### Priority Crops:\n"
                for i, rec in enumerate(recommendations_json["top_recommendations"], 1):
                    formatted += f"{i}. **{rec['crop']}** (Priority: {rec['priority']})\n"
                    formatted += f"   {rec['reasoning']}\n\n"
            
            # Market strategy
            if "market_strategy" in recommendations_json:
                formatted += f"### Market Strategy:\n{recommendations_json['market_strategy']}\n\n"
            
            # Risk assessment
            if "risk_assessment" in recommendations_json:
                formatted += f"### Risk Assessment:\n{recommendations_json['risk_assessment']}\n\n"
            
            # Timing advice
            if "timing_advice" in recommendations_json:
                formatted += f"### Timing Recommendations:\n{recommendations_json['timing_advice']}\n"
            
            return formatted
            
        except Exception as e:
            return f"Error formatting recommendations: {str(e)}"
