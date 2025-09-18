import json
import os
import streamlit as st
from openai import OpenAI

class AIAnalyzer:
    """AI-powered analysis using OpenAI for market insights and forecasting"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None
            st.warning("OpenAI API key not found. AI features will be limited.")
    
    def analyze_market_trends(self, market_data):
        """Analyze market trends using AI"""
        if not self.client:
            return "AI analysis unavailable - OpenAI API key not configured"
        
        try:
            # Prepare market data summary for AI analysis
            data_summary = self._prepare_market_summary(market_data)
            
            prompt = f"""
            As an agricultural market analyst, analyze the following market data and provide insights:
            
            Market Data Summary:
            {data_summary}
            
            Please provide a comprehensive analysis including:
            1. Current market trends and patterns
            2. Key factors driving price movements
            3. Seasonal patterns observed
            4. Market volatility assessment
            5. Potential risks and opportunities
            6. Short-term outlook (1-4 weeks)
            
            Format your response as a clear, professional market analysis report.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert agricultural market analyst with deep knowledge of commodity markets, seasonal patterns, and economic factors affecting crop prices."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating AI market analysis: {str(e)}")
            return None
    
    def analyze_forecast(self, forecast_data):
        """Analyze forecast results and provide AI insights"""
        if not self.client:
            return "AI analysis unavailable - OpenAI API key not configured"
        
        try:
            forecast_summary = self._prepare_forecast_summary(forecast_data)
            
            prompt = f"""
            Analyze this crop demand forecast and provide actionable insights:
            
            Forecast Summary:
            {forecast_summary}
            
            Please provide:
            1. Forecast reliability assessment
            2. Key driving factors for the predicted demand
            3. Potential market scenarios (best case, worst case, most likely)
            4. Recommended actions for farmers and traders
            5. Risk factors to monitor
            6. Confidence level explanation
            
            Keep the analysis practical and actionable for agricultural stakeholders.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an agricultural forecasting expert who helps farmers and traders make data-driven decisions based on demand predictions."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating forecast analysis: {str(e)}")
            return None
    
    def generate_crop_recommendations(self, comparison_data):
        """Generate crop investment/planting recommendations based on comparison"""
        if not self.client:
            return "AI recommendations unavailable - OpenAI API key not configured"
        
        try:
            comparison_summary = self._prepare_comparison_summary(comparison_data)
            
            prompt = f"""
            Based on this multi-crop comparison analysis, provide strategic recommendations:
            
            Crop Comparison Data:
            {comparison_summary}
            
            Please provide:
            1. Top 3 crops for investment/planting priority
            2. Reasoning for each recommendation
            3. Risk-adjusted return expectations
            4. Diversification strategies
            5. Timing considerations for planting/trading
            6. Market entry/exit strategies
            
            Format as clear, actionable recommendations for agricultural decision-makers.
            Respond in JSON format with this structure:
            {
                "top_recommendations": [
                    {"crop": "crop_name", "priority": "High/Medium/Low", "reasoning": "explanation"},
                ],
                "market_strategy": "overall strategy recommendation",
                "risk_assessment": "risk evaluation",
                "timing_advice": "timing recommendations"
            }
            """
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a strategic agricultural advisor who helps optimize crop selection and investment decisions based on market analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content or '{}')
            return self._format_recommendations(result)
            
        except Exception as e:
            st.error(f"Error generating crop recommendations: {str(e)}")
            return None
    
    def analyze_seasonal_patterns(self, seasonal_data):
        """Analyze seasonal patterns using AI"""
        if not self.client:
            return "AI seasonal analysis unavailable - OpenAI API key not configured"
        
        try:
            seasonal_summary = self._prepare_seasonal_summary(seasonal_data)
            
            prompt = f"""
            Analyze these seasonal patterns in agricultural commodity data:
            
            Seasonal Data:
            {seasonal_summary}
            
            Provide insights on:
            1. Strongest seasonal patterns identified
            2. Optimal timing for planting and harvesting
            3. Price peak and trough periods
            4. Seasonal trading strategies
            5. Climate and weather impact factors
            6. Year-over-year pattern consistency
            
            Focus on actionable seasonal intelligence for farming operations.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a seasonal agricultural analyst specializing in crop timing, weather patterns, and seasonal market dynamics."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error analyzing seasonal patterns: {str(e)}")
            return None
    
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
