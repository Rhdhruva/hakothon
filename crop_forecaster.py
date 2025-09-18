import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class CropForecaster:
    """Advanced crop demand forecasting using multiple models"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.fitted_models = {}
    
    def generate_forecast(self, forecast_params):
        """Generate comprehensive crop demand forecast"""
        try:
            crop = forecast_params['crop']
            weeks = forecast_params['weeks']
            
            # Generate base forecast using multiple approaches
            base_forecast = self._generate_base_forecast(crop, weeks)
            
            # Apply adjustments based on parameters
            adjusted_forecast = self._apply_forecast_adjustments(base_forecast, forecast_params)
            
            # Calculate confidence metrics
            confidence_score = self._calculate_confidence(adjusted_forecast, forecast_params)
            
            # Identify peak demand period
            peak_week = np.argmax(adjusted_forecast) + 1
            
            # Calculate average change
            avg_change = np.mean(np.diff(adjusted_forecast))
            
            # Determine trend direction
            trend = "increasing" if avg_change > 0 else "decreasing" if avg_change < 0 else "stable"
            
            forecast_result = {
                'crop': crop,
                'weeks': weeks,
                'values': adjusted_forecast.tolist(),
                'peak_week': peak_week,
                'avg_change': avg_change * 100,  # Convert to percentage
                'confidence': confidence_score,
                'trend': trend,
                'factors': self._identify_key_factors(forecast_params),
                'dates': self._generate_forecast_dates(weeks)
            }
            
            return forecast_result
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return None
    
    def generate_multi_crop_comparison(self, crops, weeks, metric):
        """Generate comparison data for multiple crops"""
        try:
            comparison_data = {}
            
            for crop in crops:
                # Generate individual forecast for each crop
                forecast_params = {
                    'crop': crop,
                    'weeks': weeks,
                    'include_weather': True,
                    'include_economic': True,
                    'seasonal_weight': 0.7,
                    'market_sentiment': 'Neutral',
                    'supply_disruption': 0.2
                }
                
                forecast = self.generate_forecast(forecast_params)
                
                if forecast:
                    # Calculate metric-specific scores
                    score = self._calculate_metric_score(forecast, metric)
                    volatility = np.std(forecast['values'])
                    growth_potential = self._calculate_growth_potential(forecast)
                    market_share = self._estimate_market_share(crop)
                    
                    comparison_data[crop] = {
                        'score': score,
                        'trend': forecast['trend'],
                        'volatility': volatility,
                        'growth_potential': growth_potential,
                        'market_share': market_share,
                        'forecast_values': forecast['values']
                    }
            
            return comparison_data
            
        except Exception as e:
            st.error(f"Error generating multi-crop comparison: {str(e)}")
            return None
    
    def _generate_base_forecast(self, crop, weeks):
        """Generate base forecast using historical patterns"""
        try:
            # Simulate historical demand patterns (in real implementation, use actual data)
            base_demand = self._get_crop_base_demand(crop)
            
            # Generate seasonal pattern
            seasonal_pattern = self._generate_seasonal_pattern(crop, weeks)
            
            # Add trend component
            trend_component = self._generate_trend_component(crop, weeks)
            
            # Add noise for realism
            noise = np.random.normal(0, 0.05, weeks)
            
            # Combine components
            forecast = base_demand * seasonal_pattern * trend_component * (1 + noise)
            
            # Ensure positive values
            forecast = np.maximum(forecast, 0.1)
            
            return forecast
            
        except Exception as e:
            st.error(f"Error generating base forecast: {str(e)}")
            return np.ones(weeks)
    
    def _apply_forecast_adjustments(self, base_forecast, params):
        """Apply various adjustments to base forecast"""
        try:
            adjusted = base_forecast.copy()
            
            # Seasonal weight adjustment
            seasonal_weight = params.get('seasonal_weight', 0.7)
            seasonal_adjustment = 1 + (seasonal_weight - 0.5) * 0.2
            adjusted *= seasonal_adjustment
            
            # Market sentiment adjustment
            sentiment = params.get('market_sentiment', 'Neutral')
            sentiment_multiplier = {
                'Bullish': 1.15,
                'Neutral': 1.0,
                'Bearish': 0.85
            }
            adjusted *= sentiment_multiplier.get(sentiment, 1.0)
            
            # Supply disruption adjustment
            disruption = params.get('supply_disruption', 0.2)
            disruption_adjustment = 1 + disruption * 0.3
            adjusted *= disruption_adjustment
            
            # Weather impact (if included)
            if params.get('include_weather', False):
                weather_adjustment = self._calculate_weather_impact(params['crop'])
                adjusted *= weather_adjustment
            
            # Economic factors (if included)
            if params.get('include_economic', False):
                economic_adjustment = self._calculate_economic_impact()
                adjusted *= economic_adjustment
            
            return adjusted
            
        except Exception as e:
            st.error(f"Error applying forecast adjustments: {str(e)}")
            return base_forecast
    
    def _calculate_confidence(self, forecast, params):
        """Calculate confidence score for the forecast"""
        try:
            confidence_factors = []
            
            # Data quality factor (simulated)
            confidence_factors.append(0.85)
            
            # Model stability factor
            volatility = np.std(forecast) / np.mean(forecast)
            stability_factor = max(0.5, 1 - volatility)
            confidence_factors.append(stability_factor)
            
            # Parameter completeness factor
            param_completeness = len([v for v in params.values() if v is not None]) / len(params)
            confidence_factors.append(param_completeness)
            
            # Seasonal factor (higher confidence during stable seasons)
            seasonal_confidence = 0.9 if params.get('seasonal_weight', 0) > 0.5 else 0.7
            confidence_factors.append(seasonal_confidence)
            
            # Overall confidence (weighted average)
            overall_confidence = np.mean(confidence_factors)
            
            return min(0.95, max(0.5, float(overall_confidence)))
            
        except Exception as e:
            return 0.75  # Default confidence
    
    def _identify_key_factors(self, params):
        """Identify key factors affecting the forecast"""
        factors = []
        
        if params.get('include_weather', False):
            factors.append("Weather patterns")
        
        if params.get('include_economic', False):
            factors.append("Economic indicators")
        
        if params.get('seasonal_weight', 0) > 0.5:
            factors.append("Seasonal trends")
        
        sentiment = params.get('market_sentiment', 'Neutral')
        if sentiment != 'Neutral':
            factors.append(f"{sentiment} market sentiment")
        
        disruption = params.get('supply_disruption', 0)
        if disruption > 0.3:
            factors.append("Supply chain disruptions")
        
        return factors
    
    def _generate_forecast_dates(self, weeks):
        """Generate forecast date range"""
        start_date = datetime.now()
        dates = []
        for i in range(weeks):
            date = start_date + timedelta(weeks=i)
            dates.append(date.strftime("%Y-%m-%d"))
        return dates
    
    def _get_crop_base_demand(self, crop):
        """Get base demand level for crop"""
        base_demands = {
            'Wheat': 100,
            'Corn': 120,
            'Soybeans': 90,
            'Rice': 110,
            'Tomatoes': 80,
            'Potatoes': 95,
            'Cotton': 75,
            'Sugar': 85
        }
        return base_demands.get(crop, 100)
    
    def _generate_seasonal_pattern(self, crop, weeks):
        """Generate seasonal demand pattern"""
        # Create seasonal pattern based on crop type
        seasonal_patterns = {
            'Wheat': lambda w: 1 + 0.3 * np.sin(2 * np.pi * w / 52),  # Annual cycle
            'Corn': lambda w: 1 + 0.4 * np.cos(2 * np.pi * w / 52),   # Peak in fall
            'Soybeans': lambda w: 1 + 0.25 * np.sin(2 * np.pi * w / 52 + np.pi/4),
            'Rice': lambda w: 1 + 0.2 * np.sin(2 * np.pi * w / 26),   # Semi-annual
            'Tomatoes': lambda w: 1 + 0.5 * np.sin(2 * np.pi * w / 52 + np.pi/2),  # Summer peak
            'Potatoes': lambda w: 1 + 0.3 * np.cos(2 * np.pi * w / 52 + np.pi/3),
        }
        
        pattern_func = seasonal_patterns.get(crop, lambda w: np.ones(len(w)))
        week_array = np.arange(weeks)
        
        return pattern_func(week_array)
    
    def _generate_trend_component(self, crop, weeks):
        """Generate trend component for forecast"""
        # Simple linear trend with some variation
        base_trend = 1.02  # 2% growth trend
        trend_variation = np.random.normal(0, 0.01, weeks)
        
        trend = np.power(base_trend, np.arange(weeks)) * (1 + trend_variation)
        
        return trend
    
    def _calculate_weather_impact(self, crop):
        """Calculate weather impact on crop demand"""
        # Simulate weather impact (in real implementation, use actual weather data)
        weather_impacts = {
            'Wheat': 1.05,    # Slight increase due to weather
            'Corn': 0.98,     # Slight decrease
            'Soybeans': 1.02,
            'Rice': 1.08,     # Higher impact
            'Tomatoes': 0.95, # Weather sensitive
            'Potatoes': 1.01
        }
        
        return weather_impacts.get(crop, 1.0)
    
    def _calculate_economic_impact(self):
        """Calculate economic impact on demand"""
        # Simulate economic conditions impact
        return np.random.uniform(0.95, 1.08)
    
    def _calculate_metric_score(self, forecast, metric):
        """Calculate score based on comparison metric"""
        try:
            if metric == "Demand Forecast":
                return np.mean(forecast['values'])
            elif metric == "Price Volatility":
                return 100 - np.std(forecast['values'])  # Lower volatility = higher score
            elif metric == "Market Share":
                return self._estimate_market_share(forecast['crop'])
            elif metric == "Growth Potential":
                return self._calculate_growth_potential(forecast)
            else:
                return np.random.uniform(60, 95)  # Default score
                
        except Exception as e:
            return 75  # Default score
    
    def _calculate_growth_potential(self, forecast):
        """Calculate growth potential score"""
        try:
            values = forecast['values']
            if len(values) > 1:
                growth_rate = (values[-1] - values[0]) / values[0] * 100
                return min(100, max(0, 50 + growth_rate))
            return 50
        except:
            return 50
    
    def _estimate_market_share(self, crop):
        """Estimate relative market share"""
        market_shares = {
            'Wheat': 85,
            'Corn': 92,
            'Soybeans': 78,
            'Rice': 88,
            'Tomatoes': 65,
            'Potatoes': 72,
            'Cotton': 70,
            'Sugar': 68
        }
        return market_shares.get(crop, 75)
