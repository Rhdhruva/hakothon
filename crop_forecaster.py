import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from apy_data_loader import APYDataLoader

class CropForecaster:
    """Advanced crop demand forecasting using multiple models and real APY data"""
    
    def __init__(self, apy_loader=None):
        self.apy_loader = apy_loader or APYDataLoader()
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.fitted_models = {}
    
    def generate_forecast(self, forecast_params):
        """Generate comprehensive crop demand forecast using real APY data"""
        try:
            crop = forecast_params['crop']
            weeks = forecast_params['weeks']
            
            # Use APY data loader for real historical-based forecasting
            forecast_result = self.apy_loader.generate_demand_forecast(crop, weeks)
            
            if forecast_result:
                # Enhance with parameter-based adjustments
                enhanced_forecast = self._apply_forecast_adjustments(
                    np.array(forecast_result['values']), forecast_params
                )
                
                # Update result with enhanced forecast
                forecast_result['values'] = enhanced_forecast.tolist()
                forecast_result['peak_week'] = int(np.argmax(enhanced_forecast) + 1)
                forecast_result['avg_change'] = float(np.mean(np.diff(enhanced_forecast)) * 100)
                forecast_result['factors'] = self._identify_key_factors(forecast_params)
                
                # Enhance confidence based on APY data quality
                apy_confidence = forecast_result.get('confidence', 0.7)
                param_confidence = self._calculate_confidence(enhanced_forecast, forecast_params)
                forecast_result['confidence'] = float((apy_confidence + param_confidence) / 2)
                
                return forecast_result
            else:
                # Fallback to model-based forecast if APY data unavailable
                st.warning(f"APY data unavailable for {crop}, using model-based forecast")
                return self._generate_fallback_forecast(forecast_params)
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return None
    
    def generate_multi_crop_comparison(self, crops, weeks, metric):
        """Generate comparison data for multiple crops using real APY data"""
        try:
            comparison_data = {}
            
            for crop in crops:
                # Get real market trends from APY data
                market_trends = self.apy_loader.get_market_trends(crop, 5)
                
                # Generate forecast with real data backing
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
                
                if forecast and market_trends:
                    # Calculate metric-specific scores using real data
                    score = self._calculate_metric_score(forecast, metric)
                    volatility = np.std(forecast['values'])
                    
                    # Use real historical growth for growth potential
                    historical_growth = market_trends.get('total_production_growth', 0)
                    growth_potential = min(100, max(0, 50 + historical_growth))
                    
                    # Get top producing states for market share estimation
                    top_states = self.apy_loader.get_top_producing_states(crop)
                    market_share = len(top_states) * 8 if top_states else 60  # Rough estimate
                    
                    comparison_data[crop] = {
                        'score': float(score),
                        'trend': forecast['trend'],
                        'volatility': float(volatility),
                        'growth_potential': float(growth_potential),
                        'market_share': float(market_share),
                        'forecast_values': forecast['values'],
                        'historical_growth': historical_growth
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
    
    def _generate_fallback_forecast(self, forecast_params):
        """Generate fallback forecast when APY data is unavailable"""
        try:
            crop = forecast_params['crop']
            weeks = forecast_params['weeks']
            
            # Generate base forecast using original method
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
            st.error(f"Error generating fallback forecast: {str(e)}")
            return None
    
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
