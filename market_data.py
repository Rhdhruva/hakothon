import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import json

class MarketDataManager:
    """Manages market data collection, storage, and retrieval"""
    
    def __init__(self):
        self.data_cache = {}
        self.initialize_sample_data()
    
    def initialize_sample_data(self):
        """Initialize with sample market data for demonstration"""
        # In production, this would connect to real data sources
        self.sample_crops = ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes"]
        self.generate_sample_historical_data()
    
    def get_historical_data(self, crop, start_date, end_date):
        """Retrieve historical market data for a crop"""
        try:
            cache_key = f"{crop}_{start_date}_{end_date}"
            
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Generate sample historical data
            data = self._generate_historical_sample(crop, start_date, end_date)
            
            if data is not None:
                self.data_cache[cache_key] = data
                return data
            else:
                st.warning(f"No historical data available for {crop} in the specified date range")
                return None
                
        except Exception as e:
            st.error(f"Error retrieving historical data: {str(e)}")
            return None
    
    def get_recent_updates(self):
        """Get recent market updates and news"""
        try:
            # Simulate recent market updates
            updates = []
            crops = ["Wheat", "Corn", "Soybeans", "Rice"]
            
            for i, crop in enumerate(crops):
                update = {
                    'crop': crop,
                    'date': (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                    'summary': f"{crop} market showing {'positive' if np.random.random() > 0.5 else 'mixed'} signals with {'increased' if np.random.random() > 0.5 else 'stable'} trading volume.",
                    'price_change': f"{np.random.uniform(-5, 5):.1f}%"
                }
                updates.append(update)
            
            return updates
            
        except Exception as e:
            st.error(f"Error getting recent updates: {str(e)}")
            return []
    
    def add_market_data(self, market_entry):
        """Add new market data entry"""
        try:
            # In production, this would save to a database
            if not hasattr(self, 'user_market_data'):
                self.user_market_data = []
            
            market_entry['timestamp'] = datetime.now()
            self.user_market_data.append(market_entry)
            
            # Update cache to include new data
            self._invalidate_related_cache(market_entry['crop'])
            
            return True
            
        except Exception as e:
            st.error(f"Error adding market data: {str(e)}")
            return False
    
    def add_weather_data(self, weather_entry):
        """Add new weather data entry"""
        try:
            if not hasattr(self, 'user_weather_data'):
                self.user_weather_data = []
            
            weather_entry['timestamp'] = datetime.now()
            self.user_weather_data.append(weather_entry)
            
            return True
            
        except Exception as e:
            st.error(f"Error adding weather data: {str(e)}")
            return False
    
    def add_consumer_trend(self, trend_entry):
        """Add new consumer trend data"""
        try:
            if not hasattr(self, 'user_trend_data'):
                self.user_trend_data = []
            
            trend_entry['timestamp'] = datetime.now()
            self.user_trend_data.append(trend_entry)
            
            return True
            
        except Exception as e:
            st.error(f"Error adding consumer trend data: {str(e)}")
            return False
    
    def get_weather_data(self, region, start_date, end_date):
        """Get weather data for forecasting"""
        try:
            # Generate sample weather data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            weather_data = []
            for date in date_range:
                weather_entry = {
                    'date': date,
                    'region': region,
                    'temperature': np.random.normal(70, 15),  # Fahrenheit
                    'rainfall': np.random.exponential(0.1),   # Inches
                    'humidity': np.random.uniform(30, 90),    # Percentage
                    'wind_speed': np.random.uniform(5, 25),   # MPH
                    'conditions': np.random.choice(['Clear', 'Partly Cloudy', 'Cloudy', 'Rainy'])
                }
                weather_data.append(weather_entry)
            
            return pd.DataFrame(weather_data)
            
        except Exception as e:
            st.error(f"Error getting weather data: {str(e)}")
            return None
    
    def get_economic_indicators(self):
        """Get economic indicators affecting agriculture"""
        try:
            indicators = {
                'inflation_rate': np.random.uniform(2, 6),      # Percentage
                'gdp_growth': np.random.uniform(-2, 4),         # Percentage
                'unemployment': np.random.uniform(3, 8),        # Percentage
                'currency_index': np.random.uniform(0.9, 1.1), # Relative strength
                'fuel_price': np.random.uniform(2.5, 4.5),     # $ per gallon
                'interest_rate': np.random.uniform(1, 6),      # Percentage
                'trade_balance': np.random.uniform(-50, 20)     # Billion $
            }
            
            return indicators
            
        except Exception as e:
            st.error(f"Error getting economic indicators: {str(e)}")
            return {}
    
    def generate_sample_historical_data(self):
        """Generate sample historical data for all crops"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            for crop in self.sample_crops:
                data = self._generate_historical_sample(crop, start_date, end_date)
                cache_key = f"{crop}_sample_data"
                self.data_cache[cache_key] = data
                
        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")
    
    def _generate_historical_sample(self, crop, start_date, end_date):
        """Generate sample historical data for a specific crop"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Base price for different crops
            base_prices = {
                'Wheat': 6.50,
                'Corn': 5.20,
                'Soybeans': 12.80,
                'Rice': 8.90,
                'Tomatoes': 3.40,
                'Potatoes': 2.10,
                'Cotton': 0.75,
                'Sugar': 0.18
            }
            
            base_price = base_prices.get(crop, 5.0)
            
            # Generate price series with trend, seasonality, and noise
            prices = []
            volumes = []
            
            for i, date in enumerate(date_range):
                # Trend component
                trend = base_price * (1 + 0.02 * i / 365)  # 2% annual growth
                
                # Seasonal component
                day_of_year = date.timetuple().tm_yday
                seasonal = 1 + 0.15 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Random walk component
                if i == 0:
                    random_walk = 1
                else:
                    random_walk = prices[-1] / (trend * seasonal) * (1 + np.random.normal(0, 0.02))
                
                # Combine components
                price = trend * seasonal * random_walk
                
                # Add some market events (random spikes/drops)
                if np.random.random() < 0.02:  # 2% chance of market event
                    price *= np.random.uniform(0.85, 1.15)
                
                prices.append(max(price, base_price * 0.5))  # Floor price
                
                # Generate volume data
                volume = np.random.lognormal(mean=10, sigma=0.5) * 1000
                volumes.append(volume)
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': date_range,
                'price': prices,
                'volume': volumes,
                'crop': crop
            })
            
            return data
            
        except Exception as e:
            st.error(f"Error generating historical sample for {crop}: {str(e)}")
            return None
    
    def _invalidate_related_cache(self, crop):
        """Invalidate cache entries related to a specific crop"""
        keys_to_remove = [key for key in self.data_cache.keys() if crop in key]
        for key in keys_to_remove:
            del self.data_cache[key]
    
    def get_market_summary(self):
        """Get overall market summary statistics"""
        try:
            summary = {
                'total_crops_tracked': len(self.sample_crops),
                'data_points_available': sum(len(data) for data in self.data_cache.values() if isinstance(data, pd.DataFrame)),
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'market_status': 'Active',
                'data_quality': 'Good'
            }
            
            return summary
            
        except Exception as e:
            st.error(f"Error getting market summary: {str(e)}")
            return {}
    
    def export_data(self, crop, format='csv'):
        """Export data for a specific crop"""
        try:
            cache_key = f"{crop}_sample_data"
            
            if cache_key in self.data_cache:
                data = self.data_cache[cache_key]
                
                if format.lower() == 'csv':
                    return data.to_csv(index=False)
                elif format.lower() == 'json':
                    return data.to_json(orient='records', date_format='iso')
                else:
                    return data
            else:
                st.warning(f"No data available for {crop}")
                return None
                
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None
    
    def validate_data_quality(self, data):
        """Validate data quality and completeness"""
        try:
            if data is None or data.empty:
                return {'quality': 'Poor', 'issues': ['No data available']}
            
            issues = []
            
            # Check for missing values
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                issues.append(f"{missing_count} missing values detected")
            
            # Check for price anomalies
            if 'price' in data.columns:
                price_std = data['price'].std()
                price_mean = data['price'].mean()
                outliers = data[abs(data['price'] - price_mean) > 3 * price_std]
                if len(outliers) > 0:
                    issues.append(f"{len(outliers)} potential price outliers detected")
            
            # Check data freshness
            if 'date' in data.columns:
                latest_date = pd.to_datetime(data['date']).max()
                days_old = (datetime.now() - latest_date).days
                if days_old > 7:
                    issues.append(f"Data is {days_old} days old")
            
            quality = 'Excellent' if len(issues) == 0 else 'Good' if len(issues) <= 2 else 'Poor'
            
            return {'quality': quality, 'issues': issues}
            
        except Exception as e:
            return {'quality': 'Unknown', 'issues': [f"Error validating data: {str(e)}"]}
