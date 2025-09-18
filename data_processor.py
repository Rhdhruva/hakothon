import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataProcessor:
    """Handles data processing and analysis for agricultural forecasting"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def process_market_data(self, raw_data):
        """Process raw market data for analysis"""
        try:
            if raw_data is None or raw_data.empty:
                return None
                
            # Create processed dataframe
            processed_data = raw_data.copy()
            
            # Calculate moving averages
            processed_data['ma_7'] = processed_data['price'].rolling(window=7).mean()
            processed_data['ma_30'] = processed_data['price'].rolling(window=30).mean()
            
            # Calculate price volatility
            processed_data['volatility'] = processed_data['price'].rolling(window=7).std()
            
            # Calculate price changes
            processed_data['price_change'] = processed_data['price'].pct_change()
            processed_data['price_change_7d'] = processed_data['price'].pct_change(periods=7)
            
            # Seasonal decomposition
            processed_data['month'] = pd.to_datetime(processed_data['date']).dt.month
            processed_data['quarter'] = pd.to_datetime(processed_data['date']).dt.quarter
            processed_data['day_of_year'] = pd.to_datetime(processed_data['date']).dt.dayofyear
            
            # Calculate seasonal factors
            monthly_avg = processed_data.groupby('month')['price'].mean()
            processed_data['seasonal_factor'] = processed_data['month'].map(monthly_avg) / processed_data['price'].mean()
            
            # Volume-weighted average price
            if 'volume' in processed_data.columns:
                processed_data['vwap'] = (processed_data['price'] * processed_data['volume']).cumsum() / processed_data['volume'].cumsum()
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error processing market data: {str(e)}")
            return None
    
    def detect_seasonal_patterns(self, data):
        """Detect seasonal patterns in crop data"""
        try:
            if data is None or data.empty:
                return None
            
            seasonal_patterns = {}
            
            # Monthly patterns
            monthly_stats = data.groupby('month').agg({
                'price': ['mean', 'std', 'min', 'max'],
                'volume': 'mean' if 'volume' in data.columns else lambda x: None
            }).round(2)
            
            seasonal_patterns['monthly'] = monthly_stats
            
            # Quarterly patterns
            quarterly_stats = data.groupby('quarter').agg({
                'price': ['mean', 'std'],
                'volatility': 'mean'
            }).round(2)
            
            seasonal_patterns['quarterly'] = quarterly_stats
            
            # Peak and trough identification
            price_series = data.set_index('date')['price']
            peaks = self._find_peaks(price_series)
            troughs = self._find_troughs(price_series)
            
            seasonal_patterns['peaks'] = peaks
            seasonal_patterns['troughs'] = troughs
            
            return seasonal_patterns
            
        except Exception as e:
            st.error(f"Error detecting seasonal patterns: {str(e)}")
            return None
    
    def _find_peaks(self, series, window=30):
        """Find price peaks in the series"""
        try:
            rolling_max = series.rolling(window=window, center=True).max()
            peaks = series[series == rolling_max]
            return peaks.head(10)  # Return top 10 peaks
        except:
            return pd.Series()
    
    def _find_troughs(self, series, window=30):
        """Find price troughs in the series"""
        try:
            rolling_min = series.rolling(window=window, center=True).min()
            troughs = series[series == rolling_min]
            return troughs.head(10)  # Return top 10 troughs
        except:
            return pd.Series()
    
    def calculate_market_indicators(self, data):
        """Calculate technical market indicators"""
        try:
            indicators = {}
            
            if data is None or data.empty:
                return indicators
            
            # RSI (Relative Strength Index)
            indicators['rsi'] = self._calculate_rsi(data['price'])
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['price'])
            indicators['bollinger_upper'] = bb_upper
            indicators['bollinger_lower'] = bb_lower
            
            # MACD
            macd_line, signal_line = self._calculate_macd(data['price'])
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            
            # Support and resistance levels
            indicators['support'] = data['price'].quantile(0.25)
            indicators['resistance'] = data['price'].quantile(0.75)
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculating market indicators: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            return upper_band.iloc[-1], lower_band.iloc[-1]
        except:
            return None, None
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            return macd_line.iloc[-1], signal_line.iloc[-1]
        except:
            return 0, 0
    
    def prepare_forecast_features(self, data, weather_data=None, economic_data=None):
        """Prepare features for forecasting models"""
        try:
            features = {}
            
            if data is not None and not data.empty:
                # Price-based features
                features['current_price'] = data['price'].iloc[-1]
                features['price_trend'] = data['price_change'].tail(7).mean()
                features['volatility'] = data['volatility'].tail(7).mean()
                features['seasonal_factor'] = data['seasonal_factor'].iloc[-1]
                
                # Technical indicators
                indicators = self.calculate_market_indicators(data)
                features.update(indicators)
                
                # Volume features
                if 'volume' in data.columns:
                    features['avg_volume'] = data['volume'].tail(7).mean()
                    features['volume_trend'] = data['volume'].pct_change().tail(7).mean()
            
            # Weather features
            if weather_data is not None:
                features['temperature'] = weather_data.get('temperature', 0)
                features['rainfall'] = weather_data.get('rainfall', 0)
                features['humidity'] = weather_data.get('humidity', 50)
            
            # Economic features
            if economic_data is not None:
                features['inflation_rate'] = economic_data.get('inflation', 0)
                features['gdp_growth'] = economic_data.get('gdp_growth', 0)
                features['currency_strength'] = economic_data.get('currency', 1.0)
            
            return features
            
        except Exception as e:
            st.error(f"Error preparing forecast features: {str(e)}")
            return {}
    
    def normalize_data(self, data):
        """Normalize data for machine learning models"""
        try:
            if isinstance(data, dict):
                values = list(data.values())
                normalized_values = self.scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
                return dict(zip(data.keys(), normalized_values))
            else:
                return self.scaler.fit_transform(data)
        except Exception as e:
            st.error(f"Error normalizing data: {str(e)}")
            return data
