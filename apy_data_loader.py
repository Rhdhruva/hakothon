import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Optional, Tuple

class APYDataLoader:
    """Load and manage real APY (Area, Production, Yield) dataset from Kaggle"""
    
    def __init__(self, data_file='crops_data.csv'):
        self.data_file = data_file
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load the APY dataset"""
        try:
            self.data = pd.read_csv(self.data_file)
            st.success(f"Loaded APY dataset with {len(self.data)} records covering {self.data['Year'].min()}-{self.data['Year'].max()}")
        except Exception as e:
            st.error(f"Error loading APY dataset: {str(e)}")
            self.data = None
    
    def get_available_crops(self) -> List[str]:
        """Get list of available crops in the dataset"""
        if self.data is None:
            return []
        
        # Extract crop names from column headers (they contain AREA, PRODUCTION, YIELD)
        crop_columns = [col for col in self.data.columns if 'AREA' in col and 'FRUITS AND VEGETABLES' not in col]
        crops = [col.replace(' AREA (1000 ha)', '') for col in crop_columns]
        
        # Filter out complex crop names and keep main crops
        main_crops = []
        for crop in crops:
            if crop not in ['KHARIF SORGHUM', 'RABI SORGHUM', 'MINOR PULSES', 'RAPESEED AND MUSTARD']:
                main_crops.append(crop.title())
        
        return sorted(main_crops)
    
    def get_available_states(self) -> List[str]:
        """Get list of available states in the dataset"""
        if self.data is None:
            return []
        
        return sorted(self.data['State Name'].unique().tolist())
    
    def get_crop_data(self, crop: str, state: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get historical data for a specific crop"""
        if self.data is None:
            return None
        
        try:
            crop_upper = crop.upper()
            
            # Find the relevant columns for this crop
            area_col = f"{crop_upper} AREA (1000 ha)"
            prod_col = f"{crop_upper} PRODUCTION (1000 tons)"
            yield_col = f"{crop_upper} YIELD (Kg per ha)"
            
            if area_col not in self.data.columns:
                st.warning(f"Data not available for {crop}")
                return None
            
            # Filter data
            crop_data = self.data[['Year', 'State Name', 'Dist Name', area_col, prod_col, yield_col]].copy()
            
            if state:
                crop_data = crop_data[crop_data['State Name'] == state]
            
            # Remove rows with all NaN values
            crop_data = crop_data.dropna(subset=[area_col, prod_col, yield_col], how='all')
            
            # Rename columns for consistency
            crop_data.columns = ['Year', 'State', 'District', 'Area', 'Production', 'Yield']
            
            # Add calculated fields
            crop_data['Price'] = self._estimate_price_from_yield(crop_data['Yield'].copy(), crop)
            crop_data['Date'] = pd.to_datetime(crop_data['Year'], format='%Y')
            
            return crop_data.sort_values('Year')
            
        except Exception as e:
            st.error(f"Error getting crop data for {crop}: {str(e)}")
            return None
    
    def get_market_trends(self, crop: str, years: int = 5) -> Dict:
        """Analyze market trends for a crop over specified years"""
        if self.data is None:
            return {}
        
        crop_data = self.get_crop_data(crop)
        if crop_data is None or crop_data.empty:
            return {}
        
        try:
            # Get recent years data
            recent_years = sorted(crop_data['Year'].unique())[-years:]
            recent_data = crop_data[crop_data['Year'].isin(recent_years)]
            
            # Calculate trends
            yearly_stats = recent_data.groupby('Year').agg({
                'Area': ['sum', 'mean'],
                'Production': ['sum', 'mean'],
                'Yield': 'mean',
                'Price': 'mean'
            }).round(2)
            
            # Calculate growth rates
            total_production = yearly_stats['Production']['sum']
            total_area = yearly_stats['Area']['sum']
            avg_yield = yearly_stats['Yield']['mean']
            avg_price = yearly_stats['Price']['mean']
            
            if len(total_production) > 1:
                prod_growth = float((total_production.iloc[-1] - total_production.iloc[0]) / total_production.iloc[0] * 100)
                area_growth = float((total_area.iloc[-1] - total_area.iloc[0]) / total_area.iloc[0] * 100)
                yield_growth = float((avg_yield.iloc[-1] - avg_yield.iloc[0]) / avg_yield.iloc[0] * 100)
                price_growth = float((avg_price.iloc[-1] - avg_price.iloc[0]) / avg_price.iloc[0] * 100)
            else:
                prod_growth = area_growth = yield_growth = price_growth = 0
            
            trends = {
                'crop': crop,
                'years_analyzed': years,
                'total_production_growth': prod_growth,
                'total_area_growth': area_growth,
                'average_yield_growth': yield_growth,
                'estimated_price_growth': price_growth,
                'yearly_data': yearly_stats,
                'trend_direction': 'increasing' if prod_growth > 2 else 'decreasing' if prod_growth < -2 else 'stable'
            }
            
            return trends
            
        except Exception as e:
            st.error(f"Error analyzing trends for {crop}: {str(e)}")
            return {}
    
    def get_seasonal_patterns(self, crop: str) -> Dict:
        """Analyze seasonal patterns from historical data"""
        crop_data = self.get_crop_data(crop)
        if crop_data is None or crop_data.empty:
            return {}
        
        try:
            # Analyze year-over-year patterns
            patterns = {}
            
            # Production patterns by year
            yearly_production = crop_data.groupby('Year')['Production'].sum()
            patterns['yearly_production'] = yearly_production.to_dict()
            
            # Yield patterns by year
            yearly_yield = crop_data.groupby('Year')['Yield'].mean()
            patterns['yearly_yield'] = yearly_yield.to_dict()
            
            # Identify peak and low production years
            patterns['peak_production_year'] = int(yearly_production.idxmax()) if len(yearly_production) > 0 else 2017
            patterns['lowest_production_year'] = int(yearly_production.idxmin()) if len(yearly_production) > 0 else 2010
            
            # Calculate volatility
            patterns['production_volatility'] = yearly_production.std()
            patterns['yield_volatility'] = yearly_yield.std()
            
            return patterns
            
        except Exception as e:
            st.error(f"Error analyzing seasonal patterns for {crop}: {str(e)}")
            return {}
    
    def generate_demand_forecast(self, crop: str, weeks: int = 8) -> Dict:
        """Generate demand forecast based on historical patterns"""
        trends = self.get_market_trends(crop)
        patterns = self.get_seasonal_patterns(crop)
        
        if not trends or not patterns:
            return {}
        
        try:
            # Base forecast on historical growth trends
            base_growth_rate = trends.get('total_production_growth', 0) / 100
            
            # Generate forecast values
            forecast_values = []
            base_demand = 100  # Normalized base
            
            for week in range(weeks):
                # Apply growth trend
                trend_factor = 1 + (base_growth_rate * week / 52)  # Weekly growth
                
                # Add seasonal variation based on historical patterns
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * week / 26)  # Bi-annual cycle
                
                # Add some realistic variation
                variation = np.random.normal(0, 0.05)
                
                forecast_value = base_demand * trend_factor * seasonal_factor * (1 + variation)
                forecast_values.append(max(forecast_value, 10))  # Minimum demand floor
            
            # Calculate forecast metrics
            peak_week = np.argmax(forecast_values) + 1
            avg_change = np.mean(np.diff(forecast_values))
            confidence = min(0.9, 0.6 + (len(patterns.get('yearly_production', {})) * 0.05))
            
            # Generate forecast dates
            start_date = datetime.now()
            dates = [(start_date + timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(weeks)]
            
            forecast_result = {
                'crop': crop,
                'weeks': weeks,
                'values': forecast_values,
                'peak_week': peak_week,
                'avg_change': avg_change * 100,
                'confidence': confidence,
                'trend': trends.get('trend_direction', 'stable'),
                'dates': dates,
                'historical_growth': trends.get('total_production_growth', 0)
            }
            
            return forecast_result
            
        except Exception as e:
            st.error(f"Error generating forecast for {crop}: {str(e)}")
            return {}
    
    def _estimate_price_from_yield(self, yield_values: pd.Series, crop: str) -> pd.Series:
        """Estimate price based on yield (inverse relationship)"""
        try:
            # Base prices per crop (estimated in INR per kg)
            base_prices = {
                'RICE': 25, 'WHEAT': 20, 'MAIZE': 18, 'SORGHUM': 22,
                'PEARL MILLET': 24, 'FINGER MILLET': 35, 'BARLEY': 19,
                'CHICKPEA': 60, 'PIGEONPEA': 75, 'GROUNDNUT': 50,
                'SESAMUM': 90, 'SUNFLOWER': 45, 'SOYABEAN': 40,
                'SUGARCANE': 3, 'COTTON': 55, 'POTATOES': 12
            }
            
            base_price = base_prices.get(crop.upper(), 30)
            
            # Inverse relationship: higher yield generally means lower price due to supply
            normalized_yield = (yield_values - yield_values.min()) / (yield_values.max() - yield_values.min() + 0.001)
            price_factor = 1.3 - (normalized_yield * 0.6)  # Price varies from 0.7x to 1.3x base
            
            return pd.Series(base_price * price_factor, index=yield_values.index)
            
        except Exception as e:
            return pd.Series([30] * len(yield_values), index=yield_values.index)
    
    def get_top_producing_states(self, crop: str, year: int = 2017) -> List[Tuple[str, float]]:
        """Get top producing states for a crop in a specific year"""
        if self.data is None:
            return []
        
        try:
            crop_upper = crop.upper()
            prod_col = f"{crop_upper} PRODUCTION (1000 tons)"
            
            if prod_col not in self.data.columns:
                return []
            
            year_data = self.data[self.data['Year'] == year]
            state_production = year_data.groupby('State Name')[prod_col].sum().sort_values(ascending=False)
            
            # Return top 10 states with their production
            return [(str(state), float(production)) for state, production in state_production.head(10).items() if production > 0]
            
        except Exception as e:
            st.error(f"Error getting top producing states for {crop}: {str(e)}")
            return []
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics of the dataset"""
        if self.data is None:
            return {}
        
        try:
            summary = {
                'total_records': len(self.data),
                'years_covered': f"{self.data['Year'].min()}-{self.data['Year'].max()}",
                'total_states': self.data['State Name'].nunique(),
                'total_districts': self.data['Dist Name'].nunique(),
                'available_crops': len(self.get_available_crops()),
                'states_list': sorted(self.data['State Name'].unique().tolist()),
                'data_completeness': (1 - self.data.isnull().sum().sum() / self.data.size) * 100
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}