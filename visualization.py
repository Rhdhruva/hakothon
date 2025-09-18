import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class Visualizer:
    """Handles all data visualization for the agricultural forecasting platform"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2E8B57',      # Sea Green
            'secondary': '#DAA520',    # Goldenrod
            'accent': '#FF6347',       # Tomato
            'neutral': '#708090',      # Slate Gray
            'success': '#32CD32',      # Lime Green
            'warning': '#FFD700',      # Gold
            'danger': '#DC143C'        # Crimson
        }
    
    def create_quick_forecast_chart(self, crops, weeks):
        """Create quick forecast overview chart for dashboard"""
        try:
            fig = go.Figure()
            
            # Generate sample forecast data for each crop
            dates = [datetime.now() + timedelta(weeks=i) for i in range(weeks)]
            
            colors = ['#2E8B57', '#DAA520', '#FF6347', '#4169E1', '#9932CC', '#FF1493']
            
            for i, crop in enumerate(crops):
                # Generate sample forecast values
                base_value = np.random.uniform(80, 120)
                trend = np.random.uniform(-0.5, 1.0)
                seasonal = [1 + 0.2 * np.sin(2 * np.pi * w / 52) for w in range(weeks)]
                noise = np.random.normal(0, 5, weeks)
                
                values = [base_value + trend * w + seasonal[w] * 10 + noise[w] for w in range(weeks)]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=crop,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=f"Crop Demand Forecast - Next {weeks} Weeks",
                xaxis_title="Date",
                yaxis_title="Demand Index",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating quick forecast chart: {str(e)}")
            return go.Figure()
    
    def create_price_trend_chart(self, data):
        """Create price trend chart with moving averages"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price Trends', 'Volume'),
                row_heights=[0.7, 0.3],
                shared_xaxes=True
            )
            
            if data is not None and not data.empty:
                dates = pd.to_datetime(data['date']) if 'date' in data.columns else data.index
                
                # Price line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=data['price'] if 'price' in data.columns else np.random.uniform(50, 150, len(dates)),
                        mode='lines',
                        name='Price',
                        line=dict(color=self.color_palette['primary'], width=2)
                    ),
                    row=1, col=1
                )
                
                # Moving averages
                if 'ma_7' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=data['ma_7'],
                            mode='lines',
                            name='7-day MA',
                            line=dict(color=self.color_palette['secondary'], width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
                
                if 'ma_30' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=data['ma_30'],
                            mode='lines',
                            name='30-day MA',
                            line=dict(color=self.color_palette['accent'], width=1, dash='dot')
                        ),
                        row=1, col=1
                    )
                
                # Volume bars
                volume_data = data['volume'] if 'volume' in data.columns else np.random.uniform(1000, 10000, len(dates))
                fig.add_trace(
                    go.Bar(
                        x=dates,
                        y=volume_data,
                        name='Volume',
                        marker_color=self.color_palette['neutral'],
                        opacity=0.6
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title="Price and Volume Analysis",
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating price trend chart: {str(e)}")
            return go.Figure()
    
    def create_seasonal_pattern_chart(self, data):
        """Create seasonal pattern visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Average Prices', 'Quarterly Patterns', 'Price Distribution', 'Seasonal Factor'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            if data is not None and not data.empty:
                # Monthly patterns
                if 'month' in data.columns and 'price' in data.columns:
                    monthly_avg = data.groupby('month')['price'].mean()
                    fig.add_trace(
                        go.Scatter(
                            x=monthly_avg.index,
                            y=monthly_avg.values,
                            mode='lines+markers',
                            name='Monthly Avg',
                            line=dict(color=self.color_palette['primary'], width=3)
                        ),
                        row=1, col=1
                    )
                
                # Quarterly patterns
                if 'quarter' in data.columns and 'price' in data.columns:
                    quarterly_avg = data.groupby('quarter')['price'].mean()
                    fig.add_trace(
                        go.Bar(
                            x=[f"Q{q}" for q in quarterly_avg.index],
                            y=quarterly_avg.values,
                            name='Quarterly Avg',
                            marker_color=self.color_palette['secondary']
                        ),
                        row=1, col=2
                    )
                
                # Price distribution
                prices = data['price'] if 'price' in data.columns else np.random.normal(100, 20, 100)
                fig.add_trace(
                    go.Histogram(
                        x=prices,
                        nbinsx=20,
                        name='Price Distribution',
                        marker_color=self.color_palette['accent'],
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                # Seasonal factor
                if 'seasonal_factor' in data.columns:
                    dates = pd.to_datetime(data['date']) if 'date' in data.columns else data.index
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=data['seasonal_factor'],
                            mode='lines',
                            name='Seasonal Factor',
                            line=dict(color=self.color_palette['success'], width=2)
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title="Seasonal Pattern Analysis",
                height=700,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating seasonal pattern chart: {str(e)}")
            return go.Figure()
    
    def create_volatility_chart(self, data):
        """Create market volatility analysis chart"""
        try:
            fig = go.Figure()
            
            if data is not None and not data.empty:
                dates = pd.to_datetime(data['date']) if 'date' in data.columns else data.index
                
                # Volatility line
                volatility = data['volatility'] if 'volatility' in data.columns else np.random.uniform(1, 10, len(dates))
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=volatility,
                        mode='lines',
                        name='Price Volatility',
                        line=dict(color=self.color_palette['danger'], width=2),
                        fill='tonexty'
                    )
                )
                
                # Add volatility bands
                avg_volatility = np.mean(volatility)
                fig.add_hline(
                    y=avg_volatility,
                    line_dash="dash",
                    line_color=self.color_palette['neutral'],
                    annotation_text="Average Volatility"
                )
                
                fig.add_hline(
                    y=avg_volatility * 1.5,
                    line_dash="dot",
                    line_color=self.color_palette['warning'],
                    annotation_text="High Volatility Threshold"
                )
            
            fig.update_layout(
                title="Market Volatility Analysis",
                xaxis_title="Date",
                yaxis_title="Volatility",
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility chart: {str(e)}")
            return go.Figure()
    
    def create_forecast_chart(self, forecast_result):
        """Create detailed forecast visualization"""
        try:
            fig = go.Figure()
            
            if forecast_result:
                dates = pd.to_datetime(forecast_result['dates'])
                values = forecast_result['values']
                
                # Main forecast line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name='Demand Forecast',
                        line=dict(color=self.color_palette['primary'], width=3),
                        marker=dict(size=8)
                    )
                )
                
                # Confidence bands (simulated)
                confidence = forecast_result.get('confidence', 0.8)
                upper_band = [v * (1 + (1 - confidence) * 0.5) for v in values]
                lower_band = [v * (1 - (1 - confidence) * 0.5) for v in values]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=upper_band,
                        mode='lines',
                        name='Upper Confidence',
                        line=dict(color=self.color_palette['primary'], width=0),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=lower_band,
                        mode='lines',
                        name='Lower Confidence',
                        line=dict(color=self.color_palette['primary'], width=0),
                        fill='tonexty',
                        fillcolor=f'rgba(46, 139, 87, 0.2)',
                        showlegend=False
                    )
                )
                
                # Highlight peak demand
                peak_week = forecast_result.get('peak_week', 1) - 1
                if 0 <= peak_week < len(values):
                    fig.add_trace(
                        go.Scatter(
                            x=[dates[peak_week]],
                            y=[values[peak_week]],
                            mode='markers',
                            name='Peak Demand',
                            marker=dict(
                                color=self.color_palette['accent'],
                                size=15,
                                symbol='star'
                            )
                        )
                    )
            
            fig.update_layout(
                title=f"Demand Forecast: {forecast_result.get('crop', 'Unknown Crop')}",
                xaxis_title="Date",
                yaxis_title="Demand Index",
                hovermode='x unified',
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating forecast chart: {str(e)}")
            return go.Figure()
    
    def create_comparison_chart(self, comparison_data, metric):
        """Create multi-crop comparison chart"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(f'{metric} Comparison', 'Trend Analysis', 'Volatility Comparison', 'Growth Potential'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            if comparison_data:
                crops = list(comparison_data.keys())
                colors = [self.color_palette['primary'], self.color_palette['secondary'], 
                         self.color_palette['accent'], self.color_palette['success'],
                         self.color_palette['warning'], self.color_palette['danger']]
                
                # Main metric comparison
                scores = [comparison_data[crop]['score'] for crop in crops]
                fig.add_trace(
                    go.Bar(
                        x=crops,
                        y=scores,
                        name=metric,
                        marker_color=colors[:len(crops)]
                    ),
                    row=1, col=1
                )
                
                # Trend analysis (forecast values over time)
                for i, crop in enumerate(crops):
                    if 'forecast_values' in comparison_data[crop]:
                        weeks = list(range(1, len(comparison_data[crop]['forecast_values']) + 1))
                        fig.add_trace(
                            go.Scatter(
                                x=weeks,
                                y=comparison_data[crop]['forecast_values'],
                                mode='lines+markers',
                                name=crop,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ),
                            row=1, col=2
                        )
                
                # Volatility comparison
                volatilities = [comparison_data[crop]['volatility'] for crop in crops]
                fig.add_trace(
                    go.Bar(
                        x=crops,
                        y=volatilities,
                        name='Volatility',
                        marker_color=colors[:len(crops)],
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                # Growth potential
                growth_potentials = [comparison_data[crop]['growth_potential'] for crop in crops]
                fig.add_trace(
                    go.Bar(
                        x=crops,
                        y=growth_potentials,
                        name='Growth Potential',
                        marker_color=colors[:len(crops)],
                        opacity=0.8
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Multi-Crop Performance Comparison",
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating comparison chart: {str(e)}")
            return go.Figure()
    
    def create_market_summary_dashboard(self, summary_data):
        """Create comprehensive market summary dashboard"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Price Trends', 'Volume Analysis', 'Top Performers', 
                              'Market Sentiment', 'Risk Indicators', 'Forecast Accuracy'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Add various dashboard components
            # This would be populated with real data in production
            
            fig.update_layout(
                title="Agricultural Market Dashboard",
                height=1000,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating market summary dashboard: {str(e)}")
            return go.Figure()
