import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_processor import DataProcessor
from ai_analyzer import AIAnalyzer
from crop_forecaster import CropForecaster
from visualization import Visualizer
from market_data import MarketDataManager

# Page configuration
st.set_page_config(
    page_title="AgriForecaster - AI Crop Demand Prediction",
    page_icon="🌾",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize application components"""
    return {
        'data_processor': DataProcessor(),
        'ai_analyzer': AIAnalyzer(),
        'forecaster': CropForecaster(),
        'visualizer': Visualizer(),
        'market_data': MarketDataManager()
    }

components = initialize_components()

# Sidebar navigation
st.sidebar.title("🌾 AgriForecaster")
st.sidebar.markdown("AI-Powered Crop Demand Prediction Platform")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard", "Market Analysis", "Demand Forecasting", "Data Input", "Multi-Crop Comparison"]
)

# Main application logic
if page == "Dashboard":
    st.title("Agricultural Forecasting Dashboard")
    st.markdown("Welcome to the AI-powered crop demand prediction platform")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Crops", "12", "↑ 2")
    with col2:
        st.metric("Forecast Accuracy", "87.5%", "↑ 2.1%")
    with col3:
        st.metric("Market Trends", "Bullish", "")
    with col4:
        st.metric("Next Update", "6 hours", "")
    
    # Quick forecast overview
    st.subheader("Quick Forecast Overview")
    
    # Sample crop selection for demo
    selected_crops = st.multiselect(
        "Select crops for quick view:",
        ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes"],
        default=["Wheat", "Corn", "Soybeans"]
    )
    
    if selected_crops:
        forecast_period = st.slider("Forecast period (weeks)", 1, 12, 4)
        
        # Generate forecast visualization
        fig = components['visualizer'].create_quick_forecast_chart(
            selected_crops, forecast_period
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent market updates
    st.subheader("Recent Market Updates")
    market_updates = components['market_data'].get_recent_updates()
    
    if market_updates:
        for update in market_updates:
            with st.expander(f"{update['crop']} - {update['date']}"):
                st.write(update['summary'])
                st.write(f"Price change: {update['price_change']}")
    else:
        st.info("No recent market updates available. Please check data connections.")

elif page == "Market Analysis":
    st.title("Historical Market Price Analysis")
    
    # Crop selection
    crop_type = st.selectbox(
        "Select crop for analysis:",
        ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes", "Cotton", "Sugar"]
    )
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date:",
            value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End date:",
            value=datetime.now()
        )
    
    if st.button("Analyze Market Data"):
        with st.spinner("Analyzing market data..."):
            # Load historical data
            historical_data = components['market_data'].get_historical_data(
                crop_type, start_date, end_date
            )
            
            if historical_data is not None:
                # Process data for trends
                processed_data = components['data_processor'].process_market_data(historical_data)
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Price Trends")
                    price_chart = components['visualizer'].create_price_trend_chart(processed_data)
                    st.plotly_chart(price_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Seasonal Patterns")
                    seasonal_chart = components['visualizer'].create_seasonal_pattern_chart(processed_data)
                    st.plotly_chart(seasonal_chart, use_container_width=True)
                
                # Volatility analysis
                st.subheader("Market Volatility Analysis")
                volatility_chart = components['visualizer'].create_volatility_chart(processed_data)
                st.plotly_chart(volatility_chart, use_container_width=True)
                
                # AI market insights
                st.subheader("AI Market Insights")
                with st.spinner("Generating AI insights..."):
                    insights = components['ai_analyzer'].analyze_market_trends(processed_data)
                    if insights:
                        st.write(insights)
                    else:
                        st.error("Unable to generate AI insights. Please check API configuration.")
            else:
                st.error("Unable to load market data. Please check data sources or try a different date range.")

elif page == "Demand Forecasting":
    st.title("AI-Powered Demand Prediction")
    
    # Forecasting parameters
    col1, col2 = st.columns(2)
    with col1:
        crop_for_forecast = st.selectbox(
            "Select crop:",
            ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes"]
        )
        forecast_weeks = st.slider("Forecast horizon (weeks):", 1, 12, 6)
    
    with col2:
        include_weather = st.checkbox("Include weather patterns", value=True)
        include_economic = st.checkbox("Include economic indicators", value=True)
    
    # Additional factors
    st.subheader("Market Factors")
    col1, col2, col3 = st.columns(3)
    with col1:
        seasonal_weight = st.slider("Seasonal influence:", 0.0, 1.0, 0.7)
    with col2:
        market_sentiment = st.selectbox("Market sentiment:", ["Bullish", "Neutral", "Bearish"])
    with col3:
        supply_disruption = st.slider("Supply disruption risk:", 0.0, 1.0, 0.2)
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating AI-powered forecast..."):
            # Collect forecast parameters
            forecast_params = {
                'crop': crop_for_forecast,
                'weeks': forecast_weeks,
                'include_weather': include_weather,
                'include_economic': include_economic,
                'seasonal_weight': seasonal_weight,
                'market_sentiment': market_sentiment,
                'supply_disruption': supply_disruption
            }
            
            # Generate forecast
            forecast_result = components['forecaster'].generate_forecast(forecast_params)
            
            if forecast_result:
                # Display forecast chart
                st.subheader(f"{crop_for_forecast} Demand Forecast - {forecast_weeks} Weeks")
                forecast_chart = components['visualizer'].create_forecast_chart(forecast_result)
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Forecast summary
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Forecast Summary")
                    st.metric("Peak Demand Week", f"Week {forecast_result['peak_week']}")
                    st.metric("Average Demand Change", f"{forecast_result['avg_change']:+.1f}%")
                    st.metric("Confidence Score", f"{forecast_result['confidence']:.1%}")
                
                with col2:
                    st.subheader("AI Analysis")
                    ai_analysis = components['ai_analyzer'].analyze_forecast(forecast_result)
                    if ai_analysis:
                        st.write(ai_analysis)
                    else:
                        st.error("Unable to generate AI analysis. Please check API configuration.")
            else:
                st.error("Unable to generate forecast. Please check input parameters and try again.")

elif page == "Data Input":
    st.title("Market Data Input System")
    
    # Data input tabs
    tab1, tab2, tab3 = st.tabs(["Market Prices", "Weather Patterns", "Consumer Trends"])
    
    with tab1:
        st.subheader("Market Price Data Entry")
        
        col1, col2 = st.columns(2)
        with col1:
            crop_input = st.selectbox("Crop:", ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes"])
            price_input = st.number_input("Price per unit ($):", min_value=0.0, step=0.01)
            volume_input = st.number_input("Trading volume:", min_value=0, step=1000)
        
        with col2:
            market_input = st.selectbox("Market:", ["Chicago", "New York", "Kansas City", "Minneapolis"])
            date_input = st.date_input("Date:", value=datetime.now())
            quality_input = st.selectbox("Quality grade:", ["Premium", "Standard", "Lower"])
        
        if st.button("Add Market Data"):
            market_entry = {
                'crop': crop_input,
                'price': price_input,
                'volume': volume_input,
                'market': market_input,
                'date': date_input,
                'quality': quality_input
            }
            
            success = components['market_data'].add_market_data(market_entry)
            if success:
                st.success("Market data added successfully!")
                st.rerun()
            else:
                st.error("Failed to add market data. Please try again.")
    
    with tab2:
        st.subheader("Weather Pattern Data")
        
        col1, col2 = st.columns(2)
        with col1:
            region_input = st.selectbox("Region:", ["Midwest", "Great Plains", "California", "Southeast"])
            temp_input = st.number_input("Average temperature (°F):", step=1.0)
            rainfall_input = st.number_input("Rainfall (inches):", min_value=0.0, step=0.1)
        
        with col2:
            humidity_input = st.slider("Humidity (%):", 0, 100, 50)
            weather_date = st.date_input("Weather date:", value=datetime.now())
            conditions_input = st.selectbox("Conditions:", ["Clear", "Partly Cloudy", "Cloudy", "Rainy", "Storm"])
        
        if st.button("Add Weather Data"):
            weather_entry = {
                'region': region_input,
                'temperature': temp_input,
                'rainfall': rainfall_input,
                'humidity': humidity_input,
                'date': weather_date,
                'conditions': conditions_input
            }
            
            success = components['market_data'].add_weather_data(weather_entry)
            if success:
                st.success("Weather data added successfully!")
                st.rerun()
            else:
                st.error("Failed to add weather data. Please try again.")
    
    with tab3:
        st.subheader("Consumer Trend Data")
        
        trend_type = st.selectbox("Trend type:", ["Demand Increase", "Demand Decrease", "Price Sensitivity", "Seasonal Preference"])
        crop_trend = st.selectbox("Affected crop:", ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes"])
        impact_score = st.slider("Impact score (1-10):", 1, 10, 5)
        trend_description = st.text_area("Trend description:")
        
        if st.button("Add Consumer Trend"):
            trend_entry = {
                'type': trend_type,
                'crop': crop_trend,
                'impact': impact_score,
                'description': trend_description,
                'date': datetime.now()
            }
            
            success = components['market_data'].add_consumer_trend(trend_entry)
            if success:
                st.success("Consumer trend data added successfully!")
                st.rerun()
            else:
                st.error("Failed to add consumer trend data. Please try again.")

elif page == "Multi-Crop Comparison":
    st.title("Multi-Crop Demand Comparison")
    
    # Crop selection for comparison
    comparison_crops = st.multiselect(
        "Select crops to compare:",
        ["Wheat", "Corn", "Soybeans", "Rice", "Tomatoes", "Potatoes", "Cotton", "Sugar"],
        default=["Wheat", "Corn", "Soybeans"]
    )
    
    if len(comparison_crops) >= 2:
        # Comparison parameters
        col1, col2 = st.columns(2)
        with col1:
            comparison_weeks = st.slider("Comparison period (weeks):", 1, 12, 8)
        with col2:
            comparison_metric = st.selectbox(
                "Comparison metric:",
                ["Demand Forecast", "Price Volatility", "Market Share", "Growth Potential"]
            )
        
        if st.button("Generate Comparison"):
            with st.spinner("Generating multi-crop comparison..."):
                # Generate comparison data
                comparison_data = components['forecaster'].generate_multi_crop_comparison(
                    comparison_crops, comparison_weeks, comparison_metric
                )
                
                if comparison_data:
                    # Create comparison visualizations
                    st.subheader(f"Crop Comparison - {comparison_metric}")
                    
                    # Side-by-side comparison chart
                    comparison_chart = components['visualizer'].create_comparison_chart(
                        comparison_data, comparison_metric
                    )
                    st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Ranking table
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Performance Ranking")
                        ranking_df = pd.DataFrame([
                            {"Crop": crop, "Score": data['score'], "Trend": data['trend']}
                            for crop, data in comparison_data.items()
                        ]).sort_values('Score', ascending=False)
                        st.dataframe(ranking_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("AI Recommendations")
                        recommendations = components['ai_analyzer'].generate_crop_recommendations(
                            comparison_data
                        )
                        if recommendations:
                            st.write(recommendations)
                        else:
                            st.error("Unable to generate recommendations. Please check API configuration.")
                else:
                    st.error("Unable to generate comparison data. Please try again.")
    else:
        st.info("Please select at least 2 crops for comparison.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources:**")
st.sidebar.markdown("• Market data APIs")
st.sidebar.markdown("• Weather services")
st.sidebar.markdown("• Consumer trend analysis")
st.sidebar.markdown("---")
st.sidebar.markdown("🤖 Powered by AI Analytics")
