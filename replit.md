# AgriForecaster - AI Crop Demand Prediction Platform

## Overview

AgriForecaster is a comprehensive agricultural analytics platform that leverages real APY (Area, Production, Yield) datasets to provide statistical analysis, market trend insights, and crop demand forecasting. The application combines multiple analytical approaches including statistical modeling, machine learning forecasting, and interactive data visualization to help agricultural stakeholders make informed decisions about crop planning and market strategies.

The platform processes real agricultural data from Kaggle's APY dataset, providing historically-grounded predictions rather than synthetic forecasts. It offers multi-crop comparison capabilities, seasonal pattern analysis, and market trend evaluation through an intuitive Streamlit-based web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with interactive dashboard
- **UI Components**: Multi-page navigation with Dashboard, Market Analysis, Demand Forecasting, Data Input, and Multi-Crop Comparison sections
- **Visualization**: Plotly-based charting system for interactive graphs and forecasting displays
- **State Management**: Streamlit session state with cached resource initialization

### Backend Architecture
- **Modular Design**: Component-based architecture with specialized classes for different functionalities
- **Data Processing Pipeline**: Segregated data loading, processing, and analysis layers
- **Analytics Engine**: Statistical analysis engine that replaces traditional AI/ML dependencies with mathematical approaches
- **Forecasting System**: Multi-model forecasting combining linear regression and random forest algorithms

### Core Components
- **APYDataLoader**: Centralized data management for real agricultural datasets
- **StatisticalAnalyzer**: Statistical computation engine for market trend analysis
- **CropForecaster**: Demand prediction system using historical APY data
- **DataProcessor**: Raw data transformation and feature engineering
- **MarketDataManager**: Historical market data retrieval and caching
- **Visualizer**: Comprehensive plotting and chart generation system

### Data Architecture
- **Primary Data Source**: CSV-based APY dataset containing historical agricultural metrics
- **Data Caching**: In-memory caching system for improved performance
- **Data Processing**: Pandas-based data manipulation with NumPy statistical computations
- **Feature Engineering**: Moving averages, volatility calculations, seasonal decomposition, and price change analytics

### Machine Learning Architecture
- **Model Types**: Linear regression for trend analysis, Random Forest for complex pattern recognition
- **Training Approach**: Historical APY data training with cross-validation
- **Prediction Pipeline**: Multi-step forecasting with confidence interval estimation
- **Model Evaluation**: Mean Absolute Error and R² score metrics for accuracy assessment

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for interactive dashboard creation
- **Pandas**: Data manipulation and analysis library for dataset processing
- **NumPy**: Numerical computation library for statistical calculations
- **Plotly**: Interactive visualization library for charts and graphs

### Machine Learning Libraries
- **scikit-learn**: Machine learning library providing LinearRegression and RandomForestRegressor models
- **sklearn.preprocessing**: Data preprocessing tools including StandardScaler
- **sklearn.decomposition**: PCA for dimensionality reduction
- **sklearn.metrics**: Model evaluation metrics (MAE, R² score)

### Data Sources
- **APY Dataset**: Kaggle-sourced CSV file containing Area, Production, and Yield data for various crops
- **Historical Market Data**: Time-series agricultural data spanning multiple years and crop types

### Utility Libraries
- **datetime/timedelta**: Date and time manipulation for temporal analysis
- **json**: Data serialization for configuration and caching
- **typing**: Type hints for improved code documentation and IDE support