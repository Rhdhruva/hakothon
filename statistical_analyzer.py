import pandas as pd
import numpy as np
from typing import Dict, List
import streamlit as st

class StatisticalAnalyzer:
    """Statistical analysis engine to replace OpenAI dependency using APY data"""
    
    def __init__(self, apy_loader):
        self.apy_loader = apy_loader
    
    def analyze_market_trends(self, market_data):
        """Analyze market trends using statistical methods"""
        try:
            if market_data is None or market_data.empty:
                return "No market data available for analysis"
            
            # Extract crop name if available
            crop_name = market_data.get('crop', 'Unknown Crop') if isinstance(market_data, dict) else 'Unknown'
            
            # Get statistical insights from the data
            analysis_parts = []
            
            # Basic statistics
            if 'price' in market_data.columns:
                price_mean = market_data['price'].mean()
                price_trend = market_data['price'].pct_change().mean()
                volatility = market_data['price'].std()
                
                analysis_parts.append(f"**Market Overview for {crop_name}:**")
                analysis_parts.append(f"• Average price: ₹{price_mean:.2f} per unit")
                analysis_parts.append(f"• Price trend: {'Increasing' if price_trend > 0 else 'Decreasing' if price_trend < 0 else 'Stable'} ({price_trend*100:.1f}%)")
                analysis_parts.append(f"• Price volatility: {'High' if volatility > price_mean * 0.2 else 'Moderate' if volatility > price_mean * 0.1 else 'Low'}")
            
            # Seasonal patterns
            if 'date' in market_data.columns:
                market_data['month'] = pd.to_datetime(market_data['date']).dt.month
                monthly_avg = market_data.groupby('month')['price'].mean()
                peak_month = monthly_avg.idxmax()
                low_month = monthly_avg.idxmin()
                
                analysis_parts.append(f"\n**Seasonal Patterns:**")
                analysis_parts.append(f"• Peak season: Month {peak_month} (₹{monthly_avg[peak_month]:.2f})")
                analysis_parts.append(f"• Low season: Month {low_month} (₹{monthly_avg[low_month]:.2f})")
                analysis_parts.append(f"• Seasonal variation: {((monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100):.1f}%")
            
            # Volume analysis
            if 'volume' in market_data.columns:
                volume_trend = market_data['volume'].pct_change().mean()
                analysis_parts.append(f"\n**Trading Volume:**")
                analysis_parts.append(f"• Volume trend: {'Increasing' if volume_trend > 0 else 'Decreasing'} ({volume_trend*100:.1f}%)")
                analysis_parts.append(f"• Average daily volume: {market_data['volume'].mean():,.0f} units")
            
            # Market outlook
            analysis_parts.append(f"\n**Market Outlook:**")
            if price_trend > 0.05:
                analysis_parts.append("• **Bullish**: Strong upward price momentum suggests good demand")
                analysis_parts.append("• Recommendation: Favorable for producers, buyers should consider forward contracts")
            elif price_trend < -0.05:
                analysis_parts.append("• **Bearish**: Declining prices may indicate oversupply or weak demand")
                analysis_parts.append("• Recommendation: Good buying opportunity, producers should focus on cost optimization")
            else:
                analysis_parts.append("• **Neutral**: Stable price environment with balanced supply-demand")
                analysis_parts.append("• Recommendation: Monitor for trend changes, maintain current strategies")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Statistical analysis unavailable: {str(e)}"
    
    def analyze_forecast(self, forecast_data):
        """Analyze forecast results using statistical methods"""
        try:
            if not forecast_data:
                return "No forecast data available for analysis"
            
            crop = forecast_data.get('crop', 'Unknown')
            confidence = forecast_data.get('confidence', 0.5)
            trend = forecast_data.get('trend', 'stable')
            values = forecast_data.get('values', [])
            
            if not values:
                return "Insufficient forecast data for analysis"
            
            analysis_parts = []
            
            # Forecast reliability assessment
            analysis_parts.append(f"**Forecast Analysis for {crop}:**")
            
            reliability = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
            analysis_parts.append(f"• **Forecast Reliability**: {reliability} ({confidence:.1%} confidence)")
            
            # Trend analysis
            value_change = (values[-1] - values[0]) / values[0] * 100
            analysis_parts.append(f"• **Trend Direction**: {trend.title()} ({value_change:+.1f}% change predicted)")
            
            # Volatility assessment
            forecast_volatility = np.std(values) / np.mean(values)
            volatility_desc = "High" if forecast_volatility > 0.15 else "Moderate" if forecast_volatility > 0.08 else "Low"
            analysis_parts.append(f"• **Demand Stability**: {volatility_desc} volatility expected")
            
            # Key drivers (based on forecast parameters)
            analysis_parts.append(f"\n**Key Driving Factors:**")
            
            historical_data = self.apy_loader.get_market_trends(crop, 3)
            if historical_data:
                historical_growth = historical_data.get('total_production_growth', 0)
                if abs(historical_growth) > 5:
                    analysis_parts.append(f"• Historical production trend: {historical_growth:+.1f}% growth over recent years")
                
            if confidence > 0.75:
                analysis_parts.append("• Strong historical data patterns support forecast reliability")
            
            # Market scenarios
            analysis_parts.append(f"\n**Scenario Analysis:**")
            
            best_case = max(values) * 1.1
            worst_case = min(values) * 0.9
            most_likely = np.mean(values)
            
            analysis_parts.append(f"• **Best Case**: Peak demand could reach {best_case:.1f} index points")
            analysis_parts.append(f"• **Most Likely**: Average demand around {most_likely:.1f} index points")
            analysis_parts.append(f"• **Worst Case**: Minimum demand may drop to {worst_case:.1f} index points")
            
            # Recommendations
            analysis_parts.append(f"\n**Strategic Recommendations:**")
            
            if trend == 'increasing':
                analysis_parts.append("• **For Farmers**: Consider expanding cultivation area for this crop")
                analysis_parts.append("• **For Traders**: Build inventory ahead of peak demand periods")
                analysis_parts.append("• **Risk Management**: Monitor supply chain capacity")
            elif trend == 'decreasing':
                analysis_parts.append("• **For Farmers**: Diversify crop portfolio, consider alternative crops")
                analysis_parts.append("• **For Traders**: Implement just-in-time inventory strategies")
                analysis_parts.append("• **Risk Management**: Focus on cost optimization")
            else:
                analysis_parts.append("• **For All Stakeholders**: Maintain current production/inventory levels")
                analysis_parts.append("• **Monitor**: Watch for trend changes in coming weeks")
            
            # Risk factors
            analysis_parts.append(f"\n**Risk Factors to Monitor:**")
            analysis_parts.append("• Weather patterns affecting crop yields")
            analysis_parts.append("• Government policy changes on agricultural subsidies")
            analysis_parts.append("• Global commodity price fluctuations")
            if forecast_volatility > 0.12:
                analysis_parts.append("• High forecast uncertainty - monitor actual vs predicted closely")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Forecast analysis unavailable: {str(e)}"
    
    def generate_crop_recommendations(self, comparison_data):
        """Generate crop recommendations based on statistical comparison"""
        try:
            if not comparison_data:
                return "No comparison data available for recommendations"
            
            # Sort crops by score
            sorted_crops = sorted(comparison_data.items(), key=lambda x: x[1].get('score', 0), reverse=True)
            
            analysis_parts = []
            analysis_parts.append("## Statistical Crop Analysis & Recommendations")
            
            # Top recommendations
            analysis_parts.append("\n### Priority Crops for Investment:")
            
            for i, (crop, data) in enumerate(sorted_crops[:3], 1):
                score = data.get('score', 0)
                trend = data.get('trend', 'neutral')
                growth_potential = data.get('growth_potential', 50)
                volatility = data.get('volatility', 0)
                
                priority = "High" if score > 80 else "Medium" if score > 65 else "Low"
                risk_level = "High" if volatility > 15 else "Moderate" if volatility > 8 else "Low"
                
                analysis_parts.append(f"\n**{i}. {crop}** (Priority: {priority})")
                analysis_parts.append(f"   • Performance Score: {score:.1f}/100")
                analysis_parts.append(f"   • Market Trend: {trend.title()}")
                analysis_parts.append(f"   • Growth Potential: {growth_potential:.1f}%")
                analysis_parts.append(f"   • Risk Level: {risk_level}")
                
                # Get historical context
                historical_data = self.apy_loader.get_market_trends(crop, 5)
                if historical_data:
                    prod_growth = historical_data.get('total_production_growth', 0)
                    analysis_parts.append(f"   • Historical Growth: {prod_growth:+.1f}% over 5 years")
                    
                    # Reasoning
                    if score > 75:
                        analysis_parts.append(f"   • **Why Recommended**: Strong performance metrics and {trend} market trend")
                    elif growth_potential > 60:
                        analysis_parts.append(f"   • **Why Recommended**: High growth potential despite moderate current performance")
                    else:
                        analysis_parts.append(f"   • **Why Considered**: Stable option with manageable risk profile")
            
            # Market strategy
            analysis_parts.append(f"\n### Overall Market Strategy:")
            
            top_3_avg_score = np.mean([data.get('score', 0) for _, data in sorted_crops[:3]])
            top_3_avg_volatility = np.mean([data.get('volatility', 0) for _, data in sorted_crops[:3]])
            
            if top_3_avg_score > 75:
                analysis_parts.append("**Aggressive Growth Strategy**: Market conditions favor expansion in top-performing crops")
                analysis_parts.append("• Recommended allocation: 60% top crop, 25% second choice, 15% diversification")
            elif top_3_avg_volatility > 12:
                analysis_parts.append("**Diversified Risk Strategy**: High volatility suggests spreading investments")
                analysis_parts.append("• Recommended allocation: Equal weighting across top 3-4 crops")
            else:
                analysis_parts.append("**Balanced Growth Strategy**: Moderate performance suggests steady expansion")
                analysis_parts.append("• Recommended allocation: 40% top crop, 35% second choice, 25% others")
            
            # Risk assessment
            analysis_parts.append(f"\n### Risk Assessment:")
            
            high_risk_crops = [crop for crop, data in comparison_data.items() if data.get('volatility', 0) > 15]
            stable_crops = [crop for crop, data in comparison_data.items() if data.get('volatility', 0) < 8]
            
            if high_risk_crops:
                analysis_parts.append(f"• **High Risk Crops**: {', '.join(high_risk_crops)} - Higher volatility requires careful monitoring")
            
            if stable_crops:
                analysis_parts.append(f"• **Stable Options**: {', '.join(stable_crops)} - Lower risk, suitable for conservative portfolios")
            
            analysis_parts.append("• **Market Risks**: Weather dependency, policy changes, global price fluctuations")
            analysis_parts.append("• **Mitigation**: Regular market monitoring, diversified crop portfolio, forward contracts")
            
            # Timing advice
            analysis_parts.append(f"\n### Timing Recommendations:")
            
            increasing_trends = [crop for crop, data in comparison_data.items() if data.get('trend') == 'increasing']
            decreasing_trends = [crop for crop, data in comparison_data.items() if data.get('trend') == 'decreasing']
            
            if increasing_trends:
                analysis_parts.append(f"• **Immediate Action**: {', '.join(increasing_trends)} showing upward trends - consider early investment")
            
            if decreasing_trends:
                analysis_parts.append(f"• **Wait and Watch**: {', '.join(decreasing_trends)} in downtrend - monitor for reversal signals")
            
            analysis_parts.append("• **Seasonal Considerations**: Align planting/investment with historical seasonal patterns")
            analysis_parts.append("• **Review Schedule**: Monthly performance reviews recommended for top 3 crops")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Crop recommendations unavailable: {str(e)}"
    
    def analyze_seasonal_patterns(self, seasonal_data):
        """Analyze seasonal patterns using statistical methods"""
        try:
            if not seasonal_data:
                return "No seasonal data available for analysis"
            
            analysis_parts = []
            analysis_parts.append("## Seasonal Pattern Analysis")
            
            # Yearly production patterns
            yearly_prod = seasonal_data.get('yearly_production', {})
            if yearly_prod:
                years = sorted(yearly_prod.keys())
                productions = [yearly_prod[year] for year in years]
                
                analysis_parts.append(f"\n**Production Trends ({min(years)}-{max(years)}):**")
                
                # Calculate trend
                if len(productions) > 1:
                    trend_slope = (productions[-1] - productions[0]) / len(productions)
                    trend_direction = "Increasing" if trend_slope > 0 else "Decreasing"
                    analysis_parts.append(f"• Overall Trend: {trend_direction}")
                    analysis_parts.append(f"• Average Annual Change: {trend_slope:+,.0f} thousand tons")
                
                # Peak and trough years
                peak_year = seasonal_data.get('peak_production_year')
                low_year = seasonal_data.get('lowest_production_year')
                
                if peak_year:
                    analysis_parts.append(f"• Peak Production Year: {peak_year} ({yearly_prod.get(peak_year, 0):,.0f} thousand tons)")
                if low_year:
                    analysis_parts.append(f"• Lowest Production Year: {low_year} ({yearly_prod.get(low_year, 0):,.0f} thousand tons)")
            
            # Yield patterns
            yearly_yield = seasonal_data.get('yearly_yield', {})
            if yearly_yield:
                yields = list(yearly_yield.values())
                avg_yield = np.mean(yields)
                yield_volatility = np.std(yields) / avg_yield if avg_yield > 0 else 0
                
                analysis_parts.append(f"\n**Yield Performance:**")
                analysis_parts.append(f"• Average Yield: {avg_yield:.0f} kg/ha")
                analysis_parts.append(f"• Yield Stability: {'High' if yield_volatility < 0.15 else 'Moderate' if yield_volatility < 0.25 else 'Low'}")
            
            # Volatility assessment
            prod_volatility = seasonal_data.get('production_volatility', 0)
            yield_volatility = seasonal_data.get('yield_volatility', 0)
            
            analysis_parts.append(f"\n**Market Stability Analysis:**")
            
            if prod_volatility > 0:
                stability_desc = "Highly Variable" if prod_volatility > 500 else "Moderately Variable" if prod_volatility > 200 else "Stable"
                analysis_parts.append(f"• Production Volatility: {stability_desc}")
            
            # Seasonal trading strategies
            analysis_parts.append(f"\n**Seasonal Intelligence:**")
            
            if peak_year and low_year:
                cycle_length = abs(peak_year - low_year)
                if cycle_length <= 3:
                    analysis_parts.append("• **Pattern**: Short-term cycles observed - likely weather-driven")
                    analysis_parts.append("• **Strategy**: Focus on year-to-year weather predictions")
                else:
                    analysis_parts.append("• **Pattern**: Long-term cycles suggest structural market changes")
                    analysis_parts.append("• **Strategy**: Consider long-term market fundamentals")
            
            analysis_parts.append("• **Optimal Timing**: Plan planting based on historical high-yield periods")
            analysis_parts.append("• **Risk Management**: Higher volatility requires diversified planning")
            
            # Climate and weather factors
            analysis_parts.append(f"\n**Environmental Impact Factors:**")
            analysis_parts.append("• **Weather Dependency**: Monitor monsoon patterns and seasonal rainfall")
            analysis_parts.append("• **Temperature Sensitivity**: Track seasonal temperature variations")
            analysis_parts.append("• **Soil Conditions**: Consider seasonal soil moisture and nutrient levels")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Seasonal analysis unavailable: {str(e)}"