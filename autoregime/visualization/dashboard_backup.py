"""
Enhanced AutoRegime Dashboard with Custom Asset Selection
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from autoregime.core.regime_detection import AutoRegimeDetector
    from autoregime.utils.data_loader import MarketDataLoader
except ImportError:
    st.error("❌ AutoRegime modules not found. Please ensure proper installation.")
    st.stop()

def main():
    st.set_page_config(
        page_title="AutoRegime Dashboard",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 AutoRegime: AI-Powered Market Analysis Dashboard")
    st.markdown("**Professional market regime detection for any asset**")
    
    # Sidebar configuration
    st.sidebar.header("📊 Asset Configuration")
    
    # Asset selection methods
    selection_method = st.sidebar.radio(
        "Choose Selection Method:",
        ["Custom Ticker", "Popular Categories"]
    )
    
    # Initialize variables
    selected_asset = None
    
    if selection_method == "Custom Ticker":
        # Custom ticker input
        custom_ticker = st.sidebar.text_input(
            "Enter Stock Ticker:",
            value="NVDA",
            help="Enter any valid ticker (NVDA, AAPL, BTC-USD, etc.)",
            placeholder="e.g., NVDA, TSLA, SPY"
        )
        selected_asset = custom_ticker.upper().strip()
        
        # Quick examples
        st.sidebar.markdown("**Quick Examples:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("NVDA"):
                selected_asset = "NVDA"
            if st.button("AAPL"):
                selected_asset = "AAPL"
        with col2:
            if st.button("TSLA"):
                selected_asset = "TSLA"
            if st.button("SPY"):
                selected_asset = "SPY"
    
    else:
        # Category selection
        category = st.sidebar.selectbox(
            "Choose Category:",
            ["Popular Stocks", "Tech Giants", "Market Indices", "Commodities", "Crypto"]
        )
        
        category_options = {
            "Popular Stocks": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX"],
            "Market Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VEA", "VWO"],
            "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC"],
            "Crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD"]
        }
        
        selected_asset = st.sidebar.selectbox(
            f"Select {category}:",
            category_options[category]
        )
    
    # Date range selection
    st.sidebar.subheader("📅 Analysis Period")
    
    # Preset periods
    period_preset = st.sidebar.selectbox(
        "Quick Periods:",
        ["Custom", "Last 2 Years", "Last 3 Years", "Last 5 Years", "COVID Period (2019-2022)", "All Available"]
    )
    
    if period_preset == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:", 
                value=datetime.now() - timedelta(days=730)  # 2 years default
            )
        with col2:
            end_date = st.date_input(
                "End Date:", 
                value=datetime.now()
            )
    else:
        period_mapping = {
            "Last 2 Years": (datetime.now() - timedelta(days=730), datetime.now()),
            "Last 3 Years": (datetime.now() - timedelta(days=1095), datetime.now()),
            "Last 5 Years": (datetime.now() - timedelta(days=1825), datetime.now()),
            "COVID Period (2019-2022)": (datetime(2019, 1, 1), datetime(2022, 12, 31)),
            "All Available": (datetime(2010, 1, 1), datetime.now())
        }
        start_date, end_date = period_mapping[period_preset]
        st.sidebar.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Analysis parameters
    st.sidebar.subheader("🔧 Analysis Settings")
    max_regimes = st.sidebar.slider("Maximum Regimes:", 3, 8, 5)
    min_duration = st.sidebar.slider("Min Regime Duration (days):", 3, 15, 5)
    
    # Analysis button
    analyze_button = st.sidebar.button("🧠 Run AI Analysis", type="primary")
    
    # Main content area
    if not selected_asset:
        st.info("👈 Please select an asset from the sidebar to begin analysis")
        return
    
    if analyze_button:
        run_analysis(selected_asset, start_date, end_date, max_regimes, min_duration)
    else:
        # Show asset info without analysis
        show_asset_preview(selected_asset)

def show_asset_preview(ticker):
    """Show basic asset information before analysis"""
    try:
        st.subheader(f"📊 Asset Preview: {ticker}")
        
        with st.spinner("Loading basic asset information..."):
            # Get basic info
            asset = yf.Ticker(ticker)
            info = asset.info
            hist = asset.history(period="5d")
            
            if hist.empty:
                st.error(f"❌ Invalid ticker: {ticker}")
                return
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${hist['Close'].iloc[-1]:.2f}" if len(hist) > 0 else "N/A"
                )
            
            with col2:
                if len(hist) > 1:
                    change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                    change_pct = (change / hist['Close'].iloc[-2]) * 100
                    st.metric(
                        "Daily Change",
                        f"${change:.2f}",
                        f"{change_pct:+.2f}%"
                    )
            
            with col3:
                st.metric(
                    "Volume",
                    f"{hist['Volume'].iloc[-1]:,.0f}" if len(hist) > 0 else "N/A"
                )
            
            with col4:
                company_name = info.get('longName', ticker)
                st.metric("Asset Name", company_name[:20] + "..." if len(company_name) > 20 else company_name)
        
        st.info("👈 Click 'Run AI Analysis' to perform regime detection")
        
    except Exception as e:
        st.error(f"❌ Error loading asset info: {str(e)}")

def run_analysis(ticker, start_date, end_date, max_regimes, min_duration):
    """Run complete AutoRegime analysis"""
    
    try:
        # Step 1: Data Loading
        st.subheader(f"🎯 AutoRegime Analysis: {ticker}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📊 Loading market data...")
        progress_bar.progress(10)
        
        # Load data
        loader = MarketDataLoader()
        data = loader.load_market_data(
            [ticker], 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if len(data) < 50:
            st.error("❌ Insufficient data. Please select a longer time period or different asset.")
            return
        
        progress_bar.progress(30)
        status_text.text("🤖 Running AI regime detection...")
        
        # Step 2: Regime Detection
        detector = AutoRegimeDetector(
            max_regimes=max_regimes,
            min_regime_duration=min_duration,
            verbose=False
        )
        detector.fit(data)
        
        progress_bar.progress(70)
        status_text.text("📈 Generating analysis...")
        
        # Step 3: Results Analysis
        timeline = detector.get_regime_timeline()
        current_regime, confidence = detector.predict_current_regime(data.tail(21))
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display Results
        display_results(ticker, data, detector, timeline, current_regime, confidence, regime_name)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Suggestions:")
        st.write("• Check if ticker symbol is correct")
        st.write("• Try a different date range")
        st.write("• Ensure internet connection is stable")

def display_results(ticker, data, detector, timeline, current_regime, confidence, regime_name):
    """Display comprehensive analysis results"""
    
    # Current Status Metrics
    st.subheader("🎯 Current Market Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🚨 Current Regime",
            regime_name,
            f"{confidence:.1%} confidence"
        )
    
    with col2:
        current_price = data.iloc[-1, 0]
        prev_price = data.iloc[-2, 0] if len(data) > 1 else current_price
        price_change = ((current_price / prev_price) - 1) * 100
        st.metric(
            "💰 Latest Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col3:
        st.metric(
            "🧠 Regimes Detected",
            detector.optimal_n_regimes,
            "AI-discovered"
        )
    
    with col4:
        total_days = len(data)
        st.metric(
            "📅 Analysis Period",
            f"{total_days} days",
            f"{total_days/252:.1f} years"
        )
    
    # Regime Timeline Table
    st.subheader("📅 Detailed Regime Timeline")
    
    # Format timeline for display
    display_timeline = timeline.copy()
    display_timeline['Start_Date'] = display_timeline['Start_Date'].dt.strftime('%Y-%m-%d')
    display_timeline['End_Date'] = display_timeline['End_Date'].dt.strftime('%Y-%m-%d')
    display_timeline['Duration_Years'] = display_timeline['Duration_Years'].round(1)
    display_timeline['Annual_Return_Pct'] = display_timeline['Annual_Return_Pct'].round(1)
    display_timeline['Sharpe_Ratio'] = display_timeline['Sharpe_Ratio'].round(2)
    
    st.dataframe(
        display_timeline[[
            'Regime_Name', 'Start_Date', 'End_Date', 'Duration_Days',
            'Annual_Return_Pct', 'Sharpe_Ratio'
        ]],
        use_container_width=True,
        hide_index=True
    )
    
    # Regime Characteristics
    st.subheader("🔍 Regime Characteristics Analysis")
    
    for regime_id, char in detector.regime_characteristics.items():
        regime_name = detector.regime_names.get(regime_id, f'Regime {regime_id}')
        
        with st.expander(f"📊 {regime_name} - Detailed Metrics", expanded=(regime_id == current_regime)):
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Annual Return", f"{char['mean_return']:.1%}")
                st.metric("📊 Frequency", f"{char['frequency']:.1%}")
            
            with col2:
                st.metric("📉 Volatility", f"{char['volatility']:.1%}")
                st.metric("⏱️ Avg Duration", f"{char['avg_duration']:.0f} days")
            
            with col3:
                st.metric("🎯 Sharpe Ratio", f"{char['sharpe_ratio']:.2f}")
                st.metric("💥 Max Drawdown", f"{char['max_drawdown']:.1%}")
            
            with col4:
                # Performance category
                if char['sharpe_ratio'] > 1.0:
                    category = "🟢 Excellent"
                elif char['sharpe_ratio'] > 0.5:
                    category = "🟡 Good" 
                elif char['sharpe_ratio'] > 0:
                    category = "🟠 Moderate"
                else:
                    category = "🔴 Poor"
                
                st.metric("Performance", category)
                
                # Risk level
                if char['volatility'] < 0.15:
                    risk = "🟢 Low Risk"
                elif char['volatility'] < 0.25:
                    risk = "🟡 Medium Risk"
                else:
                    risk = "🔴 High Risk"
                
                st.metric("Risk Level", risk)
    
    # Investment Insights
    st.subheader("💡 Investment Insights")
    
    current_char = detector.regime_characteristics.get(current_regime, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Regime Expectations:**")
        if current_char:
            expected_return = current_char.get('mean_return', 0) * 100
            expected_vol = current_char.get('volatility', 0) * 100
            
            st.write(f"• Expected annual return: **{expected_return:.1f}%**")
            st.write(f"• Expected volatility: **{expected_vol:.1f}%**")
            st.write(f"• Risk-adjusted performance: **{current_char.get('sharpe_ratio', 0):.2f}**")
    
    with col2:
        st.write("**Regime Recommendations:**")
        if current_char.get('sharpe_ratio', 0) > 1.0:
            st.success("🚀 **Favorable conditions** - Consider growth positions")
        elif current_char.get('sharpe_ratio', 0) > 0:
            st.info("⚖️ **Mixed conditions** - Balanced approach recommended")
        else:
            st.warning("⚠️ **Challenging conditions** - Consider defensive positioning")
    
    # Raw data download
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline CSV
        timeline_csv = timeline.to_csv(index=False)
        st.download_button(
            label="📊 Download Timeline CSV",
            data=timeline_csv,
            file_name=f"{ticker}_regime_timeline.csv",
            mime="text/csv"
        )
    
    with col2:
        # Analysis summary
        summary_text = f"""AutoRegime Analysis Summary - {ticker}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Current Regime: {regime_name} ({confidence:.1%} confidence)
Total Regimes Detected: {detector.optimal_n_regimes}
Analysis Period: {len(data)} days

Regime Characteristics:
"""
        for regime_id, char in detector.regime_characteristics.items():
            name = detector.regime_names.get(regime_id, f'Regime {regime_id}')
            summary_text += f"""
{name}:
- Annual Return: {char['mean_return']:.1%}
- Volatility: {char['volatility']:.1%}
- Sharpe Ratio: {char['sharpe_ratio']:.2f}
- Frequency: {char['frequency']:.1%}
"""
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=summary_text,
            file_name=f"{ticker}_autoregime_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()