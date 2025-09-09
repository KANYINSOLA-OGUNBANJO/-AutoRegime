"""
Universal AutoRegime Dashboard - Analyze ANY Asset Worldwide
Fixed version with unique keys - no duplicate key errors
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from autoregime.core.regime_detection import AutoRegimeDetector
    from autoregime.utils.data_loader import MarketDataLoader
except ImportError:
    st.error("❌ AutoRegime modules not found. Please ensure proper installation.")
    st.stop()

def validate_and_get_info(ticker):
    """Universal ticker validation for any asset worldwide"""
    try:
        asset = yf.Ticker(ticker)
        
        # Test data availability
        hist = asset.history(period="5d")
        info = asset.info
        
        if hist.empty or len(hist) == 0:
            return False, "No data available", 0, "Unknown"
        
        # Extract information
        company_name = info.get('longName', 
                              info.get('shortName', 
                                      info.get('symbol', ticker)))
        
        if not company_name or company_name == ticker:
            # Try to get name from history
            company_name = f"{ticker} Asset"
        
        current_price = hist['Close'].iloc[-1]
        
        # Determine asset type
        asset_type = "Stock"
        if ticker.endswith('-USD') or ticker.endswith('-USDT'):
            asset_type = "Cryptocurrency"
        elif ticker.startswith('^'):
            asset_type = "Index"
        elif any(word in company_name.upper() for word in ['ETF', 'FUND', 'TRUST']):
            asset_type = "ETF"
        elif any(word in company_name.upper() for word in ['GOLD', 'SILVER', 'OIL', 'COMMODITY']):
            asset_type = "Commodity"
        
        return True, company_name, current_price, asset_type
        
    except Exception as e:
        return False, f"Error: {str(e)}", 0, "Unknown"

def main():
    st.set_page_config(
        page_title="AutoRegime Universal",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🌍 AutoRegime: Universal Asset Analysis")
    st.markdown("**Analyze ANY stock, crypto, ETF, or index from any exchange worldwide**")
    st.markdown("---")
    
    # Initialize session state
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = 'NVDA'
    
    # Sidebar - Universal Search Interface
    st.sidebar.header("🔍 Universal Asset Search")
    st.sidebar.markdown("*AI-powered financial analysis for research*")
    
    # Main ticker input
    ticker_input = st.sidebar.text_input(
        "🎯 Enter ANY asset ticker:",
        value=st.session_state.get('selected_ticker', 'NVDA'),
        placeholder="Examples: GOOGL, AMZN, SHOP, BTC-USD, ^GSPC",
        help="Works with ANY ticker from Yahoo Finance - stocks, crypto, ETFs, indices, commodities",
        key="main_ticker_input"
    )
    
    selected_ticker = ticker_input.upper().strip()
    
    # Update session state
    if selected_ticker != st.session_state.get('selected_ticker'):
        st.session_state.selected_ticker = selected_ticker
    
    # Real-time validation and info display
    if selected_ticker:
        with st.sidebar:
            with st.spinner("🔍 Validating ticker..."):
                is_valid, name, price, asset_type = validate_and_get_info(selected_ticker)
            
            if is_valid:
                st.success(f"✅ **{selected_ticker}** Found!")
                st.write(f"🏢 **{name}**")
                st.write(f"📊 Type: {asset_type}")
                st.write(f"💰 Current Price: ${price:.2f}")
                
                # Show some basic stats
                try:
                    asset = yf.Ticker(selected_ticker)
                    hist_week = asset.history(period="5d")
                    if len(hist_week) >= 2:
                        week_change = ((hist_week['Close'].iloc[-1] / hist_week['Close'].iloc[0]) - 1) * 100
                        st.write(f"📈 5-Day Change: {week_change:+.2f}%")
                except:
                    pass
                    
            else:
                st.error(f"❌ **{selected_ticker}**")
                st.write(f"⚠️ {name}")
                st.write("💡 **Try:**")
                st.write("• Check spelling")
                st.write("• Use Yahoo Finance format")
                st.write("• Try examples below")
    
    # Quick Examples - Simple version without duplicates
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Popular Examples")
    st.sidebar.markdown("*Click any example to analyze:*")
    
    # Organized examples
    example_sets = {
        "🚀 Popular Stocks": ["NVDA", "AAPL", "GOOGL", "AMZN", "TSLA", "MSFT"],
        "💰 Cryptocurrencies": ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD"],
        "📈 Market Indices": ["SPY", "QQQ", "^GSPC", "^IXIC", "^DJI"],
        "🏦 ETFs & Commodities": ["GLD", "SLV", "USO", "VTI", "XLK", "XLF"],
        "🌍 International": ["ASML", "TSM", "BABA", "SAP", "NESN.SW"]
    }
    
    # Display examples
    for category, tickers in example_sets.items():
        with st.sidebar.expander(category):
            # Create columns for buttons
            cols = st.columns(2)
            for i, ticker in enumerate(tickers):
                col = cols[i % 2]
                # Create unique key for each button
                button_key = f"btn_{category.replace(' ', '').replace('🚀', '').replace('💰', '').replace('📈', '').replace('🏦', '').replace('🌍', '')}_{i}_{ticker.replace('-', '_').replace('^', 'idx_').replace('.', '_')}"
                if col.button(ticker, key=button_key):
                    st.session_state.selected_ticker = ticker
                    st.experimental_rerun()
    
    # Additional search help
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Search Tips")
    st.sidebar.write("**US Stocks**: AAPL, GOOGL, SHOP, PLTR")
    st.sidebar.write("**Crypto**: BTC-USD, ETH-USD")
    st.sidebar.write("**Indices**: ^GSPC (S&P 500), ^IXIC (NASDAQ)")
    st.sidebar.write("**International**: ASML, TSM, BABA")
    st.sidebar.write("**ETFs**: SPY, QQQ, GLD, VTI")
    
    # Date Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Analysis Period")
    
    # Preset period options
    period_options = {
        "Last 6 Months": 180,
        "Last 1 Year": 365,
        "Last 2 Years": 730, 
        "Last 3 Years": 1095,
        "Last 5 Years": 1825,
        "COVID Era (2019-2022)": "covid",
        "All Available Data": "max",
        "Custom Range": "custom"
    }
    
    selected_period = st.sidebar.selectbox(
        "Select Analysis Period:", 
        list(period_options.keys()),
        index=2  # Default to "Last 2 Years"
    )
    
    # Calculate dates based on selection
    if selected_period == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=730),
                max_value=datetime.now(),
                key="custom_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=datetime.now(),
                max_value=datetime.now(),
                key="custom_end_date"
            )
    elif period_options[selected_period] == "covid":
        start_date = datetime(2019, 1, 1).date()
        end_date = datetime(2022, 12, 31).date()
        st.sidebar.info("📅 COVID Analysis Period")
    elif period_options[selected_period] == "max":
        start_date = datetime(2010, 1, 1).date()
        end_date = datetime.now().date()
        st.sidebar.info("📅 Maximum Available Data")
    else:
        days = period_options[selected_period]
        start_date = (datetime.now() - timedelta(days=days)).date()
        end_date = datetime.now().date()
        st.sidebar.info(f"📅 {selected_period}")
    
    # AI Analysis Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ AI Analysis Settings")
    
    max_regimes = st.sidebar.slider(
        "Maximum Regimes to Detect:", 
        min_value=3, 
        max_value=8, 
        value=5,
        help="AI will find the optimal number up to this limit"
    )
    
    min_duration = st.sidebar.slider(
        "Minimum Regime Duration (days):",
        min_value=3,
        max_value=21,
        value=5,
        help="Minimum days a regime must persist to be valid"
    )
    
    # Analysis Button
    st.sidebar.markdown("---")
    analysis_ready = selected_ticker and ('is_valid' in locals() and is_valid)
    
    analyze_button = st.sidebar.button(
        "🤖 Run AI Analysis", 
        type="primary",
        disabled=not analysis_ready,
        help="Analyze the selected asset using AutoRegime AI"
    )
    
    # Main Content Area
    if not selected_ticker:
        # Welcome screen
        st.info("👈 **Enter any ticker symbol in the sidebar to begin analysis**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🎯 What AutoRegime Does")
            st.write("• **AI-Powered Analysis**: Automatically detects market regimes")
            st.write("• **Universal Coverage**: Any stock, crypto, ETF, index worldwide")
            st.write("• **Professional Insights**: Timeline, characteristics, predictions")
        
        with col2:
            st.subheader("🔍 Search Examples")
            st.write("• **US Stocks**: AAPL, GOOGL, TSLA, NVDA")
            st.write("• **Crypto**: BTC-USD, ETH-USD, DOGE-USD")
            st.write("• **International**: ASML, TSM, BABA")
        
        with col3:
            st.subheader("📊 Analysis Features")
            st.write("• **Current Regime**: What regime is active now")
            st.write("• **Historical Timeline**: Exact dates and periods")
            st.write("• **Risk Metrics**: Sharpe ratios, volatility, drawdowns")
        
        return
    
    # Asset validation status
    if 'is_valid' in locals() and selected_ticker:
        if not is_valid:
            st.error(f"❌ **{selected_ticker}** is not a valid ticker symbol")
            st.write("**Please try:**")
            st.write("• Check the spelling of the ticker symbol")
            st.write("• Use Yahoo Finance format (e.g., BTC-USD for Bitcoin)")
            st.write("• Try one of the examples from the sidebar")
            return
    
    # Run analysis or show ready state
    if analyze_button and analysis_ready:
        analyze_universal_asset(selected_ticker, name, start_date, end_date, max_regimes, min_duration)
    elif analysis_ready:
        # Show ready state
        st.info(f"👆 **Ready to analyze {name} ({selected_ticker})**")
        st.write(f"📅 Period: {start_date} to {end_date}")
        st.write(f"⚙️ Max Regimes: {max_regimes}, Min Duration: {min_duration} days")
        st.write("**Click 'Run AI Analysis' in the sidebar to begin**")

def analyze_universal_asset(ticker, name, start_date, end_date, max_regimes, min_duration):
    """Analyze any asset universally with comprehensive results"""
    
    try:
        st.subheader(f"🎯 AutoRegime AI Analysis: {ticker}")
        st.write(f"**{name}**")
        st.write(f"📅 Analysis Period: {start_date} to {end_date}")
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress = st.progress(0)
            status = st.empty()
        
        # Step 1: Load Data
        status.text("📊 Loading market data from Yahoo Finance...")
        progress.progress(10)
        
        loader = MarketDataLoader()
        data = loader.load_market_data(
            [ticker],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if len(data) < 50:
            st.error("❌ **Insufficient data for reliable analysis**")
            st.write(f"Found only {len(data)} data points. Need at least 50.")
            st.write("**Suggestions:**")
            st.write("• Try a longer time period")
            st.write("• Check if the asset has sufficient trading history")
            st.write("• Some newer assets may have limited data")
            return
        
        progress.progress(30)
        status.text("🧠 Initializing AI regime detector...")
        
        # Step 2: AI Analysis
        detector = AutoRegimeDetector(
            max_regimes=max_regimes,
            min_regime_duration=min_duration,
            verbose=False
        )
        
        progress.progress(50)
        status.text("🤖 Running machine learning analysis...")
        
        detector.fit(data)
        
        progress.progress(70)
        status.text("📈 Generating insights and predictions...")
        
        # Step 3: Extract Results
        timeline = detector.get_regime_timeline()
        current_regime, confidence = detector.predict_current_regime(data.tail(21))
        regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
        
        progress.progress(90)
        status.text("✨ Finalizing analysis...")
        
        # Step 4: Additional Calculations
        total_return = ((data.iloc[-1, 0] / data.iloc[0, 0]) - 1) * 100
        annual_volatility = data.pct_change().std().iloc[0] * (252**0.5) * 100
        
        progress.progress(100)
        status.text("✅ Analysis complete!")
        
        # Clear progress indicators
        progress.empty()
        status.empty()
        
        # Display comprehensive results
        display_universal_results(
            ticker, name, data, detector, timeline, 
            current_regime, confidence, regime_name,
            total_return, annual_volatility, start_date, end_date
        )
        
    except Exception as e:
        st.error(f"❌ **Analysis Failed**: {str(e)}")
        
        # Detailed error handling
        error_msg = str(e).lower()
        st.write("**Possible solutions:**")
        
        if "no data" in error_msg or "empty" in error_msg:
            st.write("• **Data Issue**: Try a different date range or ticker")
            st.write("• **Delisted Asset**: Check if the asset is still trading")
        elif "connection" in error_msg or "timeout" in error_msg:
            st.write("• **Network Issue**: Check internet connection")
            st.write("• **API Issue**: Try again in a few minutes")
        elif "invalid" in error_msg or "not found" in error_msg:
            st.write("• **Invalid Ticker**: Check ticker symbol spelling")
            st.write("• **Format Issue**: Use Yahoo Finance format (BTC-USD, not BTC)")
        else:
            st.write("• Try a different ticker symbol")
            st.write("• Try a different date range")
            st.write("• Check internet connection")

def display_universal_results(ticker, name, data, detector, timeline, current_regime, confidence, regime_name, total_return, annual_volatility, start_date, end_date):
    """Display comprehensive analysis results with professional formatting"""
    
    st.success("🎉 **Analysis Successfully Completed!**")
    
    # Current Status Dashboard
    st.subheader("🎯 Current Market Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🚨 Current Regime", 
            regime_name,
            f"{confidence:.1%} confidence",
            help=f"AI predicts {ticker} is currently in {regime_name} regime"
        )
    
    with col2:
        current_price = data.iloc[-1, 0]
        prev_price = data.iloc[-22, 0] if len(data) >= 22 else data.iloc[0, 0]
        recent_change = ((current_price / prev_price) - 1) * 100
        st.metric(
            "💰 Latest Price", 
            f"${current_price:.2f}",
            f"{recent_change:+.2f}% (21d)",
            help="Current price with 21-day change"
        )
    
    with col3:
        st.metric(
            "🧠 Regimes Detected", 
            detector.optimal_n_regimes,
            "AI-optimized",
            help="AI automatically found optimal number of regimes"
        )
    
    with col4:
        analysis_days = len(data)
        st.metric(
            "📊 Data Analyzed", 
            f"{analysis_days} days",
            f"{analysis_days/252:.1f} years",
            help=f"Analysis covers {analysis_days} trading days"
        )
    
    # Performance Summary
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Total Return",
            f"{total_return:+.1f}%",
            help=f"Total return from {start_date} to {end_date}"
        )
    
    with col2:
        annualized_return = (((data.iloc[-1, 0] / data.iloc[0, 0]) ** (252/len(data))) - 1) * 100
        st.metric(
            "📊 Annualized Return",
            f"{annualized_return:+.1f}%",
            help="Compound annual growth rate"
        )
    
    with col3:
        st.metric(
            "📉 Annual Volatility",
            f"{annual_volatility:.1f}%",
            help="Annualized price volatility"
        )
    
    with col4:
        sharpe_estimate = annualized_return / annual_volatility if annual_volatility > 0 else 0
        st.metric(
            "⚡ Sharpe Estimate",
            f"{sharpe_estimate:.2f}",
            help="Risk-adjusted return estimate"
        )
    
    # Regime Timeline
    st.subheader("📅 Detailed Regime Timeline")
    st.write(f"*Complete chronological breakdown of {ticker}'s market regimes*")
    
    # Enhanced timeline display
    display_timeline = timeline.copy()
    display_timeline['Start_Date'] = display_timeline['Start_Date'].dt.strftime('%Y-%m-%d')
    display_timeline['End_Date'] = display_timeline['End_Date'].dt.strftime('%Y-%m-%d')
    display_timeline['Duration_Years'] = display_timeline['Duration_Years'].round(2)
    display_timeline['Annual_Return_Pct'] = display_timeline['Annual_Return_Pct'].round(1)
    display_timeline['Sharpe_Ratio'] = display_timeline['Sharpe_Ratio'].round(2)
    
    st.dataframe(
        display_timeline[['Regime_Name', 'Start_Date', 'End_Date', 'Duration_Days', 'Annual_Return_Pct', 'Sharpe_Ratio']],
        use_container_width=True,
        hide_index=True
    )
    
    # Regime Characteristics Analysis
    st.subheader("🔍 Regime Characteristics Deep Dive")
    st.write(f"*Detailed analysis of each regime detected in {ticker}*")
    
    # Sort regimes by Sharpe ratio for better display
    regime_items = sorted(
        detector.regime_characteristics.items(),
        key=lambda x: x[1]['sharpe_ratio'],
        reverse=True
    )
    
    for regime_id, char in regime_items:
        regime_display_name = detector.regime_names.get(regime_id, f'Regime {regime_id}')
        
        # Determine if this is the current regime
        is_current = (regime_id == current_regime)
        
        with st.expander(
            f"{'🔥 ' if is_current else '📈 '}{regime_display_name}{' (CURRENT)' if is_current else ''}", 
            expanded=is_current
        ):
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Annual Return", f"{char['mean_return']:.1%}")
                st.metric("⏱️ Avg Duration", f"{char['avg_duration']:.0f} days")
                
            with col2:
                st.metric("📉 Volatility", f"{char['volatility']:.1%}")
                st.metric("🔄 Frequency", f"{char['frequency']:.1%}")
                
            with col3:
                st.metric("🎯 Sharpe Ratio", f"{char['sharpe_ratio']:.2f}")
                st.metric("💥 Max Drawdown", f"{char['max_drawdown']:.1%}")
            
            # Performance interpretation
            if char['sharpe_ratio'] > 1.5:
                performance_desc = "🟢 **Excellent** - Outstanding risk-adjusted returns"
            elif char['sharpe_ratio'] > 1.0:
                performance_desc = "🟢 **Very Good** - Strong risk-adjusted performance"
            elif char['sharpe_ratio'] > 0.5:
                performance_desc = "🟡 **Good** - Moderate risk-adjusted returns"
            elif char['sharpe_ratio'] > 0:
                performance_desc = "🟠 **Fair** - Positive but below-average performance"
            else:
                performance_desc = "🔴 **Poor** - Negative risk-adjusted returns"
            
            st.write(f"**Performance**: {performance_desc}")
            
            # Investment implications for current regime
            if is_current:
                st.write("---")
                st.write("**🎯 Current Regime Implications:**")
                
                if char['sharpe_ratio'] > 1.0:
                    st.success("🚀 **Favorable Environment** - Consider growth-oriented positions")
                elif char['sharpe_ratio'] > 0:
                    st.info("⚖️ **Mixed Environment** - Balanced approach recommended")
                else:
                    st.warning("⚠️ **Challenging Environment** - Consider defensive positioning")
    
    # Export Options
    st.subheader("📥 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Timeline CSV export
        timeline_csv = timeline.to_csv(index=False)
        st.download_button(
            label="📊 Download Timeline (CSV)",
            data=timeline_csv,
            file_name=f"{ticker}_autoregime_timeline_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download detailed regime timeline as CSV file"
        )
    
    with col2:
        # Raw data export
        raw_data_csv = data.to_csv()
        st.download_button(
            label="📈 Download Raw Data (CSV)",
            data=raw_data_csv,
            file_name=f"{ticker}_raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download raw price data used in analysis"
        )
    
    with col3:
        # Summary report
        summary_text = f"""AutoRegime Analysis Summary - {ticker}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Current Regime: {regime_name} ({confidence:.1%} confidence)
Total Regimes Detected: {detector.optimal_n_regimes}
Analysis Period: {len(data)} days ({start_date} to {end_date})
Total Return: {total_return:.1f}%
Annual Volatility: {annual_volatility:.1f}%

Regime Characteristics:
"""
        for regime_id, char in detector.regime_characteristics.items():
            name_text = detector.regime_names.get(regime_id, f'Regime {regime_id}')
            summary_text += f"""
{name_text}:
- Annual Return: {char['mean_return']:.1%}
- Volatility: {char['volatility']:.1%}
- Sharpe Ratio: {char['sharpe_ratio']:.2f}
- Frequency: {char['frequency']:.1%}
"""
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"{ticker}_autoregime_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            help="Download comprehensive analysis summary"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*Analysis powered by AutoRegime AI - Professional market intelligence for any asset worldwide*")

if __name__ == "__main__":
    main()