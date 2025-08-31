"""
AutoRegime Interactive Dashboard
Real-time market regime monitoring with professional interface
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import our AutoRegime system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from autoregime.core.regime_detection import AutoRegimeDetector
from autoregime.visualization.regime_plots import RegimeVisualizer
from autoregime.utils.data_loader import MarketDataLoader

class AutoRegimeDashboard:
    """
    Professional interactive dashboard for AutoRegime analysis.
    
    Features:
    - Real-time regime monitoring
    - Interactive visualizations
    - Portfolio recommendations
    - Risk alerts
    - Professional UI
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.detector = None
        self.visualizer = None
        self.data = None
        self.loader = MarketDataLoader()
        
        # Dashboard configuration
        st.set_page_config(
            page_title="AutoRegime Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._apply_custom_styles()
    
    def _apply_custom_styles(self):
        """Apply custom styling to the dashboard."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .regime-alert {
            padding: 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            text-align: center;
            margin: 1rem 0;
        }
        .bull-regime { background-color: #d4edda; color: #155724; }
        .bear-regime { background-color: #f8d7da; color: #721c24; }
        .goldilocks-regime { background-color: #fff3cd; color: #856404; }
        .sideways-regime { background-color: #e2e3e5; color: #383d41; }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the interactive dashboard."""
        # Main header
        st.markdown('<h1 class="main-header">🚀 AutoRegime Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("**Revolutionary AI-Powered Market Regime Detection**")
        
        # Sidebar configuration
        self._create_sidebar()
        
        # Main dashboard content
        if st.session_state.get('data_loaded', False):
            self._display_main_dashboard()
        else:
            self._display_welcome_screen()
    
    def _create_sidebar(self):
        """Create the dashboard sidebar."""
        st.sidebar.markdown("### 📊 Data Loading")
        
        universe = st.sidebar.selectbox(
            "Select Asset Universe:",
            ['indices', 'faang', 'sectors', 'all'],
            help="Choose which assets to analyze"
        )
        start_date = st.sidebar.date_input(
            "Start Date:",
            value=datetime(2000, 1, 1),  # Changed from 2010!
            min_value=datetime(1995, 1, 1),  # Allow going back to 1995
            max_value=datetime.now(),
            help="Analysis start date - can go back to 1995!"          
        )
        
        # Load data button
        if st.sidebar.button("🚀 Load Data & Detect Regimes", type="primary"):
            self._load_and_analyze_data(universe, start_date.strftime('%Y-%m-%d'))
        
        # Model parameters
        if st.session_state.get('data_loaded', False):
            st.sidebar.markdown("### 🧠 Model Parameters")
            
            max_regimes = st.sidebar.slider(
                "Max Regimes:", 2, 10, 
                value=st.session_state.get('max_regimes', 6)
            )
            
            min_duration = st.sidebar.slider(
                "Min Duration (days):", 3, 14, 
                value=st.session_state.get('min_duration', 5)
            )
            
            if st.sidebar.button("🔄 Rerun Analysis"):
                self._rerun_analysis(max_regimes, min_duration)
        
        # Export section
        if st.session_state.get('data_loaded', False):
            st.sidebar.markdown("### 💾 Export")
            if st.sidebar.button("📊 Export Analysis"):
                self._export_analysis()
    
    def _display_welcome_screen(self):
        """Display welcome screen when no data is loaded."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to AutoRegime! 🎯
            
            **Revolutionary AI-powered market regime detection system**
            
            ### What it does:
            - 🔍 **Automatically discovers** hidden market patterns
            - 📊 **Identifies regime changes** (bull, bear, crisis, recovery)
            - 🎯 **Predicts current conditions** with confidence scores
            - 📈 **Professional visualizations** for analysis
            
            ### Get started:
            1. Choose an asset universe from the sidebar
            2. Select your analysis period
            3. Click "Load Data & Detect Regimes"
            4. Explore the interactive dashboard!
            
            ---
            
            ### Recent Discoveries:
            - **COVID Crash (March 2020)**: Crisis regime detected
            - **Recovery Rally (2020-2021)**: Bull market identification
            - **Rate Hike Selloff (2022)**: Bear regime transition
            - **AI Boom (2023-2024)**: New growth regime
            """)
    
    def _load_and_analyze_data(self, universe, start_date):
        """Load data and run regime analysis."""
        with st.spinner("🔄 Loading market data and detecting regimes..."):
            try:
                # Load data
                data = self.loader.load_preset_universe(universe, start_date=start_date)
                
                # Initialize detector
                detector = AutoRegimeDetector(max_regimes=6, verbose=False)
                
                # Fit model
                detector.fit(data)
                
                # Create visualizer
                visualizer = RegimeVisualizer(detector, data)
                
                # Store in session state
                st.session_state['data'] = data
                st.session_state['detector'] = detector
                st.session_state['visualizer'] = visualizer
                st.session_state['data_loaded'] = True
                st.session_state['universe'] = universe
                st.session_state['start_date'] = start_date
                
                st.success("✅ Analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    def _rerun_analysis(self, max_regimes, min_duration):
        """Rerun analysis with new parameters."""
        with st.spinner("🔄 Rerunning analysis with new parameters..."):
            try:
                data = st.session_state['data']
                
                # Create new detector with updated parameters
                detector = AutoRegimeDetector(
                    max_regimes=max_regimes,
                    min_regime_duration=min_duration,
                    verbose=False
                )
                
                detector.fit(data)
                visualizer = RegimeVisualizer(detector, data)
                
                # Update session state
                st.session_state['detector'] = detector
                st.session_state['visualizer'] = visualizer
                st.session_state['max_regimes'] = max_regimes
                st.session_state['min_duration'] = min_duration
                
                st.success("✅ Analysis updated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error rerunning analysis: {str(e)}")
    
    def _display_main_dashboard(self):
        """Display the main dashboard content."""
        detector = st.session_state['detector']
        visualizer = st.session_state['visualizer']
        data = st.session_state['data']
        
        # Current regime prediction
        self._display_current_regime(detector, data)
        
        # Key metrics
        self._display_key_metrics(detector, data)
        
        # Interactive visualizations
        self._display_interactive_charts(visualizer)
        
        # Regime analysis
        self._display_regime_analysis(detector)
    
    def _display_current_regime(self, detector, data):
        """Display current regime prediction."""
        try:
            recent_data = data.tail(21)
            current_regime, confidence = detector.predict_current_regime(recent_data)
            regime_name = detector.regime_names.get(current_regime, f'Regime {current_regime}')
            
            # Create regime alert
            regime_class = self._get_regime_css_class(regime_name)
            
            st.markdown(f"""
            <div class="regime-alert {regime_class}">
                🚨 CURRENT MARKET REGIME: {regime_name.upper()}<br/>
                📊 Confidence: {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error predicting current regime: {str(e)}")
    
    def _get_regime_css_class(self, regime_name):
        """Get CSS class for regime styling."""
        regime_name_lower = regime_name.lower()
        if 'bull' in regime_name_lower:
            return 'bull-regime'
        elif 'bear' in regime_name_lower:
            return 'bear-regime'
        elif 'goldilocks' in regime_name_lower:
            return 'goldilocks-regime'
        else:
            return 'sideways-regime'
    
    def _display_key_metrics(self, detector, data):
        """Display key metrics in a card layout."""
        st.markdown("## 📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Optimal Regimes",
                detector.optimal_n_regimes,
                help="Number of regimes discovered"
            )
        
        with col2:
            # Calculate total observation period
            days = len(data)
            years = days / 252
            st.metric(
                "📅 Analysis Period",
                f"{years:.1f} years",
                f"{days} trading days"
            )
        
        with col3:
            # Calculate market volatility
            market_vol = data.mean(axis=1).std() * np.sqrt(252)
            st.metric(
                "📊 Market Volatility",
                f"{market_vol:.1%}",
                help="Annualized volatility"
            )
        
        with col4:
            # Calculate regime stability
            regime_predictions = detector.optimal_model.predict(
                detector._prepare_features(data)
            )
            switches = np.sum(regime_predictions[1:] != regime_predictions[:-1])
            avg_duration = len(regime_predictions) / (switches + 1)
            st.metric(
                "⏱️ Avg Regime Duration",
                f"{avg_duration:.1f} days",
                help="Average regime persistence"
            )
    
    def _display_interactive_charts(self, visualizer):
        """Display interactive charts."""
        st.markdown("## 📈 Interactive Analysis")
        
        # Chart selection tabs
        tab1, tab2, tab3 = st.tabs(["📊 Timeline", "🎯 Performance", "🔄 Transitions"])
        
        with tab1:
            st.markdown("### Market Regime Timeline")
            timeline_fig = visualizer._plot_interactive_timeline()
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Regime Performance Analysis")
            performance_fig = visualizer._plot_interactive_performance()
            st.plotly_chart(performance_fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Regime Transition Matrix")
            transition_fig = visualizer.plot_regime_transitions(interactive=True)
            st.plotly_chart(transition_fig, use_container_width=True)
    
    def _display_regime_analysis(self, detector):
        """Display detailed regime analysis."""
        st.markdown("## 🔍 Regime Analysis")
        
        # Regime summary table
        regime_data = []
        for regime_id, char in detector.regime_characteristics.items():
            regime_name = detector.regime_names.get(regime_id, f'Regime {regime_id}')
            regime_data.append({
                'Regime': regime_name,
                'Frequency': f"{char['frequency']:.1%}",
                'Avg Duration': f"{char['avg_duration']:.1f} days",
                'Annual Return': f"{char['mean_return']:.1%}",
                'Annual Volatility': f"{char['volatility']:.1%}",
                'Sharpe Ratio': f"{char['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{char['max_drawdown']:.1%}"
            })
        
        regime_df = pd.DataFrame(regime_data)
        st.dataframe(regime_df, use_container_width=True)
        
        # Detailed regime characteristics
        st.markdown("### 📋 Detailed Characteristics")
        
        selected_regime = st.selectbox(
            "Select regime for detailed analysis:",
            options=list(detector.regime_names.values())
        )
        
        # Find regime ID
        regime_id = None
        for rid, name in detector.regime_names.items():
            if name == selected_regime:
                regime_id = rid
                break
        
        if regime_id is not None and regime_id in detector.regime_characteristics:
            char = detector.regime_characteristics[regime_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **📊 {selected_regime} Statistics:**
                - **Frequency**: {char['frequency']:.1%}
                - **Average Duration**: {char['avg_duration']:.1f} days
                - **Annual Return**: {char['mean_return']:.1%}
                - **Annual Volatility**: {char['volatility']:.1%}
                """)
            
            with col2:
                st.markdown(f"""
                **🎯 Risk Metrics:**
                - **Sharpe Ratio**: {char['sharpe_ratio']:.2f}
                - **Max Drawdown**: {char['max_drawdown']:.1%}
                - **Return/Vol Ratio**: {char['mean_return']/char['volatility']:.2f}
                """)
    
    def _export_analysis(self):
        """Export analysis results."""
        try:
            visualizer = st.session_state['visualizer']
            
            with st.spinner("📊 Exporting analysis..."):
                output_dir = visualizer.export_analysis('./streamlit_export')
                
            st.success(f"✅ Analysis exported to: {output_dir}")
            
            # Provide download links
            st.markdown("### 💾 Download Files:")
            st.markdown("- 📊 regime_summary.csv")
            st.markdown("- 📈 regime_timeline.png")
            st.markdown("- 📊 regime_performance.png")
            st.markdown("- 🔄 regime_transitions.png")
            st.markdown("- 📋 regime_data.csv")
            
        except Exception as e:
            st.error(f"❌ Export error: {str(e)}")

def main():
    """Run the AutoRegime dashboard."""
    dashboard = AutoRegimeDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 
