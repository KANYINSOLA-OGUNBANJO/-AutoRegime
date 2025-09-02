"""
AutoRegime Visualization Engine
Creates stunning, professional visualizations of market regime analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Comment out the style import for now since it's not available
# from .style_config import (
#     setup_matplotlib_style, get_regime_color, create_plotly_layout,
#     REGIME_COLORS, PLOT_STYLE, PLOTLY_THEME
# )

# Simple color function to replace missing import
def get_regime_color(regime_name, regime_id=0):
    """Simple color mapping for regimes."""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    return colors[regime_id % len(colors)]

def setup_matplotlib_style():
    """Simple matplotlib style setup."""
    plt.style.use('ggplot')  # Nice clean style that you have

class RegimeVisualizer:
    """
    Professional visualization suite for AutoRegime analysis.
    
    Creates beautiful, interactive charts showing:
    - Regime timelines with color coding
    - Performance analytics and comparisons  
    - Risk-return characteristics
    - Transition matrices and patterns
    - Real-time monitoring dashboards
    """
    
    def __init__(self, detector, returns_data):
        """
        Initialize visualizer with fitted detector and data.
        
        Parameters:
        -----------
        detector : AutoRegimeDetector
            Fitted regime detection model
        returns_data : pd.DataFrame
            Historical returns data with datetime index
        """
        self.detector = detector
        self.returns_data = returns_data
        self.regime_predictions = None
        self.regime_probabilities = None
        
        # Setup styling
        setup_matplotlib_style()
        
        # Get regime data
        self._prepare_regime_data()
    
    def _prepare_regime_data(self):
        """Prepare regime predictions and probabilities for visualization."""
        if self.detector.optimal_model is None:
            raise ValueError("Detector must be fitted before visualization")
        
        # Get regime predictions
        features = self.detector._prepare_features(self.returns_data)
        self.regime_predictions = self.detector.optimal_model.predict(features)
        self.regime_probabilities = self.detector.optimal_model.predict_proba(features)
        
        # Create regime DataFrame
        self.regime_df = pd.DataFrame({
            'date': self.returns_data.index,
            'regime': self.regime_predictions,
            'regime_name': [self.detector.regime_names.get(r, f'Regime {r}') 
                          for r in self.regime_predictions],
            'market_return': self.returns_data.mean(axis=1),
            'market_volatility': self.returns_data.rolling(21).std().mean(axis=1)
        })
        
        # Add regime probabilities
        for i in range(self.detector.optimal_n_regimes):
            regime_name = self.detector.regime_names.get(i, f'Regime {i}')
            self.regime_df[f'{regime_name}_prob'] = self.regime_probabilities[:, i]
    
    def plot_regime_timeline(self, interactive=True, save_path=None):
        """
        Create beautiful regime timeline visualization.
        
        Shows market regimes over time with:
        - Color-coded regime periods
        - Market performance overlay
        - Interactive tooltips (if interactive=True)
        """
        if interactive:
            return self._plot_interactive_timeline()
        else:
            return self._plot_static_timeline(save_path)
    
    def _plot_interactive_timeline(self):
        """Create interactive Plotly timeline."""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Market Regime Timeline',
                'Cumulative Returns by Regime', 
                'Regime Confidence Levels'
            ),
            row_heights=[0.4, 0.35, 0.25]
        )
        
        # Plot 1: Regime Timeline with colored background
        dates = self.regime_df['date']
        
        # Create regime background colors
        for regime_id in range(self.detector.optimal_n_regimes):
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            regime_mask = self.regime_df['regime'] == regime_id
            color = get_regime_color(regime_name, regime_id)
            
            if np.any(regime_mask):
                regime_dates = dates[regime_mask]
                regime_returns = self.regime_df.loc[regime_mask, 'market_return']
                
                fig.add_trace(
                    go.Scatter(
                        x=regime_dates,
                        y=regime_returns,
                        mode='markers',
                        name=regime_name,
                        marker=dict(color=color, size=4, opacity=0.7),
                        hovertemplate=f'<b>{regime_name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Return: %{y:.2%}<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Cumulative returns by regime
        cumulative_returns = (1 + self.regime_df['market_return']).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Return',
                line=dict(color='black', width=2),
                hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 3: Regime probabilities
        for regime_id in range(self.detector.optimal_n_regimes):
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            color = get_regime_color(regime_name, regime_id)
            prob_col = f'{regime_name}_prob'
            
            if prob_col in self.regime_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=self.regime_df[prob_col],
                        mode='lines',
                        name=f'{regime_name} Probability',
                        line=dict(color=color, width=1),
                        opacity=0.7,
                        hovertemplate=f'{regime_name} Prob: %{{y:.1%}}<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title={
                'text': 'AutoRegime Analysis Dashboard',
                'x': 0.5,
                'font': {'size': 24}
            },
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Daily Returns", tickformat='.1%', row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
        fig.update_yaxes(title_text="Probability", tickformat='.0%', row=3, col=1)
        
        return fig
    
    def _plot_static_timeline(self, save_path=None):
        """Create static matplotlib timeline."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot 1: Regime timeline
        dates = self.regime_df['date']
        
        for regime_id in range(self.detector.optimal_n_regimes):
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            regime_mask = self.regime_df['regime'] == regime_id
            color = get_regime_color(regime_name, regime_id)
            
            if np.any(regime_mask):
                axes[0].scatter(
                    dates[regime_mask], 
                    self.regime_df.loc[regime_mask, 'market_return'],
                    c=color, label=regime_name, alpha=0.7, s=20
                )
        
        axes[0].set_title('Market Regime Timeline', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Daily Returns')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 2: Cumulative returns
        cumulative_returns = (1 + self.regime_df['market_return']).cumprod()
        axes[1].plot(dates, cumulative_returns, 'k-', linewidth=2, label='Cumulative Return')
        axes[1].set_title('📈 Cumulative Returns', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Regime probabilities
        for regime_id in range(self.detector.optimal_n_regimes):
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            color = get_regime_color(regime_name, regime_id)
            prob_col = f'{regime_name}_prob'
            
            if prob_col in self.regime_df.columns:
                axes[2].plot(
                    dates, self.regime_df[prob_col], 
                    color=color, label=f'{regime_name}', alpha=0.7
                )
        
        axes[2].set_title('Regime Confidence Levels', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Probability')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline saved to: {save_path}")
        
        return fig
    
    def plot_regime_performance(self, interactive=True):
        """
        Create regime performance comparison charts.
        
        Shows risk-return characteristics of each regime with:
        - Return vs Volatility scatter plot
        - Regime statistics table
        - Performance metrics comparison
        """
        if interactive:
            return self._plot_interactive_performance()
        else:
            return self._plot_static_performance()
    
    def _plot_interactive_performance(self):
        """Create interactive performance analysis."""
        # Prepare performance data
        perf_data = []
        for regime_id, char in self.detector.regime_characteristics.items():
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            perf_data.append({
                'Regime': regime_name,
                'Annual Return': char['mean_return'],
                'Annual Volatility': char['volatility'], 
                'Sharpe Ratio': char['sharpe_ratio'],
                'Max Drawdown': char['max_drawdown'],
                'Frequency': char['frequency'],
                'Avg Duration': char['avg_duration']
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Create simple 2x2 subplots (no pie chart to avoid errors)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk-Return Profile',
                'Sharpe Ratio Comparison', 
                'Regime Duration Analysis',  
                'Frequency Analysis'
            )
        )
        
        # Plot 1: Risk-Return Scatter
        for _, row in perf_df.iterrows():
            color = get_regime_color(row['Regime'])
            fig.add_trace(
                go.Scatter(
                    x=[row['Annual Volatility']],
                    y=[row['Annual Return']],
                    mode='markers+text',
                    name=row['Regime'],
                    marker=dict(
                        color=color,
                        size=row['Frequency'] * 500,
                        opacity=0.7
                    ),
                    text=row['Regime'],
                    textposition="top center"
                ),
                row=1, col=1
            )
        
        # Plot 2: Sharpe Ratio Bar Chart
        fig.add_trace(
            go.Bar(
                x=perf_df['Regime'],
                y=perf_df['Sharpe Ratio'],
                marker_color=[get_regime_color(regime) for regime in perf_df['Regime']],
                name='Sharpe Ratio',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Duration Analysis
        fig.add_trace(
            go.Bar(
                x=perf_df['Regime'],
                y=perf_df['Avg Duration'],
                marker_color=[get_regime_color(regime) for regime in perf_df['Regime']],
                name='Avg Duration',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Frequency Bar Chart (instead of pie)
        fig.add_trace(
            go.Bar(
                x=perf_df['Regime'],
                y=perf_df['Frequency'],
                marker_color=[get_regime_color(regime) for regime in perf_df['Regime']],
                name='Frequency',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title={
                'text': 'AutoRegime Performance Analytics',
                'x': 0.5,
                'font': {'size': 24}
            },
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_regime_transitions(self, interactive=True):
        """
        Create regime transition analysis.
        
        Shows how regimes change over time with:
        - Transition matrix heatmap
        - Transition probability analysis
        - Regime persistence metrics
        """
        # Calculate transition matrix
        regime_states = self.regime_predictions
        n_regimes = self.detector.optimal_n_regimes
        
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for i in range(len(regime_states) - 1):
            current_regime = regime_states[i]
            next_regime = regime_states[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = transition_matrix / row_sums[:, np.newaxis]
        
        # Create regime names for axes
        regime_names = [self.detector.regime_names.get(i, f'Regime {i}') 
                       for i in range(n_regimes)]
        
        if interactive:
            # Interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=transition_probs,
                x=regime_names,
                y=regime_names,
                colorscale='Viridis',
                text=[[f'{val:.1%}' for val in row] for row in transition_probs],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.1%}<extra></extra>"
            ))
            
            fig.update_layout(
                title={
                    'text': 'Regime Transition Matrix',
                    'x': 0.5,
                    'font': {'size': 20}
                },
                xaxis_title="To Regime",
                yaxis_title="From Regime",
                template='plotly_white'
            )
            
            return fig
        else:
            # Static heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                transition_probs,
                annot=True,
                fmt='.1%',
                xticklabels=regime_names,
                yticklabels=regime_names,
                cmap='viridis',
                cbar_kws={'label': 'Transition Probability'}
            )
            plt.title('Regime Transition Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('To Regime')
            plt.ylabel('From Regime')
            plt.tight_layout()
            return plt.gcf()
    
    def create_regime_summary_table(self):
        """Create comprehensive regime summary table."""
        summary_data = []
        
        for regime_id, char in self.detector.regime_characteristics.items():
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            summary_data.append({
                'Regime': regime_name,
                'Frequency': f"{char['frequency']:.1%}",
                'Avg Duration': f"{char['avg_duration']:.1f} days",
                'Annual Return': f"{char['mean_return']:.1%}",
                'Annual Volatility': f"{char['volatility']:.1%}",
                'Sharpe Ratio': f"{char['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{char['max_drawdown']:.1%}"
            })
        
        return pd.DataFrame(summary_data)
    
    def export_analysis(self, output_dir='./regime_analysis'):
        """Export comprehensive regime analysis to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting AutoRegime analysis to: {output_dir}")
        
        # Export summary table
        summary_table = self.create_regime_summary_table()
        summary_table.to_csv(f"{output_dir}/regime_summary.csv", index=False)
        print("Regime summary table exported")
        
        # Export static plots
        timeline_fig = self.plot_regime_timeline(interactive=False)
        timeline_fig.savefig(f"{output_dir}/regime_timeline.png", dpi=300, bbox_inches='tight')
        plt.close(timeline_fig)
        print("Regime timeline exported")
        
        performance_fig = self._plot_static_performance()
        performance_fig.savefig(f"{output_dir}/regime_performance.png", dpi=300, bbox_inches='tight')
        plt.close(performance_fig)
        print("Performance analysis exported")
        
        transition_fig = self.plot_regime_transitions(interactive=False)
        transition_fig.savefig(f"{output_dir}/regime_transitions.png", dpi=300, bbox_inches='tight')
        plt.close(transition_fig)
        print("Transition matrix exported")
        
        # Export data
        self.regime_df.to_csv(f"{output_dir}/regime_data.csv", index=False)
        print("Regime data exported")
        
        print("Analysis export completed!")
        return output_dir

    def _plot_static_performance(self):
        """Create static performance plots."""
        # Prepare data
        perf_data = []
        for regime_id, char in self.detector.regime_characteristics.items():
            regime_name = self.detector.regime_names.get(regime_id, f'Regime {regime_id}')
            perf_data.append({
                'Regime': regime_name,
                'Return': char['mean_return'],
                'Volatility': char['volatility'], 
                'Sharpe': char['sharpe_ratio'],
                'Frequency': char['frequency']
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Risk-Return scatter
        colors = [get_regime_color(regime) for regime in perf_df['Regime']]
        scatter = axes[0,0].scatter(
            perf_df['Volatility'], perf_df['Return'],
            c=colors, s=perf_df['Frequency']*500,
            alpha=0.7, edgecolors='black', linewidth=2
        )
        
        for i, regime in enumerate(perf_df['Regime']):
            axes[0,0].annotate(
                regime, 
                (perf_df.iloc[i]['Volatility'], perf_df.iloc[i]['Return']),
                xytext=(5, 5), textcoords='offset points'
            )
        
        axes[0,0].set_xlabel('Annual Volatility')
        axes[0,0].set_ylabel('Annual Return')
        axes[0,0].set_title('Risk-Return Profile')
        axes[0,0].grid(True, alpha=0.3)
        
        # Sharpe ratio bar chart
        bars = axes[0,1].bar(perf_df['Regime'], perf_df['Sharpe'], color=colors, alpha=0.7)
        axes[0,1].set_title('Sharpe Ratio Comparison')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Frequency pie chart
        axes[1,0].pie(
            perf_df['Frequency'], 
            labels=perf_df['Regime'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        axes[1,0].set_title('Regime Frequency')
        
        # Return distribution
        axes[1,1].bar(perf_df['Regime'], perf_df['Return'], color=colors, alpha=0.7)
        axes[1,1].set_title('Annual Returns by Regime')
        axes[1,1].set_ylabel('Annual Return')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.suptitle('📊 AutoRegime Performance Analytics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
