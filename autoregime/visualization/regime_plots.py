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

from .style_config import (
    setup_matplotlib_style, get_regime_color, create_plotly_layout,
    REGIME_COLORS, PLOT_STYLE, PLOTLY_THEME
)

class RegimeVisualizer:
    """
    Professional visualization suite for AutoRegime analysis.
    """
    
    def __init__(self, detector, returns_data):
        """Initialize visualizer with fitted detector and data."""
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
        """Create beautiful regime timeline visualization."""
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
                '🎯 Market Regime Timeline',
                '📈 Cumulative Returns', 
                '📊 Regime Confidence Levels'
            ),
            row_heights=[0.4, 0.35, 0.25]
        )
        
        # Plot 1: Regime Timeline
        dates = self.regime_df['date']
        
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
                        marker=dict(color=color, size=4, opacity=0.7)
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Cumulative returns
        cumulative_returns = (1 + self.regime_df['market_return']).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Return',
                line=dict(color='black', width=2)
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
                        opacity=0.7
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title={
                'text': '🚀 AutoRegime Analysis Dashboard',
                'x': 0.5,
                'font': {'size': 24}
            },
            showlegend=True,
            template='plotly_white'
        )
        
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
        
        axes[0].set_title('🎯 Market Regime Timeline', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Daily Returns')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        cumulative_returns = (1 + self.regime_df['market_return']).cumprod()
        axes[1].plot(dates, cumulative_returns, 'k-', linewidth=2)
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
        
        axes[2].set_title('📊 Regime Confidence Levels', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Probability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Timeline saved to: {save_path}")
        
        return fig
    
    def plot_regime_performance(self, interactive=True):
        """Create regime performance comparison charts."""
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
                'Frequency': char['frequency']
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Simple 2x2 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📊 Risk-Return Profile',
                '🎯 Sharpe Ratio', 
                '⏱️ Frequency',
                '📈 Returns'
            )
        )
        
        # Risk-Return scatter
        for _, row in perf_df.iterrows():
            color = get_regime_color(row['Regime'])
            fig.add_trace(
                go.Scatter(
                    x=[row['Annual Volatility']],
                    y=[row['Annual Return']],
                    mode='markers+text',
                    name=row['Regime'],
                    marker=dict(color=color, size=15, opacity=0.7),
                    text=row['Regime'],
                    textposition="top center"
                ),
                row=1, col=1
            )
        
        # Sharpe ratio bars
        fig.add_trace(
            go.Bar(
                x=perf_df['Regime'],
                y=perf_df['Sharpe Ratio'],
                marker_color=[get_regime_color(regime) for regime in perf_df['Regime']],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Frequency bars
        fig.add_trace(
            go.Bar(
                x=perf_df['Regime'],
                y=perf_df['Frequency'],
                marker_color=[get_regime_color(regime) for regime in perf_df['Regime']],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Return bars
        fig.add_trace(
            go.Bar(
                x=perf_df['Regime'],
                y=perf_df['Annual Return'],
                marker_color=[get_regime_color(regime) for regime in perf_df['Regime']],
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title='📊 AutoRegime Performance Analytics',
            template='plotly_white'
        )
        
        return fig
    
    def _plot_static_performance(self):
        """Create static performance plots."""
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
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = [get_regime_color(regime) for regime in perf_df['Regime']]
        
        # Risk-Return scatter
        axes[0,0].scatter(perf_df['Volatility'], perf_df['Return'], c=colors, s=100, alpha=0.7)
        axes[0,0].set_xlabel('Annual Volatility')
        axes[0,0].set_ylabel('Annual Return')
        axes[0,0].set_title('📊 Risk-Return Profile')
        axes[0,0].grid(True, alpha=0.3)
        
        # Sharpe ratio bars
        axes[0,1].bar(perf_df['Regime'], perf_df['Sharpe'], color=colors, alpha=0.7)
        axes[0,1].set_title('🎯 Sharpe Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Frequency pie
        axes[1,0].pie(perf_df['Frequency'], labels=perf_df['Regime'], colors=colors, autopct='%1.1f%%')
        axes[1,0].set_title('📈 Regime Frequency')
        
        # Return bars
        axes[1,1].bar(perf_df['Regime'], perf_df['Return'], color=colors, alpha=0.7)
        axes[1,1].set_title('💰 Annual Returns')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
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
        
        print(f"📊 Exporting AutoRegime analysis to: {output_dir}")
        
        # Export summary table
        summary_table = self.create_regime_summary_table()
        summary_table.to_csv(f"{output_dir}/regime_summary.csv", index=False)
        print("✅ Regime summary table exported")
        
        # Export static plots
        timeline_fig = self.plot_regime_timeline(interactive=False)
        timeline_fig.savefig(f"{output_dir}/regime_timeline.png", dpi=300, bbox_inches='tight')
        plt.close(timeline_fig)
        print("✅ Regime timeline exported")
        
        performance_fig = self._plot_static_performance()
        performance_fig.savefig(f"{output_dir}/regime_performance.png", dpi=300, bbox_inches='tight')
        plt.close(performance_fig)
        print("✅ Performance analysis exported")
        
        # Export data
        self.regime_df.to_csv(f"{output_dir}/regime_data.csv", index=False)
        print("✅ Regime data exported")
        
        print("🎉 Analysis export completed!")
        return output_dir
    
    def plot_regime_transitions(self, interactive=True):
        """Create regime transition analysis."""
        # Simple transition matrix
        regime_states = self.regime_predictions
        n_regimes = self.detector.optimal_n_regimes
        
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for i in range(len(regime_states) - 1):
            current_regime = regime_states[i]
            next_regime = regime_states[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = transition_matrix / row_sums[:, np.newaxis]
        
        regime_names = [self.detector.regime_names.get(i, f'Regime {i}') 
                       for i in range(n_regimes)]
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=transition_probs,
                x=regime_names,
                y=regime_names,
                colorscale='Viridis'
            ))
            fig.update_layout(title='🔄 Regime Transition Matrix')
            return fig
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_probs, annot=True, fmt='.2f',
                       xticklabels=regime_names, yticklabels=regime_names)
            plt.title('🔄 Regime Transition Matrix')
            return plt.gcf()