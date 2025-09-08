"""
AutoRegime Visualization Styling Configuration
Beautiful, professional color schemes and themes
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Professional Color Palettes
REGIME_COLORS = {
    'Goldilocks': '#FFD700',    # Gold - Perfect conditions
    'Bull Market': '#32CD32',   # Lime Green - Growth
    'Steady Growth': '#87CEEB', # Sky Blue - Stability  
    'Sideways': '#DDA0DD',      # Plum - Neutral
    'Risk-Off': '#FF6347',      # Tomato - Caution
    'Bear Market': '#DC143C',   # Crimson - Danger
    'Crisis': '#8B0000',        # Dark Red - Emergency
    'Recovery': '#00FF7F'       # Spring Green - Healing
}

# Backup colors for regimes without names
BACKUP_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]

# Professional styling settings
PLOT_STYLE = {
    'figure_size': (15, 8),
    'title_size': 16,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 11,
    'line_width': 2,
    'alpha': 0.7,
    'grid_alpha': 0.3
}

# Plotly theme settings
PLOTLY_THEME = {
    'template': 'plotly_white',
    'color_palette': px.colors.qualitative.Set3,
    'font_family': 'Arial, sans-serif',
    'title_font_size': 20,
    'axis_font_size': 14,
    'legend_font_size': 12
}

def setup_matplotlib_style():
    """Configure matplotlib for professional plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Set default figure parameters
    plt.rcParams.update({
        'figure.figsize': PLOT_STYLE['figure_size'],
        'font.size': PLOT_STYLE['label_size'],
        'axes.titlesize': PLOT_STYLE['title_size'],
        'axes.labelsize': PLOT_STYLE['label_size'],
        'xtick.labelsize': PLOT_STYLE['tick_size'],
        'ytick.labelsize': PLOT_STYLE['tick_size'],
        'legend.fontsize': PLOT_STYLE['legend_size'],
        'lines.linewidth': PLOT_STYLE['line_width'],
        'grid.alpha': PLOT_STYLE['grid_alpha']
    })

def get_regime_color(regime_name, regime_id=None):
    """Get color for a specific regime."""
    if regime_name in REGIME_COLORS:
        return REGIME_COLORS[regime_name]
    elif regime_id is not None and regime_id < len(BACKUP_COLORS):
        return BACKUP_COLORS[regime_id]
    else:
        return '#333333'  # Default gray

def create_plotly_layout(title, xaxis_title="Date", yaxis_title="Value"):
    """Create consistent Plotly layout."""
    return go.Layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'size': PLOTLY_THEME['title_font_size']}
        },
        xaxis={'title': xaxis_title, 'titlefont': {'size': PLOTLY_THEME['axis_font_size']}},
        yaxis={'title': yaxis_title, 'titlefont': {'size': PLOTLY_THEME['axis_font_size']}},
        font={'family': PLOTLY_THEME['font_family']},
        legend={'font': {'size': PLOTLY_THEME['legend_font_size']}},
        template=PLOTLY_THEME['template'],
        hovermode='x unified',
        showlegend=True
    )  
