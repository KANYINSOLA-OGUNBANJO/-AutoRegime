"""
AutoRegime Market Data Loader
Downloads and prepares real market data for regime analysis
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketDataLoader:
    """
    Professional market data loading and preparation system.
    
    Supports:
    - Major market indices (S&P 500, NASDAQ, Russell 2000)
    - Individual stocks (FAANG, banks, etc.)
    - Sector ETFs (technology, finance, healthcare)
    - International markets
    - Volatility indices (VIX)
    """
    
    def __init__(self):
        """Initialize the market data loader."""
        self.data_cache = {}
        
        # Predefined asset universes
        self.MAJOR_INDICES = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100', 
            'IWM': 'Russell 2000',
            'DIA': 'Dow Jones',
            'VTI': 'Total Stock Market'
        }
        
        self.FAANG_STOCKS = {
            'AAPL': 'Apple',
            'AMZN': 'Amazon',
            'GOOGL': 'Google',
            'META': 'Meta (Facebook)',
            'NFLX': 'Netflix',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla'
        }
        
        self.SECTOR_ETFS = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate'
        }
        
        self.VOLATILITY_INDICES = {
            'VIX': 'VIX Fear Index',
            'VXN': 'NASDAQ Volatility',
            'RVX': 'Russell 2000 Volatility'
        }
    
    def load_market_data(self, symbols, start_date='2020-01-01', end_date=None, 
                        return_type='returns'):
        """
        Load market data for specified symbols.
        
        Parameters:
        -----------
        symbols : list or str
            Stock symbols to download (e.g., ['SPY', 'QQQ'] or 'SPY')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)
        return_type : str
            'returns' for daily returns, 'prices' for price data
            
        Returns:
        --------
        pd.DataFrame
            Market data with datetime index
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📊 Loading market data for {len(symbols)} assets...")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🎯 Assets: {', '.join(symbols)}")
        
        all_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                print(f"  Downloading {symbol}...", end=' ')
                
                # Check cache first
                cache_key = f"{symbol}_{start_date}_{end_date}"
                if cache_key in self.data_cache:
                    all_data[symbol] = self.data_cache[cache_key]
                    print("✅ (cached)")
                    continue
                
                # Download from Yahoo Finance
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if len(data) == 0:
                    print("❌ No data")
                    failed_symbols.append(symbol)
                    continue
                
                # Store in cache
                self.data_cache[cache_key] = data
                all_data[symbol] = data
                print("✅")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                failed_symbols.append(symbol)
                continue
        
        if not all_data:
            raise ValueError("No data could be downloaded for any symbol")
        
        if failed_symbols:
            print(f"⚠️ Failed to download: {', '.join(failed_symbols)}")
        
        # Process data based on return_type
        if return_type == 'returns':
            return self._calculate_returns(all_data)
        else:
            return self._prepare_price_data(all_data)
    
    def _calculate_returns(self, price_data):
        """Calculate daily returns from price data."""
        print("📈 Calculating daily returns...")
        
        returns_data = {}
        
        for symbol, data in price_data.items():
            # Use adjusted close prices
            adj_close = data['Close']
            
            # Calculate daily returns
            daily_returns = adj_close.pct_change().dropna()
            returns_data[symbol] = daily_returns
        
        # Create aligned DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Remove any rows with all NaN values
        returns_df = returns_df.dropna(how='all')
        
        print(f"✅ Returns calculated for period: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        print(f"📊 Data shape: {returns_df.shape}")
        
        return returns_df
    
    def _prepare_price_data(self, price_data):
        """Prepare aligned price data."""
        print("💰 Preparing price data...")
        
        price_df_data = {}
        
        for symbol, data in price_data.items():
            price_df_data[symbol] = data['Close']
        
        price_df = pd.DataFrame(price_df_data)
        price_df = price_df.dropna(how='all')
        
        print(f"✅ Price data prepared: {price_df.shape}")
        return price_df
    
    def load_preset_universe(self, universe_name, start_date='2020-01-01', end_date=None):
        """
        Load predefined asset universes.
        
        Parameters:
        -----------
        universe_name : str
            'indices', 'faang', 'sectors', 'volatility', or 'all'
        start_date : str
            Start date for data
        end_date : str, optional
            End date for data
            
        Returns:
        --------
        pd.DataFrame
            Returns data for the specified universe
        """
        universe_map = {
            'indices': self.MAJOR_INDICES,
            'faang': self.FAANG_STOCKS,
            'sectors': self.SECTOR_ETFS,
            'volatility': self.VOLATILITY_INDICES
        }
        
        if universe_name == 'all':
            # Combine all universes
            all_symbols = []
            for universe in universe_map.values():
                all_symbols.extend(list(universe.keys()))
            symbols = all_symbols
        elif universe_name in universe_map:
            symbols = list(universe_map[universe_name].keys())
        else:
            raise ValueError(f"Unknown universe: {universe_name}. Available: {list(universe_map.keys()) + ['all']}")
        
        print(f"Loading '{universe_name}' universe ({len(symbols)} assets)")
        
        return self.load_market_data(symbols, start_date, end_date)
    
    def get_market_summary(self, returns_data):
        """
        Generate summary statistics for market data.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data
            
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        print("📊 Generating market summary statistics...")
        
        summary_stats = {}
        
        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()
            
            # Calculate statistics
            annual_return = asset_returns.mean() * 252
            annual_vol = asset_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + asset_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / rolling_max) - 1
            max_drawdown = drawdowns.min()
            
            # Skewness and kurtosis
            skewness = asset_returns.skew()
            kurtosis = asset_returns.kurtosis()
            
            summary_stats[asset] = {
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Observations': len(asset_returns)
            }
        
        summary_df = pd.DataFrame(summary_stats).T
        
        # Format for display
        summary_df['Annual Return'] = summary_df['Annual Return'].apply(lambda x: f"{x:.1%}")
        summary_df['Annual Volatility'] = summary_df['Annual Volatility'].apply(lambda x: f"{x:.1%}")
        summary_df['Sharpe Ratio'] = summary_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        summary_df['Max Drawdown'] = summary_df['Max Drawdown'].apply(lambda x: f"{x:.1%}")
        summary_df['Skewness'] = summary_df['Skewness'].apply(lambda x: f"{x:.2f}")
        summary_df['Kurtosis'] = summary_df['Kurtosis'].apply(lambda x: f"{x:.2f}")
        
        return summary_df
    
    def load_crisis_periods(self, start_date='2019-01-01'):
        """
        Load data covering major market crisis periods.
        
        Focuses on periods with significant regime changes:
        - 2020 COVID crash and recovery
        - 2022 rate hike selloff
        - 2023 banking crisis and AI boom
        """
        print("🔥 Loading crisis periods data for regime analysis...")
        
        # Load diverse asset classes for better regime detection
        crisis_symbols = [
            # Major indices
            'SPY', 'QQQ', 'IWM',
            # FAANG stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN',
            # Sectors that behave differently in crises
            'XLK', 'XLF', 'XLE', 'XLV',
            # Volatility
            'VIX'
        ]
        
        return self.load_market_data(crisis_symbols, start_date=start_date)

def demo_data_loader():
    """Demonstrate the market data loader capabilities."""
    print("🚀 AutoRegime Market Data Loader Demo")
    print("=" * 50)
    
    loader = MarketDataLoader()
    
    # Load major indices for regime analysis
    print("\n📊 Loading major market indices...")
    indices_data = loader.load_preset_universe('indices', start_date='2020-01-01')
    
    print(f"\n✅ Loaded data shape: {indices_data.shape}")
    print(f"📅 Date range: {indices_data.index[0].date()} to {indices_data.index[-1].date()}")
    print(f"📈 Assets: {', '.join(indices_data.columns)}")
    
    # Show summary statistics
    print("\n📊 Market Summary Statistics:")
    summary = loader.get_market_summary(indices_data)
    print(summary)
    
    return indices_data

if __name__ == "__main__":
    demo_data_loader() 
