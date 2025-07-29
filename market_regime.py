import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MarketRegimeDetector:
    def __init__(self, symbol="^GSPC", period="5y"):
        """
        Initialize the detector
        symbol: Yahoo Finance ticker (^GSPC for S&P 500, ^IXIC for Nasdaq)
        period: Data period (1y, 2y, 5y, 10y, max)
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.regimes = []
        
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            print(f"Fetched {len(self.data)} days of data for {self.symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def detect_regimes(self):
        """Detect bull and bear markets using 20% rule"""
        if self.data is None or self.data.empty:
            print("No data available. Please fetch data first.")
            return
        
        prices = self.data['Close'].values
        dates = self.data.index
        
        peak_price = prices[0]
        peak_date = dates[0]
        trough_price = prices[0]
        trough_date = dates[0]
        current_regime = "bull"  # Start assuming bull market
        
        regime_changes = []
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            if current_regime == "bull":
                # In bull market, track new peaks
                if price > peak_price:
                    peak_price = price
                    peak_date = date
                
                # Check for bear market (20% drop from peak)
                drawdown = (price - peak_price) / peak_price
                if drawdown <= -0.2:
                    current_regime = "bear"
                    trough_price = price
                    trough_date = date
                    regime_changes.append({
                        'date': date,
                        'price': price,
                        'regime': 'bear',
                        'change_from_peak': drawdown,
                        'peak_date': peak_date,
                        'peak_price': peak_price
                    })
                    
            elif current_regime == "bear":
                # In bear market, track new troughs
                if price < trough_price:
                    trough_price = price
                    trough_date = date
                
                # Check for bull market (20% rise from trough)
                recovery = (price - trough_price) / trough_price
                if recovery >= 0.2:
                    current_regime = "bull"
                    peak_price = price
                    peak_date = date
                    regime_changes.append({
                        'date': date,
                        'price': price,
                        'regime': 'bull',
                        'change_from_trough': recovery,
                        'trough_date': trough_date,
                        'trough_price': trough_price
                    })
        
        self.regimes = regime_changes
        return regime_changes
    
    def print_results(self):
        """Print regime changes in a readable format"""
        if not self.regimes:
            print("No regime changes detected in the given period.")
            return
            
        print(f"\n=== MARKET REGIME CHANGES FOR {self.symbol} ===\n")
        
        for i, regime in enumerate(self.regimes):
            if regime['regime'] == 'bear':
                print(f"🐻 BEAR MARKET STARTED:")
                print(f"   Date: {regime['date'].strftime('%Y-%m-%d')}")
                print(f"   Price: ${regime['price']:.2f}")
                print(f"   Drop from peak: {regime['change_from_peak']:.1%}")
                print(f"   Peak was: ${regime['peak_price']:.2f} on {regime['peak_date'].strftime('%Y-%m-%d')}")
                
            else:  # bull market
                print(f"🐂 BULL MARKET STARTED:")
                print(f"   Date: {regime['date'].strftime('%Y-%m-%d')}")
                print(f"   Price: ${regime['price']:.2f}")
                print(f"   Rise from trough: {regime['change_from_trough']:.1%}")
                print(f"   Trough was: ${regime['trough_price']:.2f} on {regime['trough_date'].strftime('%Y-%m-%d')}")
            
            print()
    
    def get_current_status(self):
        """Get current market status"""
        if self.data is None or self.data.empty:
            return "No data available"
            
        current_price = self.data['Close'].iloc[-1]
        current_date = self.data.index[-1]
        
        # Find recent peak and trough
        recent_data = self.data.tail(252)  # Last year of data
        recent_peak = recent_data['Close'].max()
        recent_trough = recent_data['Close'].min()
        
        # Calculate current drawdown from recent peak
        drawdown = (current_price - recent_peak) / recent_peak
        recovery = (current_price - recent_trough) / recent_trough
        
        if self.regimes:
            last_regime = self.regimes[-1]['regime']
        else:
            last_regime = "unknown"
            
        print(f"\n=== CURRENT STATUS ===")
        print(f"Current Price: ${current_price:.2f} ({current_date.strftime('%Y-%m-%d')})")
        print(f"Drawdown from recent peak: {drawdown:.1%}")
        print(f"Recovery from recent trough: {recovery:.1%}")
        print(f"Last detected regime change: {last_regime.upper()} market")
        
        if drawdown <= -0.2:
            print("📉 Currently in BEAR MARKET territory (>20% from peak)")
        elif recovery >= 0.2:
            print("📈 Currently in BULL MARKET territory (>20% from trough)")
        else:
            print("📊 Currently in NEUTRAL territory")
    
    def plot_regimes(self):
        """Plot price chart with regime changes marked"""
        if self.data is None or not self.regimes:
            print("No data or regime changes to plot")
            return
            
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data['Close'], 'b-', linewidth=1, alpha=0.7)
        
        # Mark regime changes
        for regime in self.regimes:
            color = 'red' if regime['regime'] == 'bear' else 'green'
            marker = 'v' if regime['regime'] == 'bear' else '^'
            plt.scatter(regime['date'], regime['price'], 
                       color=color, s=100, marker=marker, zorder=5)
            
            # Add text annotation
            label = f"Bear Start" if regime['regime'] == 'bear' else "Bull Start"
            plt.annotate(f"{label}\n{regime['date'].strftime('%Y-%m-%d')}", 
                        (regime['date'], regime['price']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                        fontsize=8)
        
        plt.title(f'{self.symbol} Price with Bull/Bear Market Transitions')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Analyze S&P 500
    detector = MarketRegimeDetector("^GSPC", "5y")  # 5 years of S&P 500 data
    
    print("Fetching data from Yahoo Finance...")
    if detector.fetch_data():
        print("Detecting market regimes...")
        detector.detect_regimes()
        detector.print_results()
        detector.get_current_status()
        
        # Uncomment to show plot
        # detector.plot_regimes()
    
    # You can also analyze other indices:
    # nasdaq_detector = MarketRegimeDetector("^IXIC", "3y")  # Nasdaq
    # dow_detector = MarketRegimeDetector("^DJI", "2y")      # Dow Jones