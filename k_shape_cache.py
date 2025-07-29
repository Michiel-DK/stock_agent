import pandas as pd
import yfinance as yf
import os
from datetime import datetime

class StockDataCache:
    def __init__(self, cache_file='stock_data_cache_nq.csv'):
        self.cache_file = cache_file
        self.data = None
        
    def build_cache(self, tickers, period='1y', force_rebuild=False):
        """
        Build cache of stock data for all tickers
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers
        period : str
            Time period for data ('1y', '2y', '5y', etc.)
        force_rebuild : bool
            Whether to rebuild cache even if it exists
        """
        if os.path.exists(self.cache_file) and not force_rebuild:
            print(f"Cache file {self.cache_file} already exists. Use force_rebuild=True to rebuild.")
            return self.load_cache()
        
        print(f"Building cache for {len(tickers)} tickers...")
        
        stock_data = {}
        successful_tickers = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(tickers)} tickers processed")
                
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                # Only keep if we have valid Close data
                if not hist.empty and 'Close' in hist.columns:
                    close_data = hist['Close']
                    if not close_data.empty and close_data.notna().sum() > 10:
                        stock_data[ticker] = close_data
                        successful_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                failed_tickers.append(ticker)
        
        # Create DataFrame and save
        if stock_data:
            df = pd.DataFrame(stock_data)
            df.to_csv(self.cache_file)
            print(f"Cache saved to {self.cache_file}")
            print(f"Successfully cached {len(successful_tickers)} tickers")
            print(f"Failed to cache {len(failed_tickers)} tickers")
            
            # Save metadata
            metadata = {
                'build_date': datetime.now().isoformat(),
                'period': period,
                'total_tickers': len(tickers),
                'successful_tickers': len(successful_tickers),
                'failed_tickers': len(failed_tickers),
                'successful_list': successful_tickers,
                'failed_list': failed_tickers
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_csv(self.cache_file.replace('.csv', '_metadata.csv'), index=False)
            
            self.data = df
            return df
        else:
            print("No valid stock data found!")
            return pd.DataFrame()
    
    def load_cache(self):
        """
        Load cached stock data
        """
        if not os.path.exists(self.cache_file):
            print(f"Cache file {self.cache_file} does not exist. Build cache first.")
            return pd.DataFrame()
        
        print(f"Loading cache from {self.cache_file}")
        self.data = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
        print(f"Loaded cache with {self.data.shape[0]} dates and {self.data.shape[1]} tickers")
        return self.data
    
    def get_available_tickers(self):
        """
        Get list of tickers available in cache
        """
        if self.data is None:
            self.load_cache()
        
        return list(self.data.columns) if self.data is not None else []
    
    def get_ticker_data(self, ticker):
        """
        Get data for a specific ticker
        """
        if self.data is None:
            self.load_cache()
        
        if ticker in self.data.columns:
            return self.data[ticker].dropna()
        else:
            print(f"Ticker {ticker} not found in cache")
            return pd.Series()
    
    def get_sample_data(self, tickers, start_date=None):
        """
        Get data for a sample of tickers, optionally filtered by start date
        
        Parameters:
        -----------
        tickers : list
            List of tickers to get data for
        start_date : str or datetime-like, optional
            Start date to filter data from (inclusive). Can be:
            - String in format 'YYYY-MM-DD' (e.g., '2023-01-01')
            - datetime object
            - pandas Timestamp
            If None, returns all available data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with only the requested tickers that exist in cache,
            filtered by start_date if specified
        """
        if self.data is None:
            self.load_cache()
        
        # Filter to only tickers that exist in cache
        available_tickers = [t for t in tickers if t in self.data.columns]
        missing_tickers = [t for t in tickers if t not in self.data.columns]
        
        if missing_tickers:
            print(f"Warning: {len(missing_tickers)} tickers not found in cache: {missing_tickers}")
        
        if available_tickers:
            # Get the data for available tickers
            result_data = self.data[available_tickers].dropna()
            
            # Apply date filter if specified
            if start_date is not None:
                try:
                    # Convert start_date to pandas datetime if it's a string
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    
                    # Handle timezone issues - make start_date compatible with the index
                    if hasattr(result_data.index, 'tz') and result_data.index.tz is not None:
                        # Index is timezone-aware, make start_date timezone-aware too
                        if start_date.tz is None:
                            start_date = start_date.tz_localize(result_data.index.tz)
                    else:
                        # Index is timezone-naive, make start_date timezone-naive too
                        if hasattr(start_date, 'tz') and start_date.tz is not None:
                            start_date = start_date.tz_localize(None)
                    
                    # Filter data from start_date onwards
                    result_data = result_data[result_data.index >= start_date]
                    
                    if result_data.empty:
                        print(f"Warning: No data found after {start_date}")
                    else:
                        print(f"Filtered data from {start_date} onwards: {result_data.shape[0]} dates")
                        
                except Exception as e:
                    print(f"Error parsing start_date '{start_date}': {e}")
                    print("Returning unfiltered data")
            
            return result_data
        else:
            print("No requested tickers found in cache")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    cache = StockDataCache(cache_file='stock_data_cache_nq_5y.csv')
    