import pandas as pd
import yfinance as yf
import os
from datetime import datetime

class StockDataCache:
    def __init__(self, cache_file='cache/stock_data_cache.csv'):
        self.cache_file = cache_file
        self.metadata_file = cache_file.replace('.csv', '_metadata.csv')
        self.data = None
        
    def fetch_and_cache_data(self, tickers, period='1y', force_rebuild=False):
        """
        Fetch stock data from Yahoo Finance and cache it
        
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
        
        print(f"Fetching and caching data for {len(tickers)} tickers...")
        
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
            # Convert index to datetime with UTC
            df.index = pd.to_datetime(df.index, utc=True)
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
            metadata_df.to_csv(self.metadata_file, index=False)
            
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
            print(f"Cache file {self.cache_file} does not exist. Use fetch_and_cache_data() first.")
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
        
        return list(self.data.columns) if self.data is not None and not self.data.empty else []
    
    def get_ticker_data(self, ticker):
        """
        Get data for a specific ticker
        """
        if self.data is None:
            self.load_cache()
        
        if not self.data.empty and ticker in self.data.columns:
            return self.data[ticker].dropna()
        else:
            print(f"Ticker {ticker} not found in cache")
            return pd.Series()
    
    def get_data(self, tickers=None, start_date=None):
        """
        Get stock data for specified tickers and date range
        
        Parameters:
        -----------
        tickers : list, optional
            List of tickers to get data for. If None, returns all available tickers
        start_date : str or datetime-like, optional
            Start date to filter data from (inclusive). If None, returns all available data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with requested tickers and date range
        """
        if self.data is None:
            self.load_cache()
        
        if self.data is None or self.data.empty:
            print("No data available in cache")
            return pd.DataFrame()
        
        # Start with all data
        result_data = self.data.copy()
        
        # Filter by tickers if specified
        if tickers is not None:
            available_tickers = [t for t in tickers if t in result_data.columns]
            missing_tickers = [t for t in tickers if t not in result_data.columns]
            
            if missing_tickers:
                print(f"Warning: {len(missing_tickers)} tickers not found in cache: {missing_tickers}")
            
            if available_tickers:
                result_data = result_data[available_tickers]
            else:
                print("No requested tickers found in cache")
                return pd.DataFrame()
        
        # Apply date filter if specified
        if start_date is not None:
            try:
                # Convert start_date to pandas datetime if it's a string
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                
                # Handle timezone issues
                if hasattr(result_data.index, 'tz') and result_data.index.tz is not None:
                    if start_date.tz is None:
                        start_date = start_date.tz_localize(result_data.index.tz)
                else:
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
        
        # Drop rows where all values are NaN
        result_data = result_data.dropna(how='all')
        
        return result_data
    
    def add_tickers(self, new_tickers, period='1y'):
        """
        Add new tickers to existing cache
        
        Parameters:
        -----------
        new_tickers : list
            List of new tickers to add
        period : str
            Time period for new data
        """
        if self.data is None:
            self.load_cache()
        
        existing_tickers = self.get_available_tickers()
        tickers_to_fetch = [t for t in new_tickers if t not in existing_tickers]
        
        if not tickers_to_fetch:
            print("All requested tickers already exist in cache")
            return self.data
        
        print(f"Adding {len(tickers_to_fetch)} new tickers to cache...")
        
        # Fetch new data
        new_data = {}
        successful_tickers = []
        failed_tickers = []
        
        for ticker in tickers_to_fetch:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty and 'Close' in hist.columns:
                    close_data = hist['Close']
                    if not close_data.empty and close_data.notna().sum() > 10:
                        new_data[ticker] = close_data
                        successful_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if new_data:
            # Create DataFrame for new data
            new_df = pd.DataFrame(new_data)
            new_df.index = pd.to_datetime(new_df.index, utc=True)
            
            # Merge with existing data
            if not self.data.empty:
                self.data = pd.concat([self.data, new_df], axis=1, sort=True)
            else:
                self.data = new_df
            
            # Save updated cache
            self.data.to_csv(self.cache_file)
            print(f"Added {len(successful_tickers)} new tickers to cache")
            
            if failed_tickers:
                print(f"Failed to add {len(failed_tickers)} tickers: {failed_tickers}")
        
        return self.data
    
    def get_cache_info(self):
        """
        Get information about the current cache
        """
        if self.data is None:
            self.load_cache()
        
        if self.data is None or self.data.empty:
            return {"status": "empty", "tickers": 0, "date_range": None}
        
        info = {
            "status": "loaded",
            "tickers": len(self.data.columns),
            "ticker_list": list(self.data.columns),
            "date_range": {
                "start": self.data.index.min().strftime('%Y-%m-%d'),
                "end": self.data.index.max().strftime('%Y-%m-%d')
            },
            "total_records": self.data.shape[0] * self.data.shape[1],
            "non_null_records": self.data.count().sum()
        }
        
        # Load metadata if available
        if os.path.exists(self.metadata_file):
            try:
                metadata = pd.read_csv(self.metadata_file).iloc[0].to_dict()
                info["metadata"] = metadata
            except:
                pass
        
        return info