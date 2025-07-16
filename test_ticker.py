import pandas as pd
import numpy as np
import yfinance as yf

def debug_stock_data(tickers, period='1y'):
    """
    Debug function to check what's happening with stock data fetching
    """
    print("=== DEBUGGING STOCK DATA FETCHING ===")
    print(f"Tickers to fetch: {tickers}")
    print(f"Period: {period}")
    
    stock_data = {}
    
    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        try:
            stock = yf.Ticker(ticker)
            print(f"Created ticker object for {ticker}")
            
            # Try to get info first
            try:
                info = stock.info
                print(f"Info available: {ticker} - {info.get('longName', 'No name')}")
            except:
                print(f"Warning: Could not get info for {ticker}")
            
            # Get historical data
            hist = stock.history(period=period)
            print(f"Raw history shape: {hist.shape}")
            print(f"Raw history columns: {hist.columns.tolist()}")
            print(f"Raw history index range: {hist.index.min()} to {hist.index.max()}")
            
            if hist.empty:
                print(f"ERROR: Empty history for {ticker}")
                continue
                
            # Check Close prices specifically
            close_prices = hist['Close']
            print(f"Close prices shape: {close_prices.shape}")
            print(f"Close prices - First 5 values:")
            print(close_prices.head())
            print(f"Close prices - Last 5 values:")
            print(close_prices.tail())
            print(f"Close prices - NaN count: {close_prices.isna().sum()}")
            print(f"Close prices - Zero count: {(close_prices == 0).sum()}")
            
            if close_prices.isna().all():
                print(f"ERROR: All Close prices are NaN for {ticker}")
                continue
            
            if close_prices.empty:
                print(f"ERROR: Close prices empty for {ticker}")
                continue
                
            stock_data[ticker] = close_prices
            print(f"SUCCESS: Added {ticker} to stock_data")
            
        except Exception as e:
            print(f"ERROR fetching {ticker}: {str(e)}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== FINAL STOCK DATA SUMMARY ===")
    print(f"Successfully fetched {len(stock_data)} out of {len(tickers)} tickers")
    print(f"Successful tickers: {list(stock_data.keys())}")
    
    if stock_data:
        # Create DataFrame
        df = pd.DataFrame(stock_data)
        print(f"DataFrame shape before dropna: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame index range: {df.index.min()} to {df.index.max()}")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        print(f"NaN counts per column:")
        for col, count in nan_counts.items():
            print(f"  {col}: {count} NaN values")
        
        # Drop NaN
        df_clean = df.dropna()
        print(f"DataFrame shape after dropna: {df_clean.shape}")
        
        if df_clean.empty:
            print("ERROR: DataFrame is empty after dropping NaN!")
            print("This suggests no overlapping valid data points across all tickers")
            
            # Try with individual tickers
            print("\n=== TESTING INDIVIDUAL TICKERS ===")
            for ticker in df.columns:
                ticker_data = df[ticker].dropna()
                print(f"{ticker}: {len(ticker_data)} valid data points")
                if len(ticker_data) > 0:
                    print(f"  Range: {ticker_data.index.min()} to {ticker_data.index.max()}")
        else:
            print("SUCCESS: Clean DataFrame created")
            print(f"Sample data:")
            print(df_clean.head())
            
        return df_clean
    else:
        print("ERROR: No stock data fetched successfully")
        return pd.DataFrame()

def debug_feature_preparation(stock_data, feature_type='normalized_prices'):
    """
    Debug feature preparation step
    """
    print(f"\n=== DEBUGGING FEATURE PREPARATION ({feature_type}) ===")
    
    if stock_data.empty:
        print("ERROR: Input stock_data is empty!")
        return pd.DataFrame()
    
    print(f"Input stock_data shape: {stock_data.shape}")
    print(f"Input stock_data columns: {stock_data.columns.tolist()}")
    
    features = {}
    
    for ticker in stock_data.columns:
        print(f"\n--- Processing {ticker} for {feature_type} ---")
        prices = stock_data[ticker]
        
        print(f"Prices shape: {prices.shape}")
        print(f"Prices type: {type(prices)}")
        print(f"First few prices: {prices.head()}")
        print(f"NaN count: {prices.isna().sum()}")
        
        if prices.empty:
            print(f"ERROR: Empty prices for {ticker}")
            continue
            
        if prices.isna().all():
            print(f"ERROR: All NaN prices for {ticker}")
            continue
        
        try:
            if feature_type == 'normalized_prices':
                first_price = prices.iloc[0]
                print(f"First price: {first_price}")
                
                if pd.isna(first_price) or first_price == 0:
                    print(f"ERROR: First price is {first_price}")
                    continue
                    
                normalized = prices / first_price
                print(f"Normalized shape: {normalized.shape}")
                print(f"Normalized first few: {normalized.head()}")
                print(f"Normalized NaN count: {normalized.isna().sum()}")
                
                features[ticker] = normalized
                print(f"SUCCESS: Added {ticker} to features")
                
            elif feature_type == 'returns':
                returns = prices.pct_change().dropna()
                print(f"Returns shape: {returns.shape}")
                print(f"Returns NaN count: {returns.isna().sum()}")
                
                if returns.empty:
                    print(f"ERROR: Empty returns for {ticker}")
                    continue
                    
                features[ticker] = returns
                print(f"SUCCESS: Added {ticker} to features")
                
        except Exception as e:
            print(f"ERROR processing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== FEATURE PREPARATION SUMMARY ===")
    print(f"Successfully processed {len(features)} out of {len(stock_data.columns)} tickers")
    
    if features:
        features_df = pd.DataFrame(features)
        print(f"Features DataFrame shape before dropna: {features_df.shape}")
        
        features_clean = features_df.dropna()
        print(f"Features DataFrame shape after dropna: {features_clean.shape}")
        
        if features_clean.empty:
            print("ERROR: Features DataFrame empty after dropna!")
        else:
            print("SUCCESS: Features DataFrame created")
            print(f"Sample features:")
            print(features_clean.head())
            
        return features_clean
    else:
        print("ERROR: No features created")
        return pd.DataFrame()

# Main debugging function
def debug_everything(tickers, period='1y'):
    """
    Run complete debugging pipeline
    """
    print("Starting complete debugging...")
    
    # Debug stock data fetching
    stock_data = debug_stock_data(tickers, period)
    
    if not stock_data.empty:
        # Debug feature preparation
        features = debug_feature_preparation(stock_data, 'normalized_prices')
        
        if not features.empty:
            print(f"\n=== FINAL SUCCESS ===")
            print(f"Stock data shape: {stock_data.shape}")
            print(f"Features shape: {features.shape}")
            return stock_data, features
    
    print(f"\n=== DEBUGGING COMPLETE - ISSUES FOUND ===")
    return stock_data, pd.DataFrame()

import pandas as pd
import numpy as np

def debug_specific_ticker_issue(clustering):
    """
    Debug why all tickers are being skipped
    """
    print("=== DEBUGGING SPECIFIC TICKER ISSUE ===")
    
    if clustering.stock_data is None:
        print("ERROR: No stock data available")
        return
        
    print(f"Stock data shape: {clustering.stock_data.shape}")
    print(f"Stock data columns: {clustering.stock_data.columns.tolist()[:10]}...")  # First 10
    print(f"Stock data index: {clustering.stock_data.index}")
    
    # Check a few specific tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL'] if any(t in clustering.stock_data.columns for t in ['AAPL', 'MSFT', 'GOOGL']) else clustering.stock_data.columns[:3]
    
    for ticker in test_tickers:
        if ticker in clustering.stock_data.columns:
            print(f"\n--- Detailed check for {ticker} ---")
            prices = clustering.stock_data[ticker]
            
            print(f"Prices type: {type(prices)}")
            print(f"Prices shape: {prices.shape}")
            print(f"Prices empty: {prices.empty}")
            print(f"Prices all NaN: {prices.isna().all()}")
            print(f"Prices dtype: {prices.dtype}")
            
            if not prices.empty:
                print(f"First 5 values:")
                print(prices.head())
                print(f"Last 5 values:")
                print(prices.tail())
                print(f"NaN count: {prices.isna().sum()}")
                print(f"Valid count: {prices.notna().sum()}")
                
                # Try the normalization that's failing
                try:
                    first_price = prices.iloc[0]
                    print(f"First price: {first_price} (type: {type(first_price)})")
                    print(f"First price is NaN: {pd.isna(first_price)}")
                    print(f"First price is zero: {first_price == 0}")
                    
                    if not pd.isna(first_price) and first_price != 0:
                        normalized = prices / first_price
                        print(f"Normalization successful: {normalized.head()}")
                    else:
                        print("Normalization would fail due to first price")
                        
                except Exception as e:
                    print(f"Error during normalization test: {e}")
            else:
                print("Prices series is empty")

# Quick fix function to bypass the error temporarily
def quick_fix_prepare_features(clustering, feature_type='normalized_prices'):
    """
    Quick fix version that won't crash
    """
    print("Using quick fix version of prepare_features...")
    
    if clustering.stock_data is None:
        print("No stock data available")
        return pd.DataFrame()
        
    # First, debug the issue
    debug_specific_ticker_issue(clustering)
    
    # Try a simple approach
    valid_data = {}
    
    for ticker in clustering.stock_data.columns:
        prices = clustering.stock_data[ticker]
        
        # More lenient checks
        if not prices.empty and prices.notna().sum() > 10:  # At least 10 valid prices
            try:
                if feature_type == 'normalized_prices':
                    # Find first valid price
                    first_valid_idx = prices.first_valid_index()
                    if first_valid_idx is not None:
                        first_price = prices.loc[first_valid_idx]
                        if first_price > 0:
                            normalized = prices / first_price
                            # Only keep if we have enough valid data
                            if normalized.notna().sum() > 10:
                                valid_data[ticker] = normalized
                                print(f"SUCCESS: {ticker} normalized successfully")
                            else:
                                print(f"SKIP: {ticker} - insufficient valid data after normalization")
                        else:
                            print(f"SKIP: {ticker} - first valid price is zero")
                    else:
                        print(f"SKIP: {ticker} - no valid price found")
                else:
                    # Handle other feature types
                    returns = prices.pct_change().dropna()
                    if len(returns) > 10:
                        valid_data[ticker] = returns
                        print(f"SUCCESS: {ticker} returns calculated")
                    else:
                        print(f"SKIP: {ticker} - insufficient returns data")
                        
            except Exception as e:
                print(f"SKIP: {ticker} - error: {e}")
        else:
            print(f"SKIP: {ticker} - empty or insufficient data")
    
    if valid_data:
        print(f"\nSUCCESS: Found {len(valid_data)} valid tickers")
        features_df = pd.DataFrame(valid_data).dropna()
        print(f"Features shape after dropna: {features_df.shape}")
        
        if not features_df.empty:
            # Store for later use
            clustering.valid_tickers = list(features_df.columns)
            # Convert to tslearn format
            from tslearn.utils import to_time_series_dataset
            clustering.features = to_time_series_dataset([features_df[col].values for col in features_df.columns])
            print(f"tslearn features shape: {clustering.features.shape}")
            return features_df
        else:
            print("Features DataFrame empty after dropna")
    else:
        print("No valid data found")
    
    return pd.DataFrame()

# Usage:
# features = quick_fix_prepare_features(clustering, 'normalized_prices')

# Usage example:
if __name__ == "__main__":
    # Test with a few known good tickers
    tickers = pd.read_csv('ticker_symbols_only.txt').Ticker.to_list()

    import random
    tickers = random.sample(tickers, 100)  # Sample 100 tickers for demonstration
    stock_data, features = debug_everything(tickers, period='1y')