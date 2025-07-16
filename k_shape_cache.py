import pandas as pd
import yfinance as yf
import os
from datetime import datetime

from k_shape import StockKShapeClustering  # Assuming this is your clustering

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
    
    def get_sample_data(self, tickers):
        """
        Get data for a sample of tickers
        
        Parameters:
        -----------
        tickers : list
            List of tickers to get data for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with only the requested tickers that exist in cache
        """
        if self.data is None:
            self.load_cache()
        
        # Filter to only tickers that exist in cache
        available_tickers = [t for t in tickers if t in self.data.columns]
        missing_tickers = [t for t in tickers if t not in self.data.columns]
        
        if missing_tickers:
            print(f"Warning: {len(missing_tickers)} tickers not found in cache: {missing_tickers}")
        
        if available_tickers:
            return self.data[available_tickers].dropna()
        else:
            print("No requested tickers found in cache")
            return pd.DataFrame()

# Updated StockKShapeClustering class to use cache
class StockKShapeClusteringWithCache(StockKShapeClustering):
    def __init__(self, n_clusters=3, random_state=42, cache_file='stock_data_cache_nq.csv'):
        super().__init__(n_clusters, random_state)
        self.cache = StockDataCache(cache_file)
        
    def fetch_stock_data_from_cache(self, tickers):
        """
        Fetch stock data from cache instead of API
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers
        """
        print("Fetching stock data from cache...")
        
        # Load cache if not already loaded
        if self.cache.data is None:
            self.cache.load_cache()
            
        # Get data for requested tickers
        self.stock_data = self.cache.get_sample_data(tickers)
        
        if self.stock_data.empty:
            print("No data available for requested tickers")
        else:
            print(f"Successfully loaded data for {len(self.stock_data.columns)} stocks from cache")
            
        return self.stock_data

# Usage functions
def build_stock_cache(ticker_file='ticker_symbols_only.txt', period='1y'):
    """
    Build cache for all tickers in file
    """
    # Load tickers from file
    tickers_df = pd.read_csv(ticker_file)
    tickers = tickers_df.Symbol.to_list()
    
    # Build cache
    cache = StockDataCache()
    cache.build_cache(tickers, period=period)
    
    return cache

def run_clustering_with_cache(sample_size=100, n_clusters=6):
    """
    Run clustering using cached data
    """
    # Initialize clustering with cache
    clustering = StockKShapeClusteringWithCache(n_clusters=n_clusters, random_state=42)
    
    # Get available tickers from cache
    available_tickers = clustering.cache.get_available_tickers()
    
    if len(available_tickers) == 0:
        print("No tickers available in cache. Build cache first.")
        return None
    
    print(f"Available tickers in cache: {len(available_tickers)}")
    
    # Sample tickers
    import random
    sample_tickers = random.sample(available_tickers, min(sample_size, len(available_tickers)))
    print(f"Sampling {len(sample_tickers)} tickers for clustering")
    
    # Fetch data from cache
    stock_data = clustering.fetch_stock_data_from_cache(sample_tickers)
    
    if stock_data.empty:
        print("No data available for clustering")
        return None
    
    # Run clustering pipeline
    print("\nPreparing features...")
    features = clustering.prepare_features(feature_type='normalized_prices')
    
    print("Fitting clustering...")
    labels = clustering.fit_clustering()
    
    print("Getting results...")
    results = clustering.get_cluster_results()
    
    print(f"\nClustering Results:")
    print(results)
    
    # Analyze performance
    print("\nAnalyzing cluster performance...")
    try:
        performance = clustering.analyze_cluster_performance()
        print("\nCluster Performance Analysis:")
        print(performance[['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio']])
        
        # Print average performance per cluster with stock lists
        print("\nDetailed Cluster Analysis:")
        for _, cluster_info in performance.iterrows():
            cluster_num = cluster_info['Cluster']
            stocks = cluster_info['Stocks']
            print(f"\n--- Cluster {cluster_num} ({cluster_info['Count']} stocks) ---")
            print(f"Stocks: {', '.join(stocks)}")
            print(f"Average Return: {cluster_info['Avg_Return']:.4f} ({cluster_info['Avg_Return']*100:.2f}%)")
            print(f"Average Volatility: {cluster_info['Avg_Volatility']:.4f} ({cluster_info['Avg_Volatility']*100:.2f}%)")
            print(f"Sharpe Ratio: {cluster_info['Sharpe_Ratio']:.4f}")
            print(f"Max Drawdown: {cluster_info['Max_Drawdown']:.4f} ({cluster_info['Max_Drawdown']*100:.2f}%)")
            
        # Calculate and print overall averages across all clusters
        print(f"\n--- Overall Portfolio Averages ---")
        print(f"Average Return: {performance['Avg_Return'].mean():.4f} ({performance['Avg_Return'].mean()*100:.2f}%)")
        print(f"Average Volatility: {performance['Avg_Volatility'].mean():.4f} ({performance['Avg_Volatility'].mean()*100:.2f}%)")
        print(f"Average Sharpe Ratio: {performance['Sharpe_Ratio'].mean():.4f}")
        print(f"Average Max Drawdown: {performance['Max_Drawdown'].mean():.4f} ({performance['Max_Drawdown'].mean()*100:.2f}%)")
        
    except Exception as e:
        print(f"Performance analysis failed: {e}")
        performance = None
    
    # Visualize results
    print("\nGenerating visualizations...")
    try:
        clustering.plot_clusters(feature_type='normalized_prices')
        print("✅ Cluster visualization completed")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save basic results
    results.to_csv(f'clusters_{timestamp}.csv', index=False)
    print(f"✅ Basic results saved to clusters_{timestamp}.csv")
    
    # Save detailed results with performance metrics
    if performance is not None:
        # Create detailed results by merging cluster results with performance
        detailed_results = results.copy()
        
        # Add performance metrics for each stock
        perf_dict = {}
        for _, cluster_info in performance.iterrows():
            cluster_num = cluster_info['Cluster']
            for stock in cluster_info['Stocks']:
                perf_dict[stock] = {
                    'Cluster_Avg_Return': cluster_info['Avg_Return'],
                    'Cluster_Avg_Volatility': cluster_info['Avg_Volatility'],
                    'Cluster_Sharpe_Ratio': cluster_info['Sharpe_Ratio'],
                    'Cluster_Max_Drawdown': cluster_info['Max_Drawdown']
                }
        
        # Add performance columns
        for col in ['Cluster_Avg_Return', 'Cluster_Avg_Volatility', 'Cluster_Sharpe_Ratio', 'Cluster_Max_Drawdown']:
            detailed_results[col] = detailed_results['Ticker'].map(lambda x: perf_dict.get(x, {}).get(col, None))
        
        detailed_results.to_csv(f'detailed_clusters_{timestamp}.csv', index=False)
        print(f"✅ Detailed results saved to detailed_clusters_{timestamp}.csv")
        
        # Save performance summary
        performance_summary = performance[['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio', 'Max_Drawdown']].copy()
        performance_summary.to_csv(f'cluster_performance_{timestamp}.csv', index=False)
        print(f"✅ Performance summary saved to cluster_performance_{timestamp}.csv")
    
    print(f"\n🎉 CLUSTERING ANALYSIS COMPLETED!")
    print(f"📊 {len(results)} stocks clustered into {n_clusters} clusters")
    
    return clustering, results, performance

# Example usage
if __name__ == "__main__":
    # Step 1: Build cache (run once)
    print("=== BUILDING CACHE ===")
    cache = build_stock_cache('nasdaq_screener.csv', period='2mo')
    
    # Step 2: Run clustering (run multiple times with different samples)
    print("\n=== RUNNING CLUSTERING ===")
    clustering, results, performance = run_clustering_with_cache(sample_size=3640, n_clusters=10)