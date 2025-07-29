import pandas as pd
from tslearn.utils import to_time_series_dataset
import numpy as np

from k_shape_clustering import StockKShapeClustering  # Assuming this is your clustering
from k_shape_cache import StockDataCache

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
        self.stock_data = self.cache.get_sample_data(tickers, start_date='2022-10-12')

        if self.stock_data.empty:
            print("No data available for requested tickers")
        else:
            print(f"Successfully loaded data for {len(self.stock_data.columns)} stocks from cache")
            
        return self.stock_data
    
    def prepare_features(self, feature_type='returns', window_size=20):
            """
            Prepare time series features for clustering
            
            Parameters:
            -----------
            feature_type : str
                Type of features to extract:
                - 'returns': Daily returns
                - 'normalized_prices': Normalized price series
                - 'ma_relative': Price relative to moving average
                - 'volatility': Rolling volatility
            window_size : int
                Window size for rolling calculations
            """
            if self.stock_data is None:
                raise ValueError("No stock data available. Run fetch_stock_data first.")
                
            features = {}
            skipped_tickers = []
            
            for ticker in self.stock_data.columns:
                prices = self.stock_data[ticker]
                
                # Skip if prices series is empty or all NaN
                if prices.empty or prices.isna().all():
                    print(f"Skipping {ticker}: Empty or all NaN price data")
                    skipped_tickers.append(ticker)
                    continue
                
                try:
                    if feature_type == 'returns':
                        # Daily returns
                        feature_series = prices.pct_change().dropna()
                        
                    elif feature_type == 'normalized_prices':
                        # Normalize prices to start at 1
                        if prices.iloc[0] == 0 or pd.isna(prices.iloc[0]):
                            print(f"Skipping {ticker}: First price is 0 or NaN, cannot normalize")
                            skipped_tickers.append(ticker)
                            continue
                        feature_series = prices / prices.iloc[0]
                        
                    elif feature_type == 'ma_relative':
                        # Price relative to moving average
                        ma = prices.rolling(window=window_size).mean()
                        feature_series = (prices / ma - 1).dropna()
                        
                    elif feature_type == 'volatility':
                        # Rolling volatility
                        returns = prices.pct_change()
                        feature_series = returns.rolling(window=window_size).std().dropna()
                        
                    else:
                        raise ValueError(f"Unknown feature_type: {feature_type}")
                    
                    # Check if resulting feature series is empty, all NaN, or has insufficient data
                    if feature_series.empty or feature_series.isna().all() or len(feature_series) < 10:
                        print(f"Skipping {ticker}: Insufficient valid data after {feature_type} calculation")
                        skipped_tickers.append(ticker)
                        continue
                    
                    # Check for infinite values
                    if np.isinf(feature_series).any():
                        print(f"Skipping {ticker}: Contains infinite values in {feature_type}")
                        skipped_tickers.append(ticker)
                        continue
                        
                    features[ticker] = feature_series
                    
                except Exception as e:
                    print(f"Skipping {ticker}: Error calculating {feature_type} - {str(e)}")
                    skipped_tickers.append(ticker)
                    continue
            
            if not features:
                print("WARNING: No valid features could be calculated for any ticker")
                # Return empty arrays but don't crash
                self.features = to_time_series_dataset([])
                return pd.DataFrame()
            
            # Convert to DataFrame and ensure all series have same length
            features_df = pd.DataFrame(features).dropna()
            
            # Check if any columns became empty after alignment
            empty_cols = features_df.columns[features_df.isna().all()].tolist()
            if empty_cols:
                print(f"Removing tickers with empty features after alignment: {empty_cols}")
                features_df = features_df.drop(columns=empty_cols)
                skipped_tickers.extend(empty_cols)
            
            if features_df.empty:
                print("WARNING: No valid aligned features remain after processing")
                # Return empty arrays but don't crash
                self.features = to_time_series_dataset([])
                return pd.DataFrame()
            
            # Convert to time series dataset format for tslearn
            self.features = to_time_series_dataset([features_df[col].values for col in features_df.columns])
            
            # Store valid tickers for later use
            self.valid_tickers = list(features_df.columns)
            
            if skipped_tickers:
                print(f"Skipped {len(skipped_tickers)} tickers: {skipped_tickers}")
            
            print(f"Prepared {feature_type} features for {len(features_df.columns)} tickers with shape: {self.features.shape}")
            return features_df

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
    
    # Convert index to datetime with UTC
    cache.data.index = pd.to_datetime(cache.data.index, utc=True)

    cache.data = cache.data.resample('1m').last()
    
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

def analyze_cluster(clustering, results, performance, n_clusters=6):
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
    cache = build_stock_cache('nasdaq_screener.csv', period='5y')
    
    n_clusters = 6
    
    # Step 2: Run clustering (run multiple times with different samples)
    print("\n=== RUNNING CLUSTERING ===")
    clustering, results, performance = run_clustering_with_cache(sample_size=3640, n_clusters=n_clusters)

    print("\n=== ANALYZING CLUSTERS ===")
    analyze_cluster(clustering, results, performance, n_clusters)