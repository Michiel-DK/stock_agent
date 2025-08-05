import pandas as pd
import random
from price_clustering.k_shape_cache import StockDataCache
from price_clustering.k_shape_clustering import StockKShapeClustering

def main():
    """
    Example usage of the refactored stock clustering system
    """
    print("=== STOCK CLUSTERING PIPELINE ===")
    
    # Step 1: Initialize cache
    print("\n1. Initializing cache...")
    cache = StockDataCache(cache_file='cache/nasdaq_cache_5y.csv')
    
    # Check if cache exists, if not build it
    cache_info = cache.get_cache_info()
    if cache_info['status'] == 'empty':
        print("Cache is empty. Building cache...")
        
        # Load tickers from file (adjust path as needed)
        try:
            tickers_df = pd.read_csv('tickers/nasdaq_screener.csv')  # or ticker_symbols_only.txt
            tickers = tickers_df['Symbol'].tolist()  # adjust column name as needed
            print(f"Found {len(tickers)} tickers to cache")
            
            # Build cache with 5 years of data
            cache.fetch_and_cache_data(tickers, period='5y')
            
        except FileNotFoundError:
            print("Ticker file not found. Using sample tickers...")
            # Sample tickers for demo
            sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
                            'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SHOP']
            cache.fetch_and_cache_data(sample_tickers, period='5y')
    else:
        print(f"Cache loaded: {cache_info['tickers']} tickers available")
        print(f"Date range: {cache_info['date_range']['start']} to {cache_info['date_range']['end']}")
    
    # Step 2: Initialize clustering
    print("\n2. Initializing clustering...")
    clustering = StockKShapeClustering(cache, n_clusters=6, random_state=42)
    
    # Step 3: Load data for clustering
    print("\n3. Loading data for clustering...")
    available_tickers = cache.get_available_tickers()
    
    # Sample tickers for clustering (adjust sample_size as needed)
    sample_size = min(100, len(available_tickers))  # Use up to 100 tickers
    sample_tickers = random.sample(available_tickers, sample_size)
    print(f"Using {len(sample_tickers)} tickers for clustering")
    
    # Load data from last 2 years for clustering
    stock_data = clustering.load_data(
        tickers=sample_tickers, 
        start_date='2022-01-01'  # Adjust as needed
    )
    
    if stock_data.empty:
        print("No data available for clustering. Exiting.")
        return
    
    # Step 4: Prepare features
    print("\n4. Preparing features...")
    features_df = clustering.prepare_features(
        feature_type='normalized_prices',  # Can be 'returns', 'normalized_prices', 'ma_relative', 'volatility'
        window_size=20
    )
    
    if features_df.empty:
        print("No valid features prepared. Exiting.")
        return
    
    # Step 5: Find optimal number of clusters (optional)
    print("\n5. Finding optimal clusters (optional)...")
    try:
        optimal_k, silhouette_scores = clustering.find_optimal_clusters(max_clusters=10)
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Optionally update n_clusters
        # clustering.n_clusters = optimal_k
    except Exception as e:
        print(f"Could not determine optimal clusters: {e}")
        print("Proceeding with default number of clusters")
    
    # Step 6: Fit clustering model
    print("\n6. Fitting clustering model...")
    cluster_labels = clustering.fit_clustering(scale_features=True)
    
    if len(cluster_labels) == 0:
        print("Clustering failed. Exiting.")
        return
    
    # Step 7: Get and display results
    print("\n7. Analyzing results...")
    results = clustering.get_cluster_results()
    print(f"\nClustering completed! {len(results)} stocks clustered into {clustering.n_clusters} clusters")
    print("\nCluster distribution:")
    print(results['Cluster'].value_counts().sort_index())
    
    # Step 8: Performance analysis
    print("\n8. Analyzing cluster performance...")
    try:
        performance = clustering.analyze_cluster_performance()
        print("\nCluster Performance Summary:")
        print(performance[['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio']].round(4))
        
        # Show stocks in each cluster
        print("\nStocks by Cluster:")
        for _, cluster_info in performance.iterrows():
            cluster_num = cluster_info['Cluster']
            stocks = cluster_info['Stocks'][:5]  # Show first 5 stocks
            stock_list = ', '.join(stocks)
            if len(cluster_info['Stocks']) > 5:
                stock_list += f" ... ({len(cluster_info['Stocks']) - 5} more)"
            print(f"Cluster {cluster_num}: {stock_list}")
            
    except Exception as e:
        print(f"Performance analysis failed: {e}")
    
    # Step 9: Visualize results
    print("\n9. Creating visualizations...")
    try:
        clustering.plot_clusters(feature_type='normalized_prices')
        print("✅ Visualizations created")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Step 10: Save results
    print("\n10. Saving results...")
    try:
        timestamp = clustering.save_results(base_filename='nasdaq_clustering')
        print(f"✅ All results saved with timestamp: {timestamp}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print(f"\n🎉 CLUSTERING ANALYSIS COMPLETED!")
    print(f"📊 {len(results)} stocks successfully clustered")
    
    return clustering, results


def demo_cache_operations():
    """
    Demo various cache operations
    """
    print("=== CACHE OPERATIONS DEMO ===")
    
    # Initialize cache
    cache = StockDataCache('cache/demo_cache.csv')
    
    # Add some sample tickers
    sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    print(f"\n1. Building cache with {len(sample_tickers)} tickers...")
    cache.fetch_and_cache_data(sample_tickers, period='1y')
    
    # Show cache info
    print("\n2. Cache information:")
    info = cache.get_cache_info()
    for key, value in info.items():
        if key != 'ticker_list':  # Don't print full ticker list
            print(f"   {key}: {value}")
    
    # Add more tickers
    new_tickers = ['META', 'NVDA', 'NFLX']
    print(f"\n3. Adding {len(new_tickers)} more tickers...")
    cache.add_tickers(new_tickers, period='1y')
    
    # Get data for specific tickers
    print("\n4. Getting data for specific tickers...")
    sample_data = cache.get_data(tickers=['AAPL', 'GOOGL'], start_date='2023-01-01')
    print(f"   Retrieved data shape: {sample_data.shape}")
    
    # Get single ticker data
    print("\n5. Getting single ticker data...")
    aapl_data = cache.get_ticker_data('AAPL')
    print(f"   AAPL data points: {len(aapl_data)}")
    
    print("✅ Cache operations demo completed")


def run_clustering_experiment():
    """
    Run clustering with different parameters to compare results
    """
    print("=== CLUSTERING EXPERIMENT ===")
    
    # Initialize cache
    cache = StockDataCache('cache/stock_data_5y.csv')
    
    # Load some sample data if cache is empty
    if cache.get_cache_info()['status'] == 'empty':
        sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
                         'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SHOP',
                         'SPOT', 'SQ', 'ZOOM', 'DOCU', 'ROKU', 'TWLO', 'OKTA', 'SNOW']
        cache.fetch_and_cache_data(sample_tickers, period='2y')
    
    available_tickers = cache.get_available_tickers()
    
    # Experiment with different feature types
    feature_types = ['normalized_prices', 'returns', 'volatility']
    cluster_counts = [3, 5, 7]
    
    results_summary = []
    
    for feature_type in feature_types:
        for n_clusters in cluster_counts:
            print(f"\n--- Experiment: {feature_type} with {n_clusters} clusters ---")
            
            # Initialize clustering
            clustering = StockKShapeClustering(cache, n_clusters=n_clusters, random_state=42)
            
            # Load data
            sample_tickers = random.sample(available_tickers, min(50, len(available_tickers)))
            clustering.load_data(tickers=sample_tickers, start_date='2023-01-01')
            
            # Prepare features and cluster
            features_df = clustering.prepare_features(feature_type=feature_type)
            
            if not features_df.empty:
                cluster_labels = clustering.fit_clustering()
                
                if len(cluster_labels) > 0:
                    # Analyze performance
                    try:
                        performance = clustering.analyze_cluster_performance()
                        avg_sharpe = performance['Sharpe_Ratio'].mean()
                        avg_return = performance['Avg_Return'].mean()
                        avg_vol = performance['Avg_Volatility'].mean()
                        
                        results_summary.append({
                            'Feature_Type': feature_type,
                            'N_Clusters': n_clusters,
                            'Valid_Stocks': len(clustering.valid_tickers),
                            'Avg_Sharpe_Ratio': avg_sharpe,
                            'Avg_Return': avg_return,
                            'Avg_Volatility': avg_vol
                        })
                        
                        print(f"   ✅ Success: {len(clustering.valid_tickers)} stocks, Avg Sharpe: {avg_sharpe:.3f}")
                    except Exception as e:
                        print(f"   ⚠️  Performance analysis failed: {e}")
                else:
                    print(f"   ❌ Clustering failed")
            else:
                print(f"   ❌ Feature preparation failed")
    
    # Summary of experiments
    if results_summary:
        print("\n=== EXPERIMENT SUMMARY ===")
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.round(4))
        
        # Find best configuration
        best_config = summary_df.loc[summary_df['Avg_Sharpe_Ratio'].idxmax()]
        print(f"\nBest configuration:")
        print(f"   Feature Type: {best_config['Feature_Type']}")
        print(f"   Number of Clusters: {best_config['N_Clusters']}")
        print(f"   Average Sharpe Ratio: {best_config['Avg_Sharpe_Ratio']:.4f}")


if __name__ == "__main__":
    # Run main clustering pipeline
    clustering, results = main()
    
    # Uncomment to run additional demos
    # demo_cache_operations()
    # run_clustering_experiment()