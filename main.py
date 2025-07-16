import pandas as pd
import random
import traceback
import ipdb

from utils import save_cluster_results_to_csv, save_detailed_cluster_results_to_csv
from StockShapeClustering import StockKShapeClustering  # Assuming this is your clustering

if __name__ == "__main__":
    try:
        # Load tickers and sample whatever size you want
        tickers = pd.read_csv('ticker_symbols_only.txt').Ticker.to_list()
        tickers = random.sample(tickers, 100)  # Back to 100 - no problem now
        
        # Initialize clustering
        clustering = StockKShapeClustering(n_clusters=6, random_state=42)
        
        # Fetch data - this now filters out bad tickers automatically
        stock_data = clustering.fetch_stock_data(tickers, period='1y')
        
        # Prepare features - this now only gets good tickers
        features = clustering.prepare_features(feature_type='normalized_prices')
        
        # Fit clustering
        labels = clustering.fit_clustering()
        
        # Get results
        results = clustering.get_cluster_results()
        print("\nClustering Results:")
        print(results)
        
        # Visualize results
        clustering.plot_clusters(feature_type='normalized_prices')
        
        # Analyze performance
        performance = clustering.analyze_cluster_performance()
        print("\nCluster Performance Analysis:")
        print(performance[['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio']])
        
        # Save results
        save_cluster_results_to_csv(clustering, filename='stock_clusters.csv')
        save_detailed_cluster_results_to_csv(clustering, filename='detailed_stock_clusters.csv')
        print("Results saved to CSV files.")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nDropping into ipdb debugger...")
        ipdb.post_mortem()