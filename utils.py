import pandas as pd
from datetime import datetime

def save_cluster_results_to_csv(clustering_model, filename=None):
    """
    Save ticker clusters to CSV file
    
    Parameters:
    -----------
    clustering_model : StockKShapeClustering instance
        Fitted clustering model
    filename : str, optional
        Output filename. If None, uses timestamp
    """
    if clustering_model.cluster_labels is None:
        raise ValueError("Model not fitted. Run fit_clustering first.")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Ticker': clustering_model.stock_data.columns,
        'Cluster': clustering_model.cluster_labels
    })
    
    # Sort by cluster for better organization
    results_df = results_df.sort_values('Cluster')
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_clusters_{timestamp}.csv"
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"Cluster results saved to: {filename}")
    
    return results_df

# Alternative: Save with additional cluster statistics
def save_detailed_cluster_results_to_csv(clustering_model, filename=None):
    """
    Save detailed ticker clusters with performance stats to CSV
    """
    if clustering_model.cluster_labels is None:
        raise ValueError("Model not fitted. Run fit_clustering first.")
    
    # Get basic results
    results_df = pd.DataFrame({
        'Ticker': clustering_model.stock_data.columns,
        'Cluster': clustering_model.cluster_labels
    })
    
    # Calculate performance stats for each stock
    returns = clustering_model.stock_data.pct_change().dropna()
    
    performance_stats = []
    for ticker in clustering_model.stock_data.columns:
        stock_returns = returns[ticker]
        stats = {
            'Ticker': ticker,
            'Avg_Return': stock_returns.mean(),
            'Volatility': stock_returns.std(),
            'Sharpe_Ratio': stock_returns.mean() / stock_returns.std() if stock_returns.std() != 0 else 0,
            'Total_Return': ((1 + stock_returns).cumprod().iloc[-1] - 1)
        }
        performance_stats.append(stats)
    
    performance_df = pd.DataFrame(performance_stats)
    
    # Merge with cluster results
    final_df = results_df.merge(performance_df, on='Ticker')
    final_df = final_df.sort_values('Cluster')
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_stock_clusters_{timestamp}.csv"
    
    # Save to CSV
    final_df.to_csv(filename, index=False)
    print(f"Detailed cluster results saved to: {filename}")
    
    return final_df

# Quick usage example:
# results = save_cluster_results_to_csv(clustering)
# detailed_results = save_detailed_cluster_results_to_csv(clustering, "my_clusters.csv")