import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy import stats
from k_shape_cache import StockKShapeClusteringWithCache, build_stock_cache

import random

from k_shape_cache import analyze_cluster



def find_optimal_clusters_for_outperformance(clustering_class, stock_data, max_clusters=10, min_clusters=4):
    """
    Find optimal number of clusters focused on identifying outperforming groups
    
    Parameters:
    -----------
    clustering_class : StockKShapeClusteringWithCache instance
        Initialized clustering object with data already loaded
    stock_data : pd.DataFrame
        Stock price data
    max_clusters : int
        Maximum number of clusters to test
    min_clusters : int
        Minimum number of clusters to test
    """
    
    print("=== FINDING OPTIMAL CLUSTERS FOR OUTPERFORMANCE ===")
    
    results = []
    cluster_range = range(min_clusters, max_clusters + 1)
    
    # Calculate market benchmark (equal-weighted portfolio return)
    returns = stock_data.pct_change().dropna()
    market_return = returns.mean(axis=1).mean()  # Average daily return
    market_volatility = returns.mean(axis=1).std()
    market_sharpe = market_return / market_volatility if market_volatility != 0 else 0
    
    print(f"Market Benchmark - Return: {market_return:.4f} ({market_return*100:.2f}%), "
          f"Volatility: {market_volatility:.4f}, Sharpe: {market_sharpe:.4f}")
    
    for n_clusters in cluster_range:
        print(f"\nTesting {n_clusters} clusters...")
        
        # Set number of clusters and prepare features
        clustering_class.n_clusters = n_clusters
        features = clustering_class.prepare_features(feature_type='normalized_prices')
        
        if features.empty:
            print(f"Skipping {n_clusters} clusters - no valid features")
            continue
        
        # Fit clustering
        try:
            labels = clustering_class.fit_clustering()
            performance = clustering_class.analyze_cluster_performance()
            
            if performance is None or performance.empty:
                print(f"Skipping {n_clusters} clusters - no performance data")
                continue
            
            # === OUTPERFORMANCE-FOCUSED METRICS ===
            
            # 1. Best cluster performance
            best_cluster_return = performance['Avg_Return'].max()
            best_cluster_sharpe = performance['Sharpe_Ratio'].max()
            worst_cluster_return = performance['Avg_Return'].min()
            
            # 2. Outperforming clusters (above market)
            outperforming_clusters = performance[performance['Avg_Return'] > market_return]
            pct_outperforming = len(outperforming_clusters) / len(performance) * 100
            
            # 3. Performance spread (how different are the clusters?)
            return_spread = performance['Avg_Return'].max() - performance['Avg_Return'].min()
            sharpe_spread = performance['Sharpe_Ratio'].max() - performance['Sharpe_Ratio'].min()
            
            # 4. Cluster size balance (avoid tiny clusters)
            cluster_sizes = performance['Count'].values
            size_balance = np.std(cluster_sizes) / np.mean(cluster_sizes)  # Lower is more balanced
            min_cluster_size = cluster_sizes.min()
            
            # 5. Statistical significance of best cluster
            best_cluster_idx = performance['Avg_Return'].idxmax()
            best_cluster_stocks = performance.loc[best_cluster_idx, 'Stocks']
            
            if len(best_cluster_stocks) > 1:
                best_cluster_returns = returns[best_cluster_stocks].mean(axis=1)
                # T-test against market
                t_stat, p_value = stats.ttest_1samp(best_cluster_returns, market_return)
                statistical_significance = p_value < 0.05
            else:
                statistical_significance = False
                p_value = 1.0
            
            # 6. Traditional clustering metric (for reference)
            if hasattr(clustering_class, 'features') and clustering_class.features is not None:
                try:
                    reshaped_features = clustering_class.features.reshape(clustering_class.features.shape[0], -1)
                    silhouette = silhouette_score(reshaped_features, labels)
                except:
                    silhouette = 0
            else:
                silhouette = 0
            
            # === COMPOSITE OUTPERFORMANCE SCORE ===
            # Higher score = better for finding outperformers
            
            outperformance_score = (
                # Reward high best cluster performance
                (best_cluster_return - market_return) * 100 +  # Excess return weight
                
                # Reward high percentage of outperforming clusters
                (pct_outperforming / 100) * 50 +
                
                # Reward performance spread (differentiation)
                return_spread * 200 +
                
                # Reward balanced cluster sizes (penalty for imbalance)
                max(0, (10 - size_balance * 50)) +
                
                # Reward statistical significance
                (25 if statistical_significance else 0) +
                
                # Penalty for too small clusters
                (-20 if min_cluster_size < 3 else 0)
            )
            
            result = {
                'n_clusters': n_clusters,
                'best_cluster_return': best_cluster_return,
                'best_cluster_sharpe': best_cluster_sharpe,
                'worst_cluster_return': worst_cluster_return,
                'return_spread': return_spread,
                'sharpe_spread': sharpe_spread,
                'pct_outperforming': pct_outperforming,
                'outperforming_clusters_count': len(outperforming_clusters),
                'size_balance': size_balance,
                'min_cluster_size': min_cluster_size,
                'statistical_significance': statistical_significance,
                'p_value': p_value,
                'silhouette_score': silhouette,
                'outperformance_score': outperformance_score,
                'cluster_performance': performance
            }
            
            results.append(result)
            
            print(f"  Best cluster return: {best_cluster_return:.4f} ({best_cluster_return*100:.2f}%)")
            print(f"  Outperforming clusters: {len(outperforming_clusters)}/{len(performance)} ({pct_outperforming:.1f}%)")
            print(f"  Return spread: {return_spread:.4f} ({return_spread*100:.2f}%)")
            print(f"  Outperformance score: {outperformance_score:.2f}")
            
        except Exception as e:
            print(f"Error with {n_clusters} clusters: {e}")
            continue
    
    if not results:
        print("No valid clustering results obtained")
        return None, None
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find optimal based on different criteria
    best_outperformance = results_df.loc[results_df['outperformance_score'].idxmax()]
    best_spread = results_df.loc[results_df['return_spread'].idxmax()]
    best_silhouette = results_df.loc[results_df['silhouette_score'].idxmax()]
    
    print(f"\n=== OPTIMAL CLUSTER RECOMMENDATIONS ===")
    print(f"🎯 Best for Outperformance: {best_outperformance['n_clusters']} clusters")
    print(f"   - Outperformance Score: {best_outperformance['outperformance_score']:.2f}")
    print(f"   - Best Cluster Return: {best_outperformance['best_cluster_return']*100:.2f}%")
    print(f"   - {best_outperformance['outperforming_clusters_count']}/{best_outperformance['n_clusters']} clusters outperform market")
    
    print(f"\n📊 Best for Differentiation: {best_spread['n_clusters']} clusters")
    print(f"   - Return Spread: {best_spread['return_spread']*100:.2f}%")
    
    print(f"\n🔗 Best Traditional Clustering: {best_silhouette['n_clusters']} clusters")
    print(f"   - Silhouette Score: {best_silhouette['silhouette_score']:.3f}")
    
    # Plot results
    plot_cluster_optimization_results(results_df, market_return)
    
    return int(best_outperformance['n_clusters']), results_df

def plot_cluster_optimization_results(results_df, market_return):
    """
    Plot cluster optimization results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cluster Optimization for Outperformance Analysis', fontsize=16)
    
    x = results_df['n_clusters']
    
    # Plot 1: Outperformance Score
    axes[0, 0].plot(x, results_df['outperformance_score'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Outperformance Score')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight best
    best_idx = results_df['outperformance_score'].idxmax()
    best_x = results_df.loc[best_idx, 'n_clusters']
    best_y = results_df.loc[best_idx, 'outperformance_score']
    axes[0, 0].plot(best_x, best_y, 'ro', markersize=12, label=f'Optimal: {best_x}')
    axes[0, 0].legend()
    
    # Plot 2: Best Cluster Performance vs Market
    axes[0, 1].plot(x, results_df['best_cluster_return'] * 100, 'go-', label='Best Cluster Return')
    axes[0, 1].axhline(y=market_return * 100, color='r', linestyle='--', label='Market Return')
    axes[0, 1].set_title('Best Cluster vs Market Return')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Daily Return (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Percentage of Outperforming Clusters
    axes[0, 2].bar(x, results_df['pct_outperforming'], alpha=0.7, color='purple')
    axes[0, 2].set_title('Percentage of Outperforming Clusters')
    axes[0, 2].set_xlabel('Number of Clusters')
    axes[0, 2].set_ylabel('Percentage (%)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Return Spread (Differentiation)
    axes[1, 0].plot(x, results_df['return_spread'] * 100, 'mo-', linewidth=2)
    axes[1, 0].set_title('Return Spread Between Best/Worst Clusters')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Return Spread (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Statistical Significance
    significance_indicator = results_df['statistical_significance'].astype(int)
    axes[1, 1].bar(x, significance_indicator, alpha=0.7, color='orange')
    axes[1, 1].set_title('Statistical Significance of Best Cluster')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Significant (1) or Not (0)')
    axes[1, 1].set_ylim(0, 1.2)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Silhouette Score (Traditional)
    axes[1, 2].plot(x, results_df['silhouette_score'], 'co-', linewidth=2)
    axes[1, 2].set_title('Silhouette Score (Traditional)')
    axes[1, 2].set_xlabel('Number of Clusters')
    axes[1, 2].set_ylabel('Silhouette Score')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_optimal_cluster_analysis(sample_size=100, max_clusters=10, period='1y'):
    """
    Complete workflow to find optimal clusters and run analysis
    """
    print("=== RUNNING OPTIMAL CLUSTER ANALYSIS ===")
    
    # Sample and load data
    
    # Load tickers and sample
    #tickers_df = pd.read_csv('ticker_symbols_only.txt')
    #all_tickers = tickers_df.Ticker.to_list()

    cache = build_stock_cache('nasdaq_screener.csv', period=period)
    clustering = StockKShapeClusteringWithCache(n_clusters=3, random_state=42)  # Temporary

    available_tickers = clustering.cache.get_available_tickers()
    sample_tickers = random.sample(available_tickers, min(sample_size, len(available_tickers)))
    
    stock_data = clustering.fetch_stock_data_from_cache(sample_tickers)
    
    # Initialize clustering with cache
    
    # Add to cache and load data
    #clustering.cache.add_tickers_to_cache(sample_tickers, period=period)
    #stock_data = clustering.fetch_stock_data_from_cache(sample_tickers)
    
    if stock_data.empty:
        print("No data available for analysis")
        return None, None, None
    
    # Find optimal clusters
    optimal_k, results_df = find_optimal_clusters_for_outperformance(clustering, stock_data, max_clusters=max_clusters)
    
    if optimal_k is None:
        print("Could not determine optimal clusters")
        return None, None, None
    
    # Run final analysis with optimal clusters
    print(f"\n=== RUNNING FINAL ANALYSIS WITH {optimal_k} CLUSTERS ===")
    clustering.n_clusters = optimal_k
    
    features = clustering.prepare_features(feature_type='normalized_prices')
    labels = clustering.fit_clustering()
    results = clustering.get_cluster_results()
    performance = clustering.analyze_cluster_performance()
    
    # Show final results
    print(f"\n🎯 FINAL RESULTS WITH {optimal_k} CLUSTERS:")
    print(performance[['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio']])
    
    return optimal_k, results_df, (clustering, results, performance)

# Example usage
if __name__ == "__main__":
    optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
        sample_size=3640, 
        max_clusters=25, 
        period='5y'
    )
    
    clustering, results, performance = final_results
    
    analyze_cluster(clustering, results, performance, n_clusters=optimal_k)