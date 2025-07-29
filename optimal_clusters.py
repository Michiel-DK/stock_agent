import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy import stats
import random
from datetime import datetime

# Import our refactored classes
from k_shape_cache import StockDataCache
from k_shape_clustering import StockKShapeClustering


def find_optimal_clusters_for_outperformance(cache, tickers, start_date=None, 
                                           max_clusters=10, min_clusters=4, 
                                           feature_type='normalized_prices'):
    """
    Find optimal number of clusters focused on identifying outperforming groups
    
    Parameters:
    -----------
    cache : StockDataCache
        Cache instance with stock data
    tickers : list
        List of tickers to analyze
    start_date : str, optional
        Start date for analysis data
    max_clusters : int
        Maximum number of clusters to test
    min_clusters : int
        Minimum number of clusters to test
    feature_type : str
        Type of features to use for clustering
    """
    
    print("=== FINDING OPTIMAL CLUSTERS FOR OUTPERFORMANCE ===")
    
    # Load stock data
    stock_data = cache.get_data(tickers=tickers, start_date=start_date)
    
    if stock_data.empty:
        print("No stock data available for analysis")
        return None, None, None
    
    print(f"Analyzing {stock_data.shape[1]} stocks from {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate market benchmark (equal-weighted portfolio return)
    returns = stock_data.pct_change().dropna()
    market_return = returns.mean(axis=1).mean()  # Average daily return
    market_volatility = returns.mean(axis=1).std()
    market_sharpe = market_return / market_volatility if market_volatility != 0 else 0
    
    print(f"Market Benchmark - Return: {market_return:.4f} ({market_return*100:.2f}%), "
          f"Volatility: {market_volatility:.4f}, Sharpe: {market_sharpe:.4f}")
    
    results = []
    cluster_range = range(min_clusters, max_clusters + 1)
    
    for n_clusters in cluster_range:
        print(f"\nTesting {n_clusters} clusters...")
        
        try:
            # Initialize clustering for this test
            clustering = StockKShapeClustering(cache, n_clusters=n_clusters, random_state=42)
            
            # Load data and prepare features
            clustering.load_data(tickers=tickers, start_date=start_date)
            features_df = clustering.prepare_features(feature_type=feature_type)
            
            if features_df.empty:
                print(f"Skipping {n_clusters} clusters - no valid features")
                continue
            
            # Fit clustering
            labels = clustering.fit_clustering()
            
            if len(labels) == 0:
                print(f"Skipping {n_clusters} clusters - clustering failed")
                continue
            
            # Analyze performance
            performance = clustering.analyze_cluster_performance()
            
            if performance.empty:
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
                # Get returns for best cluster stocks
                best_cluster_data = stock_data[best_cluster_stocks]
                best_cluster_returns = best_cluster_data.pct_change().dropna().mean(axis=1)
                # T-test against market
                t_stat, p_value = stats.ttest_1samp(best_cluster_returns, market_return)
                statistical_significance = p_value < 0.05
            else:
                statistical_significance = False
                p_value = 1.0
            
            # 6. Traditional clustering metric (for reference)
            try:
                if hasattr(clustering, 'features') and clustering.features is not None:
                    reshaped_features = clustering.features.reshape(clustering.features.shape[0], -1)
                    silhouette = silhouette_score(reshaped_features, labels)
                else:
                    silhouette = 0
            except:
                silhouette = 0
            
            # === COMPOSITE OUTPERFORMANCE SCORE ===
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
                'valid_stocks': len(clustering.valid_tickers),
                'cluster_performance': performance
            }
            
            results.append(result)
            
            print(f"  Valid stocks: {len(clustering.valid_tickers)}")
            print(f"  Best cluster return: {best_cluster_return:.4f} ({best_cluster_return*100:.2f}%)")
            print(f"  Outperforming clusters: {len(outperforming_clusters)}/{len(performance)} ({pct_outperforming:.1f}%)")
            print(f"  Return spread: {return_spread:.4f} ({return_spread*100:.2f}%)")
            print(f"  Outperformance score: {outperformance_score:.2f}")
            
        except Exception as e:
            print(f"Error with {n_clusters} clusters: {e}")
            continue
    
    if not results:
        print("No valid clustering results obtained")
        return None, None, None
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find optimal based on different criteria
    best_outperformance = results_df.loc[results_df['outperformance_score'].idxmax()]
    best_spread = results_df.loc[results_df['return_spread'].idxmax()]
    
    print(f"\n=== OPTIMAL CLUSTER RECOMMENDATIONS ===")
    print(f"🎯 Best for Outperformance: {best_outperformance['n_clusters']} clusters")
    print(f"   - Outperformance Score: {best_outperformance['outperformance_score']:.2f}")
    print(f"   - Best Cluster Return: {best_outperformance['best_cluster_return']*100:.2f}%")
    print(f"   - {best_outperformance['outperforming_clusters_count']}/{best_outperformance['n_clusters']} clusters outperform market")
    print(f"   - Valid stocks: {best_outperformance['valid_stocks']}")
    
    print(f"\n📊 Best for Differentiation: {best_spread['n_clusters']} clusters")
    print(f"   - Return Spread: {best_spread['return_spread']*100:.2f}%")
    print(f"   - Valid stocks: {best_spread['valid_stocks']}")
    
    # Plot results
    plot_cluster_optimization_results(results_df, market_return)
    
    return int(best_outperformance['n_clusters']), results_df, stock_data


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
    
    # Plot 6: Valid Stocks Count
    axes[1, 2].plot(x, results_df['valid_stocks'], 'co-', linewidth=2)
    axes[1, 2].set_title('Number of Valid Stocks')
    axes[1, 2].set_xlabel('Number of Clusters')
    axes[1, 2].set_ylabel('Valid Stocks Count')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_optimal_clusters(clustering, results, performance, n_clusters):
    """
    Analyze the results from optimal clustering
    
    Parameters:
    -----------
    clustering : StockKShapeClustering
        Fitted clustering object
    results : pd.DataFrame
        Basic clustering results
    performance : pd.DataFrame
        Cluster performance metrics
    n_clusters : int
        Number of clusters used
    """
    try:
        print(f"\n=== DETAILED CLUSTER ANALYSIS ({n_clusters} CLUSTERS) ===")
        print("\nCluster Performance Summary:")
        display_cols = ['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio']
        print(performance[display_cols].round(4))
        
        # Show stocks in each cluster
        print(f"\nDetailed Cluster Analysis:")
        for _, cluster_info in performance.iterrows():
            cluster_num = cluster_info['Cluster']
            stocks = cluster_info['Stocks']
            print(f"\n--- Cluster {cluster_num} ({cluster_info['Count']} stocks) ---")
            print(f"Stocks: {', '.join(stocks)}")
            print(f"Average Return: {cluster_info['Avg_Return']:.4f} ({cluster_info['Avg_Return']*100:.2f}%)")
            print(f"Average Volatility: {cluster_info['Avg_Volatility']:.4f} ({cluster_info['Avg_Volatility']*100:.2f}%)")
            print(f"Sharpe Ratio: {cluster_info['Sharpe_Ratio']:.4f}")
            print(f"Max Drawdown: {cluster_info['Max_Drawdown']:.4f} ({cluster_info['Max_Drawdown']*100:.2f}%)")
            
        # Calculate and print overall averages
        print(f"\n--- Overall Portfolio Averages ---")
        print(f"Average Return: {performance['Avg_Return'].mean():.4f} ({performance['Avg_Return'].mean()*100:.2f}%)")
        print(f"Average Volatility: {performance['Avg_Volatility'].mean():.4f} ({performance['Avg_Volatility'].mean()*100:.2f}%)")
        print(f"Average Sharpe Ratio: {performance['Sharpe_Ratio'].mean():.4f}")
        print(f"Average Max Drawdown: {performance['Max_Drawdown'].mean():.4f} ({performance['Max_Drawdown'].mean()*100:.2f}%)")
        
    except Exception as e:
        print(f"Performance analysis failed: {e}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    try:
        clustering.plot_clusters(feature_type='normalized_prices')
        print("✅ Cluster visualization completed")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Save results
    try:
        timestamp = clustering.save_results(base_filename='optimal_clustering')
        print(f"✅ Results saved with timestamp: {timestamp}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print(f"\n🎉 OPTIMAL CLUSTERING ANALYSIS COMPLETED!")
    print(f"📊 {len(results)} stocks clustered into {n_clusters} optimal clusters")


def run_optimal_cluster_analysis(cache_file='stock_data_5y.csv', 
                                ticker_file='nasdaq_screener.csv',
                                sample_size=100, 
                                max_clusters=15, 
                                min_clusters=4,
                                period='5y',
                                start_date='2022-01-01',
                                feature_type='normalized_prices'):
    """
    Complete workflow to find optimal clusters and run analysis
    
    Parameters:
    -----------
    cache_file : str
        Cache file name
    ticker_file : str  
        File containing ticker symbols
    sample_size : int
        Number of stocks to analyze
    max_clusters : int
        Maximum clusters to test
    min_clusters : int
        Minimum clusters to test
    period : str
        Data period to cache
    start_date : str
        Start date for analysis
    feature_type : str
        Feature type for clustering
    """
    print("=== RUNNING OPTIMAL CLUSTER ANALYSIS ===")
    
    # Step 1: Initialize cache
    cache = StockDataCache(cache_file)
    
    # Check if cache exists, build if needed
    cache_info = cache.get_cache_info()
    if cache_info['status'] == 'empty':
        print(f"Building cache from {ticker_file}...")
        try:
            # Load tickers from file
            tickers_df = pd.read_csv(ticker_file)
            # Try common column names
            ticker_col = None
            for col in ['Symbol', 'Ticker', 'symbol', 'ticker']:
                if col in tickers_df.columns:
                    ticker_col = col
                    break
            
            if ticker_col is None:
                print("Could not find ticker column. Using first column.")
                ticker_col = tickers_df.columns[0]
                
            all_tickers = tickers_df[ticker_col].tolist()
            print(f"Found {len(all_tickers)} tickers in {ticker_file}")
            
            # Build cache
            cache.fetch_and_cache_data(all_tickers, period=period)
            
        except FileNotFoundError:
            print(f"Ticker file {ticker_file} not found. Using sample tickers...")
            sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
                            'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SHOP']
            cache.fetch_and_cache_data(sample_tickers, period=period)
    else:
        print(f"Using existing cache: {cache_info['tickers']} tickers available")
    
    # Step 2: Sample tickers for analysis
    available_tickers = cache.get_available_tickers()
    if len(available_tickers) == 0:
        print("No tickers available in cache")
        return None, None, None
    
    sample_tickers = random.sample(available_tickers, min(sample_size, len(available_tickers)))
    print(f"Analyzing {len(sample_tickers)} randomly sampled tickers")
    
    # Step 3: Find optimal clusters
    optimal_k, results_df, stock_data = find_optimal_clusters_for_outperformance(
        cache=cache,
        tickers=sample_tickers,
        start_date=start_date,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        feature_type=feature_type
    )
    
    if optimal_k is None:
        print("Could not determine optimal clusters")
        return None, None, None
    
    # Step 4: Run final analysis with optimal clusters
    print(f"\n=== RUNNING FINAL ANALYSIS WITH {optimal_k} CLUSTERS ===")
    
    # Initialize clustering with optimal parameters
    clustering = StockKShapeClustering(cache, n_clusters=optimal_k, random_state=42)
    
    # Load data and run clustering
    clustering.load_data(tickers=sample_tickers, start_date=start_date)
    features_df = clustering.prepare_features(feature_type=feature_type)
    
    if features_df.empty:
        print("No valid features for final analysis")
        return optimal_k, results_df, None
    
    labels = clustering.fit_clustering()
    results = clustering.get_cluster_results()
    performance = clustering.analyze_cluster_performance()
    
    # Step 5: Detailed analysis
    analyze_optimal_clusters(clustering, results, performance, optimal_k)
    
    return optimal_k, results_df, (clustering, results, performance)


def run_quick_optimization_test():
    """
    Quick test with small sample size
    """
    print("=== QUICK OPTIMIZATION TEST ===")
    
    # Use small sample for quick testing
    optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
        cache_file='test_cache.csv',
        sample_size=50,
        max_clusters=8,
        min_clusters=3,
        period='2y',
        start_date='2023-01-01'
    )
    
    if final_results is not None:
        clustering, results, performance = final_results
        print(f"\n✅ Test completed with {optimal_k} optimal clusters")
        print(f"📊 Analyzed {len(results)} stocks")
    else:
        print("❌ Test failed")
    
    return optimal_k, optimization_results, final_results


# Example usage
if __name__ == "__main__":
    # For full analysis (comment/uncomment as needed)
    optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
        cache_file='nasdaq_cache_5y.csv',
        ticker_file='nasdaq_screener.csv',
        sample_size=3200,  # Adjust based on your needs
        max_clusters=15,
        min_clusters=4,
        period='5y',
        start_date='2022-10-12',
        feature_type='normalized_prices'
    )
    
    # For quick testing
    # optimal_k, optimization_results, final_results = run_quick_optimization_test()
    
    if final_results is not None:
        clustering, results, performance = final_results
        print(f"\n🎯 FINAL OPTIMAL ANALYSIS COMPLETE!")
        print(f"Optimal clusters: {optimal_k}")
        print(f"Stocks analyzed: {len(results)}")
        
        # Show optimization results summary
        if optimization_results is not None:
            print(f"\nOptimization tested {len(optimization_results)} different cluster counts")
            print("Top 3 configurations by outperformance score:")
            top_3 = optimization_results.nlargest(3, 'outperformance_score')
            for _, row in top_3.iterrows():
                print(f"  {row['n_clusters']} clusters: score {row['outperformance_score']:.2f}, "
                      f"best return {row['best_cluster_return']*100:.2f}%")
    else:
        print("❌ Analysis failed")