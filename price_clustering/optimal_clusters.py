import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy import stats
import random
from datetime import datetime

# Import our refactored classes
from price_clustering.k_shape_cache import StockDataCache
from price_clustering.k_shape_clustering import StockKShapeClustering


def find_optimal_clusters_for_outperformance(cache, tickers, start_date=None, 
                                           max_clusters=10, min_clusters=4, 
                                           feature_type='normalized_prices',
                                           method="kshape", metric="dtw"):
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
    method : str
        Clustering method - "kshape" or "timeserieskmeans"
    metric : str
        Distance metric for timeserieskmeans - "dtw", "softdtw", or "euclidean"
    """
    
    method_display = f"{method}" + (f"_{metric}" if method == "timeserieskmeans" else "")
    print(f"=== FINDING OPTIMAL CLUSTERS FOR OUTPERFORMANCE ({method_display.upper()}) ===")
    
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
            
            # Fit clustering with specified method
            labels = clustering.fit_clustering(method=method, metric=metric)
            
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
                'method': method,
                'metric': metric if method == "timeserieskmeans" else "N/A",
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
    print(f"   - Method: {best_outperformance['method']}" + 
          (f" ({best_outperformance['metric']})" if best_outperformance['method'] == "timeserieskmeans" else ""))
    print(f"   - Outperformance Score: {best_outperformance['outperformance_score']:.2f}")
    print(f"   - Best Cluster Return: {best_outperformance['best_cluster_return']*100:.2f}%")
    print(f"   - {best_outperformance['outperforming_clusters_count']}/{best_outperformance['n_clusters']} clusters outperform market")
    print(f"   - Valid stocks: {best_outperformance['valid_stocks']}")
    
    print(f"\n📊 Best for Differentiation: {best_spread['n_clusters']} clusters")
    print(f"   - Return Spread: {best_spread['return_spread']*100:.2f}%")
    print(f"   - Valid stocks: {best_spread['valid_stocks']}")
    
    # Plot results
    plot_cluster_optimization_results(results_df, market_return, method_display)
    
    return int(best_outperformance['n_clusters']), results_df, stock_data


def plot_cluster_optimization_results(results_df, market_return, method_display):
    """
    Plot cluster optimization results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Cluster Optimization for Outperformance Analysis ({method_display.upper()})', fontsize=16)
    
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


def analyze_optimal_clusters(clustering, results, performance, n_clusters, method, metric):
    """
    Analyze the results from optimal clustering - showing only top 3 performing clusters
    
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
    method : str
        Clustering method used
    metric : str
        Distance metric used (if applicable)
    """
    try:
        method_display = f"{method}" + (f"_{metric}" if method == "timeserieskmeans" else "")
        print(f"\n=== DETAILED CLUSTER ANALYSIS ({n_clusters} CLUSTERS - {method_display.upper()}) ===")
        
        # Show only summary table first
        print("\nAll Clusters Performance Summary:")
        display_cols = ['Cluster', 'Count', 'Avg_Return', 'Avg_Volatility', 'Sharpe_Ratio']
        performance_display = performance[display_cols].copy()
        performance_display['Avg_Return'] = performance_display['Avg_Return'] * 100  # Convert to percentage
        performance_display['Avg_Volatility'] = performance_display['Avg_Volatility'] * 100
        performance_display = performance_display.round(3)
        print(performance_display.to_string(index=False))
        
        # Calculate individual stock returns for detailed analysis
        stock_returns = {}
        if clustering.stock_data is not None and not clustering.stock_data.empty:
            returns_data = clustering.stock_data.pct_change().dropna()
            for ticker in clustering.valid_tickers:
                if ticker in returns_data.columns:
                    avg_return = returns_data[ticker].mean()
                    volatility = returns_data[ticker].std()
                    sharpe = avg_return / volatility if volatility > 0 else 0
                    
                    # Calculate total return over the period
                    prices = clustering.stock_data[ticker].dropna()
                    if len(prices) > 1:
                        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                    else:
                        total_return = 0
                    
                    stock_returns[ticker] = {
                        'avg_daily_return': avg_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        'total_return': total_return
                    }
        
        # Get top 3 performing clusters by average return
        top_3_clusters = performance.nlargest(3, 'Avg_Return')
        
        print(f"\n=== TOP 3 PERFORMING CLUSTERS (Detailed Analysis) ===")
        
        for rank, (_, cluster_info) in enumerate(top_3_clusters.iterrows(), 1):
            cluster_num = cluster_info['Cluster']
            stocks = cluster_info['Stocks']
            
            print(f"\n🏆 RANK {rank} - Cluster {cluster_num} ({cluster_info['Count']} stocks)")
            print(f"{'='*60}")
            print(f"Cluster Metrics:")
            print(f"  Average Return: {cluster_info['Avg_Return']:.4f} ({cluster_info['Avg_Return']*100:.2f}%)")
            print(f"  Average Volatility: {cluster_info['Avg_Volatility']:.4f} ({cluster_info['Avg_Volatility']*100:.2f}%)")
            print(f"  Sharpe Ratio: {cluster_info['Sharpe_Ratio']:.4f}")
            print(f"  Max Drawdown: {cluster_info['Max_Drawdown']:.4f} ({cluster_info['Max_Drawdown']*100:.2f}%)")
            
            print(f"\nTop Individual Stock Performers in this Cluster:")
            print(f"{'Ticker':<8} {'Total Return':<12} {'Avg Daily':<10} {'Volatility':<10} {'Sharpe':<8}")
            print(f"{'='*8} {'='*12} {'='*10} {'='*10} {'='*8}")
            
            # Sort stocks by total return (descending) and show top 10
            stocks_with_returns = []
            for ticker in stocks:
                if ticker in stock_returns:
                    stocks_with_returns.append((ticker, stock_returns[ticker]))
                else:
                    stocks_with_returns.append((ticker, {
                        'avg_daily_return': 0,
                        'total_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0
                    }))
            
            # Sort by total return and show top performers
            stocks_with_returns.sort(key=lambda x: x[1]['total_return'], reverse=True)
            top_stocks_to_show = min(10, len(stocks_with_returns))  # Show top 10 or all if less
            
            for i, (ticker, metrics) in enumerate(stocks_with_returns[:top_stocks_to_show]):
                print(f"{ticker:<8} {metrics['total_return']*100:>11.1f}% "
                      f"{metrics['avg_daily_return']*100:>9.2f}% "
                      f"{metrics['volatility']*100:>9.2f}% "
                      f"{metrics['sharpe_ratio']:>7.3f}")
            
            if len(stocks_with_returns) > top_stocks_to_show:
                print(f"   ... and {len(stocks_with_returns) - top_stocks_to_show} more stocks")
                
        # Calculate and print overall averages
        print(f"\n=== OVERALL PORTFOLIO SUMMARY ===")
        print(f"Average Return: {performance['Avg_Return'].mean():.4f} ({performance['Avg_Return'].mean()*100:.2f}%)")
        print(f"Average Volatility: {performance['Avg_Volatility'].mean():.4f} ({performance['Avg_Volatility'].mean()*100:.2f}%)")
        print(f"Average Sharpe Ratio: {performance['Sharpe_Ratio'].mean():.4f}")
        print(f"Average Max Drawdown: {performance['Max_Drawdown'].mean():.4f} ({performance['Max_Drawdown'].mean()*100:.2f}%)")
        
        # Show top individual performers across all clusters
        if stock_returns:
            print(f"\n=== TOP 10 INDIVIDUAL STOCK PERFORMERS (All Clusters) ===")
            all_stocks_sorted = sorted(stock_returns.items(), key=lambda x: x[1]['total_return'], reverse=True)
            for i, (ticker, metrics) in enumerate(all_stocks_sorted[:10]):
                cluster_num = results[results['Ticker'] == ticker]['Cluster'].iloc[0] if ticker in results['Ticker'].values else 'N/A'
                print(f"{i+1:2d}. {ticker:<6} (Cluster {cluster_num}): {metrics['total_return']*100:>6.1f}% total, "
                      f"Sharpe: {metrics['sharpe_ratio']:>6.3f}")
        
    except Exception as e:
        print(f"Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Enhanced results saving with method in filename
    try:
        method_str = f"{method}" + (f"_{metric}" if method == "timeserieskmeans" else "")
        timestamp = _save_enhanced_results(clustering, results, performance, stock_returns, method_str)
        print(f"✅ Enhanced results saved with timestamp: {timestamp}")
    except Exception as e:
        print(f"⚠️  Could not save enhanced results: {e}")
        # Fallback to basic save
        try:
            timestamp = clustering.save_results(base_filename=f'optimal_clustering_{method_str}')
            print(f"✅ Basic results saved with timestamp: {timestamp}")
        except Exception as e2:
            print(f"⚠️  Could not save results: {e2}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    try:
        clustering.plot_clusters(feature_type='normalized_prices')
        print("✅ Cluster visualization completed")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    print(f"\n🎉 OPTIMAL CLUSTERING ANALYSIS COMPLETED!")
    print(f"📊 {len(results)} stocks clustered into {n_clusters} optimal clusters using {method_str.upper()}")


def _save_enhanced_results(clustering, results, performance, stock_returns, method_str):
    """
    Save enhanced results with individual stock performance and method in filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create enhanced results DataFrame
    enhanced_results = results.copy()
    
    # Add method information
    enhanced_results['Method'] = method_str
    
    # Add individual stock metrics
    for ticker in enhanced_results['Ticker']:
        if ticker in stock_returns:
            metrics = stock_returns[ticker]
            enhanced_results.loc[enhanced_results['Ticker'] == ticker, 'Individual_Avg_Daily_Return'] = metrics['avg_daily_return']
            enhanced_results.loc[enhanced_results['Ticker'] == ticker, 'Individual_Total_Return'] = metrics['total_return']
            enhanced_results.loc[enhanced_results['Ticker'] == ticker, 'Individual_Volatility'] = metrics['volatility']
            enhanced_results.loc[enhanced_results['Ticker'] == ticker, 'Individual_Sharpe_Ratio'] = metrics['sharpe_ratio']
    
    # Add cluster-level metrics
    cluster_metrics = {}
    for _, cluster_info in performance.iterrows():
        cluster_num = cluster_info['Cluster']
        cluster_metrics[cluster_num] = {
            'Cluster_Avg_Return': cluster_info['Avg_Return'],
            'Cluster_Avg_Volatility': cluster_info['Avg_Volatility'],
            'Cluster_Sharpe_Ratio': cluster_info['Sharpe_Ratio'],
            'Cluster_Max_Drawdown': cluster_info['Max_Drawdown'],
            'Cluster_Count': cluster_info['Count']
        }
    
    # Add cluster metrics to each stock
    for cluster_num, metrics in cluster_metrics.items():
        mask = enhanced_results['Cluster'] == cluster_num
        for metric_name, metric_value in metrics.items():
            enhanced_results.loc[mask, metric_name] = metric_value
    
    # Save enhanced results
    enhanced_file = f'output_data/optimal_clustering_{method_str}_enhanced_{timestamp}.csv'
    enhanced_results.to_csv(enhanced_file, index=False)
    print(f"✅ Enhanced results saved to {enhanced_file}")
    
    # Save cluster performance summary
    perf_summary = performance.drop('Stocks', axis=1, errors='ignore').copy()
    perf_summary['Method'] = method_str
    perf_file = f'output_data/optimal_clustering_{method_str}_performance_{timestamp}.csv'
    perf_summary.to_csv(perf_file, index=False)
    print(f"✅ Performance summary saved to {perf_file}")
    
    # Save individual stock performance ranking
    if stock_returns:
        stock_ranking = []
        for ticker, metrics in stock_returns.items():
            cluster_num = results[results['Ticker'] == ticker]['Cluster'].iloc[0] if ticker in results['Ticker'].values else None
            stock_ranking.append({
                'Ticker': ticker,
                'Cluster': cluster_num,
                'Method': method_str,
                'Avg_Daily_Return': metrics['avg_daily_return'],
                'Total_Return': metrics['total_return'],
                'Volatility': metrics['volatility'],
                'Sharpe_Ratio': metrics['sharpe_ratio']
            })
        
        ranking_df = pd.DataFrame(stock_ranking)
        ranking_df = ranking_df.sort_values('Total_Return', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        ranking_file = f'output_data/optimal_clustering_{method_str}_stock_ranking_{timestamp}.csv'
        ranking_df.to_csv(ranking_file, index=False)
        print(f"✅ Stock performance ranking saved to {ranking_file}")
    
    return timestamp


def run_optimal_cluster_analysis(cache_file='stock_data_5y.csv', 
                                ticker_file='nasdaq_screener.csv',
                                sample_size=100, 
                                max_clusters=15, 
                                min_clusters=4,
                                period='5y',
                                start_date='2022-01-01',
                                feature_type='normalized_prices',
                                method="kshape",
                                metric="dtw"):
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
    method : str
        Clustering method - "kshape" or "timeserieskmeans"
    metric : str
        Distance metric for timeserieskmeans - "dtw", "softdtw", or "euclidean"
    """
    method_display = f"{method}" + (f"_{metric}" if method == "timeserieskmeans" else "")
    print(f"=== RUNNING OPTIMAL CLUSTER ANALYSIS ({method_display.upper()}) ===")
    
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
        feature_type=feature_type,
        method=method,
        metric=metric
    )
    
    if optimal_k is None:
        print("Could not determine optimal clusters")
        return None, None, None
    
    # Step 4: Run final analysis with optimal clusters
    print(f"\n=== RUNNING FINAL ANALYSIS WITH {optimal_k} CLUSTERS ({method_display.upper()}) ===")
    
    # Initialize clustering with optimal parameters
    clustering = StockKShapeClustering(cache, n_clusters=optimal_k, random_state=42)
    
    # Load data and run clustering
    clustering.load_data(tickers=sample_tickers, start_date=start_date)
    features_df = clustering.prepare_features(feature_type=feature_type)
    
    if features_df.empty:
        print("No valid features for final analysis")
        return optimal_k, results_df, None
    
    labels = clustering.fit_clustering(method=method, metric=metric)
    results = clustering.get_cluster_results()
    performance = clustering.analyze_cluster_performance()
    
    # Step 5: Detailed analysis
    analyze_optimal_clusters(clustering, results, performance, optimal_k, method, metric)
    
    return optimal_k, results_df, (clustering, results, performance)


def run_quick_optimization_test(method="kshape", metric="dtw"):
    """
    Quick test with small sample size
    """
    method_display = f"{method}" + (f"_{metric}" if method == "timeserieskmeans" else "")
    print(f"=== QUICK OPTIMIZATION TEST ({method_display.upper()}) ===")
    
    # Use small sample for quick testing
    optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
        cache_file='test_cache.csv',
        sample_size=50,
        max_clusters=8,
        min_clusters=3,
        period='2y',
        start_date='2023-01-01',
        method=method,
        metric=metric
    )
    
    if final_results is not None:
        clustering, results, performance = final_results
        print(f"\n✅ Test completed with {optimal_k} optimal clusters using {method_display.upper()}")
        print(f"📊 Analyzed {len(results)} stocks")
    else:
        print("❌ Test failed")
    
    return optimal_k, optimization_results, final_results

# Example usage
if __name__ == "__main__":
    # For full analysis with different methods (choose one)
    
    # K-Shape analysis
    # optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
    #     cache_file='cache/nasdaq_cache_5y.csv',
    #     ticker_file='tickers/nasdaq_screener.csv',
    #     sample_size=200,  # Adjust based on your needs
    #     max_clusters=16,
    #     min_clusters=4,
    #     period='5y',
    #     start_date='2022-10-12',
    #     feature_type='normalized_prices',
    #     method="kshape"
    # )
    
    # DTW-based TimeSeriesKMeans analysis (recommended for price patterns)
    # optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
    #     cache_file='cache/nasdaq_cache_5y.csv',
    #     ticker_file='tickers/nasdaq_screener.csv',
    #     sample_size=3200,
    #     max_clusters=16,
    #     min_clusters=4,
    #     period='5y',
    #     start_date='2022-10-12',
    #     feature_type='normalized_prices',
    #     method="timeserieskmeans",
    #     metric="dtw"
    # )
    
    # Fast Euclidean-based analysis
    optimal_k, optimization_results, final_results = run_optimal_cluster_analysis(
        cache_file='cache/nasdaq_cache_5y.csv',
        ticker_file='tickers/nasdaq_screener.csv',
        sample_size=3200,
        max_clusters=16,
        min_clusters=4,
        period='5y',
        start_date='2022-10-12',
        feature_type='normalized_prices',
        method="timeserieskmeans",
        metric="euclidean"
    )
    
    # For quick testing with different methods
    # optimal_k, optimization_results, final_results = run_quick_optimization_test(method="kshape")
    # optimal_k, optimization_results, final_results = run_quick_optimization_test(method="timeserieskmeans", metric="dtw")
    
    if final_results is not None:
        clustering, results, performance = final_results
        method_str = f"{clustering.model.__class__.__name__.lower()}"
        if hasattr(clustering.model, 'metric'):
            method_str += f"_{clustering.model.metric}"
        
        print(f"\n🎯 FINAL OPTIMAL ANALYSIS COMPLETE!")
        print(f"Method: {method_str.upper()}")
        print(f"Optimal clusters: {optimal_k}")
        print(f"Stocks analyzed: {len(results)}")
        
        # Show optimization results summary
        if optimization_results is not None:
            print(f"\nOptimization tested {len(optimization_results)} different cluster counts")
            print("Top 3 configurations by outperformance score:")
            top_3 = optimization_results.nlargest(3, 'outperformance_score')
            for _, row in top_3.iterrows():
                method_display = f"{row['method']}" + (f"_{row['metric']}" if row['method'] == "timeserieskmeans" else "")
                print(f"  {row['n_clusters']} clusters ({method_display}): score {row['outperformance_score']:.2f}, "
                      f"best return {row['best_cluster_return']*100:.2f}%")
    else:
        print("❌ Analysis failed")