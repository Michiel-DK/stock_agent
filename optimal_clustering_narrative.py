import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy import stats

def find_optimal_clusters_for_narratives(clustering_class, stock_data, max_clusters=15, min_clusters=3):
    """
    Find optimal number of clusters focused on building coherent narratives/themes
    
    This approach favors:
    - Smaller, focused clusters (5-15 stocks each)
    - Clear performance differentiation 
    - Actionable cluster sizes for narrative building
    - Quality over quantity of clusters
    
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
    
    print("=== FINDING OPTIMAL CLUSTERS FOR NARRATIVE BUILDING ===")
    print("Focus: Small, coherent clusters suitable for thematic analysis")
    
    results = []
    cluster_range = range(min_clusters, max_clusters + 1)
    
    # Calculate market benchmark
    returns = stock_data.pct_change().dropna()
    market_return = returns.mean(axis=1).mean()
    market_volatility = returns.mean(axis=1).std()
    market_sharpe = market_return / market_volatility if market_volatility != 0 else 0
    
    print(f"Market Benchmark - Return: {market_return:.4f} ({market_return*100:.2f}%), "
          f"Volatility: {market_volatility:.4f}, Sharpe: {market_sharpe:.4f}")
    
    total_stocks = len(stock_data.columns)
    print(f"Total stocks to cluster: {total_stocks}")
    
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
            
            # === NARRATIVE-FOCUSED METRICS ===
            
            cluster_sizes = performance['Count'].values
            avg_cluster_size = np.mean(cluster_sizes)
            min_cluster_size = cluster_sizes.min()
            max_cluster_size = cluster_sizes.max()
            
            # 1. IDEAL CLUSTER SIZE SCORE (5-15 stocks per cluster)
            # Sweet spot for narrative building - not too small, not too large
            ideal_min, ideal_max = 5, 15
            size_scores = []
            
            for size in cluster_sizes:
                if ideal_min <= size <= ideal_max:
                    size_score = 100  # Perfect size
                elif size < ideal_min:
                    # Penalty for being too small (harder to build narrative)
                    size_score = max(0, (size / ideal_min) * 100)
                else:
                    # Penalty for being too large (diluted narrative)
                    size_score = max(0, 100 - (size - ideal_max) * 5)
                size_scores.append(size_score)
            
            avg_size_score = np.mean(size_scores)
            pct_ideal_sized = sum(1 for s in cluster_sizes if ideal_min <= s <= ideal_max) / len(cluster_sizes) * 100
            
            # 2. PERFORMANCE CLARITY (clear winners and losers)
            returns_list = performance['Avg_Return'].values
            return_spread = np.max(returns_list) - np.min(returns_list)
            
            # Reward clear performance tiers
            performance_clarity = return_spread * 1000  # Scale up
            
            # 3. ACTIONABLE CLUSTER COUNT 
            # Prefer 4-8 clusters (manageable for analysis)
            if 4 <= n_clusters <= 8:
                cluster_count_score = 100
            elif n_clusters < 4:
                cluster_count_score = n_clusters * 25  # Too few clusters
            else:
                cluster_count_score = max(0, 100 - (n_clusters - 8) * 10)  # Too many clusters
            
            # 4. PERFORMANCE DIVERSITY (avoid all clusters performing similarly)
            return_std = np.std(returns_list)
            diversity_score = min(100, return_std * 5000)  # Scale appropriately
            
            # 5. TOP PERFORMER IDENTIFICATION
            best_cluster_return = performance['Avg_Return'].max()
            best_vs_market = (best_cluster_return - market_return) * 100
            top_performer_score = max(0, best_vs_market * 20)  # 20x multiplier
            
            # 6. CLUSTER BALANCE (not too extreme in sizes)
            size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes)  # Coefficient of variation
            balance_score = max(0, 100 - size_cv * 50)  # Penalty for high variation
            
            # 7. MINIMUM VIABLE CLUSTERS (avoid tiny clusters)
            min_viable_threshold = 3
            viable_clusters = sum(1 for s in cluster_sizes if s >= min_viable_threshold)
            viability_score = (viable_clusters / len(cluster_sizes)) * 100
            
            # === NARRATIVE COMPOSITE SCORE ===
            # Weighted combination favoring narrative building
            
            narrative_score = (
                avg_size_score * 0.25 +           # 25% - Ideal cluster sizes
                performance_clarity * 0.20 +      # 20% - Clear performance differences  
                cluster_count_score * 0.15 +      # 15% - Manageable number of clusters
                diversity_score * 0.15 +          # 15% - Performance diversity
                top_performer_score * 0.10 +      # 10% - Strong top performer
                balance_score * 0.10 +            # 10% - Reasonable balance
                viability_score * 0.05            # 5%  - All clusters viable
            )
            
            # Traditional clustering quality (for reference)
            if hasattr(clustering_class, 'features') and clustering_class.features is not None:
                try:
                    reshaped_features = clustering_class.features.reshape(clustering_class.features.shape[0], -1)
                    silhouette = silhouette_score(reshaped_features, labels)
                except:
                    silhouette = 0
            else:
                silhouette = 0
            
            result = {
                'n_clusters': n_clusters,
                'avg_cluster_size': avg_cluster_size,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'pct_ideal_sized': pct_ideal_sized,
                'return_spread': return_spread,
                'best_cluster_return': best_cluster_return,
                'best_vs_market': best_vs_market,
                'performance_diversity': return_std,
                'avg_size_score': avg_size_score,
                'performance_clarity': performance_clarity,
                'cluster_count_score': cluster_count_score,
                'diversity_score': diversity_score,
                'top_performer_score': top_performer_score,
                'balance_score': balance_score,
                'viability_score': viability_score,
                'narrative_score': narrative_score,
                'silhouette_score': silhouette,
                'cluster_performance': performance,
                'cluster_sizes': cluster_sizes
            }
            
            results.append(result)
            
            print(f"  Avg cluster size: {avg_cluster_size:.1f} stocks ({pct_ideal_sized:.0f}% in ideal range)")
            print(f"  Return spread: {return_spread:.4f} ({return_spread*100:.2f}%)")
            print(f"  Best cluster: {best_cluster_return*100:.2f}% vs market {market_return*100:.2f}%")
            print(f"  Narrative score: {narrative_score:.1f}")
            
        except Exception as e:
            print(f"Error with {n_clusters} clusters: {e}")
            continue
    
    if not results:
        print("No valid clustering results obtained")
        return None, None
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find optimal based on narrative score
    best_narrative = results_df.loc[results_df['narrative_score'].idxmax()]
    best_silhouette = results_df.loc[results_df['silhouette_score'].idxmax()]
    
    # Find best for different size preferences
    best_small_clusters = results_df[results_df['avg_cluster_size'] <= 10]
    if not best_small_clusters.empty:
        best_small = best_small_clusters.loc[best_small_clusters['narrative_score'].idxmax()]
    else:
        best_small = best_narrative
    
    print(f"\n=== OPTIMAL CLUSTER RECOMMENDATIONS FOR NARRATIVES ===")
    print(f"🎯 Best Overall: {best_narrative['n_clusters']} clusters")
    print(f"   - Narrative Score: {best_narrative['narrative_score']:.1f}")
    print(f"   - Avg Cluster Size: {best_narrative['avg_cluster_size']:.1f} stocks")
    print(f"   - {best_narrative['pct_ideal_sized']:.0f}% of clusters in ideal size range")
    print(f"   - Return Spread: {best_narrative['return_spread']*100:.2f}%")
    
    if best_small['n_clusters'] != best_narrative['n_clusters']:
        print(f"\n📊 Best for Small Clusters: {best_small['n_clusters']} clusters")
        print(f"   - Avg Cluster Size: {best_small['avg_cluster_size']:.1f} stocks")
        print(f"   - Narrative Score: {best_small['narrative_score']:.1f}")
    
    print(f"\n🔗 Traditional Best: {best_silhouette['n_clusters']} clusters")
    print(f"   - Silhouette Score: {best_silhouette['silhouette_score']:.3f}")
    
    # Plot results
    plot_narrative_optimization_results(results_df, market_return)
    
    return int(best_narrative['n_clusters']), results_df

def plot_narrative_optimization_results(results_df, market_return):
    """
    Plot narrative-focused cluster optimization results
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Cluster Optimization for Narrative Building', fontsize=16)
    
    x = results_df['n_clusters']
    
    # Plot 1: Narrative Score (main metric)
    axes[0, 0].plot(x, results_df['narrative_score'], 'bo-', linewidth=3, markersize=8)
    best_idx = results_df['narrative_score'].idxmax()
    best_x = results_df.loc[best_idx, 'n_clusters']
    best_y = results_df.loc[best_idx, 'narrative_score']
    axes[0, 0].plot(best_x, best_y, 'ro', markersize=15, label=f'Optimal: {best_x}')
    axes[0, 0].set_title('Narrative Score (Main Metric)', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Cluster Size
    axes[0, 1].plot(x, results_df['avg_cluster_size'], 'go-', linewidth=2)
    axes[0, 1].axhline(y=5, color='orange', linestyle='--', label='Ideal Min (5)')
    axes[0, 1].axhline(y=15, color='orange', linestyle='--', label='Ideal Max (15)')
    axes[0, 1].set_title('Average Cluster Size')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Average Stocks per Cluster')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: % Clusters in Ideal Size Range
    axes[1, 0].bar(x, results_df['pct_ideal_sized'], alpha=0.7, color='purple')
    axes[1, 0].set_title('Percentage of Clusters in Ideal Size Range (5-15 stocks)')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Return Spread (Performance Differentiation)
    axes[1, 1].plot(x, results_df['return_spread'] * 100, 'mo-', linewidth=2)
    axes[1, 1].set_title('Return Spread (Performance Differentiation)')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Return Spread (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Best Cluster Performance vs Market
    axes[2, 0].plot(x, results_df['best_cluster_return'] * 100, 'co-', label='Best Cluster')
    axes[2, 0].axhline(y=market_return * 100, color='r', linestyle='--', label='Market')
    axes[2, 0].set_title('Best Cluster vs Market Performance')
    axes[2, 0].set_xlabel('Number of Clusters')
    axes[2, 0].set_ylabel('Daily Return (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Cluster Size Distribution (box plot style visualization)
    cluster_size_ranges = []
    for _, row in results_df.iterrows():
        sizes = row['cluster_sizes']
        cluster_size_ranges.append([np.min(sizes), np.max(sizes)])
    
    cluster_size_ranges = np.array(cluster_size_ranges)
    axes[2, 1].fill_between(x, cluster_size_ranges[:, 0], cluster_size_ranges[:, 1], 
                           alpha=0.3, label='Size Range')
    axes[2, 1].plot(x, results_df['avg_cluster_size'], 'ro-', label='Average Size')
    axes[2, 1].axhline(y=5, color='orange', linestyle='--', alpha=0.7)
    axes[2, 1].axhline(y=15, color='orange', linestyle='--', alpha=0.7)
    axes[2, 1].set_title('Cluster Size Distribution')
    axes[2, 1].set_xlabel('Number of Clusters')
    axes[2, 1].set_ylabel('Cluster Size (stocks)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed recommendations
    print(f"\n=== DETAILED ANALYSIS ===")
    best_row = results_df.loc[results_df['narrative_score'].idxmax()]
    
    print(f"📋 Recommended Configuration:")
    print(f"   • {best_row['n_clusters']} clusters")
    print(f"   • Average {best_row['avg_cluster_size']:.1f} stocks per cluster")
    print(f"   • Size range: {best_row['min_cluster_size']}-{best_row['max_cluster_size']} stocks")
    print(f"   • {best_row['pct_ideal_sized']:.0f}% of clusters in ideal size range")
    
    print(f"\n📈 Performance Characteristics:")
    print(f"   • Return spread: {best_row['return_spread']*100:.2f}% (higher = more differentiated)")
    print(f"   • Best cluster outperformance: +{best_row['best_vs_market']:.2f}% vs market")
    print(f"   • Performance diversity score: {best_row['diversity_score']:.1f}")
    
    print(f"\n🎯 Why This Works for Narratives:")
    print(f"   • Manageable number of themes to analyze")
    print(f"   • Each cluster has enough stocks to build a story")
    print(f"   • Clear performance differentiation between themes")
    print(f"   • Suitable for qualitative analysis and news correlation")

# Example usage with narrative focus
def run_narrative_cluster_analysis(sample_size=100, max_clusters=12, period='1y'):
    """
    Run clustering analysis optimized for narrative building
    """
    print("=== NARRATIVE-FOCUSED CLUSTERING ANALYSIS ===")
    print("Optimizing for: Small coherent clusters suitable for thematic analysis")
    
    # Your existing data loading code here...
    # This would integrate with your existing cache system
    
    pass  # Implement based on your existing structure

if __name__ == "__main__":
    # This would integrate with your existing cache and clustering system
    print("Narrative-focused cluster optimization ready!")
    print("Key differences from traditional approach:")
    print("- Favors 5-15 stocks per cluster (ideal for narrative building)")
    print("- Rewards 4-8 total clusters (manageable for analysis)")
    print("- Emphasizes performance differentiation over statistical optimization")
    print("- Designed for qualitative analysis and news correlation")