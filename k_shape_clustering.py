import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
import random

warnings.filterwarnings('ignore')

class StockKShapeClustering:
    def __init__(self, cache, n_clusters=3, random_state=42):
        """
        Initialize K-Shape clustering for stock price analysis
        
        Parameters:
        -----------
        cache : StockDataCache
            Instance of StockDataCache for data access
        n_clusters : int
            Number of clusters to form
        random_state : int
            Random state for reproducibility
        """
        self.cache = cache
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.stock_data = None
        self.features = None
        self.cluster_labels = None
        self.valid_tickers = None
        
    def load_data(self, tickers=None, start_date=None):
        """
        Load stock data from cache
        
        Parameters:
        -----------
        tickers : list, optional
            List of stock tickers. If None, uses all available tickers
        start_date : str or datetime-like, optional
            Start date to filter data from
        """
        print(f"Loading stock data from cache...")
        
        if tickers is None:
            available_tickers = self.cache.get_available_tickers()
            if not available_tickers:
                print("No tickers available in cache")
                self.stock_data = pd.DataFrame()
                return self.stock_data
            tickers = available_tickers
        
        self.stock_data = self.cache.get_data(tickers=tickers, start_date=start_date)
        
        if self.stock_data.empty:
            print("No data available for requested tickers and date range")
        else:
            print(f"Loaded data for {self.stock_data.shape[1]} stocks from {self.stock_data.index[0].strftime('%Y-%m-%d')} to {self.stock_data.index[-1].strftime('%Y-%m-%d')}")
            
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
        if self.stock_data is None or self.stock_data.empty:
            raise ValueError("No stock data available. Run load_data() first.")
            
        features = {}
        skipped_tickers = []
        
        for ticker in self.stock_data.columns:
            prices = self.stock_data[ticker]
            
            # Skip if prices series is empty or all NaN
            if prices.empty or prices.isna().all():
                skipped_tickers.append(ticker)
                continue
            
            try:
                if feature_type == 'returns':
                    feature_series = prices.pct_change().dropna()
                    
                elif feature_type == 'normalized_prices':
                    if prices.iloc[0] == 0 or pd.isna(prices.iloc[0]):
                        skipped_tickers.append(ticker)
                        continue
                    feature_series = prices / prices.iloc[0]
                    
                elif feature_type == 'ma_relative':
                    ma = prices.rolling(window=window_size).mean()
                    feature_series = (prices / ma - 1).dropna()
                    
                elif feature_type == 'volatility':
                    returns = prices.pct_change()
                    feature_series = returns.rolling(window=window_size).std().dropna()
                    
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")
                
                # Validation checks
                if (feature_series.empty or feature_series.isna().all() or 
                    len(feature_series) < 10 or np.isinf(feature_series).any()):
                    skipped_tickers.append(ticker)
                    continue
                    
                features[ticker] = feature_series
                
            except Exception as e:
                print(f"Skipping {ticker}: Error calculating {feature_type} - {str(e)}")
                skipped_tickers.append(ticker)
                continue
        
        if not features:
            print("WARNING: No valid features could be calculated for any ticker")
            self.features = to_time_series_dataset([])
            self.valid_tickers = []
            return pd.DataFrame()
        
        # Convert to DataFrame and ensure all series have same length
        features_df = pd.DataFrame(features).dropna()
        
        # Remove any columns that became empty after alignment
        empty_cols = features_df.columns[features_df.isna().all()].tolist()
        if empty_cols:
            features_df = features_df.drop(columns=empty_cols)
            skipped_tickers.extend(empty_cols)
        
        if features_df.empty:
            print("WARNING: No valid aligned features remain after processing")
            self.features = to_time_series_dataset([])
            self.valid_tickers = []
            return pd.DataFrame()
        
        # Convert to time series dataset format for tslearn
        self.features = to_time_series_dataset([features_df[col].values for col in features_df.columns])
        self.valid_tickers = list(features_df.columns)
        
        if skipped_tickers:
            print(f"Skipped {len(skipped_tickers)} tickers due to insufficient/invalid data")
        
        print(f"Prepared {feature_type} features for {len(features_df.columns)} tickers")
        return features_df
    
    def find_optimal_clusters(self, max_clusters=10, min_clusters=2):
        """
        Find optimal number of clusters using silhouette score
        """
        if self.features is None or len(self.features) == 0:
            raise ValueError("No features prepared. Run prepare_features() first.")
            
        # Adjust max_clusters based on available data
        max_possible_clusters = min(max_clusters, len(self.features))
        if max_possible_clusters < min_clusters:
            print(f"Not enough data for cluster optimization. Only {len(self.features)} samples available.")
            return len(self.features), []
        
        # Scale the features
        scaler = TimeSeriesScalerMeanVariance()
        scaled_features = scaler.fit_transform(self.features)
        
        silhouette_scores = []
        cluster_range = range(min_clusters, max_possible_clusters + 1)
        
        print("Finding optimal number of clusters...")
        for n_clusters in cluster_range:
            kshape = KShape(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = kshape.fit_predict(scaled_features)
            
            # Calculate silhouette score
            reshaped_features = scaled_features.reshape(scaled_features.shape[0], -1)
            score = silhouette_score(reshaped_features, cluster_labels)
            silhouette_scores.append(score)
            
            print(f"Clusters: {n_clusters}, Silhouette Score: {score:.3f}")
        
        # Find optimal number of clusters
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters, silhouette_scores
    
    def fit_clustering(self, scale_features=True):
        """
        Fit K-Shape clustering model
        """
        if self.features is None or len(self.features) == 0:
            print("WARNING: No valid features available for clustering")
            self.cluster_labels = np.array([])
            return self.cluster_labels
            
        # Adjust number of clusters if necessary
        if len(self.features) < self.n_clusters:
            print(f"WARNING: Only {len(self.features)} samples, reducing clusters from {self.n_clusters} to {len(self.features)}")
            self.n_clusters = len(self.features)
            
        # Scale features if requested
        if scale_features:
            self.scaler = TimeSeriesScalerMeanVariance()
            scaled_features = self.scaler.fit_transform(self.features)
        else:
            scaled_features = self.features
            
        # Fit K-Shape model
        print(f"Fitting K-Shape clustering with {self.n_clusters} clusters...")
        self.model = KShape(n_clusters=self.n_clusters, random_state=self.random_state)
        self.cluster_labels = self.model.fit_predict(scaled_features)
        
        # Print cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("Cluster distribution:")
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} stocks")
            
        return self.cluster_labels
    
    def get_cluster_results(self):
        """
        Get clustering results as a DataFrame
        """
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            print("WARNING: No cluster results available")
            return pd.DataFrame(columns=['Ticker', 'Cluster'])
            
        if not self.valid_tickers:
            print("WARNING: No valid tickers available")
            return pd.DataFrame(columns=['Ticker', 'Cluster'])
            
        results = pd.DataFrame({
            'Ticker': self.valid_tickers,
            'Cluster': self.cluster_labels
        })
        
        return results.sort_values('Cluster')
    
    def analyze_cluster_performance(self):
        """
        Analyze performance characteristics of each cluster
        """
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            raise ValueError("Model not fitted. Run fit_clustering() first.")
            
        if not self.valid_tickers:
            print("WARNING: No valid tickers for performance analysis")
            return pd.DataFrame()
        
        # Get returns data for valid tickers only
        valid_stock_data = self.stock_data[self.valid_tickers]
        returns = valid_stock_data.pct_change().dropna()
        
        cluster_stats = []
        
        for cluster in range(self.n_clusters):
            cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == cluster]
            cluster_stocks = [self.valid_tickers[i] for i in cluster_indices]
            
            if cluster_stocks:
                cluster_returns = returns[cluster_stocks]
                
                stats = {
                    'Cluster': cluster,
                    'Stocks': cluster_stocks,
                    'Count': len(cluster_stocks),
                    'Avg_Return': cluster_returns.mean().mean(),
                    'Avg_Volatility': cluster_returns.std().mean(),
                }
                
                # Calculate Sharpe ratio safely
                avg_vol = stats['Avg_Volatility']
                if avg_vol > 0:
                    stats['Sharpe_Ratio'] = stats['Avg_Return'] / avg_vol
                else:
                    stats['Sharpe_Ratio'] = 0
                
                # Calculate max drawdown
                if len(cluster_stocks) > 1:
                    portfolio_returns = cluster_returns.mean(axis=1)
                else:
                    portfolio_returns = cluster_returns.iloc[:, 0]
                
                stats['Max_Drawdown'] = self._calculate_max_drawdown(portfolio_returns)
                cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        if returns.empty or returns.isna().all():
            return 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def plot_clusters(self, feature_type='returns', figsize=(15, 10)):
        """
        Visualize clustering results
        """
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            print("No clustering results to plot")
            return
            
        # Prepare features for plotting
        features_df = self.prepare_features(feature_type)
        
        if features_df.empty:
            print("No features available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'K-Shape Clustering Results - {feature_type.title()}', fontsize=16)
        
        # Plot 1: All time series colored by cluster
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        
        for i, ticker in enumerate(self.valid_tickers):
            if ticker in features_df.columns:
                cluster = self.cluster_labels[i]
                ax1.plot(features_df.index, features_df[ticker], 
                        color=colors[cluster], alpha=0.7, linewidth=1)
                
        ax1.set_title('All Time Series by Cluster')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(feature_type.title())
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster centroids
        ax2 = axes[0, 1]
        if hasattr(self.model, 'cluster_centers_'):
            for i, centroid in enumerate(self.model.cluster_centers_):
                ax2.plot(centroid.ravel(), color=colors[i], 
                        linewidth=3, label=f'Cluster {i}')
            ax2.set_title('Cluster Centroids')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Normalized Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cluster composition
        ax3 = axes[1, 0]
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        bars = ax3.bar(unique, counts, color=colors[:len(unique)])
        ax3.set_title('Cluster Sizes')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Stocks')
        ax3.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # Plot 4: Sample from each cluster
        ax4 = axes[1, 1]
        for cluster in range(self.n_clusters):
            cluster_stocks = [self.valid_tickers[i] for i, label in enumerate(self.cluster_labels) 
                            if label == cluster]
            
            if cluster_stocks and cluster_stocks[0] in features_df.columns:
                sample_stock = cluster_stocks[0]
                ax4.plot(features_df.index, features_df[sample_stock], 
                        color=colors[cluster], linewidth=2, 
                        label=f'Cluster {cluster} ({sample_stock})')
        
        ax4.set_title('Representative Stock from Each Cluster')
        ax4.set_xlabel('Date')
        ax4.set_ylabel(feature_type.title())
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, base_filename='stock_clustering'):
        """
        Save clustering results to CSV files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save basic results
        results = self.get_cluster_results()
        if not results.empty:
            results_file = f'{base_filename}_results_{timestamp}.csv'
            results.to_csv(results_file, index=False)
            print(f"✅ Basic results saved to {results_file}")
        
        # Save performance analysis
        try:
            performance = self.analyze_cluster_performance()
            if not performance.empty:
                # Save performance summary
                perf_summary = performance.drop('Stocks', axis=1)  # Remove stocks list for CSV
                perf_file = f'{base_filename}_performance_{timestamp}.csv'
                perf_summary.to_csv(perf_file, index=False)
                print(f"✅ Performance summary saved to {perf_file}")
                
                # Save detailed results with performance metrics
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
                
                detailed_file = f'{base_filename}_detailed_{timestamp}.csv'
                detailed_results.to_csv(detailed_file, index=False)
                print(f"✅ Detailed results saved to {detailed_file}")
                
        except Exception as e:
            print(f"⚠️  Could not save performance analysis: {e}")
        
        return timestamp