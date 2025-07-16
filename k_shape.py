import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import silhouette_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockKShapeClustering:
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize K-Shape clustering for stock price analysis
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.stock_data = None
        self.features = None
        self.cluster_labels = None
        
    def fetch_stock_data(self, tickers, period='1y'):
        """
        Fetch stock data from Yahoo Finance
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers
        period : str
            Time period for data ('1y', '2y', '5y', etc.)
        """
        print("Fetching stock data...")
        stock_data = {}
        skipped_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                # Only keep if we have valid Close data
                if not hist.empty and 'Close' in hist.columns:
                    close_data = hist['Close']
                    if not close_data.empty and close_data.notna().sum() > 10:  # At least 10 valid prices
                        stock_data[ticker] = close_data
                    else:
                        skipped_tickers.append(ticker)
                else:
                    skipped_tickers.append(ticker)
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                skipped_tickers.append(ticker)
                
        if skipped_tickers:
            print(f"Skipped {len(skipped_tickers)} tickers with no/insufficient data")
            
        self.stock_data = pd.DataFrame(stock_data).dropna()
        print(f"Successfully fetched data for {len(self.stock_data.columns)} stocks")
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
    
    def find_optimal_clusters(self, max_clusters=10, min_clusters=2):
        """
        Find optimal number of clusters using silhouette score
        
        Parameters:
        -----------
        max_clusters : int
            Maximum number of clusters to try
        min_clusters : int
            Minimum number of clusters to try
        """
        if self.features is None:
            raise ValueError("No features prepared. Run prepare_features first.")
            
        # Scale the features
        scaler = TimeSeriesScalerMeanVariance()
        scaled_features = scaler.fit_transform(self.features)
        
        silhouette_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)
        
        print("Finding optimal number of clusters...")
        for n_clusters in cluster_range:
            kshape = KShape(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = kshape.fit_predict(scaled_features)
            
            # Calculate silhouette score
            # Reshape for silhouette calculation
            reshaped_features = scaled_features.reshape(scaled_features.shape[0], -1)
            score = silhouette_score(reshaped_features, cluster_labels)
            silhouette_scores.append(score)
            
            print(f"Clusters: {n_clusters}, Silhouette Score: {score:.3f}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True)
        plt.show()
        
        # Find optimal number of clusters
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters, silhouette_scores
    
    def fit_clustering(self, scale_features=True):
        """
        Fit K-Shape clustering model
        
        Parameters:
        -----------
        scale_features : bool
            Whether to scale features before clustering
        """
        if self.features is None:
            raise ValueError("No features prepared. Run prepare_features first.")
            
        # Check if we have any valid features
        if len(self.features) == 0:
            print("WARNING: No valid features available for clustering. Skipping clustering.")
            self.cluster_labels = np.array([])
            return self.cluster_labels
            
        # Check if we have enough samples for the requested number of clusters
        if len(self.features) < self.n_clusters:
            print(f"WARNING: Only {len(self.features)} valid stocks, but {self.n_clusters} clusters requested.")
            print(f"Reducing number of clusters to {len(self.features)}")
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
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Run fit_clustering first.")
            
        # Handle case where no clustering was performed due to insufficient data
        if len(self.cluster_labels) == 0:
            print("WARNING: No cluster results available - no valid stocks were clustered")
            return pd.DataFrame(columns=['Ticker', 'Cluster'])
            
        # Use valid_tickers if available, otherwise fallback to original logic
        if hasattr(self, 'valid_tickers') and self.valid_tickers:
            valid_tickers = self.valid_tickers
        else:
            # Fallback: use first N tickers where N = number of cluster labels
            valid_tickers = self.stock_data.columns[:len(self.cluster_labels)]
            
        results = pd.DataFrame({
            'Ticker': valid_tickers,
            'Cluster': self.cluster_labels
        })
        
        return results.sort_values('Cluster')
    
    def plot_clusters(self, feature_type='returns', figsize=(15, 10)):
        """
        Visualize clustering results
        
        Parameters:
        -----------
        feature_type : str
            Type of features to plot
        figsize : tuple
            Figure size
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Run fit_clustering first.")
            
        # Prepare features for plotting
        features_df = self.prepare_features(feature_type)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'K-Shape Clustering Results - {feature_type.title()}', fontsize=16)
        
        # Plot 1: All time series colored by cluster
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        
        for i, ticker in enumerate(features_df.columns):
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
            cluster_stocks = [ticker for i, ticker in enumerate(features_df.columns) 
                            if self.cluster_labels[i] == cluster]
            
            if cluster_stocks:
                # Plot first stock from each cluster as representative
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
    
    def analyze_cluster_performance(self):
        """
        Analyze performance characteristics of each cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Run fit_clustering first.")
            
        # Calculate returns for performance analysis
        returns = self.stock_data.pct_change().dropna()
        
        cluster_stats = []
        
        for cluster in range(self.n_clusters):
            cluster_stocks = [ticker for i, ticker in enumerate(self.stock_data.columns) 
                            if self.cluster_labels[i] == cluster]
            
            if cluster_stocks:
                cluster_returns = returns[cluster_stocks]
                
                stats = {
                    'Cluster': cluster,
                    'Stocks': cluster_stocks,
                    'Count': len(cluster_stocks),
                    'Avg_Return': cluster_returns.mean().mean(),
                    'Avg_Volatility': cluster_returns.std().mean(),
                    'Sharpe_Ratio': cluster_returns.mean().mean() / cluster_returns.std().mean(),
                    'Max_Drawdown': self._calculate_max_drawdown(cluster_returns.mean(axis=1))
                }
                
                cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

# Example usage
if __name__ == "__main__":
    # Example stock tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 
               'JPM', 'JNJ', 'PG', 'KO', 'XOM', 'GE', 'BA']
    
    # Initialize clustering
    clustering = StockKShapeClustering(n_clusters=3)
    
    # Fetch data and prepare features
    stock_data = clustering.fetch_stock_data(tickers, period='1y')
    features = clustering.prepare_features(feature_type='normalized_prices')
    
    # Find optimal clusters (optional)
    # optimal_k, scores = clustering.find_optimal_clusters()
    
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