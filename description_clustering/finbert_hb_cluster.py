import yfinance as yf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class FinBERTClusterer:
    def __init__(self, model_name='ProsusAI/finbert'):
        """
        Initialize FinBERT model for text clustering
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embeddings(self, texts, batch_size=16):
        """
        Generate embeddings for a list of texts
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def cluster_companies(self, company_descriptions, company_tickers=None, 
                         n_clusters=None, use_hdbscan=True, reduce_dims=True):
        """
        Cluster company descriptions and return results
        """
        print("Generating embeddings...")
        embeddings = self.get_embeddings(company_descriptions)
        
        # Optional dimensionality reduction
        if reduce_dims:
            print("Reducing dimensions...")
            reducer = umap.UMAP(n_components=50, random_state=42)
            embeddings_reduced = reducer.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # Clustering
        if use_hdbscan:
            print("Clustering with HDBSCAN...")
            clusterer = HDBSCAN(min_cluster_size=3, metric='euclidean')
            cluster_labels = clusterer.fit_predict(embeddings_reduced)
        else:
            if n_clusters is None:
                n_clusters = self._find_optimal_clusters(embeddings_reduced)
            print(f"Clustering with K-means (k={n_clusters})...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings_reduced)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(embeddings_reduced, cluster_labels)
            print(f"Silhouette Score: {sil_score:.3f}")
        
        return {
            'embeddings': embeddings,
            'embeddings_reduced': embeddings_reduced,
            'cluster_labels': cluster_labels,
            'clusterer': clusterer
        }
    
    def _find_optimal_clusters(self, embeddings, max_k=15):
        """
        Find optimal number of clusters using elbow method
        """
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(embeddings)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(embeddings, cluster_labels))
        
        # Return k with highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal k based on silhouette score: {optimal_k}")
        return optimal_k
    
    def visualize_clusters(self, results, company_tickers=None, company_descriptions=None):
        """
        Visualize clusters in 2D space
        """
        # Reduce to 2D for visualization
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer_2d.fit_transform(results['embeddings_reduced'])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=results['cluster_labels'], cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Company Clusters (FinBERT Embeddings)')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        
        # Add labels for some points if tickers provided
        if company_tickers:
            for i, ticker in enumerate(company_tickers[:20]):  # Show first 20
                plt.annotate(ticker, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_clusters(self, results, company_descriptions, company_tickers=None):
        """
        Analyze and display cluster characteristics
        """
        df = pd.DataFrame({
            'ticker': company_tickers if company_tickers else range(len(company_descriptions)),
            'description': company_descriptions,
            'cluster': results['cluster_labels']
        })
        
        print("\nCluster Analysis:")
        print("=" * 50)
        
        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id == -1:  # HDBSCAN noise
                continue
                
            cluster_companies = df[df['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_companies)} companies):")
            print("-" * 30)
            
            # Show sample companies
            for _, row in cluster_companies.head(5).iterrows():
                ticker_str = f"({row['ticker']}) " if company_tickers else ""
                desc_preview = row['description'][:100] + "..." if len(row['description']) > 100 else row['description']
                print(f"  {ticker_str}{desc_preview}")
            
            if len(cluster_companies) > 5:
                print(f"  ... and {len(cluster_companies) - 5} more")
        
        return df

# Example usage
def main():
    # Sample company tickers - replace with your list
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'JPM', 'BAC', 'XOM', 'CVX', 
               'JNJ', 'PFE', 'WMT', 'AMZN', 'META', 'NFLX']
    
    # Fetch company descriptions
    print("Fetching company data...")
    companies_data = []
    descriptions = []
    valid_tickers = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            if 'longBusinessSummary' in info and info['longBusinessSummary']:
                descriptions.append(info['longBusinessSummary'])
                valid_tickers.append(ticker)
                companies_data.append(info)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    print(f"Successfully fetched {len(descriptions)} company descriptions")
    
    # Initialize clusterer
    clusterer = FinBERTClusterer()
    
    # Perform clustering
    results = clusterer.cluster_companies(
        descriptions, 
        company_tickers=valid_tickers,
        use_hdbscan=True,  # Set to False to use K-means
        reduce_dims=True
    )
    
    # Analyze results
    cluster_df = clusterer.analyze_clusters(results, descriptions, valid_tickers)
    
    # Visualize
    clusterer.visualize_clusters(results, valid_tickers, descriptions)
    
    import ipdb;ipdb.set_trace()  # For debugging purposes
    
    return cluster_df, results

if __name__ == "__main__":
    cluster_df, results = main()