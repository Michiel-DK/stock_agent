import yfinance as yf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx

class FinBERTClusterer:
    def __init__(self, model_name='ProsusAI/finbert'):
        """
        Initialize FinBERT model for text clustering
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cluster_centroids = None
        self.cluster_distance_matrix = None
        
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
        
        # Calculate cluster distance metrics
        distance_metrics = self._calculate_cluster_distances(embeddings_reduced, cluster_labels)
        
        return {
            'embeddings': embeddings,
            'embeddings_reduced': embeddings_reduced,
            'cluster_labels': cluster_labels,
            'clusterer': clusterer,
            'distance_metrics': distance_metrics
        }
    
    def _calculate_cluster_distances(self, embeddings, cluster_labels):
        """
        Calculate various distance metrics between clusters
        """
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]  # Exclude noise (-1)
        n_clusters = len(unique_clusters)
        
        if n_clusters < 2:
            return None
        
        # 1. Centroid-based distances
        centroids = []
        cluster_sizes = []
        cluster_variances = []
        
        for cluster_id in unique_clusters:
            cluster_points = embeddings[cluster_labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            cluster_sizes.append(len(cluster_points))
            
            # Calculate within-cluster variance
            variance = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
            cluster_variances.append(variance)
        
        centroids = np.array(centroids)
        self.cluster_centroids = centroids
        
        # Distance matrix between centroids (Euclidean)
        centroid_distances = pairwise_distances(centroids, metric='euclidean')
        
        # Cosine similarity between centroids
        centroid_cosine_sim = 1 - pairwise_distances(centroids, metric='cosine')
        
        # 2. Minimum distance between clusters (closest points)
        min_distances = np.full((n_clusters, n_clusters), np.inf)
        max_distances = np.full((n_clusters, n_clusters), 0)
        avg_distances = np.full((n_clusters, n_clusters), 0)
        
        for i, cluster_i in enumerate(unique_clusters):
            for j, cluster_j in enumerate(unique_clusters):
                if i != j:
                    points_i = embeddings[cluster_labels == cluster_i]
                    points_j = embeddings[cluster_labels == cluster_j]
                    
                    # Calculate all pairwise distances
                    distances = pairwise_distances(points_i, points_j, metric='euclidean')
                    
                    min_distances[i, j] = np.min(distances)
                    max_distances[i, j] = np.max(distances)
                    avg_distances[i, j] = np.mean(distances)
        
        # 3. Normalized distances (accounting for cluster size and variance)
        normalized_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    # Normalize by average of both cluster variances
                    norm_factor = np.sqrt(cluster_variances[i] + cluster_variances[j])
                    normalized_distances[i, j] = centroid_distances[i, j] / (norm_factor + 1e-8)
        
        # 4. Weighted distances (accounting for cluster sizes)
        weighted_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    # Weight by harmonic mean of cluster sizes
                    weight = 2 / (1/cluster_sizes[i] + 1/cluster_sizes[j])
                    weighted_distances[i, j] = centroid_distances[i, j] * np.log(weight + 1)
        
        self.cluster_distance_matrix = centroid_distances
        
        return {
            'unique_clusters': unique_clusters,
            'centroids': centroids,
            'cluster_sizes': cluster_sizes,
            'cluster_variances': cluster_variances,
            'centroid_distances': centroid_distances,
            'centroid_cosine_similarity': centroid_cosine_sim,
            'min_distances': min_distances,
            'max_distances': max_distances,
            'avg_distances': avg_distances,
            'normalized_distances': normalized_distances,
            'weighted_distances': weighted_distances
        }
    
    def get_cluster_similarity_features(self, results):
        """
        Extract features that can be used in ML models to predict cluster similarity
        """
        if results['distance_metrics'] is None:
            return None
        
        dm = results['distance_metrics']
        n_clusters = len(dm['unique_clusters'])
        
        features = []
        cluster_pairs = []
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):  # Only upper triangle
                cluster_i = dm['unique_clusters'][i]
                cluster_j = dm['unique_clusters'][j]
                
                feature_vector = {
                    'cluster_i': cluster_i,
                    'cluster_j': cluster_j,
                    'centroid_distance': dm['centroid_distances'][i, j],
                    'cosine_similarity': dm['centroid_cosine_similarity'][i, j],
                    'min_distance': dm['min_distances'][i, j],
                    'max_distance': dm['max_distances'][i, j],
                    'avg_distance': dm['avg_distances'][i, j],
                    'normalized_distance': dm['normalized_distances'][i, j],
                    'weighted_distance': dm['weighted_distances'][i, j],
                    'size_ratio': min(dm['cluster_sizes'][i], dm['cluster_sizes'][j]) / max(dm['cluster_sizes'][i], dm['cluster_sizes'][j]),
                    'combined_size': dm['cluster_sizes'][i] + dm['cluster_sizes'][j],
                    'variance_ratio': min(dm['cluster_variances'][i], dm['cluster_variances'][j]) / max(dm['cluster_variances'][i], dm['cluster_variances'][j]),
                    'combined_variance': dm['cluster_variances'][i] + dm['cluster_variances'][j],
                }
                
                features.append(feature_vector)
                cluster_pairs.append((cluster_i, cluster_j))
        
        return pd.DataFrame(features), cluster_pairs
    
    def visualize_cluster_distances(self, results, method='heatmap'):
        """
        Visualize cluster distances using various methods
        """
        if results['distance_metrics'] is None:
            print("No distance metrics available")
            return
        
        dm = results['distance_metrics']
        cluster_labels = [f"Cluster {c}" for c in dm['unique_clusters']]
        
        if method == 'heatmap':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Centroid distances
            sns.heatmap(dm['centroid_distances'], annot=True, fmt='.3f', 
                       xticklabels=cluster_labels, yticklabels=cluster_labels,
                       ax=axes[0, 0], cmap='viridis')
            axes[0, 0].set_title('Centroid Distances')
            
            # Cosine similarity
            sns.heatmap(dm['centroid_cosine_similarity'], annot=True, fmt='.3f',
                       xticklabels=cluster_labels, yticklabels=cluster_labels,
                       ax=axes[0, 1], cmap='RdYlBu')
            axes[0, 1].set_title('Cosine Similarity')
            
            # Normalized distances
            sns.heatmap(dm['normalized_distances'], annot=True, fmt='.3f',
                       xticklabels=cluster_labels, yticklabels=cluster_labels,
                       ax=axes[1, 0], cmap='plasma')
            axes[1, 0].set_title('Normalized Distances')
            
            # Min distances
            sns.heatmap(dm['min_distances'], annot=True, fmt='.3f',
                       xticklabels=cluster_labels, yticklabels=cluster_labels,
                       ax=axes[1, 1], cmap='magma')
            axes[1, 1].set_title('Minimum Distances')
            
            plt.tight_layout()
            plt.show()
        
        elif method == 'dendrogram':
            # Create dendrogram based on centroid distances
            condensed_distances = squareform(dm['centroid_distances'])
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            plt.figure(figsize=(10, 6))
            dendrogram(linkage_matrix, labels=cluster_labels)
            plt.title('Cluster Dendrogram (Based on Centroid Distances)')
            plt.ylabel('Distance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        elif method == 'network':
            # Create network graph where edge weights represent similarity
            G = nx.Graph()
            
            # Add nodes
            for i, cluster_id in enumerate(dm['unique_clusters']):
                G.add_node(cluster_id, size=dm['cluster_sizes'][i])
            
            # Add edges (higher similarity = thicker edge)
            n_clusters = len(dm['unique_clusters'])
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    similarity = dm['centroid_cosine_similarity'][i, j]
                    if similarity > 0.1:  # Only show meaningful similarities
                        G.add_edge(dm['unique_clusters'][i], dm['unique_clusters'][j], 
                                 weight=similarity)
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
            
            # Draw edges
            edges = G.edges()
            weights = [G[u][v]['weight'] * 5 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos)
            
            plt.title('Cluster Similarity Network')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    def get_most_similar_clusters(self, results, top_k=5):
        """
        Get the most similar cluster pairs
        """
        features_df, cluster_pairs = self.get_cluster_similarity_features(results)
        if features_df is None:
            return None
        
        # Sort by cosine similarity (descending) and centroid distance (ascending)
        features_df['similarity_score'] = (
            features_df['cosine_similarity'] * 0.6 + 
            (1 / (1 + features_df['normalized_distance'])) * 0.4
        )
        
        top_similar = features_df.nlargest(top_k, 'similarity_score')
        
        print(f"Top {top_k} Most Similar Cluster Pairs:")
        print("=" * 50)
        for idx, row in top_similar.iterrows():
            print(f"Clusters {row['cluster_i']} & {row['cluster_j']}:")
            print(f"  Similarity Score: {row['similarity_score']:.3f}")
            print(f"  Cosine Similarity: {row['cosine_similarity']:.3f}")
            print(f"  Centroid Distance: {row['centroid_distance']:.3f}")
            print(f"  Normalized Distance: {row['normalized_distance']:.3f}")
            print("-" * 30)
        
        return top_similar
    
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

# Example usage with distance analysis
def main():
    import json
    
    file_name = 'company_descriptions_cluster_timeserieskmeans_euclidean_stock_ranking_20250804_224954.json'

    base_name = file_name.replace('.json', '')  # Remove .json extension for output files


    with open(f'description_data/{file_name}', 'r', encoding='utf-8') as f:
            js = json.load(f)
    
    valid_tickers = [doc['ticker'] for doc in js if 'ticker' in doc]
    descriptions = [doc['businessSummary'] for doc in js if 'businessSummary' in doc]
    
    # Initialize clusterer
    clusterer = FinBERTClusterer()
    
    # Perform clustering
    results = clusterer.cluster_companies(
        descriptions, 
        company_tickers=valid_tickers,
        use_hdbscan=True,
        reduce_dims=True
    )
    
    # Analyze clusters
    cluster_df = clusterer.analyze_clusters(results, descriptions, valid_tickers)
    
    # Get similarity features for ML
    features_df, cluster_pairs = clusterer.get_cluster_similarity_features(results)
    if features_df is not None:
        print(f"\nGenerated {len(features_df)} cluster pair features for ML modeling")
        features_df.to_csv(f'description_data/cluster_similarity_features_{base_name}_full_new.csv', index=False)
    
    # Visualize cluster distances
    clusterer.visualize_cluster_distances(results, method='heatmap')
    clusterer.visualize_cluster_distances(results, method='dendrogram')
    clusterer.visualize_cluster_distances(results, method='network')
    
    # Get most similar clusters
    similar_clusters = clusterer.get_most_similar_clusters(results, top_k=5)
    
    cluster_df.to_csv(f'description_data/finbert_cluster_results_{base_name}_full.csv', index=False)
    
    # Visualize original clusters
    clusterer.visualize_clusters(results, valid_tickers, descriptions)
    
    return cluster_df, results, features_df

if __name__ == "__main__":
    cluster_df, results, features_df = main()
    import ipdb; ipdb.set_trace()  # For debugging purposes 