import pandas as pd
from typing import List, Dict

class ClusterLookupTool:
    """
    Ultra simple tool - just lookup cluster similarities from the CSV
    """
    
    def __init__(self):
        self.df = pd.read_csv("description_data/cluster_similarity_features_company_descriptions_cluster_timeserieskmeans_euclidean_stock_ranking_20250804_224954_full_new.csv")
    
    def get_most_similar_to_cluster(self, target_cluster: int, top_k: int = 5) -> List[Dict]:
        """
        Find clusters most similar to target_cluster
        Just lookup and sort!
        """
        # Find all rows with our target cluster
        matches = self.df[
            (self.df['cluster_i'] == target_cluster) | 
            (self.df['cluster_j'] == target_cluster)
        ].copy()
        
        # Get the "other" cluster for each row
        matches['other_cluster'] = matches.apply(
            lambda row: row['cluster_j'] if row['cluster_i'] == target_cluster else row['cluster_i'], 
            axis=1
        )
        
        # Sort by similarity (highest first) and return top k
        top_matches = matches.nlargest(top_k, 'cosine_similarity')
        
        results = []
        for _, row in top_matches.iterrows():
            results.append({
                'cluster': int(row['other_cluster']),
                'similarity': round(row['cosine_similarity'], 4),
                'distance': round(row['centroid_distance'], 2)
            })
        
        return results
    
    def get_top_similar_pairs(self, top_k: int = 5) -> List[Dict]:
        """Get the top K most similar cluster pairs overall"""
        top_pairs = self.df.nlargest(top_k, 'cosine_similarity')
        
        results = []
        for _, row in top_pairs.iterrows():
            results.append({
                'cluster_1': int(row['cluster_i']),
                'cluster_2': int(row['cluster_j']),
                'similarity': round(row['cosine_similarity'], 4),
                'distance': round(row['centroid_distance'], 2)
            })
        
        return results
    
    def check_similarity(self, cluster_1: int, cluster_2: int) -> Dict:
        """Check similarity between two specific clusters"""
        # Find the row (could be in either order)
        match = self.df[
            ((self.df['cluster_i'] == cluster_1) & (self.df['cluster_j'] == cluster_2)) |
            ((self.df['cluster_i'] == cluster_2) & (self.df['cluster_j'] == cluster_1))
        ]
        
        if match.empty:
            return {'error': f'No data found for clusters {cluster_1} and {cluster_2}'}
        
        row = match.iloc[0]
        return {
            'cluster_1': cluster_1,
            'cluster_2': cluster_2,
            'similarity': round(row['cosine_similarity'], 4),
            'distance': round(row['centroid_distance'], 2),
            'size_ratio': round(row['size_ratio'], 3)
        }

# Super simple functions for LLM agents - no path needed!
def find_similar_clusters(cluster_id: int, top_k: int = 3):
    """Find clusters similar to cluster_id"""
    tool = ClusterLookupTool()
    return tool.get_most_similar_to_cluster(cluster_id, top_k)

def get_most_similar_pairs(top_k: int = 5):
    """Get most similar cluster pairs"""
    tool = ClusterLookupTool()
    return tool.get_top_similar_pairs(top_k)

def check_clusters(cluster_1: int, cluster_2: int):
    """Check if two clusters are similar"""
    tool = ClusterLookupTool()
    return tool.check_similarity(cluster_1, cluster_2)

# Example usage
if __name__ == "__main__":
    
    # Find what's similar to cluster 0
    similar_to_0 = find_similar_clusters(0, 3)
    print("Most similar to cluster 0:")
    for result in similar_to_0:
        print(f"  Cluster {result['cluster']}: {result['similarity']} similarity")
    
    # Get top similar pairs overall
    top_pairs = get_most_similar_pairs(3)
    print("\nMost similar cluster pairs:")
    for pair in top_pairs:
        print(f"  Clusters {pair['cluster_1']} & {pair['cluster_2']}: {pair['similarity']}")
    
    # Check specific clusters
    check = check_clusters(0, 1)
    print(f"\nClusters 0 & 1: {check['similarity']} similarity")