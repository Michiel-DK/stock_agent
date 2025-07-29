import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class StockClusteringModel:
    """
    A comprehensive clustering model for stock data focusing on business summaries
    with TF-IDF text analysis and multiple clustering algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,  # Limit features for manageable clustering
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        self.lemmatizer = WordNetLemmatizer()
        self.pca = None
        self.models = {}
        self.cluster_results = {}
        self.feature_names = []
        self.processed_data = None
        self.business_summary_features = None
        
    def clean_business_summary(self, text: str) -> str:
        """Clean and preprocess business summary text"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        financial_stopwords = {
            'company', 'inc', 'corporation', 'corp', 'ltd', 'limited', 
            'llc', 'co', 'group', 'holdings', 'services', 'solutions',
            'technologies', 'systems', 'international', 'global'
        }
        stop_words.update(financial_stopwords)
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_business_summary_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract TF-IDF features from business summaries"""
        print("Processing business summaries...")
        
        # Clean business summaries
        cleaned_summaries = df['businessSummary'].apply(self.clean_business_summary)
        
        # Handle empty summaries
        cleaned_summaries = cleaned_summaries.fillna("")
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_summaries)
        
        print(f"TF-IDF feature shape: {tfidf_features.shape}")
        print(f"Top TF-IDF features: {self.tfidf_vectorizer.get_feature_names_out()[:20]}")
        
        return tfidf_features.toarray()
    def load_data(self, data: List[Dict]) -> pd.DataFrame:
        """Load and convert stock data to DataFrame"""
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} stocks")
        return df

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        processed_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['businessSummary', 'marketCap', 'sector', 'industry']
        for col in required_cols:
            if col not in processed_df.columns:
                if col == 'businessSummary':
                    processed_df[col] = ""
                elif col == 'marketCap':
                    processed_df[col] = 1000000000  # Default 1B
                else:
                    processed_df[col] = "Unknown"
        
        # Handle employeeCount specifically
        if 'employeeCount' not in processed_df.columns:
            processed_df['employeeCount'] = 100  # Default value
        
        # Convert data types
        if processed_df['marketCap'].dtype == 'object':
            processed_df['marketCap'] = pd.to_numeric(processed_df['marketCap'], errors='coerce')
        
        if processed_df['employeeCount'].dtype == 'object':
            processed_df['employeeCount'] = pd.to_numeric(processed_df['employeeCount'], errors='coerce')
        
        return processed_df
    
    def preprocess_data(self, df: pd.DataFrame, text_weight: float = 0.7) -> np.ndarray:
        """
        Preprocess stock data for clustering with heavy focus on business summaries:
        - Extract TF-IDF features from business summaries (70% weight by default)
        - Add complementary numerical and categorical features (30% weight)
        """
        # Validate and clean data first
        processed_df = self.validate_and_clean_data(df)
        
        # Handle missing values more comprehensively
        print("Handling missing values...")
        
        # Fill missing employeeCount with median
        if processed_df['employeeCount'].isna().any():
            median_employees = processed_df['employeeCount'].median()
            if pd.isna(median_employees):  # All values are NaN
                processed_df['employeeCount'] = processed_df['employeeCount'].fillna(100)  # Default value
            else:
                processed_df['employeeCount'] = processed_df['employeeCount'].fillna(median_employees)
        
        # Fill missing marketCap with median
        if processed_df['marketCap'].isna().any():
            median_market_cap = processed_df['marketCap'].median()
            if pd.isna(median_market_cap):  # All values are NaN
                processed_df['marketCap'] = processed_df['marketCap'].fillna(1000000000)  # Default 1B
            else:
                processed_df['marketCap'] = processed_df['marketCap'].fillna(median_market_cap)
        
        # Handle missing business summaries
        processed_df['businessSummary'] = processed_df['businessSummary'].fillna("")
        
        # Handle missing categorical data
        processed_df['sector'] = processed_df['sector'].fillna("Unknown")
        processed_df['industry'] = processed_df['industry'].fillna("Unknown")
        
        # Extract business summary features (main focus)
        business_features = self.extract_business_summary_features(processed_df)
        self.business_summary_features = business_features
        
        # Create supplementary numerical features
        processed_df['log_market_cap'] = np.log1p(processed_df['marketCap'].astype(float))
        processed_df['log_employees'] = np.log1p(processed_df['employeeCount'].astype(float))
        
        # Select minimal supplementary features
        numerical_features = ['log_market_cap', 'log_employees']
        categorical_features = ['sector', 'industry']  # Reduced to most important
        
        # Process numerical features with NaN check
        X_numerical = processed_df[numerical_features].values.astype(float)
        
        # Check for any remaining NaN values in numerical features
        if np.isnan(X_numerical).any():
            print("Warning: NaN values found in numerical features, filling with zeros...")
            X_numerical = np.nan_to_num(X_numerical, nan=0.0)
        
        X_numerical = self.scaler.fit_transform(X_numerical)
        
        # Process categorical features
        X_categorical = []
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            
            # Convert to string and handle any remaining NaN
            feature_values = processed_df[feature].astype(str).fillna("Unknown")
            encoded = self.label_encoders[feature].fit_transform(feature_values)
            X_categorical.append(encoded.reshape(-1, 1))
        
        if X_categorical:
            X_categorical = np.hstack(X_categorical)
            # Combine supplementary features
            X_supplementary = np.hstack([X_numerical, X_categorical])
        else:
            X_supplementary = X_numerical
        
        # Check for NaN in business features
        if np.isnan(business_features).any():
            print("Warning: NaN values found in business features, filling with zeros...")
            business_features = np.nan_to_num(business_features, nan=0.0)
        
        # Combine with weighted approach - business summary gets higher weight
        X_text_weighted = business_features * text_weight
        X_supp_weighted = X_supplementary * (1 - text_weight)
        
        # Final feature matrix
        X = np.hstack([X_text_weighted, X_supp_weighted])
        
        # Final NaN check
        if np.isnan(X).any():
            print("Warning: NaN values found in final feature matrix, filling with zeros...")
            X = np.nan_to_num(X, nan=0.0)
        
        # Store feature information
        tfidf_feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        self.feature_names = tfidf_feature_names + numerical_features + categorical_features
        self.processed_data = processed_df
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Business summary features: {business_features.shape[1]} (weight: {text_weight})")
        print(f"Supplementary features: {X_supplementary.shape[1]} (weight: {1-text_weight})")
        print(f"NaN values in final matrix: {np.isnan(X).sum()}")
        
        return X
    
    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> Dict:
        """Find optimal number of clusters using multiple methods"""
        k_range = range(2, min(max_k + 1, len(X) // 2))
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            calinski_scores.append(calinski_harabasz_score(X, labels))
        
    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> Dict:
        """Find optimal number of clusters using multiple methods"""
        k_range = range(2, min(max_k + 1, len(X) // 2))
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            calinski_scores.append(calinski_harabasz_score(X, labels))
        
        # Find elbow point using rate of change
        elbow_k = 3  # default
        if len(inertias) > 2:
            # Calculate second derivatives (acceleration)
            second_derivatives = np.diff(np.diff(inertias))
            # Find the point where acceleration is maximum (most negative)
            elbow_k = list(k_range)[np.argmax(second_derivatives) + 2]
        
        # Best silhouette score
        best_silhouette_k = list(k_range)[np.argmax(silhouette_scores)]
        
        # Best Calinski-Harabasz score
        best_calinski_k = list(k_range)[np.argmax(calinski_scores)]
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'elbow_k': elbow_k,
            'best_silhouette_k': best_silhouette_k,
            'best_calinski_k': best_calinski_k,
            'recommended_k': best_silhouette_k  # Prefer silhouette for text data
        }
    
    def train_models(self, X: np.ndarray, n_clusters: Optional[int] = None) -> Dict:
        """Train multiple clustering models"""
        if n_clusters is None:
            # Find optimal number of clusters
            optimization_results = self.find_optimal_clusters(X)
            n_clusters = optimization_results['recommended_k']
            print(f"Using {n_clusters} clusters based on optimization")
        
        models_config = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10),
            'gaussian_mixture': GaussianMixture(n_components=n_clusters, random_state=self.random_state),
            'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=3)  # Will be tuned
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name == 'dbscan':
                # Tune DBSCAN parameters
                best_eps = self._tune_dbscan(X)
                model.set_params(eps=best_eps)
            
            # Fit the model
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.predict(X)
            
            # Calculate metrics
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters_found > 1:
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
            else:
                silhouette = -1
                calinski = -1
            
            results[name] = {
                'model': model,
                'labels': labels,
                'n_clusters': n_clusters_found,
                'silhouette_score': silhouette,
                'calinski_score': calinski
            }
            
            print(f"{name}: {n_clusters_found} clusters, silhouette: {silhouette:.3f}")
        
        self.models = results
        return results
    
    def _tune_dbscan(self, X: np.ndarray) -> float:
        """Tune DBSCAN epsilon parameter"""
        from sklearn.neighbors import NearestNeighbors
        
        # Find k-distance graph to choose eps
        k = 4
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Use elbow method or return 75th percentile
        return np.percentile(distances, 75)
    
    def analyze_clusters(self, model_name: str = 'kmeans') -> Dict:
        """Analyze cluster characteristics focusing on business summaries"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        labels = self.models[model_name]['labels']
        df = self.processed_data.copy()
        df['cluster'] = labels
        
        analysis = {'model': model_name, 'clusters': {}}
        
        unique_labels = sorted([l for l in set(labels) if l != -1])
        
        for cluster_id in unique_labels:
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Basic stats
            cluster_info = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'avg_market_cap': cluster_data['marketCap'].mean(),
                'avg_employees': cluster_data['employeeCount'].mean(),
            }
            
            # Sector/Industry distribution
            cluster_info['top_sectors'] = cluster_data['sector'].value_counts().head(3).to_dict()
            cluster_info['top_industries'] = cluster_data['industry'].value_counts().head(3).to_dict()
            
            # Business summary analysis - key terms
            cluster_summaries = cluster_data['businessSummary'].fillna('')
            all_text = ' '.join(cluster_summaries)
            
            # Extract key terms from business summaries for this cluster
            if all_text.strip():
                cluster_tfidf = TfidfVectorizer(
                    max_features=20,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                try:
                    cluster_features = cluster_tfidf.fit_transform([all_text])
                    feature_names = cluster_tfidf.get_feature_names_out()
                    scores = cluster_features.toarray()[0]
                    
                    # Get top terms
                    term_scores = list(zip(feature_names, scores))
                    term_scores.sort(key=lambda x: x[1], reverse=True)
                    cluster_info['key_business_terms'] = term_scores[:10]
                except:
                    cluster_info['key_business_terms'] = []
            else:
                cluster_info['key_business_terms'] = []
            
            # Sample companies
            cluster_info['sample_companies'] = cluster_data[['ticker', 'companyName', 'sector']].head(5).to_dict('records')
            
            analysis['clusters'][cluster_id] = cluster_info
        
        return analysis
    
    def get_cluster_summary(self, model_name: str = 'kmeans') -> str:
        """Get a human-readable summary of clusters"""
        analysis = self.analyze_clusters(model_name)
        
        summary = f"\n=== Cluster Analysis ({model_name.upper()}) ===\n"
        summary += f"Total clusters: {len(analysis['clusters'])}\n\n"
        
        for cluster_id, info in analysis['clusters'].items():
            summary += f"CLUSTER {cluster_id} ({info['size']} companies, {info['percentage']:.1f}%):\n"
            summary += f"  Avg Market Cap: ${info['avg_market_cap']:,.0f}\n"
            summary += f"  Avg Employees: {info['avg_employees']:,.0f}\n"
            
            summary += f"  Top Sectors: {', '.join([f'{k}({v})' for k, v in info['top_sectors'].items()])}\n"
            
            summary += f"  Key Business Terms: {', '.join([term for term, score in info['key_business_terms'][:5]])}\n"
            
            summary += "  Sample Companies:\n"
            for company in info['sample_companies']:
                summary += f"    - {company['ticker']}: {company['companyName']} ({company['sector']})\n"
            summary += "\n"
        
        return summary
    
    def predict_cluster(self, business_summary: str, model_name: str = 'kmeans') -> int:
        """Predict cluster for a new business summary"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Clean the business summary
        cleaned_summary = self.clean_business_summary(business_summary)
        
        # Transform using existing TF-IDF vectorizer
        tfidf_features = self.tfidf_vectorizer.transform([cleaned_summary]).toarray()
        
        # Create dummy supplementary features (use mean values)
        dummy_numerical = np.array([[0, 0]])  # Will be scaled to mean
        dummy_categorical = np.zeros((1, 2))   # Assume 2 categorical features
        
        # Combine features with same weighting as training
        text_weight = 0.7
        X_text_weighted = tfidf_features * text_weight
        X_supp_weighted = np.hstack([dummy_numerical, dummy_categorical]) * (1 - text_weight)
        X = np.hstack([X_text_weighted, X_supp_weighted])
        
        # Predict
        model = self.models[model_name]['model']
        if hasattr(model, 'predict'):
            return model.predict(X)[0]
        else:
            # For models without predict method, find closest cluster center
            return 0  # Placeholder
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        import pickle
        
        model_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'models': self.models,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")


# Example usage and training script
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    
    import json

    with open('company_descriptions.json', 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # Initialize and train the model
    model = StockClusteringModel(random_state=42)
    
    # Load data
    df = model.load_data(sample_data)
    
    # Preprocess with 80% weight on business summaries
    X = model.preprocess_data(df, text_weight=0.8)
    
    # Train models
    results = model.train_models(X, n_clusters=3)  # Or let it auto-determine
    
    # Analyze results
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Clusters: {result['n_clusters']}")
        print(f"  Silhouette Score: {result['silhouette_score']:.3f}")
    
    # Get detailed cluster analysis
    print(model.get_cluster_summary('kmeans'))
    
    # Save the model
    # model.save_model('stock_clustering_model.pkl')
    
    # Example prediction for new business summary
    new_summary = "A technology company that develops software solutions for enterprise customers."
    predicted_cluster = model.predict_cluster(new_summary)
    print(f"Predicted cluster for new company: {predicted_cluster}")