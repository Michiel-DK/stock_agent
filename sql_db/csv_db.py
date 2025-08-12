from langchain.sql_database import SQLDatabase
from langchain.tools import Tool
import sqlite3
import pandas as pd

class FinancialDatabaseTool:
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    def query(self, sql_query: str) -> str:
        """Execute SQL query on financial database"""
        try:
            result = self.db.run(sql_query)
            if not result:
                return "No results found for the query."
            return str(result)
        except Exception as e:
            return f"SQL Error: {str(e)}. Check your query syntax."
    
    def get_schema(self) -> str:
        """Get database schema information"""
        return self.db.get_table_info()
    
    def add_csv_to_db(self, csv_path: str, table_name: str):
        """
        Add CSV file to SQLite database
        
        Args:
            csv_path: Path to CSV file
            table_name: Name for the database table
            db_path: Path to database file
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Connect to database and add table
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"Added {len(df)} rows to table '{table_name}'")

    def create_financial_tables(self):
        """Create tables for stock analysis data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stock analysis table
        stock_analysis_sql = """
        CREATE TABLE IF NOT EXISTS stock_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            cluster_price INTEGER,
            method TEXT,
            avg_daily_return REAL,
            total_return REAL,
            volatility REAL,
            sharpe_ratio REAL,
            rank INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Cluster summary table
        cluster_summary_sql = """
        CREATE TABLE IF NOT EXISTS cluster_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_price INTEGER NOT NULL,
            count INTEGER,
            avg_return REAL,
            avg_volatility REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Company descriptions table
        company_descriptions_sql = """
        CREATE TABLE IF NOT EXISTS company_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            description TEXT,
            cluster_description INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Cluster similarity table
        cluster_similarity_sql = """
        CREATE TABLE IF NOT EXISTS cluster_similarity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_i INTEGER NOT NULL,
            cluster_j INTEGER NOT NULL,
            centroid_distance REAL,
            cosine_similarity REAL,
            min_distance REAL,
            max_distance REAL,
            avg_distance REAL,
            normalized_distance REAL,
            weighted_distance REAL,
            size_ratio REAL,
            combined_size INTEGER,
            variance_ratio REAL,
            combined_variance REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(stock_analysis_sql)
        cursor.execute(cluster_summary_sql)
        cursor.execute(company_descriptions_sql)
        cursor.execute(cluster_similarity_sql)
        conn.commit()
        conn.close()
        
        
        print("Created tables: stock_analysis, cluster_summary, company_descriptions")

    def setup_database(self, csv_files: dict):
        """
        Setup database with tables and load CSV data
        
        Args:
            csv_files: Dictionary with table_name -> csv_path mapping
            db_path: Path to database file
        """
        # Create tables
        self.create_financial_tables()
        
        # Load CSV files
        for table_name, csv_path in csv_files.items():
            self.add_csv_to_db(csv_path, table_name)

        print(f"Database setup complete: {self.db_path}")
        
    def get_similar_clusters_sql(self, cluster_id: int, top_k: int = 5) -> str:
        """Helper method to generate SQL for finding similar clusters"""
        return f"""
        SELECT 
            CASE 
                WHEN cluster_i = {cluster_id} THEN cluster_j 
                ELSE cluster_i 
            END as similar_cluster,
            cosine_similarity,
            centroid_distance,
            size_ratio
        FROM cluster_similarity 
        WHERE cluster_i = {cluster_id} OR cluster_j = {cluster_id}
        ORDER BY cosine_similarity DESC 
        LIMIT {top_k};
        """

    def get_top_similar_pairs_sql(self, top_k: int = 5) -> str:
        """Helper method to generate SQL for top similar cluster pairs"""
        return f"""
        SELECT 
            cluster_i,
            cluster_j,
            cosine_similarity,
            centroid_distance,
            combined_size
        FROM cluster_similarity 
        ORDER BY cosine_similarity DESC 
        LIMIT {top_k};
        """

if __name__ == "__main__":
    # Example usage
    db_tool = FinancialDatabaseTool()
    
    # Setup database with your CSV files
    csv_files = {
        "stock_analysis": "output_data/optimal_clustering_timeserieskmeans_euclidean_stock_ranking_20250804_224954.csv",
        "cluster_summary": "output_data/optimal_clustering_timeserieskmeans_euclidean_performance_20250804_224954.csv", 
        "company_descriptions": "description_data/finbert_cluster_results_company_descriptions_cluster_timeserieskmeans_euclidean_stock_ranking_20250804_224954_full.csv",
        "cluster_similarity": "description_data/cluster_similarity_features_company_descriptions_cluster_timeserieskmeans_euclidean_stock_ranking_20250804_224954_full_new.csv",
    }
    #db_tool.setup_database(csv_files)
    
    # Example query
    query_result = db_tool.query("SELECT * FROM stock_analysis LIMIT 5;")
    print(query_result)
    