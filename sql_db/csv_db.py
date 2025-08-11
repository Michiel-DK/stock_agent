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
        
        cursor.execute(stock_analysis_sql)
        cursor.execute(cluster_summary_sql)
        cursor.execute(company_descriptions_sql)
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


# Initialize database tool
db_tool = FinancialDatabaseTool()

# Setup database with your CSV files
csv_files = {
    "stock_analysis": "stock_data.csv",
    "cluster_summary": "cluster_data.csv", 
    "company_descriptions": "company_descriptions.csv"
}
db_tool.setup_database(csv_files)

financial_query_tool = Tool(
    name="sql_query",
    description="""Execute SQL queries on financial database. 
    Contains tables: 
    - stock_analysis: ticker, cluster_price, method, avg_daily_return, total_return, volatility, sharpe_ratio, rank
    - cluster_summary: cluster_price, count, avg_return, avg_volatility, sharpe_ratio, max_drawdown, method  
    - company_descriptions: ticker, description, cluster_description
    
    Note: cluster_price (price-based clustering) and cluster_description (description-based clustering) are different clustering methods.
    
    Examples: 
    - SELECT * FROM stock_analysis WHERE ticker='AAPL' ORDER BY rank LIMIT 5
    - SELECT sa.ticker, sa.sharpe_ratio, cs.avg_return FROM stock_analysis sa JOIN cluster_summary cs ON sa.cluster_price = cs.cluster_price WHERE sa.ticker='AAPL'
    - SELECT sa.ticker, sa.cluster_price, cd.cluster_description FROM stock_analysis sa JOIN company_descriptions cd ON sa.ticker = cd.ticker WHERE sa.ticker='AAPL'
    - SELECT sa.*, cs.max_drawdown FROM stock_analysis sa LEFT JOIN cluster_summary cs ON sa.cluster_price = cs.cluster_price ORDER BY sa.rank LIMIT 10""",
    func=db_tool.query
)

schema_tool = Tool(
    name="database_info", 
    description="Get information about database tables and their columns",
    func=lambda x: db_tool.get_schema()
)
