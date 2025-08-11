import yfinance as yf
import json
from datetime import datetime
import time

def fetch_company_descriptions(tickers):
    """
    Fetch company descriptions for a list of tickers using yfinance
    
    Args:
        tickers (list): List of stock ticker symbols
    
    Returns:
        list: List of dictionaries ready for MongoDB insertion
    """
    company_data = []
    
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Get company info
            info = stock.info
            
            # Extract relevant information for MongoDB
            company_doc = {
                "ticker": ticker.upper(),
                "companyName": info.get("longName", "N/A"),
                "shortName": info.get("shortName", "N/A"),
                "businessSummary": info.get("longBusinessSummary", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "website": info.get("website", "N/A"),
                "country": info.get("country", "N/A"),
                "city": info.get("city", "N/A"),
                "state": info.get("state", "N/A"),
                "marketCap": info.get("marketCap", None),
                "employeeCount": info.get("fullTimeEmployees", None),
                "exchange": info.get("exchange", "N/A"),
                "currency": info.get("currency", "N/A"),
                "fetchedAt": datetime.utcnow().isoformat(),
                "dataSource": "yfinance"
            }
            
            company_data.append(company_doc)
            
            # Add a small delay to be respectful to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            # Add error document to maintain data integrity
            error_doc = {
                "ticker": ticker.upper(),
                "error": str(e),
                "fetchedAt": datetime.utcnow().isoformat(),
                "dataSource": "yfinance"
            }
            company_data.append(error_doc)
    
    return company_data

def save_to_json(data, filename="description_data/company_descriptions.json"):
    """
    Save company data to JSON file
    
    Args:
        data (list): List of company dictionaries
        filename (str): Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
        print(f"Total records: {len(data)}")
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")

def main():
    
    #cluster = 6
    cluster = [10, 0, 19, 8]
    
    import pandas as pd
    # Example list of tickers - replace with your actual list
    #tickers = pd.read_csv('optimal_clustering_results_20250729_155103.csv')
    tickers = pd.read_csv('output_data/optimal_clustering_timeserieskmeans_euclidean_stock_ranking_20250804_224954.csv')
    
    tickers = tickers[tickers['Cluster'].isin(cluster)]['Ticker'].tolist()
    
    print("Starting to fetch company descriptions...")
    print(f"Processing {len(tickers)} tickers")
    
    # Fetch company data
    company_data = fetch_company_descriptions(tickers)
    
    # Save to JSON file
    save_to_json(company_data, filename=f"description_data/company_descriptions_cluster_timeserieskmeans_euclidean_stock_ranking_20250804_224954.json")

    # Optional: Print sample data
    if company_data:
        print("\nSample record:")
        print(json.dumps(company_data[0], indent=2))

if __name__ == "__main__":
    main()

# Alternative function for direct MongoDB insertion (requires pymongo)
def insert_to_mongodb(data, db_name="finance_db", collection_name="companies"):
    """
    Optional function to directly insert data into MongoDB
    Uncomment and modify as needed
    """
    # from pymongo import MongoClient
    # 
    # try:
    #     client = MongoClient('mongodb://localhost:27017/')  # Adjust connection string
    #     db = client[db_name]
    #     collection = db[collection_name]
    #     
    #     # Insert documents
    #     result = collection.insert_many(data)
    #     print(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
    #     
    #     client.close()
    # except Exception as e:
    #     print(f"Error inserting to MongoDB: {str(e)}")

# Usage with custom ticker list:
# tickers = ["YOUR", "TICKER", "LIST", "HERE"]
# company_data = fetch_company_descriptions(tickers)
# save_to_json(company_data, "my_companies.json")