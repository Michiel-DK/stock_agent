"""My Personal Agent
"""

import os
from datetime import datetime
from dotenv import load_dotenv

## Import necessary libraries
# $CHALLENGIFY_BEGIN
from langchain.chat_models import init_chat_model

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from news import get_extra_info
from sql_db.csv_db import FinancialDatabaseTool
from description_clustering.clusterlookup import ClusterLookupTool

import json

# $CHALLENGIFY_END

## Configurations
ABORT_VALUES = ("quit", "exit", "quit()", "exit()")
load_dotenv()  # Load environment variables (API keys)
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

## Instantiate all the tools

# Get the current date
# $CHALLENGIFY_BEGIN
@tool
def get_today() -> str:
    """Get today's date."""
    return datetime.today().strftime("%Y-%m-%d")
# $CHALLENGIFY_END

@tool
def get_news(ticker: str) -> list:
    """Fetch recent new articles for a list of tickers."""
    return get_extra_info(ticker)

# Initialize database tool (you might want to do this once globally)
db_tool = FinancialDatabaseTool()

@tool
def sql_query(query: str) -> str:
    """Execute SQL queries on financial database. 
    Contains tables: 
    - stock_analysis: ticker, cluster_price, method, avg_daily_return, total_return, volatility, sharpe_ratio, rank
    - cluster_summary: cluster_price, count, avg_return, avg_volatility, sharpe_ratio, max_drawdown, method  
    - company_descriptions: ticker, description, cluster_description
    
    Note: cluster_price (price-based clustering) and cluster_description (description-based clustering) are different clustering methods.
    Examples: 
        - SELECT * FROM stock_analysis WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
        - SELECT ticker, cluster_price FROM stock_analysis WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
        - SELECT ticker, description, cluster_description FROM company_descriptions WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
        - SELECT sa.ticker, sa.cluster_price, cs.avg_return, cs.max_drawdown FROM stock_analysis sa JOIN cluster_summary cs ON sa.cluster_price = cs.cluster_price WHERE sa.ticker IN ('AAPL', 'MSFT')
        - SELECT sa.ticker, sa.cluster_price, cd.cluster_description FROM stock_analysis sa JOIN company_descriptions cd ON sa.ticker = cd.ticker WHERE sa.ticker IN ('AAPL', 'MSFT', 'GOOGL')"""
    return db_tool.query(query)

@tool
def list_columns(table_name: str) -> str:
    """List all column names for a specific table."""
    try:
        result = db_tool.db.run(f"PRAGMA table_info({table_name})")
        return f"Columns in {table_name}: {result}"
    except Exception as e:
        return f"Error getting columns for {table_name}: {str(e)}"

@tool
def show_sample_data(table_name: str, limit: int = 3) -> str:
    """Show sample data from a table to see column names and data types."""
    try:
        result = db_tool.db.run(f"SELECT * FROM {table_name} LIMIT {limit}")
        return f"Sample data from {table_name}:\n{result}"
    except Exception as e:
        return f"Error getting sample data: {str(e)}"
    
@tool
def find_similar_clusters(cluster_id: int, top_k: int = 3):
    """Find clusters similar to cluster_id"""
    tool = ClusterLookupTool()
    return tool.get_most_similar_to_cluster(cluster_id, top_k)

@tool
def get_most_similar_pairs(top_k: int = 5):
    """Get most similar cluster pairs"""
    tool = ClusterLookupTool()
    return tool.get_top_similar_pairs(top_k)

@tool
def check_clusters(cluster_1: int, cluster_2: int):
    """Check if two clusters are similar"""
    tool = ClusterLookupTool()
    return tool.check_similarity(cluster_1, cluster_2)

# Requests tool
requests_tool = RequestsGetTool(
    requests_wrapper=TextRequestsWrapper(headers={}), allow_dangerous_requests=True
)
# $CHALLENGIFY_END

@tool
def cluster_companies_by_news(tickers: str) -> str:
    """
    Cluster companies based on their recent news articles and business activities.
    
    Args:
        tickers: Comma-separated list of company tickers (e.g. "AAPL,MSFT,GOOGL")
    
    Returns:
        JSON string with clusters based on recent news analysis
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    
    # Fetch news for all tickers
    all_news = {}
    for ticker in ticker_list:
        try:
            all_news[ticker] = get_news.invoke({"ticker": ticker})
        except Exception as e:
            all_news[ticker] = f"Error fetching news: {str(e)}"
    
    # Create clustering prompt
    prompt = f"""You are an expert financial analyst. Cluster these companies based on their recent news and business activities.

        COMPANY NEWS DATA:
        {json.dumps(all_news, indent=2)}

        OUTPUT FORMAT:
        {{
            "clusters": [
                {{
                    "cluster_name": "Descriptive name",
                    "theme": "What unites these companies based on news",
                    "companies": ["AAPL", "MSFT"],
                    "reasoning": "Why grouped together based on recent activities"
                }}
            ],
            "summary": "Key insights from news-based clustering"
        }}

        Focus on: recent business moves, industry trends, partnerships, product launches, market focus."""

    try:
        # Use the same model that's used by the agent
        response = model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return json.dumps({"error": f"Clustering failed: {str(e)}", "tickers": ticker_list})

# Wikipedia tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def cluster_companies_by_wikipedia(tickers: str) -> str:
    """
    Cluster companies based on their Wikipedia information and business descriptions.
    
    Args:
        tickers: Comma-separated list of company tickers (e.g. "AAPL,MSFT,GOOGL")
    
    Returns:
        JSON string with clusters based on Wikipedia business analysis
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    
    # Fetch Wikipedia info for all tickers
    all_wiki_info = {}
    for ticker in ticker_list:
        try:
            # Search for company name + ticker to get better results
            search_query = f"{ticker} company stock"
            all_wiki_info[ticker] = wikipedia_tool.invoke({"query": search_query})
        except Exception as e:
            all_wiki_info[ticker] = f"Error fetching Wikipedia info: {str(e)}"
    
    # Create clustering prompt
    prompt = f"""You are an expert financial analyst. Cluster these companies based on their Wikipedia business information and descriptions.

COMPANY WIKIPEDIA DATA:
{json.dumps(all_wiki_info, indent=2)}

OUTPUT FORMAT:
{{
    "clusters": [
        {{
            "cluster_name": "Descriptive name",
            "theme": "What unites these companies based on Wikipedia info",
            "companies": ["AAPL", "MSFT"],
            "reasoning": "Why grouped together based on business models and industries"
        }}
    ],
    "summary": "Key insights from Wikipedia-based clustering"
}}

Focus on: industry sectors, business models, target markets, company size, geographic focus, founding history."""

    try:
        response = model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return json.dumps({"error": f"Clustering failed: {str(e)}", "tickers": ticker_list})

## Instantiate the Model
model = init_chat_model(
    "gemini-2.0-flash",
    model_provider="google_genai",
    # api_key=GEMINI_API_KEY
)
# $CHALLENGIFY_END

## Instantiate the agent
# $CHALLENGIFY_BEGIN
tools = [
    requests_tool,
    wikipedia_tool,
    get_news,
    sql_query,
    get_today,
    list_columns,
    show_sample_data,
    find_similar_clusters,
    get_most_similar_pairs,
    check_clusters,
    cluster_companies_by_news,
    cluster_companies_by_wikipedia
]
memory = MemorySaver()

# Set the system prompt
SYSTEM_PROMPT = """
You are a financial assistant that helps users with stock market queries.

You can:
- Look up news articles based on a ticker provided
- Cluster companies based on these news articles
- Query financial database for stock analysis, cluster data, and company descriptions  
- Use Wikipedia for general company information
- Use Wikipedia company descriptions to cluster companies based on business activities
- Find similar clusters using dedicated tools OR SQL queries

Database workflow:
1. ALWAYS use database_info(), list_columns(), or show_sample_data() FIRST to see exact column names
2. Then write SQL queries using the correct column names
3. Use sql_query() to execute your queries

Available database tables:
- stock_analysis: contains ticker performance and price-based clustering
- cluster_summary: contains price cluster statistics  
- company_descriptions: contains company info and description-based clustering
- cluster_similarity: contains pre-computed similarity scores between all cluster pairs

For cluster similarity queries, you can either:
- Use dedicated tools: find_similar_clusters(), get_most_similar_pairs(), check_clusters()
- Or write SQL queries against the cluster_similarity table
- Use recent news articles to cluster companies

IMPORTANT: Column names must be exact - check them before querying to avoid errors.
"""

agent_executor = create_react_agent(
    model, tools, checkpointer=memory, prompt=SYSTEM_PROMPT
)
# $CHALLENGIFY_END


def use_agent(user_message, thread_id="abc123"):
    """Use the agent to get a response.
    Returns the response from the agent.
    """
    # $CHALLENGIFY_BEGIN
    # Set the configuration for the agent
    config = {"configurable": {"thread_id": thread_id}}

    # Set the system message in the agent executor
    # Use the agent to get a response
    messages = [HumanMessage(content=user_message)]
    if DEBUG:
        for step in agent_executor.stream(
            {"messages": messages},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()
        return "Finished"
    response = agent_executor.invoke({"messages": messages}, config=config)
    return response["messages"][-1].content
    # $CHALLENGIFY_END


def main():
    """Main loop of the program
    """
    print("\nWelcome! Type your questions below. Use `quit` or `exit` to stop.")
    print("\n> ", end="")

    while (query := input()).lower() not in ABORT_VALUES:
        print(use_agent(query))
        print("\n> ", end="")


if __name__ == "__main__":
    main()

