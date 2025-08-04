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
    """Fetch company descriptions for a list of tickers."""
    return get_extra_info(ticker)

# Polygon API
# $CHALLENGIFY_BEGIN
# polygon = PolygonAPIWrapper()
# polygon_toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)
# $CHALLENGIFY_END

# Requests tool
# $CHALLENGIFY_BEGIN
requests_tool = RequestsGetTool(
    requests_wrapper=TextRequestsWrapper(headers={}), allow_dangerous_requests=True
)
# $CHALLENGIFY_END

# Wikipedia tool
# $CHALLENGIFY_BEGIN
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# $CHALLENGIFY_END

## Instantiate the Model
# $CHALLENGIFY_BEGIN
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
    get_news
]
memory = MemorySaver()

# Set the system prompt
SYSTEM_PROMPT = """
    You are a financial assistant that helps users with stock market queries.
    You can look up news articles based on a ticker provided. Use these to answer questions about stocks.
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
