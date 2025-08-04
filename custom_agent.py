from langgraph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

# Define the state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the agent node
def call_model(state: State):
    messages = state["messages"]
    
    # Add system message if not already present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the workflow
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Compile the agent
agent_executor = workflow.compile(checkpointer=memory)