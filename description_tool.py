from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import json


def init_chat_model(model_name):    

    with open('company_descriptions.json', 'r', encoding='utf-8') as f:
            js = json.load(f)

    retrieved_docs = [doc['businessSummary'] for doc in js]
    

    # Initialize the model
    llm = ChatGoogleGenerativeAI(model=model_name)

    # Create your prompt
    company_descriptions = '\n\n'.join([f"Company {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""You are an expert financial analyst specializing in sector classification and company clustering.

    TASK: Analyze the following company descriptions and group them into meaningful clusters based on their business models, industries, and operational similarities.

    COMPANY DESCRIPTIONS:
    {company_descriptions}

    OUTPUT FORMAT: Please provide your analysis in the following JSON structure:
    {{
        "clusters": [
            {{
                "cluster_name": "Descriptive name for the cluster",
                "theme": "Brief explanation of what unites these companies",
                "companies": ["AAPL", "MSFT", "GOOGL"],
                "reasoning": "Why these companies belong together"
            }}
        ],
        "summary": "Overall analysis of the clustering patterns and key insights"
    }}

    GUIDELINES:
    - Aim for 3-7 clusters depending on the data
    - Each cluster should have clear business/industry logic
    - Consider factors like: industry sector, business model, target market, technology focus
    - Provide clear reasoning for each grouping"""

    # Use LangChain's invoke method
    response = llm.invoke(prompt)
    
        # Save the response content to a JSON file
    output_data = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "clustering_analysis": response.content
    }
    
    with open('clustering_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return response.content


if __name__ == "__main__":
    model_name = "gemini-2.0-flash"
    response = init_chat_model(model_name)
    print(response)