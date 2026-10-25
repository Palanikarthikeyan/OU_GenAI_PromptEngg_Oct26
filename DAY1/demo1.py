import cohere

co = cohere.ClientV2(
    api_key="YOUR_COHERE_API_KEY"
)

query = "How do I handle exceptions in Python?"

documents = [
    "Python functions help organize reusable code.",
    "Use try and except blocks to handle exceptions.",
    "Python classes support object-oriented programming.",
    "Use finally to execute cleanup code."
]

response = co.rerank(
    model="rerank-v4.0-fast",
    query=query,
    documents=documents,
    top_n=3
)

for result in response.results:
    print(
        result.relevance_score,
        documents[result.index]
    )
