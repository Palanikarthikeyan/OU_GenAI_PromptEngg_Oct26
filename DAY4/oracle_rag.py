from langchain_community.document_loaders import TextLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OracleVS # old version
# from langchain_oracledb.vectorstores.oraclevs import OracleVS # updated version
from langchain_community.vectorstores.utils import DistanceStrategy
import oracledb


# --------------------------------------------------
#  SETUP FUNCTION — run once to create the vector store
# --------------------------------------------------
def setup_oracle_vectorstore():
    # Load and split documents
    loader = TextLoader("my_docs.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Embedding model (Ollama must be running!)
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    # Oracle client pool
    client = oracledb.create_pool(
        user="student",
        password="apelix",
        dsn="localhost:1521/FREEPDB1",
        min=1,
        max=5,
        increment=1,
    )

    # Create vector store
    vector_store = OracleVS.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        table_name="LANGCHAIN_VSTORE",
        distance_strategy=DistanceStrategy.COSINE
    )

    print("Oracle Vector Store setup complete.")
    return vector_store


# --------------------------------------------------
#  SEARCH FUNCTION — call this repeatedly for queries
# --------------------------------------------------
def search_oracle_vector(vector_store, query, k=3, threshold=0.4):
    results = vector_store.similarity_search_with_score(query, k=k)
    print("-->",results)
    print("\n")
    # Filter by threshold score
    filtered = [r for r, score in results if score >= threshold]
    print(filtered)
    '''
    if not filtered:
        print(f" No relevant match found for query: '{query}'")
    else:
        print(f" Found {len(filtered)} relevant result(s):\n")
        for r in filtered:
            print(r.page_content[:200], "\n---\n")
    '''

# --------------------------------------------------
#  MAIN — Example usage
# --------------------------------------------------
if __name__ == "__main__":
    vs = setup_oracle_vectorstore()

    # Example queries
    search_oracle_vector(vs, "what is langchain?")
    search_oracle_vector(vs, "@&^$&#%%^")
