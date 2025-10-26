
# Graph RAG

Graph RAG is a framework for building and querying knowledge graphs from unstructured documents using Retrieval-Augmented Generation (RAG) techniques. It leverages document embeddings, named entity recognition, and graph-based retrieval to enable advanced semantic search and context retrieval for LLMs.

## Features
- Extracts concepts/entities from documents using GLiNER
- Generates document embeddings (supports LangChain embedding models)
- Builds a kNN knowledge graph with concept-enhanced edge weights
- Graph-based retrieval with semantic and centrality-aware ranking
- Integrates with LangChain for RetrievalQA pipelines


## Installation

```bash
uv venv --python 3.12
uv sync
```

## Usage Example

Below is a minimal example of how to use Graph RAG to build a knowledge graph and perform retrieval:

```python
from graph_rag import KnowledgeGraph, GraphRAGRetriever
from langchain_google_vertexai import VertexAIEmbeddings

# 1. Prepare your documents (list of strings)
docs = ["Document 1 text...", "Document 2 text...", ...]

# 2. Create embeddings
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")
embeddings = embedding_model.embed_documents(docs)

# 3. Extract concepts/entities
kg = KnowledgeGraph()
concepts_per_chunk = kg.extract_concepts_batch(docs)

# 4. Build the knowledge graph
G = kg.create_graph(
	embeddings=embeddings,
	docs=docs,
	concepts_per_chunk=concepts_per_chunk,
	k=5,
	metric="IP",
	save_dir="./data/graph",
	similarity_threshold=0.5,
	concept_weight=0.3,
)

# 5. Create a retriever
retriever = GraphRAGRetriever(
	kg=kg,
	embedding_model=embedding_model,
	top_k=5,
	k_hops=1,
)

# 6. Retrieve relevant documents for a query
query = "How can I configure the Actionbar in Allplan?"
docs = retriever.get_relevant_documents(query)
for d in docs:
	print(d.metadata, d.page_content[:100])
```

You can also integrate the retriever with LangChain's `RetrievalQA` for end-to-end question answering.

---
