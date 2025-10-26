# GraphRAG implementation

<p align="right">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Alpha-orange.svg" alt="Status">
</p>

**GraphRAG** is a modern framework for building and querying knowledge graphs from unstructured documents using Retrieval-Augmented Generation (RAG) techniques. It combines state-of-the-art document embeddings, named entity recognition, and graph-based retrieval to enable advanced semantic search and context retrieval for large language models (LLMs).

---

## 🚀 Key Features

- **Graph Construction:** Built on [networkx](https://networkx.org/) for flexible, in-memory knowledge graph creation and manipulation.
- **Named Entity Recognition:** Extracts concepts/entities using [GLiNER](https://github.com/urchade/gliner) for robust NER.
- **Semantic Search:** Uses in-memory [FAISS](https://github.com/facebookresearch/faiss) for fast semantic similarity search (swap for a production vector database as needed).
- **LangChain Integration:** Includes a LangChain-compatible retriever wrapper for seamless integration with [LangChain](https://python.langchain.com/) RetrievalQA pipelines.
- **Production-Ready Guidance:** For large-scale or production deployments, migrate to a graph database such as [Neo4j](https://neo4j.com/) and a dedicated vector database for scalable, persistent storage and search.

---

## 📦 Installation

```bash
uv venv --python 3.12
uv sync
```

---

## 📝 Usage Example

Below is a minimal example of how to use GraphRAG to build a knowledge graph and perform retrieval:

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

You can also integrate the retriever with LangChain's `RetrievalQA` for end-to-end question answering:

```python
from langchain.chains import RetrievalQA
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0.1, thinking_budget=0)

qa_chain = RetrievalQA.from_chain_type(
	llm=llm,
	retriever=retriever,
)
response = qa_chain.invoke({"query": "How do I configure the Actionbar in Allplan?"})
print(response["result"])
```

---

## 📚 References

- [networkx documentation](https://networkx.org/)
- [GLiNER: Generalist Named Entity Recognizer](https://github.com/urchade/gliner)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [LangChain documentation](https://python.langchain.com/)
- [Neo4j: Graph Database](https://neo4j.com/)

---

## 🛡️ License

This project is licensed under the MIT License.

---
