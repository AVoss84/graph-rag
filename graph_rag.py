import os
from typing import Callable, Dict, List, Optional, Tuple, Any
from pydantic import PrivateAttr
import warnings
from tqdm import tqdm
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gliner import GLiNER
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from custom_logger import LoggerFactory


logger = LoggerFactory().create_module_logger()


class KnowledgeGraph:

    default_entity_groups: List[str] = [
        "Software Product / Component",
        "Trademark / Format",
        "Operating System / Platform",
        "Hardware Requirement",
        "Installation / Licensing Entity",
        "User Interface Entity",
        "Palette",
        "Library Element Type",
        "Data",
        "Interchange / Standard",
        "Workflow / Tool",
        "Web Resource",
    ]

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if self.verbose:
            logger.info("Initializing KnowledgeGraph 📈")
            logger.info("Loading GLiNER NER model...")
        try:
            self.ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            self.ner_model = None

    def extract_concepts_batch(
        self,
        docs: List[str],
        entity_groups: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> List[List[str]]:
        """
        Extract named entities (concepts) from a batch of documents using NER model.

        Processes documents in batches to extract named entities based on specified
        entity groups. Suppresses GLiNER truncation warnings during processing.

        Args:
            docs (List[str]): List of document strings to process for entity extraction.
            entity_groups (Optional[List[str]], optional): List of entity group names
                to extract. If None, uses default_entity_groups. Defaults to None.
            batch_size (int, optional): Number of documents to process per batch
                iteration. Defaults to 100.

        Returns:
            List[List[str]]: List containing extracted concepts for each document.
                Each inner list contains the entity text strings found in the
                corresponding document.

        Note:
            Uses a threshold of 0.5 for entity prediction confidence. GLiNER
            truncation warnings are temporarily suppressed during processing.
        """
        if entity_groups is None:
            entity_groups = self.default_entity_groups

        all_concepts = []

        if self.ner_model is None:
            return [[] for _ in docs]

        # Temporarily suppress GLiNER truncation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Sentence of length .* has been truncated to .*"
            )

            for i in tqdm(
                range(0, len(docs), batch_size),
                desc="Extracting concepts / NER",
                disable=not self.verbose,
            ):
                batch = docs[i : i + batch_size]
                for doc in batch:
                    entities = self.ner_model.predict_entities(
                        doc, entity_groups, threshold=0.5
                    )
                    concepts = [entity["text"] for entity in entities]
                    all_concepts.append(concepts)

        return all_concepts

    def create_adjacency_list(
        self,
        embeddings: np.ndarray,
        k: int = 10,
        metric: str = "L2",
        save_dir: Optional[str] = None,
        filename: Optional[str] = "graph_adjlist.txt",
        threshold: Optional[float] = None,
    ) -> Tuple[List[float], float]:
        """
        Generate kNN adjacency list using FAISS for NetworkX graph construction.

        For metric="IP" (cosine similarity):
        - Higher values = more similar
        - Edge weight in graph = similarity score
        - Threshold: minimum similarity required (e.g., 0.5)

        For metric="L2" (Euclidean distance):
        - Lower values = more similar
        - Edge weight in graph = 1 / (1 + distance) to convert to similarity
        - Threshold: maximum distance allowed (e.g., 2.0)
        """
        assert isinstance(embeddings, np.ndarray), "Embeddings must be a numpy array."
        X = embeddings.astype("float32")  # keep it small
        n_items, dim = X.shape

        if save_dir:
            logger.info(f"Saving adjacency list to directory: {save_dir}")

        if self.verbose:
            logger.info(f"Building kNN graph: {n_items} nodes, k={k}, metric={metric}")
            if threshold is not None:
                thresh_str = f"<={threshold}" if metric == "L2" else f">={threshold}"
                print(f"  Applying threshold: {metric} {thresh_str}")

        # Create FAISS index
        self.build_faiss_index(embeddings=X, metric=metric)

        X_indexed = self.embeddings_indexed

        # Build adjacency list:
        # ----------------------
        distances = []
        edge_count = 0
        filtered_count = 0

        if save_dir and filename:
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)

        with open(filepath, "w") as f:
            for i in tqdm(
                range(n_items), desc="Building edges", disable=not self.verbose
            ):
                # Find k+1 neighbors (including self)
                dists, neighbors = self.index.search(X_indexed[i : i + 1], k + 1)

                # Skip self (first neighbor) and add edges
                for j in range(1, len(neighbors[0])):
                    if neighbors[0][j] >= 0:  # Valid neighbor
                        dist = dists[0][j]

                        # Apply threshold if specified
                        include_edge = True
                        if threshold is not None:
                            if metric == "L2":
                                include_edge = dist <= threshold
                            else:  # IP (cosine similarity)
                                include_edge = dist >= threshold

                            if not include_edge:
                                filtered_count += 1
                                continue

                        # Convert distance to weight for NetworkX
                        if metric == "L2":
                            # Convert L2 distance to similarity-like weight
                            # Closer nodes get higher weights
                            weight = 1.0 / (1.0 + dist)
                        else:  # IP
                            # Already a similarity score [0, 1]
                            weight = dist

                        if include_edge:
                            # if save_dir and filename:
                            #     with open(filepath, "w") as f:
                            # Write as: node1 node2 weight
                            f.write(f"{i} {neighbors[0][j]} {weight:.6f}\n")
                            distances.append(dist)
                            edge_count += 1

        # Calculate final threshold:
        if threshold is not None:
            final_threshold = threshold
        elif distances:
            final_threshold = np.mean(distances)
        else:
            final_threshold = 0

        if self.verbose:
            print(f"✓ Created {edge_count} edges")
            if threshold is not None and filtered_count > 0:
                print(f"  Filtered out {filtered_count} edges by threshold")
            print(f"  Saved to: {filepath}")
            if distances:
                print(
                    f"  Distance stats: mean={np.mean(distances):.3f}, std={np.std(distances):.3f}"
                )
                print(
                    f"  Distance range: [{np.min(distances):.3f}, {np.max(distances):.3f}]"
                )

        return distances, final_threshold

    def create_graph(
        self,
        embeddings: np.ndarray,
        docs: List[str],
        concepts_per_chunk: List[List[str]],
        k: int = 10,
        metric: str = "IP",
        similarity_threshold: float = 0.5,
        concept_weight: float = 0.3,
        save_dir: str = None,
        with_concepts: bool = True,
    ) -> nx.Graph:
        """
        Creates a graph from document embeddings, optionally enhancing edge weights with concept overlap.

        This method constructs a graph where nodes represent documents and edges represent similarity between document embeddings. Optionally, it enhances edge weights based on the overlap of named entity recognition (NER) concepts between documents.

        Args:
            embeddings (np.ndarray): Array of document embeddings.
            docs (List[str]): List of document texts.
            concepts_per_chunk (List[List[str]]): List of NER concepts for each document.
            k (int, optional): Number of nearest neighbors for graph construction. Defaults to 10.
            metric (str, optional): Similarity metric to use ("IP" for inner product, etc.). Defaults to "IP".
            similarity_threshold (float, optional): Minimum similarity threshold for edge creation. Defaults to 0.5.
            concept_weight (float, optional): Weight for concept overlap in edge weight calculation (between 0 and 1). Defaults to 0.3.
            save_dir (str): Directory to save/load the adjacency list. If None, uses default package directory.
            with_concepts (bool, optional): Whether to enhance edge weights with concept overlap. Defaults to True.

        Returns:
            nx.Graph: A NetworkX graph with nodes representing documents and edges weighted by similarity (and optionally concept overlap).
        """
        # Create initial adjacency list based on embeddings
        # This will be saved to disk for loading into NetworkX
        self.create_adjacency_list(
            embeddings=embeddings,
            k=k,
            metric=metric,
            threshold=similarity_threshold,
            save_dir=save_dir,
        )

        # Load graph adjacency list
        filepath = os.path.join(save_dir, "graph_adjlist.txt")

        # Create NetworkX graph from adjacency list
        G = nx.read_weighted_edgelist(filepath, nodetype=int)

        # Without NER concepts, just add document texts as node attributes
        # -----------------------------------------------------------------
        if not with_concepts:

            # Add node attributes:
            for i, doc in enumerate(docs):
                if i in G.nodes():
                    G.nodes[i]["content"] = doc
                    G.nodes[i]["embedding"] = embeddings[i].tolist()

            if self.verbose:
                print("Graph topology without concept enhancement:")
                print(f"  Nodes: {G.number_of_nodes()}")
                print(f"  Edges: {G.number_of_edges()}")
                print(
                    f"  Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}"
                )
            return G

        # With NER concepts, enhance edge weights
        # ----------------------------------------
        # Add node attributes:
        for i, (doc, concepts) in enumerate(zip(docs, concepts_per_chunk)):
            if i in G.nodes():
                G.nodes[i]["content"] = doc
                G.nodes[i]["embedding"] = embeddings[i].tolist()
                G.nodes[i]["concepts"] = concepts

        # Enhance edge weights with concept overlap
        edges_to_update = []
        for u, v in G.edges():
            concepts_u = (
                set(concepts_per_chunk[u]) if u < len(concepts_per_chunk) else set()
            )
            concepts_v = (
                set(concepts_per_chunk[v]) if v < len(concepts_per_chunk) else set()
            )

            # Calculate concept overlap
            shared = concepts_u.intersection(concepts_v)
            max_concepts = max(len(concepts_u), len(concepts_v))
            overlap = len(shared) / max_concepts if max_concepts > 0 else 0

            # Get current similarity weight
            current_weight = G[u][v]["weight"]

            # Linearly combine similarity and concept overlap
            new_weight = (
                1 - concept_weight
            ) * current_weight + concept_weight * overlap

            edges_to_update.append((u, v, new_weight, list(shared)))

        # Update edges with new weights and shared concepts
        for u, v, weight, shared in edges_to_update:
            G[u][v]["weight"] = weight
            G[u][v]["shared_concepts"] = shared

        if self.verbose:
            print("Graph topology with concept enhancement:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            print(
                f"  Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}"
            )
        self.nx_graph = G
        return self.nx_graph

    def build_faiss_index(self, embeddings: np.ndarray, metric: str = "IP") -> None:
        """
        Build a FAISS index over node embeddings for fast semantic seed lookup.

        Args:
            embeddings: Array of shape (N, d) aligned with node ids 0..N-1
            metric: "IP" for cosine (vectors will be L2-normalized) or "L2"
        """
        self.metric = metric
        self.embeddings = embeddings.astype("float32")
        self.faiss_index_dim = self.embeddings.shape[1]

        if metric == "IP":
            self.index = faiss.IndexFlatIP(self.faiss_index_dim)
            self.embeddings_indexed = self.embeddings.copy()
            faiss.normalize_L2(self.embeddings_indexed)  # cosine similarity
        elif metric == "L2":
            self.index = faiss.IndexFlatL2(self.faiss_index_dim)
            self.embeddings_indexed = self.embeddings.copy()
        else:
            raise ValueError("metric must be 'IP' or 'L2'")

        self.index.add(self.embeddings_indexed)

    @staticmethod
    def compute_centrality(G: nx.Graph) -> Dict[str, Dict[int, float]]:
        """
        Compute various centrality metrics (degree, closeness, betweenness, eigenvector)
        and store them as node attributes in G.

        Returns:
            A dict mapping metric_name -> {node_id: score}
        """
        # Degree centrality (NetworkX normalises by default for undirected graphs)
        deg_cent = nx.degree_centrality(G)
        # Closeness centrality
        clos_cent = nx.closeness_centrality(G)
        # Betweenness centrality
        betw_cent = nx.betweenness_centrality(G, normalized=True)
        # Eigenvector centrality (might not converge on large graphs)
        try:
            eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception as e:
            logger.warning(f"Eigenvector centrality did not converge: {e}")
            eig_cent = {}

        # Set as node attributes
        nx.set_node_attributes(G, deg_cent, "deg_centrality")
        nx.set_node_attributes(G, clos_cent, "closeness_centrality")
        nx.set_node_attributes(G, betw_cent, "betweenness_centrality")
        if eig_cent:
            nx.set_node_attributes(G, eig_cent, "eigenvector_centrality")

        return {
            "degree": deg_cent,
            "closeness": clos_cent,
            "betweenness": betw_cent,
            "eigenvector": eig_cent,
        }

    @staticmethod
    def visualize_graph_simple(G: nx.Graph, max_nodes: int = 50) -> None:
        """
        Visualizes a NetworkX graph using a simple spring layout.

        If the graph has more nodes than `max_nodes`, only the top `max_nodes` nodes with the highest degree are shown.
        Nodes are displayed as light blue circles, and edges are drawn with transparency and width proportional to their weights.
        Node labels are included for clarity.

        Args:
            G (nx.Graph): The NetworkX graph to visualize. Edges must have a 'weight' attribute.
            max_nodes (int, optional): Maximum number of nodes to display. Defaults to 50.

        Returns:
            None: Displays the graph using matplotlib.
        """

        # Create a subgraph if too large
        if G.number_of_nodes() > max_nodes:
            # Get most connected nodes
            degree_dict = dict(G.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[
                :max_nodes
            ]
            nodes_to_show = [node for node, degree in top_nodes]
            G_sub = G.subgraph(nodes_to_show)
        else:
            G_sub = G

        plt.figure(figsize=(10, 5))

        # Use spring layout for better visualization
        pos = nx.spring_layout(G_sub, k=0.5, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            G_sub, pos, node_size=300, node_color="lightblue", alpha=0.7
        )

        # Draw edges with transparency based on weight
        edges = G_sub.edges()
        weights = [G_sub[u][v]["weight"] for u, v in edges]

        nx.draw_networkx_edges(G_sub, pos, alpha=0.3, width=weights)

        # Add labels
        nx.draw_networkx_labels(G_sub, pos, font_size=8)

        plt.title(f"Allplan Manual Knowledge Graph ({G_sub.number_of_nodes()} nodes)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def retrieve(
        self,
        G: nx.Graph,
        query: str,
        embedding_model: Embeddings,
        top_k: int = 5,
        k_hops: int = 1,
        text_attr: str = "content",
        centrality_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Graph RAG retrieval with centrality re‐ranking:
        1) Embed query and FAISS-search top_k seed nodes by semantic similarity.
        2) Adjust seed scores by node centrality.
        3) Expand k hops in graph to collect neighbourhood.
        4) Return seeds, induced subgraph, and node texts as LLM context.

        Args:
        G: NetworkX graph
        query: question string
        embedding_model: function that returns (1, d) embedding for query
        top_k: number of initial semantic seed nodes (k-NN search)
        k_hops: neighborhood expansion radius
        text_attr: node attribute holding node text
        centrality_weight: multiplier for centrality bonus (0.0 = no centrality effect)

        Returns:
        dict with keys: "top-k-nodes", "neighborhood", "subgraph", "contexts"
        """
        assert hasattr(self, "index"), "Call build_faiss_index() first."
        assert 0.0 <= centrality_weight <= 1.0, "centrality_weight must be in [0, 1]"
        assert callable(
            getattr(embedding_model, "embed_query", None)
        ), "embed_query must be callable"

        # 1) embed query
        vec = embedding_model.embed_query(query)
        q = np.array(vec, dtype="float32").reshape(1, -1)

        if self.metric == "IP":
            faiss.normalize_L2(q)

        scores, idxs = self.index.search(q, top_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        # 2) semantic seed list
        seeds: List[Tuple[int, float]] = []
        for i, node in enumerate(idxs):
            if node == -1:  # invalid node
                continue
            sim_score = float(scores[i])  # raw similarity score
            central = G.nodes[node].get(
                "deg_centrality", 0.0
            )  # use degree centrality as bonus
            # If centrality is zero, only semantic similarity counts
            adj_score = sim_score * (1.0 + centrality_weight * central)
            seeds.append((node, adj_score))

        # Sort top nodes by adjusted score descending
        seeds.sort(key=lambda x: x[1], reverse=True)
        seed_nodes = [n for n, _ in seeds]

        # 3) BFS k-hop expansion / create retrieved contexts
        visited = set(seed_nodes)
        frontier = list(seed_nodes)
        for _ in range(k_hops):
            nxt = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break

        neighborhood = sorted(visited)
        SG = G.subgraph(neighborhood).copy()

        # 4) gather texts in order (seeds first)
        seed_set = set(seed_nodes)
        ordered_nodes = seed_nodes + [n for n in neighborhood if n not in seed_set]
        contexts = [(n, str(G.nodes[n].get(text_attr, ""))) for n in ordered_nodes]

        return {
            "top-k-nodes": seeds,  # List of the top-k most similar nodes found by FAISS
            "neighborhood": ordered_nodes,  # all node IDs included in the expanded subgraph
            "subgraph": SG,  # subgraph object containing just the retrieved nodes and their edges
            "contexts": contexts,  # already ordered by relevance
        }


class GraphRAGRetriever(BaseRetriever):
    """
    GraphRAGRetriever is a retriever class that leverages a knowledge graph and graph-based retrieval techniques to find relevant documents for a given query.

    Args:
        kg (KnowledgeGraph): The knowledge graph object used for retrieval.
        embedding_model (Embeddings): The embedding function used to encode the query.
        top_k (int, optional): The number of top relevant nodes/documents to retrieve. Defaults to 5.
        k_hops (int, optional): The number of hops to consider in the graph for neighborhood expansion. Defaults to 1.
        centrality_weight (float, optional): Weight factor for centrality in the retrieval scoring. Defaults to 0.5.

    Methods:
        _get_relevant_documents(query: str) -> List[Document]:
            Retrieves a list of relevant Document objects for the given query using the knowledge graph and embedding function.

        _aget_relevant_documents(query: str) -> List[Document]:
            Asynchronous version of _get_relevant_documents.

    Attributes:
        _kg: The knowledge graph instance.
        _embedding_model: The embedding function used for encoding.
        _top_k: Number of top results to return.
        _k_hops: Number of hops for neighborhood expansion.
        _centrality_weight: Weight for centrality in scoring.
    """

    _kg: Any = PrivateAttr()
    _embedding_model: Any = PrivateAttr()
    _top_k: int = PrivateAttr()
    _k_hops: int = PrivateAttr()
    _centrality_weight: float = PrivateAttr()

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_model: Embeddings,
        top_k: int = 5,
        k_hops: int = 1,
        centrality_weight: float = 0.5,
    ):
        super().__init__()
        self._kg = kg
        assert isinstance(kg, KnowledgeGraph)
        self._embedding_model = embedding_model
        self._G = kg.nx_graph
        self._top_k = top_k
        self._k_hops = k_hops
        self._centrality_weight = centrality_weight

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents from the knowledge graph based on a query.

        Args:
            query (str): The input query string used to search for relevant documents.

        Returns:
            List[Document]: A list of Document objects containing the relevant text and associated node metadata.

        This method uses the knowledge graph's retrieval mechanism with the specified embedding model, number of top results, number of hops, and centrality weighting to find and return documents most relevant to the query.
        """
        result = self._kg.retrieve(
            G=self._G,
            query=query,
            embedding_model=self._embedding_model,
            top_k=self._top_k,
            k_hops=self._k_hops,
            centrality_weight=self._centrality_weight,
        )
        docs: List[Document] = []
        for node_id, text in result["contexts"]:
            docs.append(Document(page_content=text, metadata={"node_id": node_id}))
        return docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Asynchronously retrieves relevant documents for a given query.

        Args:
            query (str): The search query string.

        Returns:
            List[Document]: A list of relevant Document objects matching the query.
        """
        return self._get_relevant_documents(query, **kwargs)
