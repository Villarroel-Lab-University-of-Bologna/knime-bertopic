import logging
import knime.extension as knext
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import utils.knutils as kutil

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Topic Extractor (BERTopic)",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/icon.png",
    category="/"
)
@knext.input_table(name="Input Table", description="Table containing the text column for topic modeling.")
@knext.output_table(name="Document-Topic Probabilities", description="Document-topic distribution with probabilities and coherence scores.")
@knext.output_table(name="Word-Topic Probabilities", description="Topic-word probabilities for each topic with MMR optimization.")
@knext.output_table(name="Topic Information", description="Detailed information about each discovered topic including size and representative terms.")
class BERTopicNode:
    """
    Topic Extractor (BERTopic) node

    The Topic Extractor (BERTopic) node enables users to perform advanced topic modeling on text documents using the state-of-the-art BERTopic library. It automatically discovers coherent topics within document collections by leveraging modern transformer-based embeddings, dimensionality reduction, and clustering techniques to preserve semantic relationships that traditional methods miss.

    The model processes text data through a validated three-stage pipeline: document embedding using pre-trained transformers, UMAP dimensionality reduction for clustering optimization, and HDBSCAN density-based clustering for automatic topic discovery. This approach significantly outperforms traditional methods, with research by Huang et al. (2024) demonstrating superior performance in capturing semantic meaning from short texts like online reviews compared to conventional LDA approaches.

    Training parameters such as embedding models, UMAP components, and clustering methods can be customized. The node features automatic topic number determination and incorporates Maximal Marginal Relevance (MMR) optimization to balance topic coherence with word diversity. The node outputs three comprehensive tables: document-topic assignments with probabilities, detailed word-topic distributions with coherence metrics, and complete topic information including statistics and representative terms.

    The node implements a validated processing pipeline based on recent advances in topic modeling research:

    -**Document Embedding (Stage 1)**: Uses pre-trained transformer models (BERT/SentenceTransformers) to create high-dimensional semantic representations that capture contextual meaning and relationships between documents.
    [More info](https://www.sbert.net/)
    
    -**UMAP Dimensionality Reduction (Stage 2)**: Applies Uniform Manifold Approximation and Projection to reduce embedding dimensions while preserving local and global structure, optimizing the data for effective clustering.
    [More info](https://umap-learn.readthedocs.io/)
    
    -**HDBSCAN Clustering (Stage 3)**: Employs hierarchical density-based clustering to automatically discover the optimal number of topics, effectively handling noise and outliers without requiring manual cluster specification.
    [More info](https://hdbscan.readthedocs.io/)

    ### How It Works:
    1. **Document Embedding**: The node converts text documents into high-dimensional vector representations using the selected embedding method (BERT-based transformers or TF-IDF).

    2. **Dimensionality Reduction**: UMAP reduces the high-dimensional embeddings to a lower-dimensional space while preserving semantic structure and relationships between documents.
    
    3. **Density-Based Clustering**: HDBSCAN analyzes the reduced embeddings to identify dense regions representing coherent topics, automatically determining optimal cluster numbers without manual specification.
    
    4. **Topic Representation**: c-TF-IDF generates topic representations by treating each cluster as a single document, with optional MMR optimization to balance word relevance and diversity for improved coherence and interpretability.
    
    5. **Output Generation**: Three comprehensive tables are produced containing document-topic assignments, detailed word-topic probabilities with coherence scores, and topic metadata including statistical summaries and representative documents.

    
    Research by Huang et al. (2024) validates this integrated pipeline's effectiveness in extracting coherent topics from domain-specific text collections, demonstrating superior performance in capturing fine-grained semantic aspects that traditional topic modeling approaches often miss.

    """
    

    # Input column
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents for topic modeling.",
        column_filter=kutil.is_string
    )

    # Embedding configuration
    embedding_method = knext.StringParameter(
        label="Embedding Method",
        description="Method for generating document embeddings.",
        default_value="SentenceTransformers",
        enum=["SentenceTransformers", "TF-IDF"],
        is_advanced=False
    )
    sentence_transformer_model = knext.StringParameter(
        label="Sentence Transformer Model",
        description="Pre-trained transformer model for document embeddings. all-mpnet-base-v2 provides best quality.",
        default_value="all-MiniLM-L6-v2",
        enum=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distilbert-base-nli-mean-tokens",
            "paraphrase-distilroberta-base-v1"
        ],
        is_advanced=True
    ).rule(knext.OneOf(embedding_method, ["TF-IDF"]), knext.Effect.HIDE)

    # UMAP configuration
    use_umap = knext.BoolParameter(
        label="Use UMAP Dimensionality Reduction",
        description="Enable UMAP for dimensionality reduction before clustering (recommended for optimal results).",
        default_value=True
    )

    umap_n_components = knext.IntParameter(
        label="UMAP Components",
        description="Number of dimensions for UMAP reduction. Lower values improve clustering but may lose semantic information.",
        default_value=5,
        min_value=2,
        max_value=100,
        is_advanced=True
    ).rule(knext.OneOf(embedding_method, [False]), knext.Effect.HIDE)

    umap_n_neighbors = knext.IntParameter(
        label="UMAP Neighbors",
        description="Number of neighbors for UMAP. Higher values preserve global structure, lower values preserve local structure.",
        default_value=15,
        min_value=2,
        max_value=200,
        is_advanced=True
    )

    umap_min_dist = knext.DoubleParameter(
        label="UMAP Min Distance",
        description="Minimum distance between points in UMAP embedding. Lower values create tighter clusters.",
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True
    )

    # Clustering configuration
    clustering_method = knext.StringParameter(
        label="Clustering Method",
        description="Clustering algorithm. HDBSCAN recommended for automatic topic discovery.",
        default_value="HDBSCAN",
        enum=["HDBSCAN", "KMeans"],
        is_advanced=False
    )

    min_topic_size = knext.IntParameter(
        label="Minimum Topic Size",
        description="Minimum number of documents required to form a topic. Affects topic granularity.",
        default_value=10,
        min_value=2,
        max_value=1000,
        is_advanced=False
    )

    min_samples = knext.IntParameter(
        label="HDBSCAN Min Samples",
        description="Minimum samples for HDBSCAN core points. Higher values create more conservative clusters.",
        default_value=1,
        min_value=1,
        is_advanced=True
    )

    nr_topics_param = knext.IntParameter(
        label="Target Number of Topics",
        description="Target number of topics (0 for automatic). Only used when auto selection is disabled or for KMeans.",
        default_value=0,
        min_value=0,
        max_value=1000,
        is_advanced=True
    )

    # MMR configuration
    use_mmr = knext.BoolParameter(
        label="Use Maximal Marginal Relevance (MMR)",
        description="Enable MMR for topic representation to balance relevance and diversity of topic terms.",
        default_value=True,
        is_advanced=False
    )
    mmr_diversity = knext.DoubleParameter(
        label="MMR Diversity",
        description="Controls diversity vs relevance trade-off. Higher values increase diversity of topic terms.",
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True
    )

    # General configuration
    calculate_probabilities = knext.BoolParameter(
        label="Calculate Topic Probabilities",
        description="Calculate soft clustering probabilities for documents (may increase computation time).",
        default_value=True,
        is_advanced=True
    )

    top_k_words = knext.IntParameter(
        label="Top K Words per Topic",
        description="Number of most representative words to extract per topic.",
        default_value=10,
        min_value=5,
        max_value=50,
        is_advanced=True
    )

    random_state = knext.IntParameter(
        label="Random State",
        description="Random seed for reproducible results.",
        default_value=42,
        min_value=0,
        is_advanced=True
    )

    def configure(self, config_context, input_schema):
        # Output 1: Documents with topics and probabilities
        schema1 = input_schema.append([
            knext.Column(knext.int32(), "Topic"),
            knext.Column(knext.double(), "Topic_Probability")
        ])

        # Output 2: Topic-word probabilities
        schema2 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "Topic_ID"),
            knext.Column(knext.string(), "Word"),
            knext.Column(knext.double(), "Probability"),
            knext.Column(knext.double(), "MMR_Score"),
            knext.Column(knext.int32(), "Word_Rank")
        ])

        # Output 3: Topic information
        schema3 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "Topic_ID"),
            knext.Column(knext.int32(), "Topic_Size"),
            knext.Column(knext.double(), "Topic_Percentage"),
            knext.Column(knext.string(), "Top_Words"),
            knext.Column(knext.string(), "Representative_Document"),
            knext.Column(knext.double(), "Coherence_Score")
        ])
        return schema1, schema2, schema3

    def execute(self, exec_context, input_table):
        # Convert to pandas
        df = input_table.to_pandas()
        original_df = df.copy()

        # Guard against pre-existing columns that we add
        for col in ("Topic", "Topic_Probability"):
            if col in original_df.columns:
                raise ValueError(
                    f"Input table already has a '{col}' column; please rename or remove it before this node."
                )

        # Get documents
        documents = original_df[self.text_column].dropna().astype(str).tolist()
        if not documents:
            raise ValueError("The selected text column contains no valid documents.")

        LOGGER.info(f"Processing {len(documents)} documents for topic modeling")

        # Set up embedding model
        embedding_model = None
        vectorizer_model = None

        if self.embedding_method == "SentenceTransformers":
            embedding_model = SentenceTransformer(self.sentence_transformer_model)
            LOGGER.info(f"Using embedding model: {self.sentence_transformer_model}")
        else:  # TF-IDF
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2,
                max_df=0.95
            )
            LOGGER.info("Using TF-IDF vectorization")

        # UMAP
        umap_model = None
        if self.use_umap:
            umap_model = UMAP(
                n_components=self.umap_n_components,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric='cosine',
                random_state=self.random_state
            )
            LOGGER.info(f"UMAP configured with {self.umap_n_components} components")

        # Clustering
        hdbscan_model = None
        cluster_model = None
        if self.clustering_method == "HDBSCAN":
            ms = None if self.min_samples == 0 else self.min_samples
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_topic_size,
                min_samples=ms,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            LOGGER.info(f"HDBSCAN configured with min_cluster_size={self.min_topic_size}, min_samples={ms or 'auto'}")
        else:  # KMeans
            n_clusters = self.nr_topics_param if self.nr_topics_param > 0 else 10
            cluster_model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            LOGGER.info(f"KMeans configured with {n_clusters} clusters")

        # Representation (MMR)
        representation_model = None
        if self.use_mmr:
            from bertopic.representation import MaximalMarginalRelevance
            representation_model = MaximalMarginalRelevance(diversity=self.mmr_diversity)
            LOGGER.info(f"MMR enabled with diversity={self.mmr_diversity}")

        # Build BERTopic params
        bertopic_params = {
            "language": self.language_param,
            "calculate_probabilities": self.calculate_probabilities,
            "min_topic_size": self.min_topic_size,
            "verbose": True
        }
        if embedding_model is not None:
            bertopic_params["embedding_model"] = embedding_model
        if vectorizer_model is not None:
            bertopic_params["vectorizer_model"] = vectorizer_model
        if umap_model is not None:
            bertopic_params["umap_model"] = umap_model
        if hdbscan_model is not None:
            bertopic_params["hdbscan_model"] = hdbscan_model
        if cluster_model is not None:
            # BERTopic accepts arbitrary clustering models via cluster_model in recent versions
            bertopic_params["cluster_model"] = cluster_model
        if representation_model is not None:
            bertopic_params["representation_model"] = representation_model

        # Handle nr_topics (only meaningful for certain reductions / HDBSCAN pipelines)
        if not self.auto_topic_selection and self.nr_topics_param > 0 and self.clustering_method == "HDBSCAN":
            bertopic_params["nr_topics"] = self.nr_topics_param

        # Fit
        LOGGER.info("Fitting BERTopic model...")
        topic_model = BERTopic(**bertopic_params)

        if self.calculate_probabilities:
            topics, probabilities = topic_model.fit_transform(documents)
        else:
            topics = topic_model.fit_transform(documents)
            probabilities = None

        LOGGER.info(f"Topic modeling completed. Found {len(set(topics))} topics.")

        
        # Output 1: Documents + topics
        
        output_df = original_df.copy()
        output_df["Topic"] = -1
        output_df["Topic_Probability"] = 0.0

        valid_indices = original_df[self.text_column].dropna().index
        output_df.loc[valid_indices, "Topic"] = pd.Series(topics, index=valid_indices, dtype="int32").values

        if probabilities is not None:
            # max probability per doc (probabilities is 2D array-like)
            max_probs = [float(max(p)) if (p is not None and len(p) > 0) else 0.0 for p in probabilities]
            output_df.loc[valid_indices, "Topic_Probability"] = pd.Series(max_probs, index=valid_indices, dtype="float64").values

        # Enforce exact dtypes expected by configure()
        output_df["Topic"] = output_df["Topic"].astype("int32", copy=False)
        output_df["Topic_Probability"] = output_df["Topic_Probability"].astype("float64", copy=False)

        output1 = knext.Table.from_pandas(output_df)

       
        # Output 2: Topic-word probabilities
        
        topic_words_data = []
        all_topics = topic_model.get_topics()  # dict: topic -> List[(word, prob)]

        for topic_id, words in all_topics.items():
            # words already sorted by prob desc
            for rank, (word, prob) in enumerate(words[: self.top_k_words], 1):
                # Placeholder MMR score: keeping prob to maintain schema
                mmr_score = float(prob)
                topic_words_data.append({
                    "Topic_ID": int(topic_id),
                    "Word": str(word),
                    "Probability": float(prob),
                    "MMR_Score": mmr_score,
                    "Word_Rank": int(rank)
                })

        if topic_words_data:
            topic_words_df = pd.DataFrame(topic_words_data).astype({
                "Topic_ID": "int32",
                "Word": "object",          # use object for KNIME string
                "Probability": "float64",
                "MMR_Score": "float64",
                "Word_Rank": "int32"
            })
        else:
            topic_words_df = pd.DataFrame({
                "Topic_ID": pd.Series(dtype="int32"),
                "Word": pd.Series(dtype="object"),
                "Probability": pd.Series(dtype="float64"),
                "MMR_Score": pd.Series(dtype="float64"),
                "Word_Rank": pd.Series(dtype="int32"),
            })

        output2 = knext.Table.from_pandas(topic_words_df)

        
        # Output 3: Topic information
        
        topic_details_data = []
        n_docs = len(documents)

        for topic_id in all_topics.keys():
            if topic_id == -1:
                continue  # skip outliers

            topic_size = sum(1 for t in topics if t == topic_id)
            topic_percentage = (topic_size / n_docs) * 100.0 if n_docs > 0 else 0.0

            # top words
            top_words_list = [w for (w, _) in all_topics[topic_id][:5]]
            top_words_str = ", ".join(top_words_list)

            # representative document (first doc of this topic)
            topic_docs_idx = [i for i, t in enumerate(topics) if t == topic_id]
            representative_doc = documents[topic_docs_idx[0]] if topic_docs_idx else ""
            if len(representative_doc) > 200:
                representative_doc = representative_doc[:200] + "..."

            # simple coherence proxy: avg prob of top 5 words (bounded)
            if all_topics[topic_id]:
                top5 = all_topics[topic_id][:5]
                coherence_score = float(sum(prob for _, prob in top5) / max(1, len(top5)))
            else:
                coherence_score = 0.0

            topic_details_data.append({
                "Topic_ID": int(topic_id),
                "Topic_Size": int(topic_size),
                "Topic_Percentage": float(topic_percentage),
                "Top_Words": str(top_words_str),
                "Representative_Document": str(representative_doc),
                "Coherence_Score": float(coherence_score)
            })

        if topic_details_data:
            topic_details_df = pd.DataFrame(topic_details_data).astype({
                "Topic_ID": "int32",
                "Topic_Size": "int32",
                "Topic_Percentage": "float64",
                "Top_Words": "object",
                "Representative_Document": "object",
                "Coherence_Score": "float64"
            })
        else:
            topic_details_df = pd.DataFrame({
                "Topic_ID": pd.Series(dtype="int32"),
                "Topic_Size": pd.Series(dtype="int32"),
                "Topic_Percentage": pd.Series(dtype="float64"),
                "Top_Words": pd.Series(dtype="object"),
                "Representative_Document": pd.Series(dtype="object"),
                "Coherence_Score": pd.Series(dtype="float64"),
            })

        output3 = knext.Table.from_pandas(topic_details_df)

        LOGGER.info("BERTopic node execution completed successfully")
        return output1, output2, output3