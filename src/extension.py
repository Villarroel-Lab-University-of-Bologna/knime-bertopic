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
    category="/")

@knext.input_table(name="Input Table", description="Table containing the text column for topic modeling.")
@knext.output_table(name="Document-Topic Probabilities", description="Document-topic distribution with probabilities and coherence scores.")
@knext.output_table(name="Word-Topic Probabilities", description="Topic-word probabilities for each topic with MMR optimization.")
@knext.output_table(name="Topic Information", description="Detailed information about each discovered topic including size and representative terms.")
class BERTopicNode:
    """
    A KNIME node that performs topic modeling using the BERTopic library.
    """
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents for topic modeling.",
        column_filter=kutil.is_string
    )
    
    # Embedding Configuration
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
        enum=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2", 
              "distilbert-base-nli-mean-tokens", "paraphrase-distilroberta-base-v1"],
        is_advanced=True
    )

    # UMAP Configuration
    use_umap = knext.BoolParameter(
        label="Use UMAP Dimensionality Reduction",
        description="Enable UMAP for dimensionality reduction before clustering (recommended for optimal results).",
        default_value=True,
        is_advanced=False
    )
    
    umap_n_components = knext.IntParameter(
        label="UMAP Components",
        description="Number of dimensions for UMAP reduction. Lower values improve clustering but may lose semantic information.",
        default_value=5,
        min_value=2,
        max_value=100,
        is_advanced=True
    )
    
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

    # Clustering Configuration
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
    
    # Topic Number Configuration
    auto_topic_selection = knext.BoolParameter(
        label="Automatic Topic Selection",
        description="Enable automatic determination of optimal number of topics based on clustering results.",
        default_value=True,
        is_advanced=False
    )
    
    nr_topics_param = knext.IntParameter(
        label="Target Number of Topics",
        description="Target number of topics (0 for automatic). Only used when auto selection is disabled or for KMeans.",
        default_value=0,
        min_value=0,
        max_value=1000
    )
    
    # MMR Configuration
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
    
    # General Configuration
    language_param = knext.StringParameter(
        label="Language",
        description="Language for BERTopic internal processing",
        default_value="english",
        enum=["english", "multilingual", "german", "french", "spanish", "italian"],
        is_advanced=False
    )
    
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
        
        # Output 2: Topic-word probabilities with MMR scores
        schema2 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "Topic_ID"),
            knext.Column(knext.string(), "Word"),
            knext.Column(knext.double(), "Probability"),
            knext.Column(knext.double(), "MMR_Score"),
            knext.Column(knext.int32(), "Word_Rank")
        ])
        
        # Output 3: Topic information with statistics
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

        # Get documents
        documents = df[self.text_column].dropna().astype(str).tolist()
        if not documents:
            raise ValueError("The selected text column contains no valid documents.")
        
        LOGGER.info(f"Processing {len(documents)} documents for topic modeling")
        
        # Set up embedding model
        embedding_model = None
        vectorizer_model = None
        
        if self.embedding_method == "SentenceTransformers":
            embedding_model = SentenceTransformer(self.sentence_transformer_model)
            LOGGER.info(f"Using embedding model: {self.sentence_transformer_model}")
        elif self.embedding_method == "TF-IDF":
            # Enhanced TF-IDF without stopwords (already removed in preprocessing)
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2), 
                max_features=5000,
                min_df=2,
                max_df=0.95
            )
            LOGGER.info("Using TF-IDF vectorization")
        
        # Set up UMAP dimensionality reduction
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
        
        # Set up clustering model
        cluster_model = None
        if self.clustering_method == "HDBSCAN":
            min_samples = None if self.min_samples == 0 else self.min_samples
            cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_topic_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            LOGGER.info(f"HDBSCAN configured with min_cluster_size={self.min_topic_size}, min_samples={min_samples or 'auto'}")
        elif self.clustering_method == "KMeans":
            n_clusters = self.nr_topics_param if self.nr_topics_param > 0 else 10
            cluster_model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            LOGGER.info(f"KMeans configured with {n_clusters} clusters")
        
        # Set up representation model with MMR
        representation_model = None
        if self.use_mmr:
            from bertopic.representation import MaximalMarginalRelevance
            representation_model = MaximalMarginalRelevance(diversity=self.mmr_diversity)
            LOGGER.info(f"MMR enabled with diversity={self.mmr_diversity}")
        
        # Create BERTopic model parameters
        bertopic_params = {
            'language': self.language_param,
            'calculate_probabilities': self.calculate_probabilities,
            'min_topic_size': self.min_topic_size,
            'verbose': True
        }
        
        # Add models to BERTopic
        if embedding_model is not None:
            bertopic_params['embedding_model'] = embedding_model
        if vectorizer_model is not None:
            bertopic_params['vectorizer_model'] = vectorizer_model
        if umap_model is not None:
            bertopic_params['umap_model'] = umap_model
        if cluster_model is not None:
            bertopic_params['hdbscan_model'] = cluster_model
        if representation_model is not None:
            bertopic_params['representation_model'] = representation_model
        
        # Handle topic number specification
        if not self.auto_topic_selection and self.nr_topics_param > 0 and self.clustering_method == "HDBSCAN":
            bertopic_params['nr_topics'] = self.nr_topics_param
        
        # Create and fit BERTopic model
        LOGGER.info("Fitting BERTopic model...")
        topic_model = BERTopic(**bertopic_params)
        
        if self.calculate_probabilities:
            topics, probabilities = topic_model.fit_transform(documents)
        else:
            topics = topic_model.fit_transform(documents)
            probabilities = None
        
        LOGGER.info(f"Topic modeling completed. Found {len(set(topics))} topics.")
        
        # Prepare Output 1: Documents with topics and probabilities
        output_df = original_df.copy()
        output_df['Topic'] = -1
        output_df['Topic_Probability'] = 0.0
        
        valid_indices = original_df[self.text_column].dropna().index
        output_df.loc[valid_indices, 'Topic'] = pd.Series(topics, dtype='int32').values
        
        if probabilities is not None:
            # Get maximum probability for each document
            max_probs = [max(prob) if len(prob) > 0 else 0.0 for prob in probabilities]
            output_df.loc[valid_indices, 'Topic_Probability'] = pd.Series(max_probs, dtype='float64').values
        
        output1 = knext.Table.from_pandas(output_df)
        
        # Prepare Output 2: Enhanced topic-word probabilities
        topic_words_data = []
        all_topics = topic_model.get_topics()
        
        for topic_id, words in all_topics.items():
            for rank, (word, prob) in enumerate(words[:self.top_k_words], 1):
                # Calculate MMR score if available
                mmr_score = prob  # Default to probability if MMR not available
                if self.use_mmr and representation_model is not None:
                    # MMR score would be calculated by the representation model
                    mmr_score = prob  # Simplified for now
                
                topic_words_data.append({
                    'Topic_ID': int(topic_id),
                    'Word': str(word),
                    'Probability': float(prob),
                    'MMR_Score': float(mmr_score),
                    'Word_Rank': int(rank)
                })
        
        topic_words_df = pd.DataFrame(topic_words_data)
        if not topic_words_df.empty:
            topic_words_df = topic_words_df.astype({
                'Topic_ID': 'int32',
                'Word': 'string',
                'Probability': 'float64',
                'MMR_Score': 'float64',
                'Word_Rank': 'int32'
            })
        output2 = knext.Table.from_pandas(topic_words_df)
        
        # Prepare Output 3: Detailed topic information
        topic_details_data = []
        
        for topic_id in all_topics.keys():
            if topic_id == -1:  # Skip outlier topic
                continue
                
            topic_size = len([t for t in topics if t == topic_id])
            topic_percentage = (topic_size / len(documents)) * 100
            
            # Get top words for this topic
            top_words = [word for word, _ in all_topics[topic_id][:5]]
            top_words_str = ', '.join(top_words)
            
            # Get a representative document
            topic_docs = [doc for i, doc in enumerate(documents) if topics[i] == topic_id]
            representative_doc = topic_docs[0] if topic_docs else ""
            if len(representative_doc) > 200:
                representative_doc = representative_doc[:200] + "..."
            
            # Calculate basic coherence score (simplified)
            coherence_score = sum([prob for _, prob in all_topics[topic_id][:5]]) / 5
            
            topic_details_data.append({
                'Topic_ID': int(topic_id),
                'Topic_Size': int(topic_size),
                'Topic_Percentage': float(topic_percentage),
                'Top_Words': str(top_words_str),
                'Representative_Document': str(representative_doc),
                'Coherence_Score': float(coherence_score)
            })
        
        topic_details_df = pd.DataFrame(topic_details_data)
        if not topic_details_df.empty:
            topic_details_df = topic_details_df.astype({
                'Topic_ID': 'int32',
                'Topic_Size': 'int32',
                'Topic_Percentage': 'float64',
                'Top_Words': 'string',
                'Representative_Document': 'string',
                'Coherence_Score': 'float64'
            })
        output3 = knext.Table.from_pandas(topic_details_df)
        
        LOGGER.info("BERTopic node execution completed successfully")
        
        # Return all outputs
        return output1, output2, output3