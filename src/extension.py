import logging
import knime.extension as knext
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
import utils.knutils as kutil

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Topic Extractor (BERTopic)", 
    node_type=knext.NodeType.LEARNER, 
    icon_path="icons/icon.png", 
    category="/")

@knext.input_table(name="Input Table", description="Table containing the text column for topic modeling.")
@knext.output_table(name="Document-Topic Probabilities", description="Document-topic distribution with probabilities.")
@knext.output_table(name="Word-Topic Probabilities", description="Topic-word probabilities for each topic.")
@knext.output_table(name="Model Fit Summary", description="Basic statistics and evaluation metrics from model fitting.")
class BERTopicNode:
    """Use BERTopic to extract topics from documents."""
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
        column_filter=kutil.is_string
    )

    embedding_method = knext.StringParameter(
        label="Embedding Model",
        description="Type of embedding model to use.",
        default_value="SentenceTransformers",
        enum=["SentenceTransformers", "TF-IDF"],
        is_advanced=True
    )

    sentence_transformer_model = knext.StringParameter(
        label="SentenceTransformer Model",
        description="Specific SentenceTransformer model to use.",
        default_value="all-MiniLM-L6-v2",
        enum=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
            "distilbert-base-nli-stsb-mean-tokens"
        ],
        is_advanced=True
    )

    clustering_method = knext.StringParameter(
        label="Clustering Method",
        description="Clustering algorithm to use.",
        default_value="HDBSCAN",
        enum=["HDBSCAN", "KMeans"],
        is_advanced=True
    )
    
    language_param = knext.StringParameter(
        label="Language",
        description="Language for topic modeling.",
        default_value="english",
        enum=["english", "multilingual"]
    )
    
    calculate_probabilities = knext.BoolParameter(
        label="Calculate Probabilities",
        description="Whether to calculate topic probabilities for documents.",
        default_value=True
    )
    
    nr_topics = knext.IntParameter(
        label="Number of Topics",
        description="Number of topics to extract. Use 'auto' for automatic selection.",
        default_value=20
    )

    min_topic_size = knext.IntParameter(
        label="Minimum Topic Size",
        description="Minimum number of documents required to form a topic.",
        default_value=10,
        is_advanced=True
    )

    def configure(self, config_context, input_schema):
        # Output 1: Documents with topics
        schema1 = input_schema.append(knext.Column(knext.int64(), "Topic"))
        
        # Output 2: Topic-word probabilities
        schema2 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "Topic_ID"),
            knext.Column(knext.string(), "Word"),
            knext.Column(knext.double(), "Probability")
        ])
        
        # Output 3: Model summary
        schema3 = knext.Schema.from_columns([
            knext.Column(knext.string(), "Metric"),
            knext.Column(knext.string(), "Value")
        ])
        
        return schema1, schema2, schema3

    def execute(self, exec_context, input_table):
        # Convert to pandas
        df = input_table.to_pandas()
        
        # Get documents
        documents = df[self.text_column].dropna().astype(str).tolist()
        
        if not documents:
            raise ValueError("No valid documents found.")
        
        # Set up embedding model
        embedding_model = None
        if self.embedding_method == "SentenceTransformers":
            embedding_model = SentenceTransformer(self.sentence_transformer_model)
        elif self.embedding_method == "TF-IDF":
            # TF-IDF will be handled by BERTopic's vectorizer model
            pass
        
        # Set up clustering model
        cluster_model = None
        if self.clustering_method == "HDBSCAN":
            cluster_model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_topic_size,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
        elif self.clustering_method == "KMeans":
            cluster_model = KMeans(n_clusters=self.nr_topics, random_state=42)
        
        # Set up vectorizer for TF-IDF option
        vectorizer_model = None
        if self.embedding_method == "TF-IDF":
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        
        # Create BERTopic model parameters
        bertopic_params = {
            'language': self.language_param,
            'calculate_probabilities': self.calculate_probabilities,
            'min_topic_size': self.min_topic_size
        }
        
        # Add optional parameters
        if embedding_model is not None:
            bertopic_params['embedding_model'] = embedding_model
        if cluster_model is not None:
            bertopic_params['hdbscan_model'] = cluster_model
        if vectorizer_model is not None:
            bertopic_params['vectorizer_model'] = vectorizer_model
        if self.nr_topics > 0 and self.clustering_method != "KMeans":
            bertopic_params['nr_topics'] = self.nr_topics
        
        # Create and fit BERTopic model
        topic_model = BERTopic(**bertopic_params)
        topics, probabilities = topic_model.fit_transform(documents)
        
        # Prepare Output 1: Documents with topics
        df['Topic'] = -1  # Default for all rows
        valid_indices = df[self.text_column].dropna().index
        df.loc[valid_indices, 'Topic'] = topics
        output1 = knext.Table.from_pandas(df)
        
        # Prepare Output 2: Topic-word probabilities
        topic_words_data = []
        all_topics = topic_model.get_topics()
        
        for topic_id, words in all_topics.items():
            for word, prob in words:
                topic_words_data.append({
                    'Topic_ID': topic_id,
                    'Word': word,
                    'Probability': prob
                })
        
        topic_words_df = pd.DataFrame(topic_words_data)
        output2 = knext.Table.from_pandas(topic_words_df)
        
        # Prepare Output 3: Model summary
        summary_data = pd.DataFrame({
            'Metric': [
                'Number of Topics', 
                'Number of Documents', 
                'Number of Outliers',
                'Embedding Model',
                'Clustering Method',
                'Language',
                'Min Topic Size'
            ],
            'Value': [
                str(len(all_topics)),
                str(len(documents)),
                str(sum(1 for t in topics if t == -1)),
                self.embedding_method,
                self.clustering_method,
                self.language_param,
                str(self.min_topic_size)
            ]
        })
        output3 = knext.Table.from_pandas(summary_data)
        
        # Return outputs
        return output1, output2, output3