import logging
import knime.extension as knext
from bertopic import BERTopic
import pandas as pd
from utils import knutils as kutil
from sentence_transformers import SentenceTransformer
from flair.embeddings import TransformerDocumentEmbeddings
import gensim.downloader as api
import hdbscan
from sklearn.cluster import KMeans

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
class TemplateNode:
    """Use BERTopic to extract topics from documents. 
    
    This node uses the BERTopic library to perform topic modeling on text documents.
    It supports different embedding models and clustering algorithms to extract
    meaningful topics from your text data.
    """
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
        port_index=0, 
        column_filter=kutil.is_string
    )

    embedding_model_param = knext.StringParameter(
        label="Embedding Model",
        description="Type of embedding model to use for BERTopic.",
        default_value="SentenceTransformers",
        enum=["SentenceTransformers", "Flair", "Spacy", "Gensim"],
        is_advanced=True
    )

    clustering_method = knext.StringParameter(
        label="Clustering Method",
        description="Clustering algorithm to use within BERTopic.",
        default_value="HDBSCAN",
        enum=["HDBSCAN", "KMeans"],
        is_advanced=True
    )
    
    language_param = knext.StringParameter(
        label="Language",
        description="Language for topic modeling.",
        default_value="english",
        enum=["english", "multilingual"],
        is_advanced=True
    )
    
    probabilities_param = knext.BoolParameter(
        label="Calculate Probabilities",
        description="Whether to calculate topic probabilities for documents.",
        default_value=True,
        is_advanced=True
    )
    
    nr_topics_param = knext.IntParameter(
        label="Number of Topics",
        description="Number of topics to extract. Use 'auto' for automatic selection.",
        default_value=20,
        is_advanced=True
    )

    def configure(self, config_context, input_schema):
        # Output schema 1: Input table with added Topics column
        schema1 = input_schema.append(knext.Column(knext.int64(), "Topics"))
        
        # Output schema 2: Topic-word probabilities
        schema2 = knext.Schema([
            knext.Column(knext.int64(), "Topic_ID"),
            knext.Column(knext.string(), "Term"),
            knext.Column(knext.double(), "Weight")
        ])
        
        # Output schema 3: Model summary statistics
        schema3 = knext.Schema([
            knext.Column(knext.int64(), "Num_Topics"),
            knext.Column(knext.int64(), "Num_Documents"),
            knext.Column(knext.string(), "Embedding_Model"),
            knext.Column(knext.string(), "Clustering_Method")
        ])
        
        return schema1, schema2, schema3

    def execute(self, exec_context, input_1):
        # Convert input to pandas DataFrame
        input_1_pandas = input_1.to_pandas()
        
        # Check if execution was canceled
        kutil.check_canceled(exec_context)
        
        # Get the text documents
        documents = input_1_pandas[self.text_column].dropna().tolist()
        
        if not documents:
            raise ValueError("No valid text documents found in the selected column.")
        
        # Set up embedding model
        embedding_model = None
        if self.embedding_model_param == "SentenceTransformers":
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.embedding_model_param == "Flair":
            embedding_model = TransformerDocumentEmbeddings('bert-base-uncased')
        # Add other embedding models as needed
        
        # Set up clustering model
        cluster_model = None
        if self.clustering_method == "HDBSCAN":
            cluster_model = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', 
                                          cluster_selection_method='eom', prediction_data=True)
        elif self.clustering_method == "KMeans":
            cluster_model = KMeans(n_clusters=self.nr_topics_param, random_state=42)
        
        # Create and fit BERTopic model
        topic_model = BERTopic(
            language=self.language_param,
            calculate_probabilities=self.probabilities_param,
            nr_topics=self.nr_topics_param,
            embedding_model=embedding_model,
            hdbscan_model=cluster_model if self.clustering_method == "HDBSCAN" else None,
            # Note: KMeans integration might need custom setup depending on BERTopic version
        )
        
        # Fit the model and get topics
        topics, probs = topic_model.fit_transform(documents)
        
        # Add topics to the input dataframe
        # Handle case where input has more rows than valid documents
        input_1_pandas['Topics'] = -1  # Default for missing/invalid documents
        valid_indices = input_1_pandas[self.text_column].dropna().index
        input_1_pandas.loc[valid_indices, 'Topics'] = topics
        
        # Prepare output table 1: Documents with topics
        output_1 = knext.Table.from_pandas(input_1_pandas)
        
        # Prepare output table 2: Topic-word probabilities
        topic_info = []
        all_topics = topic_model.get_topics()
        
        for topic_id, words in all_topics.items():
            for word, weight in words:
                topic_info.append({
                    'Topic_ID': topic_id,
                    'Term': word,
                    'Weight': weight
                })
        
        topic_info_df = pd.DataFrame(topic_info)
        output_2 = knext.Table.from_pandas(topic_info_df)
        
        # Prepare output table 3: Model summary
        summary_data = pd.DataFrame({
            'Num_Topics': [len(all_topics)],
            'Num_Documents': [len(documents)],
            'Embedding_Model': [self.embedding_model_param],
            'Clustering_Method': [self.clustering_method]
        })
        output_3 = knext.Table.from_pandas(summary_data)
        
        return output_1, output_2, output_3