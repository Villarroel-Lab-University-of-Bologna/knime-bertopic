import logging
import knime.extension as knext
import pandas as pd

# Test basic BERTopic import
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    print(f"BERTopic import failed: {e}")

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Topic Extractor (BERTopic)", 
    node_type=knext.NodeType.LEARNER, 
    icon_path="icons/icon.png", 
    category="/")
@knext.input_table(name="Input Table", description="Table containing the text column for topic modeling.")
@knext.output_table(name="Document Topics", description="Documents with topic assignments.")
@knext.output_table(name="Topic Words", description="Topic-word probabilities for each topic.")
@knext.output_table(name="Model Summary", description="Basic statistics from model fitting.")
class BERTopicNode:
    """Use BERTopic to extract topics from documents."""
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
        port_index=0,
        column_filter=lambda col: col.ktype == knext.string()
    )

    def configure(self, config_context, input_schema):
        # Output 1: Documents with topics
        schema1 = input_schema.append(knext.Column(knext.int64(), "Topic"))
        
        # Output 2: Topic-word probabilities
        schema2 = knext.Schema([
            knext.Column(knext.int64(), "Topic_ID"),
            knext.Column(knext.string(), "Word"),
            knext.Column(knext.double(), "Probability")
        ])
        
        # Output 3: Model summary
        schema3 = knext.Schema([
            knext.Column(knext.string(), "Metric"),
            knext.Column(knext.string(), "Value")
        ])
        
        return schema1, schema2, schema3

    def execute(self, exec_context, input_table):
        if not BERTOPIC_AVAILABLE:
            raise RuntimeError("BERTopic is not available. Please check installation.")
            
        # Convert to pandas
        df = input_table.to_pandas()
        
        # Get documents
        documents = df[self.text_column].dropna().astype(str).tolist()
        
        if not documents:
            raise ValueError("No valid documents found.")
        
        # Simple BERTopic usage with defaults
        topic_model = BERTopic()
        topics, _ = topic_model.fit_transform(documents)
        
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
            'Metric': ['Number of Topics', 'Number of Documents', 'Number of Outliers'],
            'Value': [
                str(len(all_topics)),
                str(len(documents)),
                str(sum(1 for t in topics if t == -1))
            ]
        })
        output3 = knext.Table.from_pandas(summary_data)
        
        return output1, output2, output3