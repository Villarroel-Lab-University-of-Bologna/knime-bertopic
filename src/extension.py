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
@knext.output_table(name="Output Table", description="Table with topic assignments.")
class BERTopicNode:
    """Use BERTopic to extract topics from documents."""
    
    text_column = knext.ColumnParameter(
        "Text Column",
        "Column containing input documents.",
        port_index=0,
        column_filter=lambda col: col.ktype == knext.string()
    )

    def configure(self, config_context, input_schema):
        # Simple output: input + topic column
        return input_schema.append(knext.Column(knext.int64(), "Topic"))

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
        
        # Add topics back to dataframe
        df['Topic'] = -1  # Default for all rows
        valid_indices = df[self.text_column].dropna().index
        df.loc[valid_indices, 'Topic'] = topics
        
        return knext.Table.from_pandas(df)