# KNIME Topic Modeling Extension using BERT
[![License: Apache License 2.0](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Code Quality Check](https://github.com/Albdl03/knime-bertopic/actions/workflows/code-quality-check.yml/badge.svg)](https://github.com/Albdl03/knime-bertopic/actions/workflows/code-quality-check.yml)
[![Extension Bundling](https://github.com/Albdl03/knime-bertopic/workflows/bundle-extension.yml/badge.svg)](https://github.com/Albdl03/knime-bertopic/actions/workflows/bundle-extension.yml)

This repository contains the source code for the KNIME Topic Modeling node, a pure Python-based KNIME extension that implements advanced topic modeling capabilities using BERTopic methodology.

## Overview

The Topic Modeling node provides an end-to-end pipeline for discovering topics in text documents through a three-stage process:
1. **Document Embedding**: Convert text to numerical representations using sentence transformers or TF-IDF
2. **Dimensionality Reduction**: Apply UMAP to reduce embedding dimensionality for better clustering
3. **Clustering & Topic Extraction**: Use HDBSCAN or K-Means clustering with c-TF-IDF topic representation

## Features

- **Multiple Embedding Methods**: Support for sentence transformers and TF-IDF vectorization
- **Flexible Clustering**: Choose between HDBSCAN (automatic cluster detection) or K-Means (fixed number of clusters)
- **Advanced Topic Representation**: Uses c-TF-IDF (class-based TF-IDF) for coherent topic extraction
- **Topic Probability Calculation**: Optional computation of document-topic probabilities
- **MMR Diversification**: Maximal Marginal Relevance for diverse topic keyword selection
- **Automatic Topic Selection**: Intelligent determination of optimal topic count

## Usage

### Input
- **Input Table**: Table containing text documents with at least one string column for analysis

### Configuration Options

The node provides comprehensive configuration options organized by processing stages:

#### Document Embedding (Stage 1)
- **Text Column**: Select the column containing text documents to analyze
- **Embedding Method**: Choose between "SentenceTransformers" (recommended) or "TF-IDF"
- **Sentence Transformer Model**: Select from pre-trained models:
  - `all-MiniLM-L6-v2`
  - `all-mpnet-base-v2`
  - `paraphrase-multilingual-MiniLM-L12-v2`
  - `distilbert-base-nli-mean-tokens`
  - `paraphrase-distilroberta-base-v1`

#### UMAP Dimensionality Reduction (Stage 2)
- **Use UMAP Dimensionality Reduction**: Enable/disable UMAP (recommended: enabled)
- **UMAP Components**: Number of dimensions for reduction (default: 5)
- **UMAP Neighbors**: Number of neighbors for local structure preservation (default: 15)
- **UMAP Min Distance**: Minimum distance between points in embedding (default: 0.0)

#### Clustering & Topic Extraction (Stage 3)
- **Clustering Method**: Choose "HDBSCAN" (automatic) or "KMeans" (fixed number)
- **Minimum Topic Size**: Minimum documents required per topic (default: 10)
- **HDBSCAN Min Samples**: Minimum samples for core points (default: 1)
- **Automatic Topic Selection**: Enable automatic topic number determination
- **Target Number of Topics**: Specify target topics when auto-selection disabled

#### Advanced Options
- **Use Maximal Marginal Relevance (MMR)**: Enable MMR for topic representation
- **MMR Diversity**: Balance between relevance and diversity (default: 0.3)
- **Language**: Select processing language (English, German, French, Spanish, Italian, Multilingual)
- **Calculate Topic Probabilities**: Enable soft clustering probabilities
- **Top K Words per Topic**: Number of representative words per topic (default: 10)
- **Random State**: Random seed for reproducible results (default: 42)

### Output
The node produces three comprehensive output tables:

1. **Document-Topic Probabilities**: Original input data with added topic assignments and probability scores
   - `Topic`: Assigned topic ID (-1 for outliers)
   - `Topic_Probability`: Maximum probability score for the assigned topic

2. **Word-Topic Probabilities**: Detailed word-topic distributions
   - `Topic_ID`: Topic identifier
   - `Word`: Representative word
   - `Probability`: Word probability within topic
   - `MMR_Score`: MMR optimization score
   - `Word_Rank`: Rank within topic (1 = most representative)

3. **Topic Information**: Complete topic metadata
   - `Topic_ID`: Topic identifier
   - `Topic_Size`: Number of documents in topic
   - `Topic_Percentage`: Percentage of total documents
   - `Top_Words`: Top 5 representative words
   - `Representative_Document`: Sample document from topic
   - `Coherence_Score`: Topic coherence metric