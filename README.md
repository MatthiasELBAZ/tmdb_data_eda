# tmdb_data_eda

## Setup

To enable the virtual environment and install all dependencies, run:

```bash
uv sync
```

This will set up your environment as specified in the `pyproject.toml` and `uv.lock` files.

---

**Note:** This notebook is also intended as an introduction to the Superlinked framework, demonstrating its capabilities for building advanced search and retrieval systems with structured and unstructured data.

This project explores and analyzes movie data using the `tmbdb_data_eda.ipynb` notebook. The workflow consists of two main parts:

## 1. Exploratory Data Analysis (EDA)
The notebook begins with a comprehensive EDA of the TMDB movie dataset, focusing on understanding the data, cleaning, and uncovering trends over time. Key steps include:

- **Data Cleaning & Filtering:**
  - Removal of non-released and adult movies.
  - Focus on English-language films with valid release dates.
  - Filtering for modern cinema (release year >= 1990) and excluding recent years to avoid recency bias.
  - Dropping unnecessary columns and handling missing values.

- **Feature Engineering:**
  - Extraction of release year, month, and day from release dates.
  - Creation of categorical variables for time-based analysis.

- **Statistical Analysis & Visualization:**
  - Distribution analysis of vote counts and averages, including log transformations and scatter plots.
  - Investigation of how release timing (year, month, day) affects votes, ratings, popularity, budget, revenue, and runtime.
  - Use of statistical tests (Kolmogorov-Smirnov, Mann-Whitney U) to compare distributions and test hypotheses (e.g., the 'dump month' effect in January).
  - Analysis of genre trends over time and by month using vectorization and stacked bar charts.

- **Key Insights:**
  - Identification of patterns such as the over-representation of older movies in January, recency bias in votes, and the impact of release timing on movie success.
  - Exploration of how budget, revenue, runtime, and genre distributions evolve over time and across months.

## 2. Retrieval-Augmented Generation (RAG) System with Superlinked
After EDA, the notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) system using the [Superlinked](https://superlinked.com/) framework. Superlinked is a Python framework and cloud infrastructure designed for high-performance search and recommendation applications that combine structured and unstructured data.

### RAG System Features:
- **Schema Definition:**
  - Custom schema for movies, including fields like title, rating, release date, runtime, overview, genres, and keywords.
- **Vector Spaces:**
  - Construction of multiple vector spaces for text similarity (title, overview), numerical features (rating, runtime), recency (release date), and categorical similarity (genres, keywords).
- **Indexing & Querying:**
  - Creation of a multi-modal index combining all defined spaces.
  - Support for both semantic (text similarity) and relational (structured filtering) queries.
  - Example queries include searching for movies by natural language, filtering by release date, genre, and keywords, and ranking by relevance.
- **Natural Language Querying:**
  - Integration with OpenAI models to allow natural language queries that are automatically translated into structured search parameters.

### Example Use Cases:
- Find movies similar to a given title or description.
- Filter results by release year, genre, or specific keywords.
- Retrieve top-rated or most popular movies within a certain timeframe.
- Use natural language to ask complex questions about the dataset.
- **Example Query:**
  - Try queries like: "10 most rating of drama movies between the 01-01-2005 and 31-12-2005".
  - Superlinked uses the `NumberSpace` with `mode=MAXIMUM` to retrieve the best-rated movies.
  - The OpenAI LLM (gpt-4o) is leveraged to automatically define the appropriate filters from the natural language query, making the search process intuitive and powerful.

To learn more about Superlinked and its capabilities, visit their [official website](https://superlinked.com/).

---

**Note:** This notebook is intended for data exploration, experimentation, and as a demonstration of integrating modern vector search and RAG techniques with movie data. It provides a template for combining EDA with advanced retrieval systems for rich, interactive data analysis.

