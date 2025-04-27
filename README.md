# NIC Code Search Web Application

This is a Flask web application that allows you to search for NIC (National Industrial Classification) codes using different models.

## Features

- Select from 4 different models for NIC code prediction
- Enter a query in natural language
- View top 5 matching NIC codes with confidence scores
- Performance metrics (query processing time)

## Setup and Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install flask pandas numpy
```

3. Run the Flask application:

```bash
python app.py
```

4. Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

## Models

The application supports 5 different models:

1. **Model 1**: Hierarchical search model
2. **Model 2**: Sentence Transformer model
3. **Model 3**: Sentence Transformer with OpenAI query enhancement
4. **Model 4**: Sentence Transformer with enhanced data
5. **Model 5**: Sentence Transformer with DeepSeek query enhancement

Each model is loaded only once when needed to optimize performance.

## Usage

1. Select a model from the dropdown menu
2. Enter your query in the input field
3. Click "Search" to get the results
4. View the top 5 matching NIC codes with confidence scores

## Notes

- The first query for each model might take longer as the model needs to be loaded
- Subsequent queries using the same model will be faster

## Evaluation Dataset

In the evaluation dataset:
- `nic_paraphrased_openai.txt` file contains Semantic queries
- `nic_subclass_codes.txt` file contains Exact queries
- `Vague_Queries.txt` file contains vague queries
