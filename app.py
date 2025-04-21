from flask import Flask, render_template, request, jsonify
import importlib
import pandas as pd
import time
import os
import sys
import traceback

app = Flask(__name__)

# Dictionary to store loaded models
loaded_models = {}

def load_model(model_number):
    """Load a specific model if not already loaded"""
    if model_number in loaded_models:
        return loaded_models[model_number]

    try:
        # Import the model module
        model_module = importlib.import_module(f"model{model_number}")

        # Get the get_top5_nic_codes function
        get_top5_nic_codes = getattr(model_module, "get_top5_nic_codes")

        # Store the function in the loaded_models dictionary
        loaded_models[model_number] = get_top5_nic_codes

        return get_top5_nic_codes
    except Exception as e:
        print(f"Error loading model {model_number}: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        # Get the model number and query from the form
        model_number = int(request.form.get('model', 1))
        query = request.form.get('query', '')

        if not query:
            return jsonify({
                'success': False,
                'error': 'Please enter a query'
            })

        # Load the model
        get_top5_nic_codes = load_model(model_number)

        if get_top5_nic_codes is None:
            return jsonify({
                'success': False,
                'error': f'Failed to load model {model_number}'
            })

        # Time the prediction
        start_time = time.time()
        results = get_top5_nic_codes(query)
        elapsed_time = time.time() - start_time

        # Convert results to a list of dictionaries
        if isinstance(results, pd.DataFrame):
            results_list = []
            for _, row in results.iterrows():
                # Ensure confidence is a positive value between 0 and 1
                confidence = float(row['confidence'])
                if confidence < 0:
                    confidence = 0
                elif confidence > 1:
                    confidence = confidence / 100 if confidence > 1 else confidence

                results_list.append({
                    'code': row['Sub-class Code'],
                    'name': row['Sub-class Name'],
                    'confidence': confidence
                })

            # Try to get the enhanced query if available (for models that enhance queries)
            enhanced_query = query
            try:
                # Check if the model has an enhanced_query attribute
                if hasattr(results, 'enhanced_query'):
                    enhanced_query = results.enhanced_query
                # For model4, check if there's an enhanced query in the global scope
                elif model_number == 4 and 'model4' in sys.modules:
                    model4 = sys.modules['model4']
                    if hasattr(model4, 'last_enhanced_query') and model4.last_enhanced_query:
                        enhanced_query = model4.last_enhanced_query
            except Exception as e:
                print(f"Error getting enhanced query: {e}")

            return jsonify({
                'success': True,
                'results': results_list,
                'time': elapsed_time,
                'query': query,
                'enhanced_query': enhanced_query if enhanced_query != query else None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model did not return expected results'
            })

    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Run the Flask app
    app.run(debug=True)
