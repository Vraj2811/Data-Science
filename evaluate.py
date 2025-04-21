import importlib
import pandas as pd
import time
import sys
import os
import datetime

# Import all models dynamically
model_modules = {}
for i in range(1, 5):  # Models 1 to 4
    try:
        model_modules[i] = importlib.import_module(f"model{i}")
        print(f"Successfully imported model{i}")
    except Exception as e:
        print(f"Error importing model{i}: {e}")
        model_modules[i] = None

def get_predictions(query, model_number):
    """Get predictions from the specified model"""
    if model_modules[model_number] is None:
        print(f"Model {model_number} is not available")
        return [], 0

    try:
        # Get the get_top5_nic_codes function from the module
        get_top5_nic_codes = getattr(model_modules[model_number], "get_top5_nic_codes")

        # Time the prediction
        start_time = time.time()
        results = get_top5_nic_codes(query)
        elapsed_time = time.time() - start_time

        # Check if results is a DataFrame
        if isinstance(results, pd.DataFrame):
            return results["Sub-class Code"].tolist(), elapsed_time
        else:
            print(f"Warning: Model {model_number} did not return a DataFrame")
            return [], elapsed_time
    except Exception as e:
        print(f"Error getting predictions from model {model_number}: {e}")
        return [], 0


def calculate_score(true_code, predicted_codes):

    weights = [1.0, 0.95, 0.9, 0.85, 0.8]  # More emphasis on top predictions
    score = 0.0
    max_possible_score = sum(weights)  # Maximum possible score

    # Clean the true code (remove any quotes if present)
    true_code = true_code.strip("'\"")

    # If no predictions, return 0
    if not predicted_codes:
        return 0.0

    for i, pred_code in enumerate(predicted_codes):
        if i >= len(weights):  # Only consider up to 5 predictions
            break

        # Clean the predicted code (remove quotes)
        clean_pred = pred_code.strip("'\"")

        # Calculate similarity based on matching digits from left to right
        match_length = 0
        for j in range(min(len(true_code), len(clean_pred))):
            if true_code[j] == clean_pred[j]:
                match_length += 1
            else:
                break

        # Calculate match score based on the number of matching digits
        if match_length == 0:
            match_score = 0.0
        elif match_length == 1:  # Section level match
            match_score = 0.0
        elif match_length == 2:  # Division level match
            match_score = 0.90
        elif match_length == 3:  # Group level match
            match_score = 0.95
        elif match_length == 4:  # Class level match
            match_score = 0.98
        else:  # Exact match (all 5 digits)
            match_score = 1.0

        # Apply position weight
        score += weights[i] * match_score

    # Normalize the score
    return score / max_possible_score


def evaluate_model_on_file(model_number, file_path, output_file=None, limit=None):
    """Evaluate a specific model on a specific file"""
    total_score = 0.0
    num_queries = 0
    total_time = 0.0
    results = []

    print(f"\nEvaluating Model {model_number} on {file_path}")
    print("-" * 50)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header lines if they exist
            if lines and lines[0].startswith('#'):
                lines = lines[3:]

            # Limit the number of queries if specified
            if limit and limit > 0:
                lines = lines[:limit]

            for line in lines:
                if ' - ' not in line:
                    continue

                true_code, query = line.strip().split(' - ', 1)
                predicted_codes, elapsed_time = get_predictions(query, model_number)

                score = calculate_score(true_code, predicted_codes)
                total_score += score
                total_time += elapsed_time
                num_queries += 1

                result = {
                    "model": f"model{model_number}",
                    "query": query,
                    "true_code": true_code,
                    "predicted_codes": predicted_codes,
                    "score": score,
                    "time": elapsed_time
                }
                results.append(result)

                print(f"Query: {query}\nTrue Code: {true_code}\nPredicted Codes: {predicted_codes}\nScore: {score:.4f}\nTime: {elapsed_time:.4f}s\n")

        accuracy = total_score / num_queries if num_queries > 0 else 0
        avg_time = total_time / num_queries if num_queries > 0 else 0
        print(f"Model {model_number} on {file_path}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average time per query: {avg_time:.4f}s")
        print(f"Total queries: {num_queries}")

        # Write results to output file if specified
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\nModel {model_number} on {file_path}:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Average time per query: {avg_time:.4f}s\n")
                f.write(f"Total queries: {num_queries}\n")
                f.write("-" * 50 + "\n")

        return accuracy, avg_time, num_queries, results

    except Exception as e:
        print(f"Error evaluating model {model_number} on {file_path}: {e}")
        return 0, 0, 0, []

def evaluate_all_models_on_all_files(files, output_file="evaluation_results.txt", limit=None):
    """Evaluate all models on all files"""
    # Create a timestamp for the results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"models1-4_results_{timestamp}.txt")

    # Clear the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results for Models 1-4\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")

    all_results = {}

    for model_number in range(1, 5):  # Models 1 to 4
        if model_modules[model_number] is None:
            continue

        model_results = {}
        for file_path in files:
            accuracy, avg_time, num_queries, results = evaluate_model_on_file(
                model_number, file_path, output_file, limit
            )
            model_results[file_path] = {
                "accuracy": accuracy,
                "avg_time": avg_time,
                "num_queries": num_queries,
                "results": results
            }
        all_results[f"model{model_number}"] = model_results

    # Print summary table
    print("\nSummary of Results:")
    print("-" * 81)
    print(f"{'Model':<10} | {'File':<26} | {'Accuracy':<10} | {'Avg Time (s)':<15} | {'Queries':<10}")
    print("-" * 81)

    for model_name, model_results in all_results.items():
        for file_path, results in model_results.items():
            print(f"{model_name:<10} | {file_path[-26:]:<26} | {results['accuracy']:<10.4f} | {results['avg_time']:<15.4f} | {results['num_queries']:<10}")

    print("-" * 81)

    # Write summary to output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\nSummary of Results:\n")
        f.write("-" * 81 + "\n")
        f.write(f"{'Model':<10} | {'File':<26} | {'Accuracy':<10} | {'Avg Time (s)':<15} | {'Queries':<10}\n")
        f.write("-" * 81 + "\n")

        for model_name, model_results in all_results.items():
            for file_path, results in model_results.items():
                f.write(f"{model_name:<10} | {file_path[-26:]:<26} | {results['accuracy']:<10.4f} | {results['avg_time']:<15.4f} | {results['num_queries']:<10}\n")

        f.write("-" * 81 + "\n")

    print(f"\nDetailed results saved to: {output_file}")
    return all_results

# Files to evaluate
files_to_evaluate = [
    "nic_queries.txt",
    "nic_subclass_codes.txt",
    "nic_paraphrased_openai.txt"
]

# Check if a limit was provided as a command-line argument
limit = None
if len(sys.argv) > 1:
    try:
        limit = int(sys.argv[1])
        print(f"Limiting evaluation to {limit} queries per file")
    except ValueError:
        print(f"Invalid limit argument: {sys.argv[1]}")

# Run the evaluation
if __name__ == "__main__":
    evaluate_all_models_on_all_files(files_to_evaluate, limit=limit)