import subprocess
import argparse
import re



def modify_and_run_token(file_path, similarity_directions, similarity_strictness):
    # Read the original file
    with open(file_path, 'r') as file:
        original_code = file.read()
    
    try:
        # Modify and run for each combination of strictness, direction, and threshold
        for strictness in similarity_strictness:
            for direction in similarity_directions:
                # Determine threshold values based on strictness
                if strictness == 0:
                    thresholds = [0, 100]
                else:
                    thresholds = range(5, 100, 1)
                
                for threshold in thresholds:
                    # Modify the hyper-parameters in the file
                    modified_code = re.sub(r'TOKEN_SIMILARITY_STRICTNESS\s*=\s*\d+', 
                                           f'TOKEN_SIMILARITY_STRICTNESS = {strictness}', 
                                           original_code)
                    modified_code = re.sub(r'TOKEN_SIMILARITY_DIRECTION\s*=\s*\'(greater|lesser)\'', 
                                           f'TOKEN_SIMILARITY_DIRECTION = \'{direction}\'', 
                                           modified_code)
                    modified_code = re.sub(r'TOKEN_SIMILARITY_THRESHOLD\s*=\s*\d+', 
                                           f'TOKEN_SIMILARITY_THRESHOLD = {threshold}', 
                                           modified_code)

                    # Write the modified code back to the file
                    with open(file_path, 'w') as file:
                        file.write(modified_code)
                    
                    # Run the script for the current combination
                    try:
                        # Run the Python file with the '-a token' argument
                        result = subprocess.run(['python3', file_path, '-a', 'token'], 
                                                capture_output=True, text=True, check=True)
                        # Output the result of the run
                        print(f"Run with STRICTNESS = {strictness}, DIRECTION = {direction}, THRESHOLD = {threshold}:")
                        print(result.stdout)
                        
                    except subprocess.CalledProcessError as e:
                        print(f"Error occurred during run with STRICTNESS = {strictness}, DIRECTION = {direction}, THRESHOLD = {threshold}")
                        print(e.stdout)
                        print(e.stderr)
                        # Break after an error
                        break
    
    except Exception as e:
        print("An error occurred. Restoring the original file.")
        print(f"Error: {e}")

    finally:
        # Restore the original code after all runs
        with open(file_path, 'w') as file:
            file.write(original_code)



def modify_and_run_bayes(file_path, percentage_values, repetitions):
    # Read the original file
    with open(file_path, 'r') as file:
        original_code = file.read()

    try: 
        # Modify and run for each percentage value
        for percentage in percentage_values:
            # Replace the BAYES_TRAINING_PERCENTAGE in the code
            modified_code = re.sub(r'BAYES_TRAINING_PERCENTAGE\s*=\s*\d+', 
                                f'BAYES_TRAINING_PERCENTAGE = {percentage}', 
                                original_code)
            
            # Write the modified code back to the file
            with open(file_path, 'w') as file:
                file.write(modified_code)


            # Run the script the specified times for the current combination
            for run_num in range(repetitions):
                try:
                    # Run the Python file with the '-a bayes' argument
                    result = subprocess.run(['python3', file_path, '-a', 'bayes'], capture_output=True, text=True, check=True)
                    # Output the result of the run
                    print(f"Run {run_num + 1} with BAYES_TRAINING_PERCENTAGE = {percentage}:")
                    print(result.stdout)

                except subprocess.CalledProcessError as e:
                    # Handle errors during the run
                    print(f"Error during run {run_num + 1} with BAYES_TRAINING_PERCENTAGE = {percentage}:")
                    print(e.stdout)
                    print(e.stderr)
                    # Break after an error
                    break
    
    except Exception as e:
        print("An error occurred. Restoring the original file.")
        print(f"Error: {e}")

    finally:
        # Restore the original code after all runs
        with open(file_path, 'w') as file:
            file.write(original_code)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Algorithms to run multiple times")
    parser.add_argument("-a", "--algorithm", required=True, choices=['bayes', 'token'], help="Type of algorithm to compute derivation graph: ['bayes', 'token']")
    args = parser.parse_args()

    file = 'derivation_graph.py'
    
    algo = args.algorithm.lower()
    if algo == 'bayes':
        percentage_values = range(10, 95, 5)
        repetitions = 30
        modify_and_run_bayes(file, percentage_values, repetitions)
    elif algo == 'token':
        similarity_directions = ['greater', 'lesser']
        similarity_strictness = [0, 1, 2]
        modify_and_run_token(file, similarity_directions, similarity_strictness)

    
