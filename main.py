from help_function import process_multiple_columns

# Process multiple columns (example)
columns_to_process = list(range(20, 27)) # Process columns 20, 21, 22, 23, 24, 25, 26
process_multiple_columns(columns_to_process, use_cuda=False)
