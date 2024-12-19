import argparse
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

def remove_non_numeric_chars(column):
    # Use regex to remove non-numeric characters
    numeric_only = pc.replace_substring_regex(column, r'\D', '')  # \D matches any non-digit character
    return numeric_only

def convert_parquet(input_file, output_file, old_id_col, old_text_col):
    # Read the input Parquet file
    table = pq.read_table(input_file)

    # Rename columns to new names
    table = table.rename_columns([old_id_col, old_text_col])

    # Ensure the 'Id' column is treated as a string to apply regex replacement
    table = table.cast(pa.schema([(old_id_col, pa.string()), (old_text_col, pa.string())]))

    # Remove non-numeric characters from 'Id' column
    cleaned_id_column = remove_non_numeric_chars(table[old_id_col])
    cleaned_table = table.set_column(
        table.schema.get_field_index(old_id_col),
        old_id_col,
        cleaned_id_column
    )

    # Convert 'Id' to uint32
    cleaned_table = cleaned_table.cast(pa.schema([(old_id_col, pa.uint32()), (old_text_col, pa.string())]))

    # Reorder columns: First Text, then Id
    reordered_table = cleaned_table.select([old_text_col, old_id_col])

    # Write the cleaned and converted table to the output Parquet file
    pq.write_table(reordered_table, output_file)

    print(f"Parquet file saved to {output_file}")

def process_directory(input_dir, output_dir, old_id_col, old_text_col):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".parquet"):
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)
            convert_parquet(input_file, output_file, old_id_col, old_text_col)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert Parquet to Parquet with correct columns")

    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument("-i", "--input", help="Input path to a parquet file or a directory of parquet files", required=True)
    requiredArgs.add_argument("-o", "--output", help="Path to the output parquet file or an output directory", required=True)
    requiredArgs.add_argument("-n", "--text", help="Column-name of the text column", required=True)
    requiredArgs.add_argument("-d", "--id", help="Column-name of the id column", required=True)

    args = vars(parser.parse_args())

    if os.path.isdir(args["input"]):
        os.makedirs(args["output"], exist_ok=True)
        process_directory(args["input"], args["output"], args["id"], args["text"])
    else:
        convert_parquet(args["input"], args["output"], args["id"], args["text"])
