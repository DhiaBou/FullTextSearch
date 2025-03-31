import argparse
import os
import pyarrow as pa
import pyarrow.parquet as pq

global_id_counter = 0

def convert_parquet(input_file, output_file, old_text_col):
    global global_id_counter  # Zugriff auf die globale Variable

    # Read the input Parquet file
    table = pq.read_table(input_file)

    # Ensure the text column exists
    if old_text_col not in table.column_names:
        raise ValueError(f"Column '{old_text_col}' not found in {input_file}")

    # Create a new ID column with sequential values continuing from global counter
    num_rows = table.num_rows
    new_id_column = pa.array(range(global_id_counter, global_id_counter + num_rows), type=pa.uint32())

    # Update global counter
    global_id_counter += num_rows

    # Create new table with the new ID column
    new_table = pa.Table.from_arrays([table[old_text_col], new_id_column], names=[old_text_col, "Id"])

    # Write the transformed table to the output Parquet file
    pq.write_table(new_table, output_file)

    print(f"Parquet file saved to {output_file}")

def process_directory(input_dir, output_dir, old_text_col):
    global global_id_counter  # Sicherstellen, dass die ID global bleibt

    for file_name in sorted(os.listdir(input_dir)):  # Sortierte Reihenfolge für Konsistenz
        if file_name.endswith(".parquet"):
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)
            convert_parquet(input_file, output_file, old_text_col)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet to Parquet with a new sequential ID column")

    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument("-i", "--input", help="Input path to a parquet file or a directory of parquet files", required=True)
    requiredArgs.add_argument("-o", "--output", help="Path to the output parquet file or an output directory", required=True)
    requiredArgs.add_argument("-n", "--text", help="Column-name of the text column", required=True)

    args = vars(parser.parse_args())

    if os.path.isdir(args["input"]):
        os.makedirs(args["output"], exist_ok=True)
        process_directory(args["input"], args["output"], args["text"])
    else:
        convert_parquet(args["input"], args["output"], args["text"])
