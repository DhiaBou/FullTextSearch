import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

parser = argparse.ArgumentParser(description="Convert CSV to Parquet")

requiredArgs = parser.add_argument_group('required arguments')
requiredArgs.add_argument("-i", "--input", help="Input path to a CSV file", required=True)
requiredArgs.add_argument("-o", "--output", help="Path to the output parquet file", required=True)
requiredArgs.add_argument("-n", "--name", help="Column-name of the text column", required=True)

args = vars(parser.parse_args())

try:

    df = pd.read_csv(args["input"], header=0)
    df_filtered = df[[args["name"]]].rename(columns={args["name"]: "text"})

    print(df_filtered)

    metadata = { b'creator': b'CSVParquetConvert'}
    schema = pa.schema([("text", pa.binary())])

    table = pa.Table.from_pandas(df_filtered, preserve_index=False)
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, args["output"], compression=None)
    print("Conversion successful!")
except Exception as e:
    print(f"An error occurred: {e}")