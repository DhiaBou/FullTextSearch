import pandas as pd
import argparse
import os

def get_document_rank(queries_file, output_file, result_dir):
    queries_df = pd.read_csv(queries_file, sep='|', header=None, names=["query", "document"], index_col=False)
    output_df = pd.read_csv(output_file, sep='|', header=None, names=["query", "rank", "document", "score"], index_col=False)
    
    results = []
    for _, row in queries_df.iterrows():
        query = row["query"]
        doc_id = row["document"]  # Assuming "rank" in queries_file corresponds to document ID

        print(query)
        print(doc_id)
        
        match = output_df[(output_df["query"] == query) & (output_df["document"] == doc_id)]
        rank = match["rank"].values[0] if not match.empty else -1
        
        results.append({"query": query, "rank": rank, "document": doc_id})
    
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "query_ranks.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the rank of documents for given queries.")
    parser.add_argument("queries_file", type=str, help="Path to the queries TBL file")
    parser.add_argument("output_file", type=str, help="Path to the output TBL file")
    parser.add_argument("result_dir", type=str, help="Directory to save the result CSV file")
    
    args = parser.parse_args()
    get_document_rank(args.queries_file, args.output_file, args.result_dir)
