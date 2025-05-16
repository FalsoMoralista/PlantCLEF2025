#!/usr/bin/env /usr/bin/python3
import pandas as pd
import csv
import argparse
import sys

args = argparse.ArgumentParser(description="Postprocess the submission file.")
args.add_argument(
    "--input_file",
    type=str,
    default="submission.csv",
    help="Path to the input CSV file.",
)
args.add_argument(
    "--output_file",
    type=str,
    default="submission_postprocessed.csv",
    help="Path to the output CSV file.",
)

args = args.parse_args(sys.argv[1:])

if __name__ == '__main__':
    data = pd.read_csv(args.input_file)

    def process(row):
        ids = row["species_ids"]
        ids = ids.replace('["', "")
        ids = ids.replace('"]', "")
        ids = ids.replace("'", "")
        ids = ids.replace("[", "")
        ids = ids.replace("]", "")
        ids = ids.split(", ")
        ids = [int(i) for i in ids]
        row["species_ids"] = ids
        return row

    data.apply(process, axis=1)

    data.to_csv(args.output_file, index=False, sep=",", quoting=csv.QUOTE_ALL)
    print(f"Postprocessed submission file saved to {args.output_file}")

