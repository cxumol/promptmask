import csv, os, os.path

current_directory = os.path.dirname(os.path.abspath(__file__))
CSV_FPATH = os.path.join(current_directory, "data/eval_result.csv")
MD_FPATH = os.path.join(current_directory, "benchmark.md")

def main(markdown_table:str):
    tmpl = """# Benchmark Report

## Evaluation Results

{table}

## Metric Definitions

- **err_rate:** `Number_of_Errors / Total_Number_of_Samples` The proportion of samples that failed to process due to system-level errors (e.g., parsing, timeout). 
  > The total number of samples is derived from the source file using the following logic:
    ```python
    sum(len(max_map) for line in read(jsonl_fpath)[:TOTAL_LINES])
    ```

*The following classification metrics are calculated exclusively on successfully processed samples.*

- **recall (TPR):** `TP / (TP + FN)` The proportion of actual positives that were correctly identified. 

- **fnr (False Negative Rate):** `FN / (TP + FN)` The proportion of actual positives that were misclassified as negative. 
    > `fnr = 1 - recall`

- **fp_rate (False Positive Rate):** `FP / (FP + TN)` The proportion of actual negatives that were misclassified as positive. 
"""
    final_report = tmpl.format(table=markdown_table)
    try:
        with open(MD_FPATH, 'w', encoding='utf-8') as mdfile:
            mdfile.write(final_report)
        print(f"Benchmark report successfully generated at: {MD_FPATH}")
    except IOError as e:
        print(f"Error writing to file {MD_FPATH}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")

def csv_to_markdown(input_filepath):
    try:
        with open(input_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            try:
                header = next(reader)
            except StopIteration:
                return "The CSV file is empty."

            data_rows = list(reader)

            md_header = "|" + "|".join(str(cell).strip() for cell in header) + "|"
            md_separator = "|" + "|".join(["---"] * len(header)) + "|"

            md_data_rows = []
            for row in data_rows:
                md_row_cells = []
                for cell_value in row:
                    md_row_cells.append(str(cell_value).strip())
                
                md_data_rows.append("|" + "|".join(md_row_cells) + "|")

            markdown_table_parts = [md_header, md_separator] + md_data_rows
            return "\n".join(markdown_table_parts)

    except FileNotFoundError:
        return f"Error: File not found at '{input_filepath}'"
    except csv.Error as e:
        return f"Error parsing CSV file: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    markdown_table_content = csv_to_markdown(CSV_FPATH)
    
    if markdown_table_content.startswith("Error:") or markdown_table_content.startswith("The CSV file is empty."):
        print(markdown_table_content)
    else:
        main(markdown_table_content)