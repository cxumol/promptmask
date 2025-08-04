import csv,json
from glob import glob
import os.path

from typing import List,Dict

from util import tomllib, mkdirp, load_dataset_masks, TOTAL_LINES, RAW_RESULT_DIR, DATASET_DIR

# DEBUG_RESULT_DIR = "data/debug_result"

def load_result_masks(fpath):
    with open(fpath) as f:
        return [list(json.loads(l).get("mask_map", {}).keys()) for l in f]

def main():
    eval_data = []
    eval_csv_path = "data/eval_result.csv"

    # mkdirp(DEBUG_RESULT_DIR)
    # debug_fpath = os.path.join(DEBUG_RESULT_DIR, "debug.jsonl")

    print("load dataset")
    dataset_masks : List[List[str]] = load_dataset_masks()
    result_fpaths = glob(os.path.join(RAW_RESULT_DIR, "*.masked.jsonl"))
    if not result_fpaths:
        print(f"No eval result files found from {RAW_RESULT_DIR}. Exiting.")
        return

    for fpath in result_fpaths:
        basefname = os.path.basename(fpath)
        print(f"Processing file: {basefname}")
        result_masks = load_result_masks(fpath)

        # debug_writer = open(debug_fpath, 'a')

        tp,fn,fp=0,0,0 # False Negative = Missed by model; False Positive = Incorrectly identified by model
        gt_total, pred_total = 0,0     # ground truth = (TP + FN); prediction = (TP + FP)
        num_lines = min(len(result_masks), len(dataset_masks), TOTAL_LINES)
        valid_lines = num_lines
        for i in range(num_lines):
            gt,pred = set(dataset_masks[i]),set(result_masks[i])
            if not pred or tuple(pred)==("err",):
                valid_lines-=1
                continue
            _tp = gt.intersection(pred) # correct
            _fn = gt.difference(pred) # missed
            _fp = pred.difference(gt) # extra

            tp+=len(_tp)
            fn+=len(_fn)
            fp+=len(_fp)

            gt_total+=len(gt)
            pred_total+=len(pred)

            # debug_writer.write(json.dumps(sorted(list(gt)))+'\n')
            # debug_writer.write(json.dumps(sorted(list(pred)))+'\n')
            # debug_writer.write('\n')
        
        recall = tp / gt_total if gt_total > 0 else 0.0 # TPR
        fnr = fn / gt_total if gt_total > 0 else 0.0 
        fpr = fp / pred_total if pred_total > 0 else 0.0 
        errr = (num_lines-valid_lines) / num_lines if num_lines else 0.0 # error rate

        eval_data.append({
            "model": basefname.split(".masked.jsonl")[0],
            "recall": f"{recall:.2%}", # True Positive Rate, correct rate
            "fnr": f"{fnr:.2%}",      # False Negative Rate, missed rate
            "fp_rate": f"{fpr:.2%}", # False Positive Rate (relative to predictions), wrong pred rate
            "err_rate": f"{errr:.2%}"
        })
        print(f"{basefname}  - Recall: {recall:.2%}, FN: {fnr:.2%}, FP: {fpr:.2%}, Err: {errr:.2%}")

    # Write to a CSV file
    if eval_data:
        print(f"\nWriting eval_data to '{eval_csv_path}'...")
        # Use English abbreviations for headers as requested
        fieldnames = ["model", "recall", "fnr", "fp_rate", "err_rate"]
        with open(eval_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eval_data)
        print("Done.")
    # debug_writer.close()

if __name__ == "__main__":
    main()