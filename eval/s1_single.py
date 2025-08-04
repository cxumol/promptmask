from promptmask import PromptMask

import os.path
import json

import httpx
from icecream import ic
from tqdm import trange

from typing import Dict

from util import tomllib, mkdirp, prepare_dataset, TOTAL_LINES, RAW_RESULT_DIR, DATASET_DIR

CONFIG_PATH = "promptmask.config.user.toml"

def main():
    src_txts = prepare_dataset()

    model = tomllib.load(open(CONFIG_PATH,'rb'))['llm_api']['model']
    mkdirp(RAW_RESULT_DIR)
    eval_result_path=RAW_RESULT_DIR+model.replace('/','_')+'.masked.jsonl'
    eval_result_len=sum(1 for _ in open(eval_result_path)) if os.path.isfile(eval_result_path) else 0 # len()-1 <-newlineEOF
    if eval_result_len >= TOTAL_LINES:
        print("Evaluation has been completed. To re-run, please delete result files or change config.")
        exit(0)

    pm = PromptMask(config_file=CONFIG_PATH)
    with open(eval_result_path, 'a+') as f:
        for l in trange(eval_result_len,TOTAL_LINES):
            masked_text , mask_map = pm.mask_str(src_txts[l]) # mask_map:Dict[str,str]
            json.dump({"masked_text":masked_text, "mask_map":mask_map},f)
            f.write('\n')

if __name__ == "__main__":
    main()