from promptmask import PromptMask

import os.path
import json

import httpx
from icecream import ic
from tqdm import trange

from util import tomllib, mkdirp, prepare_dataset, fpath_sanitize, TOTAL_LINES, RAW_RESULT_DIR, DATASET_DIR

prepare_pm = lambda model="": PromptMask(config={"llm_api":{"model":model},"general":{"verbose":False}})

CONFIG_PATH = "promptmask.config.batch.toml"

def get_model_list():
    llm_cfg=tomllib.load(open(CONFIG_PATH,'rb'))['llm_api']
    models=llm_cfg.get("models",None)
    if models and len(models)>0:
        if isinstance(models, list): return models
        if isinstance(models, str): return models.split(',')
    url=llm_cfg['base']+'/models'
    r = httpx.get(url).json()
    return [x['id'] for x in r['data']]

def main():
    src_txts = prepare_dataset()
    models: [str] = get_model_list()
    ic(models)
    mkdirp(RAW_RESULT_DIR)
    for model in models:
        eval_result_path=RAW_RESULT_DIR+fpath_sanitize(model)+'.masked.jsonl'
        eval_result_len=sum(1 for _ in open(eval_result_path)) if os.path.isfile(eval_result_path) else 0 # len()-1 <-newlineEOF
        if eval_result_len >= TOTAL_LINES: continue

        pm = prepare_pm(model=model)
        with open(eval_result_path, 'a+') as f:
            for l in trange(eval_result_len,TOTAL_LINES):
                masked_text, mask_map = pm.mask_str(src_txts[l])
                json.dump({"masked_text":masked_text, "mask_map":mask_map},f)
                f.write('\n')

if __name__ == "__main__":
    main()