from promptmask import PromptMask

import os.path
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
from icecream import ic
from tqdm import tqdm

from util import tomllib, mkdirp, prepare_dataset, fpath_sanitize, fn_timer, TOTAL_LINES, RAW_RESULT_DIR, DATASET_DIR

CONFIG_PATH = "promptmask.config.batch-wrapped.toml"
BATCH_SIZE = 10

prepare_pm = lambda model="": PromptMask(config={"llm_api":{"model":model}}, config_file=CONFIG_PATH)
def get_model_list():
    llm_cfg=tomllib.load(open(CONFIG_PATH,'rb'))['llm_api']
    models=llm_cfg.get("models",None)
    if models and len(models)>0:
        if isinstance(models, list): return models
        if isinstance(models, str): return models.split(',')
    url=llm_cfg['base']+'/models'
    r = httpx.get(url).json()
    return [x['id'] for x in r['data']]

@fn_timer
def main():
    src_txts = prepare_dataset()
    models: [str] = get_model_list()
    ic(models)
    mkdirp(RAW_RESULT_DIR)

    for model in models:
        eval_result_path = RAW_RESULT_DIR + fpath_sanitize(model) + '.masked.jsonl'
        eval_result_len = sum(1 for _ in open(eval_result_path)) if os.path.isfile(eval_result_path) else 0
        if eval_result_len >= TOTAL_LINES:
            print(f"Skipping model '{model}', results already complete.")
            continue

        pm = prepare_pm(model=model)
        
        with open(eval_result_path, 'a+') as f, tqdm(total=TOTAL_LINES, initial=eval_result_len, desc=f"Masking for {model}") as pbar:
            for i in range(eval_result_len, TOTAL_LINES, BATCH_SIZE):
                batch_texts = src_txts[i : i + BATCH_SIZE]
                
                num_workers = min(BATCH_SIZE, len(batch_texts))
                
                results = []
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(pm.mask_str, text) for text in batch_texts]
                    
                    for future in futures:
                        masked_text, mask_map = future.result()
                        results.append({"masked_text": masked_text, "mask_map": mask_map})

                for res in results:
                    json.dump(res, f)
                    f.write('\n')
                
                pbar.update(len(results))

if __name__ == "__main__":
    main()