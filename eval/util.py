import os,sys
import json
import urllib.request

import functools,time

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

TOTAL_LINES=200
DATASET_DIR="data/"
RAW_RESULT_DIR="data/result_raw/"

mkdirp = lambda dir:os.makedirs(dir, exist_ok=True)
fpath_sanitize = lambda fpath:fpath.replace('/','_').replace(':','_')

def _load_dataset():
    url='https://huggingface.co/datasets/ai4privacy/pii-masking-300k/resolve/main/data/validation/1english_openpii_8k.jsonl'
    mkdirp(DATASET_DIR)
    local_path=DATASET_DIR+url.split('/')[-1]
    if not os.path.isfile(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path

def prepare_dataset()->list:
    local_path=_load_dataset()
    # take .source_text from first TOTAL_LINES lines
    src_txts = []
    with open(local_path, 'r') as f:
        count=0
        for line in f:
            src_txts+=[json.loads(line).get('source_text','')]
            count+=1
            if count==TOTAL_LINES: break
    return src_txts

def load_dataset_masks()->list:
    local_path=_load_dataset()
    # take .source_text from first TOTAL_LINES lines
    mask_vals = []
    with open(local_path, 'r') as f:
        count=0
        for line in f:
            mask_vals+=[ [x['value'] for x in json.loads(line).get('privacy_mask','')] ]
            count+=1
            if count==TOTAL_LINES: break
    return mask_vals


def fn_timer(f):
    @functools.wraps(f)
    def w(*a, **kw):
        s = time.perf_counter()
        r = f(*a, **kw)
        e = time.perf_counter()
        print(f"'{f.__name__}' took {e-s:.6f}s")
        return r
    return w