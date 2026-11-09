"""Download only the explicitly requested, pinned BGE checkpoint locally."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
os.environ['HF_HOME'] = str(HERE / '.hf_home')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from huggingface_hub import snapshot_download

MODEL = 'BAAI/bge-small-zh-v1.5'
REVISION = '7999e1d3359715c523056ef9478215996d62a620'

if __name__ == '__main__':
    path = snapshot_download(repo_id=MODEL, revision=REVISION,
        local_dir=str(HERE / 'bge_model'),
        allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'README.md'])
    print('Pinned model location:', path, flush=True)
