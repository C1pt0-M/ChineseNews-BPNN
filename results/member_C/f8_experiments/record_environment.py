"""Record the F.8 experiment environment without installing or altering packages."""
import importlib.metadata
import json
import platform
import sys

import pipeline as p
import torch

def main():
    target=p.HERE/'environment.json'
    if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
    packages=['torch','transformers','tokenizers','huggingface_hub','safetensors','regex','tqdm',
              'numpy','scipy','scikit-learn','pandas','jieba','matplotlib']
    versions={}
    for package in packages:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    files=[p.HERE/'pipeline.py',p.HERE/'download_model.py',p.HERE/'review_seed42.py',
           p.HERE/'verify_encoder.py',p.HERE/'analyze_f8.py',p.ROOT/'models/bp.py',
           p.ROOT/'models/trainer.py',p.ROOT/'models/f_features.py',p.ROOT/'models/features.py',
           p.ROOT/'model_config.py']
    data={'created_after_training':True,'source_hash_scope':'recorded at audit time after training',
          'python':sys.version,'platform':platform.platform(),'torch_threads':torch.get_num_threads(),
          'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'package_versions':versions,
          'source_sha256':{str(f.relative_to(p.ROOT)):p.digest(f) for f in files},
          'model_weights_sha256':p.digest(p.MODEL/'model.safetensors'),
          'cache_manifest_sha256':p.digest(p.CACHE/'manifest.json'),
          'local_dependencies':'results/member_C/f8_experiments/vendor; no system-level package install'}
    p.save_json(target,data)
    print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
