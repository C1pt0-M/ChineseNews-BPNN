"""Recompute a small sample with the pinned frozen encoder and compare its cache."""
import json
import numpy as np
import torch

import pipeline as p
from transformers import AutoModel, AutoTokenizer

def main():
    target=p.HERE/'encoder_audit.json'
    if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
    splits,_=p.splits_and_baseline()
    vectors,manifest=p.load_vectors(splits)
    tokenizer=AutoTokenizer.from_pretrained(p.MODEL,local_files_only=True)
    model,info=AutoModel.from_pretrained(p.MODEL,local_files_only=True,output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys'] and not info['error_msgs']
    model.requires_grad_(False).eval()
    assert not any(x.requires_grad for x in model.parameters())
    assert sum(x.numel() for x in model.parameters())==manifest['frozen_encoder_parameters']
    dev=torch.device('cuda' if manifest['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    model.to(dev)
    diffs={}
    for name in ('train','val'):
        batch=tokenizer(splits[name][0][:32],padding=True,truncation=True,max_length=512,return_tensors='pt')
        with torch.inference_mode():
            z=model(**{k:v.to(dev) for k,v in batch.items()}).last_hidden_state[:,0,:]
            z=torch.nn.functional.normalize(z,p=2,dim=1).cpu().numpy()
        maxdiff=float(np.max(np.abs(z-vectors[name][:32])))
        assert maxdiff<1e-5, f'{name} first 32 embeddings mismatch: {maxdiff}'
        diffs[name]=maxdiff
    result={'frozen':True,'encoder_parameters':manifest['frozen_encoder_parameters'],
            'batch_per_split':32,'maximum_absolute_difference':diffs,'device':str(dev),
            'cache_manifest_sha256':p.digest(p.CACHE/'manifest.json'), 'test_access':False}
    p.save_json(target,result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
