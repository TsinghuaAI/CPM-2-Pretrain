import sys
import os
import torch
import copy
import tqdm

filenames = ["cpm-2/32000/mp_rank_0{}_model_states.pt".format(i) for i in range(4)]

output_dir = "expert-32/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "32000"), exist_ok=True)

with open(os.path.join(output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
    f.write(str(32000) + "\n")

preserve_keys = [
    "lr_scheduler",
    "skipped_steps",
    "global_steps",
    "global_samples",
    "dp_world_size",
    "iteration",
    "np_rng_state",
    "random_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
    
]


d_centroids = {}

for layer in range(24):

    k = 'encoder.blocks.{}.ff.expert_centroids'.format(layer)
    d_centroids[k] = torch.empty(32, 4096)
    torch.nn.init.orthogonal_(d_centroids[k], gain=0.01)
    d_centroids[k] = d_centroids[k].half()

    k = 'decoder.blocks.{}.ff.expert_centroids'.format(layer)
    d_centroids[k] = torch.empty(32, 4096)
    torch.nn.init.orthogonal_(d_centroids[k], gain=0.01)
    d_centroids[k] = d_centroids[k].half()

dd = torch.load('cpm-2/32000/mp_rank_00_model_states.pt', map_location='cpu')

print("Increase MP size.")
ratio = 2
for i in range(len(filenames)):
    start = ratio * i
    end = ratio * (i+1)
    d = torch.load(filenames[i], map_location='cpu')
    for j in tqdm.tqdm(range(start, end)):
        d_new = {}
        shift = j - start
        for k, v in dd.items():
            if k != 'module':
                if k in preserve_keys:
                    d_new[k] = copy.deepcopy(dd[k])
                elif k == "mp_world_size":
                    d_new[k] = ratio * len(filenames)
                else:
                    d_new[k] = None
        d_new['model'] = {}
        for k, v in d['module'].items():
            assert len(v.shape) < 3
            if len(v.shape) == 2:
                if 'project.weight' in k:
                    part = v.shape[0] // ratio // 3
                    d_new['model'][k] = torch.cat([v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(shift+1+ratio)*part, :], v[(shift+2*ratio)*part:(shift+1+2*ratio)*part, :]], 0)
                elif 'project_q.weight' in k:
                    part = v.shape[0] // ratio
                    d_new['model'][k] = v[shift*part:(shift+1)*part, :]
                elif 'project_kv.weight' in k:
                    part = v.shape[0] // ratio // 2
                    d_new['model'][k] = torch.cat([v[shift*part:(shift+1)*part, :], v[(shift+ratio)*part:(shift+1+ratio)*part, :]], 0)
                elif 'word_embeds.weight' in k or 'dense_relu_dense.wi_1.weight' in k or 'dense_relu_dense.wi_0.weight' in k or 'lm_head.weight' in k:
                    part = v.shape[0] // ratio
                    d_new['model'][k] = v[shift*part:(shift+1)*part, :]
                    # if j == 0:
                    #     print(k)
                else:
                    part = v.shape[1] // ratio
                    d_new['model'][k] = v[:, shift*part:(shift+1)*part]
            else:
                d_new['model'][k] = v
            
            d_new['model'][k] = d_new['model'][k].half()
            
        keys = list(d_new['model'].keys())
        for k in keys:
            if 'ff' in k:
                new_k = k.replace('ff', 'ff.expert_network')
                d_new['model'][new_k] = d_new['model'][k].clone()
                del d_new['model'][k]
        
        for k, v in d_centroids.items():
            d_new['model'][k] = v

        filename = os.path.join(output_dir, "32000", "mp_rank_{:02d}_model_states.pt".format(j))
        torch.save(d_new, filename)
