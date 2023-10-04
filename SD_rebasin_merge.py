import argparse
import torch
import os
from pathlib import Path
from safetensors.torch import load_file, save_file

from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation


parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", help="Whether to use half precision", action="store_true", required=False)
parser.add_argument("--usefp32", dest="usefp16", help="Whether to use full precision", action="store_false", required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)
parser.add_argument("--safetensors", type=str, help="Save as safetensors", default=True, required=False)
parser.add_argument("--prune", help="Pruning before merge", action='store_true', default=False, required=False)
parser.add_argument("--fixclip", help="Force to fix clip to int64", action='store_true', default=False, required=False)
parser.set_defaults(usefp16=True)
args = parser.parse_args()   
device = args.device
usefp16 = args.usefp16 

if device == "cpu":
    usefp16 = False
    
def load_model(path, device):
    if path.suffix == ".safetensors":
        return load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device)
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

def prune(model):
    keys = list(model.keys())
    for k in keys:
        if "diffusion_model." not in k and "first_stage_model." not in k and "cond_stage_model." not in k:
            model.pop(k, None)
    return model

if args.model_a is None or args.model_b is None:
    parser.print_help()
    exit(-1)
model_a = load_model(Path(args.model_a), device)
model_b = load_model(Path(args.model_b), device)

if args.prune:
    model_a = prune(model_a)
    model_b = prune(model_b)

theta_0 = model_a
theta_1 = model_b

alpha = float(args.alpha)
iterations = int(args.iterations)
step = alpha/iterations
permutation_spec = sdunet_permutation_spec()
special_keys = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", 
"first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]

if args.usefp16:
    print("Using half precision")
else:
    print("Using full precision")

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

for x in range(iterations):
    print(f"""
    ---------------------
         ITERATION {x+1}
    ---------------------
    """)

    # In order to reach a certain alpha value with a given number of steps,
    # You have to calculate an alpha for each individual iteration
    if x > 0:
        new_alpha = 1 - (1 - step*(1+x)) / (1 - step*(x))
    else:
        new_alpha = step
    print(f"new alpha = {new_alpha}\n")

    # Add the models together in specific ratio to reach final ratio
    for key in theta_0.keys():
        if "model_" in key:
            continue
        if key in checkpoint_dict_skip_on_merge:
            continue
        if "model" in key and key in theta_1:
            theta_0[key] = (1 - new_alpha) * theta_0[key] + new_alpha * theta_1[key]

    if x == 0:
        for key in theta_1.keys():
            if "model" in key and key not in theta_0:
                theta_0[key] = theta_1[key]

    print("FINDING PERMUTATIONS")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(permutation_spec, model_a, theta_0, usefp16=usefp16, device=device)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    second_permutation, z = weight_matching(permutation_spec, model_b, theta_0, usefp16=usefp16, device=device)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)

    new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

    # Weighted sum of the permutations
    
    for key in special_keys:
        theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])

# fix/check bad clip
position_id_key = 'cond_stage_model.transformer.text_model.embeddings.position_ids'
if position_id_key in theta_0:
    correct = torch.tensor([list(range(77))], dtype=torch.int64, device="cpu")
    current = theta_0[position_id_key].to(torch.int64)
    broken = correct.ne(current)
    broken = [i for i in range(77) if broken[0][i]]
    if len(broken) != 0:
        if args.fixclip:
            theta_0[position_id_key] = correct
            print(f"Fixed broken clip\n{broken}")
        else:
            print(f"Broken clip!\n{broken}")
    else:
        print("Clip is fine")

ext = "ckpt" if not args.safetensors else "safetensors"
output_file = f'{args.output}.{ext}'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")

print("\nSaving...")

try:
    if ext == "safetensors":
        save_file(theta_0, output_file, metadata={"format":"pt"})
    else:
        torch.save({"state_dict": theta_0}, output_file)
    print("Done!")
except Exception as e:
    print(f"ERROR: Couldn't save {output_file} - {e}")
