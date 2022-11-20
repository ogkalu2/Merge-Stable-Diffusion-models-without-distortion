import argparse
import torch
import os

from jax import random 
from weight_matching import sdunet_permutation_spec, weight_matching

parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--seed", type=int, default=0, help="Random seed")

args = parser.parse_args()   
device = args.device
seed = args.seed

model_a = torch.load(args.model_a, map_location=device)
model_b = torch.load(args.model_b, map_location=device)
state_a = model_a["state_dict"]
state_b = model_b["state_dict"]


permutation_spec = sdunet_permutation_spec()
final_permutation = weight_matching(random.PRNGKey(seed), permutation_spec, state_a, state_b)
              
for a in state_b.keys():
    w = state_b[a]
    for axis, p in enumerate(permutation_spec.axes_to_perm[a]):
        # Skip the axis we're trying to permute.
        if axis == None:
            continue
        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, final_permutation[p].int())
    state_b[a] = w

output_file = f'{args.output}.ckpt'

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

print("Saving...")

torch.save({
        "state_dict": state_b
            }, output_file)

print("Done!")
