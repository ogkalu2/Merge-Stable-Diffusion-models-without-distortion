import argparse
import torch
import os

from weight_matching import sdunet_permutation_spec, weight_matching

parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)

args = parser.parse_args()   
device = args.device

model_a = torch.load(args.model_a, map_location=device)
model_b = torch.load(args.model_b, map_location=device)
state_a = model_a["state_dict"]
state_b = model_b["state_dict"]


permutation_spec = sdunet_permutation_spec()
final_permutation = weight_matching(permutation_spec, state_a, state_b)
              
for w in state_b.keys():
    for axis, p in enumerate(permutation_spec.axes_to_perm[w]):
        if axis == None:
         continue

        if p is not None:
            w = torch.index_select(w, axis, final_permutation[p].int())

for key in state_b.keys():
    state_b[key] = w


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
