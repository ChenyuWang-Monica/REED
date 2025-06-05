# generate folders of json files
import json
import os
import pickle
from tqdm import tqdm

base_path = '/your/path/to/data/'
data_path = os.path.join(base_path, 'pmpnn/raw/pdb_2021aug02')  # 'pmpnn/raw/pdb_2021aug02', or 'pmpnn/raw/pdb_2021aug02_sample'


with open(f"{data_path}/cluster_seq_dict_removeX.pkl", "rb") as f:
    cluster_seq_dict_removeX = pickle.load(f)
    

def generate_protein_json(protein_name, protein_sequence, output_file):
    # Create the JSON structure
    data = {
        "name": protein_name,
        "modelSeeds": [0],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_sequence,
                    "unpairedMsa": "",
                    "pairedMsa": ""
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1
    }

    # Write the JSON structure to the file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    # print(f"JSON file generated: {output_file}")

# Example usage
# protein_name = "my_protein"
# protein_sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"
# output_file = "protein_data.json"

# generate_protein_json(protein_name, protein_sequence, output_file)
save_path = '/your/save/path/data/pmpnn/raw/seq_json'

for i, item in tqdm(enumerate(cluster_seq_dict_removeX.items())):
    protein_name = f"cluster_{item[0]}"
    protein_sequence = item[1]
    folderid = int(i // 1000)
    output_path = os.path.join(save_path, f"{folderid}")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{protein_name}.json")
    generate_protein_json(protein_name, protein_sequence, output_file)
