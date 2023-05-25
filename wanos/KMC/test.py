import json, yaml

# Open the JSON file
with open('PF.json', 'r') as f:
    data = json.load(f)


# num_keys = len(data["e14"])
# print("Number of keys:", num_keys)

num_index = len(data["e1"])

outdict = {"iter": 0, "max_index": num_index}
with open("output_dict.yml",'w') as out:
    yaml.dump(outdict, out, default_flow_style=False)

# # Compute the length of the JSON data
# length = len(json.dumps(data))

# print("Length of JSON data:", length)

