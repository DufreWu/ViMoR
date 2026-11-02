import json
import random

input_file = "dataset.jsonl"
train_file = "train.jsonl"
test_file = "test.jsonl"

split_ratio = 0.8

# load data
data = []
with open(input_file, "r") as f:
    for line in f:
        item = json.loads(line)

        new_item = {
            "conversations": [
                {"from": "human", "value": item["prompt"]},
                {"from": "gpt", "value": item["response"]}
            ],
            "prefill_length": item["prefill_length"],
            "decode_length": item["decode_length"]
        }

        data.append(new_item)

# shuffle
random.shuffle(data)

# split
split_idx = int(len(data) * split_ratio)

train_data = data[:split_idx]
test_data = data[split_idx:]

# save
with open(train_file, "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(test_file, "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print(f"Train: {len(train_data)}, Test: {len(test_data)}")