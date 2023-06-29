import os
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--dataset")

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = os.path.join(args.path, f"{args.dataset}.jsonl")
    with open(data_path) as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line.strip()))

    base_indices_file_path = os.path.join(args.path, "splits", args.dataset)

    output_path = os.path.join(args.path, args.dataset)
    os.makedirs(output_path)

    for split in ["train", "valid", "test"]:

        indices_file_path = base_indices_file_path + f".{split}.txt"
        with open(indices_file_path) as f:
            indices = list(map(lambda x: int(x.strip()), f.readlines()))

        examples = [data[i] for i in indices]

        output_file_path = os.path.join(output_path, f"{split}.tsv")

        with open(output_file_path, "w") as output_file:
            for example in examples:
                src = " ".join(example["src"])
                tgt = " ".join(example["tgt"])

                label = None
                if "x" in data_path:
                    label = example["category"]
                else:
                    labels_dict = example["tgt_class"]
                    assert sum(labels_dict.values()) == 1
                    for label, value in labels_dict.items():
                        if value == 1:
                            break
                if "wi_plus_locness" in data_path and label == "N":
                    continue
                assert label is not None

                # add ## just to make it look like wiki atomics
                output_line = f"{src}\t{label}\t{tgt}\n"
                output_file.write(output_line)
