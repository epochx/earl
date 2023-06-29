import os
import json
from argparse import ArgumentParser
from tqdm import tqdm


def read_jsonl(file_path):
    with open(file_path) as f:
        data = []
        for line in f.readlines():
            try:
                datum = json.loads(line.strip())
                data.append(datum)
            except json.JSONDecodeError:
                pass
    return data


parser = ArgumentParser()

parser.add_argument("--data")

if __name__ == "__main__":

    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        input_file_path = os.path.join(args.data, f"{split}.json")
        output_file_path = os.path.join(args.data, f"{split}.tsv")

        histories = read_jsonl(input_file_path)
        labels = []
        with open(output_file_path, "w") as f:
            for history in tqdm(histories):
                for edit in history:
                    before = edit["previous_text"]
                    after = edit["text"]
                    line = f"{before}\t###\t{after}\n"
                    f.write(line)
                    label = edit["comment"][0]
                    if label == "":
                        label = "__EMPTY__"
                    labels.append(label)

        labels_output_file_path = os.path.join(args.data, f"{split}.labels.txt")
        with open(labels_output_file_path, "w") as f:
            for label in labels:
                f.write(f"{label}\n")

