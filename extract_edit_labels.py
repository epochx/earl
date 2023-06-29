import json
import re
import tqdm
from itertools import chain
import transformers
from multiprocessing import Pool, cpu_count
import sys, os
from characters import characters
import Levenshtein

from argparse import ArgumentParser

tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")

parser = ArgumentParser()
parser.add_argument("--data")


def extract_edit(string1, string2):
    str1 = tokenizer.tokenize(string1)
    if str1 == []:
        str1 = tokenizer.tokenize("<mask>")
    str2 = tokenizer.tokenize(string2)
    tokens = {j: i for i, j in enumerate(list(set(str1 + str2)))}
    st1 = "".join([characters[tokens[x]] for x in str1])
    st2 = "".join([characters[tokens[x]] for x in str2])
    output_list = ["KEEP"] * len(str1)
    output = Levenshtein.editops(st1, st2)
    #     print(output,"\n",str1,"\n",str2)
    indices = [i[1] for i in output if (i[0] != "insert" and i[0] != "delete")]
    other_indices = [
        i[2] for i in output if (i[0] != "insert" and i[0] != "delete")
    ]
    delete_indices = [i[1] for i in output if i[0] == "delete"]

    new_str2 = {}
    tag_per_span = {}
    for something in output:
        if something[0] != "delete":
            try:
                new_str2[something[1]].append((str2[something[2]]))
                tag_per_span[something[1]].append(something[0])
            except KeyError:
                new_str2[something[1]] = [str2[something[2]]]
                tag_per_span[something[1]] = [something[0]]
    for key in new_str2:
        new_str2[key] = tokenizer.convert_tokens_to_string(new_str2[key])
    only_insert_indices = []
    for key in tag_per_span:
        if (
            len(set(tag_per_span[key])) == 1
            and tag_per_span[key][0] == "insert"
        ):
            only_insert_indices.append(key - 1)

    # m2f = [str2[min(i, len(str2) - 1)] for i in other_indices]
    kept_mask_indices = []
    spans = []
    span_info = []
    for x in indices:
        try:
            output_list[x] = "REPLACE"
            kept_mask_indices.append(x)
        except:
            pass
    kept_mask_indices = set(kept_mask_indices)
    kept_deletion_indices = []
    for x in delete_indices:
        try:
            output_list[x] = "DELETE"
            kept_deletion_indices.append(x)
        except:
            pass

    kept_insert_indices = []
    for x in only_insert_indices:
        try:
            output_list[x] = "INSERT"
            kept_insert_indices.append(x)
        except:
            pass

    if output_list == []:
        output_list.append("REPLACE")
    if set(output_list) == {"KEEP"}:
        return [None] * 4

    for x in range(len(output_list)):
        if output_list[x] == "INSERT":
            spans.append([x])
            continue
        if output_list[x] != "KEEP" and x == 0:
            spans.append([x])
        elif output_list[x] != "KEEP" and output_list[x - 1] != output_list[x]:
            spans.append([x])

        if output_list[x] != "KEEP" and x == (len(output_list) - 1):
            spans[-1].append(x)
        elif (
            output_list[x] != "KEEP"
            and output_list[x + 1] != output_list[x]
            and output_list[x - 1] == output_list[x]
        ):
            spans[-1].append(x)

    mask_spans = []
    for i, span in enumerate(spans):
        if len(span) == 1:
            span_info.append(
                [
                    tokenizer.convert_tokens_to_string(str1[span[0]]),
                    output_list[span[0]],
                ]
            )
        elif len(span) == 2:
            span_info.append(
                [
                    tokenizer.convert_tokens_to_string(
                        str1[span[0] : span[1] + 1]
                    ),
                    output_list[span[0]],
                ]
            )

        if span_info[-1][1] == "REPLACE" or span_info[-1][1] == "INSERT":
            mask_spans.append(i)

    kept_deletion_indices = set(kept_deletion_indices)
    # new_list = output_list.copy()
    # for i, j in enumerate(output_list):
    #     if j == "KEEP":
    #         new_list[i] = str1[i]

    mask2fill = [""]  # str1.copy()
    # print(mask2fill[:5])

    for i, token in enumerate(str1):
        if i in kept_deletion_indices:
            continue
        elif i in kept_mask_indices:
            if mask2fill[-1] != "<mask>":
                mask2fill.append("<mask>")
            else:
                continue
        elif i in kept_insert_indices:
            mask2fill.append(token)
            mask2fill.append("<mask>")
        else:
            mask2fill.append(token)
    try:
        # former_key = 0
        for j, i in enumerate(sorted(new_str2.keys())):
            # sep = ["</s>"] if (i - former_key > 1 and j != 0) else []
            # try:
            span_info[mask_spans[j]].append(new_str2[i])
            # except IndexError:
            #     output_list[i] == "INSERT"
            #     mask2fill = mask2fill[:i]+["<mask>"]+mask2fill[i:]
            #     span_info[].append(new_str2[i])
            #     span_info
            #     # former_key = i
    except IndexError:
        pass

    mask2fill = tokenizer.convert_tokens_to_string(mask2fill)

    return mask2fill, output_list, spans, span_info


def edison_extract_edit(tokenizer, string_before, string_after):
    # this is a partial re-write of Machel's extract edit

    # tokenizing inputs
    tokenized_before = tokenizer.tokenize(string_before)
    if tokenized_before == []:
        tokenized_before = tokenizer.tokenize("<mask>")
    tokenized_after = tokenizer.tokenize(string_after)

    # create sef of unique indices for tokens
    # replace tokens with unique characters to feeed into Levenshtein module
    token2index = {
        j: i
        for i, j in enumerate(list(set(tokenized_before + tokenized_after)))
    }
    chars_before = "".join(
        [characters[token2index[token]] for token in tokenized_before]
    )
    chars_after = "".join(
        [characters[token2index[token]] for token in tokenized_after]
    )
    editops = Levenshtein.editops(chars_before, chars_after)
    # editops looks like: [('replace', 2, 2), ('replace', 3, 3)]

    # prepare outputs
    output_before = ["KEEP"] * len(chars_before)
    output_after = ["KEEP"] * len(chars_after)

    new_str2 = defaultdict(list)
    tag_per_span = defaultdict(list)
    for editop, i, j in editops:
        if editop != "delete":
            new_str2[i].append(chars_after[j])
            tag_per_span[i].append(editop)
    new_str2 = dict(new_str2)
    tag_per_span = dict(new_str2)

    for key in new_str2:
        new_str2[key] = tokenizer.convert_tokens_to_string(new_str2[key])

    only_insert_indices = []
    for key in tag_per_span:
        if (
            len(set(tag_per_span[key])) == 1
            and tag_per_span[key][0] == "insert"
        ):
            only_insert_indices.append(key - 1)

    kept_mask_indices = []
    kept_deletion_indices = []
    kept_insert_indices = []
    for editop, i, j in editops:
        try:
            if editop == "replace":
                output_before[i] = "REPLACE"
                output_after[j] = "REPLACER"
                kept_mask_indices.append(i)
            elif editop == "delete":
                output_before[i] = "DELETE"
                kept_deletion_indices.append(i)
            else:
                output_before[i] = "INSERT"
                output_after[j] = "INSERTER"
                kept_insert_indices.append(i)
        except:
            pass
    kept_mask_indices = set(kept_mask_indices)

    if output_before == []:
        output_before.append("REPLACE")
    if set(output_before) == {"KEEP"}:
        return [None] * 4, [None] * 4

    return output_before, output_after


def extract_edit_labels(tokenizer, string_before, string_after):
    # tokenizing inputs
    tokenized_before = tokenizer.tokenize(string_before)
    if tokenized_before == []:
        tokenized_before = tokenizer.tokenize("<mask>")
    tokenized_after = tokenizer.tokenize(string_after)

    # create sef of unique indices for tokens
    # replace tokens with unique characters to feeed into Levenshtein module
    token2index = {
        j: i
        for i, j in enumerate(list(set(tokenized_before + tokenized_after)))
    }
    chars_before = "".join(
        [characters[token2index[token]] for token in tokenized_before]
    )
    chars_after = "".join(
        [characters[token2index[token]] for token in tokenized_after]
    )
    opcodes = Levenshtein.opcodes(chars_before, chars_after)

    output_before = [None] * len(chars_before)
    output_after = [None] * len(chars_after)
    changed_tokens = []
    for opcode, start_before, end_before, start_after, end_after in opcodes:
        if opcode == "equal":
            for index in range(start_before, end_before):
                output_before[index] = "KEEP"
            for index in range(start_after, end_after):
                output_after[index] = "KEEP"
        elif opcode == "replace":
            for index in range(start_before, end_before):
                output_before[index] = "REPLACE"
                changed_tokens.append(tokenized_before[index])
            for index in range(start_after, end_after):
                output_after[index] = "REPLACER"
                changed_tokens.append(tokenized_after[index])
        elif opcode == "delete":
            for index in range(start_before, end_before):
                output_before[index] = "DELETE"
                changed_tokens.append(tokenized_before[index])

    # second pass for inserts
    for opcode, start_before, end_before, start_after, end_after in opcodes:
        if opcode == "insert":
            # mark location of single-token insertion in output_before
            try:
                if start_before == end_before:
                    output_before[start_before - 1] = "INSERT"
                else:
                    for index in range(start_before, end_before):
                        output_before[index - 1] = "INSERT"
            except IndexError:
                import ipdb

                ipdb.set_trace()
            for index in range(start_after, end_after):
                output_after[index] = "INSERTER"
                changed_tokens.append(tokenized_after[index])

    assert None not in output_before
    assert None not in output_after

    changed_tokens = list(set(changed_tokens))
    changed_tokens = tokenizer.convert_tokens_to_string(changed_tokens)

    return output_before, output_after, changed_tokens


if __name__ == "__main__":
    args = parser.parse_args()

    file = args.data
    labels_0 = []
    labels_1 = []
    labels_2 = []

    with open(file, "r") as f:
        for line in tqdm.tqdm(f):
            string_before, _, string_after = line.strip().split("\t")

            (
                labels_before,
                labels_after,
                changed_tokens_string,
            ) = extract_edit_labels(tokenizer, string_before, string_after)
            labels_0.append(labels_before)
            labels_1.append(labels_after)
            labels_2.append(changed_tokens_string)

    labels_0_path = file.replace(".tsv", ".label0")
    with open(labels_0_path, "w") as f:
        # f.write("source\ttarget\n")
        f.write("\n".join([" ".join(s) for s in labels_0]))
    print(f"Written {labels_0_path}")

    labels_1_path = file.replace(".tsv", ".label1")
    with open(labels_1_path, "w") as f:
        # f.write("source\ttarget\n")
        f.write("\n".join([" ".join(t) for t in labels_1]))
    print(f"Written {labels_1_path}")

    labels_2_path = file.replace(".tsv", ".label2")
    with open(labels_2_path, "w") as f:
        # f.write("source\ttarget\n")
        f.write("\n".join([t for t in labels_2]))
    print(f"Written {labels_2_path}")

