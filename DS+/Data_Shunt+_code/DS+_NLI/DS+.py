import json
import csv
import numpy as np


def softmax(x):
    """Compute the softmax of a vector."""
    # Compute the exponential of each element in x
    exp_x = np.exp(x)
 
    # Take the sum of all the elements in exp_x
    sum_exp_x = np.sum(exp_x)
 
    # Compute the softmax
    softmax = exp_x / sum_exp_x
 
    return softmax


small_path = "model_predictions_for_snli_mnli.json"
test_path = "chaosNLI_mnli_m.jsonl"
large_pre = "llama3-mnli.tsv"

with open(small_path, 'r') as file:
    data = json.load(file)

test_id = []
label = []
large_predict = []
fine_predict=[]
with open(test_path, 'r') as file:
    for line in file:
        # read json data
        json_data = json.loads(line)
        test_id.append(json_data["example"]["uid"])

        if json_data["majority_label"] == "e":
            label.append(0)
        elif json_data["majority_label"] == "n":
            label.append(1)
        else: 
            label.append(2)

# Large Model Acc
with open(large_pre, 'r', encoding='utf-8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for row in reader:
        if row[-1] == 'Entailment':
            large_predict.append(0)
        elif row[-1] == 'Neutral':
            large_predict.append(1)
        else:
            large_predict.append(2)



count = 0
for i in range(1599):
    if large_predict[i] == label[i]:
        count += 1
print("large:",count/1599)

#DS+ ACC  
count = 0
count_id = 0
large_count = 0
for key in data["xlnet-large"]:
    if key in test_id:
        count_id += 1
        logits = data["xlnet-large"][key]["logits"]
        predict = logits.index(max(logits))
        # if predict == label[count_id-1]:
        #     count += 1

        if max(softmax(logits))>0.9:
            if predict == label[count_id-1]:
                count += 1
        else:
            predict = large_predict[count_id-1]
            if predict == label[count_id-1]:
                count += 1
                large_count += 1
print("xlnet acc:",count/1599)
print("xlnet proportion:",large_count/1599)

#Small Model ACC
count = 0
count_id = 0
large_count = 0
for key in data["xlnet-large"]:#distilbert
    if key in test_id:
        count_id += 1
        logits = data["xlnet-large"][key]["logits"] # di
        predict = logits.index(max(logits))
        if predict == label[count_id-1]:
            count += 1

print("xlnet-large acc:",count/1599)







