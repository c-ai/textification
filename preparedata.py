import json
from pathlib import Path
import re
import random

tickets_data = Path("dataset") / "Training.txt"
training_data = Path("fasttext_dataset_training.txt")
test_data = Path("fasttext_dataset_test.txt")

# What percent of data to save separately as test data
percent_test_data = 0.10

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

with tickets_data.open(errors='ignore') as input, \
     training_data.open("w") as train_output, \
     test_data.open("w") as test_output:
    
    for line in input:
        fasttext_line = strip_formatting(line.replace("\n", " "))

        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")
    
    lines = open('fasttext_dataset_training.txt').readlines()
    random.shuffle(lines)
    open('fasttext_dataset_training.txt', 'w').writelines(lines)
    
    lines = open('fasttext_dataset_test.txt').readlines()
    random.shuffle(lines)
    open('fasttext_dataset_test.txt', 'w').writelines(lines)