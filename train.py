from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pathlib import Path
from fastText import train_supervised


training_data = os.path.join("fasttext_dataset_training.txt")
test_data = os.path.join("fasttext_dataset_test.txt")

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

if __name__ == "__main__":
    train_data = training_data
    valid_data = test_data

    model = train_supervised(
        input=train_data, epoch=40, lr=0.75, wordNgrams=3, verbose=2, minCount=1, lrUpdateRate=50,
        bucket=100000, dim=100, loss="softmax"
    )
    print_results(*model.test(valid_data))
    
    model.save_model("tickets.bin")

    model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    print_results(*model.test(valid_data))
    model.save_model("tickets.ftz")