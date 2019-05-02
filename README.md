# textification
Simplest text classification implemented with Faceboo FastText library

# Instructions
Make sure you have `fastText` python library installed

1. Fill in the `dataset\Training.txt` file with your raw training data in the format of `__label__{customized} {Text need be classfied}` 

2. Run the `preparedata.py` 
  This will generate two txt files: `fasttext_dataset_test.txt` and `fasttext_dataset_training.txt`

3. Run the `train.py` which will read the two txt files generated from step 2 and then generate two models `trained_model.bin` and `trained_model.ftz`(quantized,compact size) 
