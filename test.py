import fastText
import re
import sys
from argparse import ArgumentParser


def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

if __name__ == "__main__":

    subject = str(sys.argv[1:])
    # Load the model
    classifier = fastText.load_model('trained_model.ftz')
    # Get fastText to classify each review with the model
    label, probability = classifier.predict(strip_formatting(subject), 1)
    
    print("{} ({}% confidence)".format(label, int(probability * 100)))
    print(subject)
    print()
