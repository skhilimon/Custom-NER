# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:39:02 2023

@author: Oles
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:10:08 2023

@author: Oles
"""

import os
import spacy
from spacy.scorer import Scorer
from spacy.training import Example
import json

class Custom_Scorer:
    def __init__(self):
        # Default scoring pipeline
        self.scorer = Scorer()
        os.chdir(r'C:\Users\Final_Project\Labelling_Outputs')
        output_dir = r'C:\Users\Final_Project\content'
        
        self.nlp = spacy.load(output_dir)
        
    def convert_to_spacy_format(self, json_data):
        training_data = []
        for item in json_data:
            text = item["text"]
            entities = [(label["start"], label["end"], label["labels"][0]) for label in item["label"]]
            example = (text, {"entities": entities})
            training_data.append(example)
        return training_data
    
    def evaluate(self):
        with open('golden_dataset.json', 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
        
        golden_data_converted = self.convert_to_spacy_format(golden_data)
        TEST_DATA = golden_data_converted
        
        examples = []
        for text, annotations in TEST_DATA:
            doc_pred = self.nlp(text)
            example = Example.from_dict(doc_pred, annotations)
            examples.append(example)
        
        # Print or use the scores as needed
        scores = self.scorer.score(examples)
        return scores

# Example usage
custom_scorer = Custom_Scorer()

CONTROL_1 = custom_scorer.evaluate()
print(type(CONTROL_1))

print("Precision:", CONTROL_1['ents_p'])
print("Recall:", CONTROL_1['ents_r'])
print("F1 Score:", CONTROL_1['ents_f'])
print("Scores per type:", CONTROL_1['ents_per_type'])

import warnings
warnings.filterwarnings("ignore")
