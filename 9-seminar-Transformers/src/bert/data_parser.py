from typing import List, Tuple
import random
import csv
import math
import os
import urllib.request


class AppURLopener(urllib.request.FancyURLopener):
        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
    

class DataParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.download_data()
        self.question_pairs = self._read_file_()

    def download_data(self):
        QUORA_DATA_LINK = 'http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv'
        DIR = './data/'
        QUORA_FILE_PATH = DIR + 'questions.tsv'
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
        if os.path.exists(QUORA_FILE_PATH):
           return
        urllib._urlopener = AppURLopener()
        urllib._urlopener.retrieve(QUORA_DATA_LINK, QUORA_FILE_PATH)
    
    def _read_file_(self) -> List[Tuple[str, str, int]]:
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            for row in reader:
                data.append((row['question1'], row['question2'], row['is_duplicate']))
        return data

    def train_test_split(self, train_part: float = 0.9) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
        train, test = self.question_pairs, None

        ################### INSERT YOUR CODE HERE ###################
        
        ################### INSERT YOUR CODE HERE ###################

        return train, test