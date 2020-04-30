from typing import List, Tuple
import random
import csv
import math


class DataParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.question_pairs = self._read_file_()

    
    def _read_file_(self) -> List[Tuple[str, str, int]]:
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append((row['question1'], row['question2'], row['is_duplicate']))
        return data

    def train_test_split(self, train_part: float = 0.9) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
        train, test = self.question_pairs, None

        ################### INSERT YOUR CODE HERE ###################
        
        ################### INSERT YOUR CODE HERE ###################

        return train, test