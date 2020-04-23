# Created by: c00k1ez (https://github.com/c00k1ez)

from typing import List, Tuple
import random


class DataParser:

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.raw_data = self._read_file()

    def _read_file(self) -> List[Tuple[str, str]]:
        raw_data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                line = line.split('\t')
                raw_data.append((line[0], line[1]))
        return raw_data

    def train_test_split(self, train_part: float = 0.7) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        random.shuffle(self.raw_data)
        train_len = int(train_part * len(self.raw_data))
        train, test = self.raw_data[:train_len], self.raw_data[train_len:]
        return train, test

    def split_by_languages(self) -> Tuple[List[str], List[str]]:
        ru = []
        eng = []
        for sample in self.raw_data:
            ru.append(sample[1])
            eng.append(sample[0])
        return eng, ru
