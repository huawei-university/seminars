# Created by: c00k1ez (https://github.com/c00k1ez)

from typing import List, Dict, Tuple


class Dialogue:
    def __init__(self, raw_dialog: str) -> None:
        self.raw_dialog = raw_dialog
        self.sentencies = self._parse_sententices_()

    def _parse_sententices_(self) -> List[str]:
        sentencies = self.raw_dialog.split('\n')
        sentencies = [sentence.replace('\n', '') for sentence in sentencies if len(sentence) > 1]
        return sentencies
    
    def get_pairs(self) -> List[Dict[str, str]]:
        pairs = []
        for ind in range(len(self.sentencies) - 1):
            pairs.append({
                'context': self.sentencies[ind],
                'answer': self.sentencies[ind + 1]
            })
        return pairs


class DataParser:
    def __init__(self, file_path: str) -> None:
      self.file_path = file_path
      self.all_pairs, self.dialogues = self._read_file_()
  
    def _read_file_(self) -> Tuple[List[Dict[str, str]], List[Dialogue]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_data = f.read().split('\n\n\n')
        dialogues = [Dialogue(dialog) for dialog in raw_data]
        all_pairs = []
        for dialog in dialogues:
            pairs = dialog.get_pairs()
            if len(pairs) > 0:
                all_pairs.extend(pairs)
        return all_pairs, dialogues
    
    def train_test_split(self, train_part: float = 0.7):
        train, test = self.all_pairs, None
        ################### INSERT YOUR CODE HERE ###################
        
        ################### INSERT YOUR CODE HERE ###################
        return train, test