# Neural Machine Translation
To download dataset you can use `bash get_data.sh` command, or `python3 get_data.py` (special for Windows users).  

Check `machine_translation.ipynb` file for baseline solution.  

After downloading dataset your working directory should looks like:
```
├── data
│   ├── _about.txt
│   └── rus.txt
├── get_data.py
├── get_data.sh
├── machine_translation.ipynb
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── data_parser.py
    ├── tokenizer.py
    ├── dataset.py
    ├── models.py
    └── utils.py
```

`src/data_parser.py` file contains `DataParser` class with methods for splitting dataset by language and generating train and validation sets.  

`src/tokenizer.py` file contains `Tokenizer` class for creating vocabulary for each language.  

`src/dataset.py` contains `torch.utils.data.Dataset` child class. For each index, `__getitem__` method returns
```
{
    'source_sentence': torch.LongTensor,
    'source_sentence_mask': torch.LongTensor,
    'target_language_sentence': torch.LongTensor,
    'target_sentence_mask': torch.LongTensor,
    'target_for_loss': torch.LongTensor
}
```
'`source_sentence`' and  '`target_language_sentence`' fiels are used for model training and '`target_for_loss`' - for loss computation between model predicts and real values. '`source_sentence_mask`' and '`target_sentence_mask`' fields aren't used now, but you can apply them for attention mechanism.  

`src/models.py` contains `SpatialDropout`, `Encoder`, `Decoder`, `NMTModel` classes.  