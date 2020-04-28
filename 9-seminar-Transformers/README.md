# Transformers
This seminar contains _ notebooks with independent tasks. Yo download dataset you have to use `python3 get_data.py` command.
## 1. Dialogue generation with GPT2
You can find everything you need in `gpt2_finetune.ipynb` file, `src/utils.py`  and `src/gpt2` directory.  
`src/gpt2/data_parser.py` file contains `Dialogue` and `DataParser` classes. `Dialogue` class store raw dialogue sentencies and has method to transform it into context-answer format. `DataParser` class parse raw Twitter dialogues and has a method to split data into test and train (you have to implement it).  
`src/gpt2/dataset.py` file contains `torch.utils.data.Dataset` child class. For each index, `__getitem__` method returns
```
{
    'sample': torch.LongTensor, 
    'mask': torch.LongTensor, 
    'label': torch.LongTensor
}
``` 
