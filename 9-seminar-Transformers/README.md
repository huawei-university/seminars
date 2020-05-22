# Transformers
This seminar contains a notebook with 2 independent tasks. Dataset will be automatically downloaded when you initialize first time the DataParser class for each task.
## 1. Quora question pairs classification with BERT
A few words about the task: we have to identify question pairs that have the same intent. 
Check first part of [transformers_demo.ipynb](./transformers_demo.ipynb) for a baseline solution. As in the previous seminar, you can find data preprocessing and models code in the [src/bert](./src/bert) directory.  

## 2. Dialogue generation with GPT2
The task description and an interface to play with are in [transformers_demo.ipynb](./transformers_demo.ipynb) file, althe there are [src/utils.py](./src/utils.py)  and [src/gpt2](./src/gpt2) directory with data preprocessing code.  

[src/gpt2/data_parser.py](src/gpt2/data_parser.py) file contains `Dialogue` and `DataParser` classes. `Dialogue` class store raw dialogue sentencies and has method to transform it into context-answer format. `DataParser` class parse raw Twitter dialogues and has a method to split data into test and train (you have to implement it).  
[src/gpt2/dataset.py](./src/gpt2/dataset.py) file contains `torch.utils.data.Dataset` child class. For each index, `__getitem__` method returns
```
{
    'sample': torch.LongTensor, 
    'mask': torch.LongTensor, 
    'label': torch.LongTensor
}
``` 
