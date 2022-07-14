# multitask-learning-transformers
A simple recipe for training and inferencing Transformer architecture for Multi-Task Learning on custom datasets. You can find two approaches for achieving this in this repo.

### Colab Notebook
[Colab Notebook](https://colab.research.google.com/drive/1R7WcLHxDsVvZXPhr5HBgIWa3BlSZKY6p?usp=sharing#scrollTo=3Fzv6lYA0wFw)

### Trained Huggingface Model
[HF Model](https://huggingface.co/shahrukhx01/bert-multitask-query-classifiers)

### Install depedencies
```bash
pip install -r requirements.txt
```
### Run training
```python
python3 main.py \
        --model_name_or_path='roberta-base' \
        --per_device_train_batch_size=8 \
        --output_dir=output --num_train_epochs=1
```

## Single Encoder Multiple Output Heads
A multi-task model in the age of BERT works by having a shared BERT-style encoder transformer, and different task heads for each task.


![mt1](https://user-images.githubusercontent.com/6007894/134903082-64247dd0-fc6f-4b85-a936-b9607ade5a14.png)

## Shared Encoder
Separate models for each task, but we make them share same encoder. 

![mt2](https://user-images.githubusercontent.com/6007894/134903512-ce42e2d0-b5b1-4269-97de-7255d0cf5a52.png)

References:
[Multi-task Training with Transformers+NLP](https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=CQ39AbTAPAUi)
