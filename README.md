# multitask-learning-transformers
A simple recipe for training and inferencing Transformer architecture on custom datasets.
You can find two approaches for achieving this in this repo.

* Single Encoder Multiple Output Heads
A multi-task model in the age of BERT works by having a shared BERT-style encoder transformer, and different task heads for each task.
![mt1](https://user-images.githubusercontent.com/6007894/134903082-64247dd0-fc6f-4b85-a936-b9607ade5a14.png)

* Shared Encoder
![mt2](https://user-images.githubusercontent.com/6007894/134903512-ce42e2d0-b5b1-4269-97de-7255d0cf5a52.png)
Separate models for each task, but we make them share the same encoder. 
 

```python
python3 main.py \
        --model_name_or_path='roberta-base' \
        --per_device_train_batch_size=8 \
        --output_dir=output --num_train_epochs=1
```

References:
[Multi-task Training with Transformers+NLP](https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=CQ39AbTAPAUi)
