import transformers
import torch


def save_model(model_name, multitask_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    multitask_model.config.to_json_file(f"./model/config.json")
    torch.save(
        multitask_model.state_dict(),
        f"./model/pytorch_model.bin",
    )
    tokenizer.save_pretrained(f"./model/")
