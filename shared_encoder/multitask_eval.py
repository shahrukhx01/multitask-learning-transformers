from multitask_data_collator import DataLoaderWithTaskname
import nlp
import numpy as np
import torch
import transformers


def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in ["quora_keyword_pairs", "spaadia_squad_pairs"]:
        val_len = len(features_dict[task_name]["validation"])
        acc = 0.0
        for index in range(0, val_len, batch_size):

            batch = features_dict[task_name]["validation"][
                index : min(index + batch_size, val_len)
            ]["doc"]
            labels = features_dict[task_name]["validation"][
                index : min(index + batch_size, val_len)
            ]["target"]
            inputs = tokenizer(batch, max_length=512, padding=True)
            inputs["input_ids"] = torch.LongTensor(inputs["input_ids"])
            inputs["attention_mask"] = torch.LongTensor(inputs["attention_mask"])
            logits = multitask_model("quora_keyword_pairs", **inputs)[0]

            predictions = torch.argmax(
                torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),
                dim=1,
            )
            acc += sum(np.array(predictions) == np.array(labels))
        acc = acc / val_len
        print(f"Task name: {task_name} \t Accuracy: {acc}")
    """
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer.prediction_loop(
        eval_dataloader,
        description=f"Validation: {task_name}",
    )
    for x in eval_dataloader:
        print(x)
    # Evalute task
    nlp.load_metric("glue", name="rte").compute(
        np.argmax(preds_dict[task_name].predictions),
        preds_dict[task_name].label_ids,
    )"""
