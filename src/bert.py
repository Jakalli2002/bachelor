"""
bert approach: tokenizes the raw texts with the models own subword tokenizer and fine-tunes a pretrained german bert with the transformers trainer.
uses the raw text column, not the lemmas, bert relies on word order, context and capitalization, all of which the classic preprocessing removes.
"""
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from config import (MODEL_NAME, MAX_LENGTH, CATEGORIES, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY)


# -------------------- PRE LOADING --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label2id = {category: i for i, category in enumerate(CATEGORIES)}
id2label = {i: category for category, i in label2id.items()}


# -------------------- PREPROCESSING --------------------
def make_dataset(texts, labels=None) -> Dataset:
    """
    builds a huggingface Dataset from raw texts and tokenizes them.
    labels are optional: the training set needs them, the test set does not
    (the true test labels are only used in experiment.py).
    """
    data = {"text": list(texts)}
    if labels is not None:
        data["label"] = [label2id[c] for c in labels]

    dataset = Dataset.from_dict(data)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


# -------------------- MODEL --------------------
def run_bert(train_texts, train_labels, test_texts, seed: int = 42):
    train_ds = make_dataset(train_texts, train_labels)
    test_ds = make_dataset(test_texts)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CATEGORIES),
        label2id=label2id,
        id2label=id2label,
    )

    args = TrainingArguments(
        output_dir="/tmp/bert_run",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        seed=seed,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    logits = trainer.predict(test_ds).predictions
    pred_ids = np.argmax(logits, axis=-1)
    return np.array([id2label[i] for i in pred_ids])


# -------------------- TESTING --------------------
if __name__ == "__main__":
    import time
    from data import load_data, get_data_subset
    from config import TRAINING_PATH, TEST_PATH

    train_df = load_data(TRAINING_PATH)
    test_df = load_data(TEST_PATH)

    subset = get_data_subset(train_df, 0.1, 1)

    start = time.perf_counter()
    preds = run_bert(subset["text"], subset["category"], test_df["text"])
    dauer = time.perf_counter() - start

    print("Trainingsartikel:", len(subset))
    print("Accuracy:", (preds == test_df["category"].values).mean())
    print(f"Dauer: {dauer:.1f}s")