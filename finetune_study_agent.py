from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

data = {
    "input_text": [
        "Notes: Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "Notes: Newton's First Law states that an object remains at rest or in motion unless acted on by a force."
    ],
    "target_text": [
        "Summary: Photosynthesis converts sunlight to food.\nQuiz: 1) What does photosynthesis do? 2) What gas is released?",
        "Summary: Newton's First Law explains inertia.\nQuiz: 1) What is inertia? 2) State Newton's First Law."
    ]
}

dataset = Dataset.from_dict(data)

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

def preprocess(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_ds = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds
)

trainer.train()
model.save_pretrained("fine_tuned_study_agent")
tokenizer.save_pretrained("fine_tuned_study_agent")
