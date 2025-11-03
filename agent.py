from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# Use fine-tuned model if available
model_name = "fine_tuned_study_agent" if os.path.isdir("fine_tuned_study_agent") else "google/flan-t5-base"

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

study_agent = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def study_helper(notes):
    input_text = f"Notes: {notes}"
    result = study_agent(input_text, max_length=200)[0]["generated_text"]
    return result

def evaluate_summary(notes, output):
    key_words = [w for w in notes.split() if len(w) > 6]
    covered = sum([1 for k in key_words if k.lower() in output.lower()])
    coverage = round((covered / len(key_words)) * 100, 2) if key_words else 0
    return coverage

if __name__ == "__main__":
    notes = input("Paste your study notes:\n")
    output = study_helper(notes)
    score = evaluate_summary(notes, output)
    print("\n=== AI Summary + Quiz ===\n")
    print(output)
    print(f"\nCoverage Score: {score}%")
