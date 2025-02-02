from transformers import VisionEncoderDecoderModel, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from PIL import Image
from torchvision import transforms

# 1) Load Pretrained OCR Model (using a seq2seq model, e.g. TrOCR)
tokenizer = AutoTokenizer.from_pretrained("datalab-to/ocr_error_detection")
model = VisionEncoderDecoderModel.from_pretrained("datalab-to/ocr_error_detection")

# 2) Define image transforms
image_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 3) Load Train & Test HF Datasets
train_dataset_path = "synthetic_dataset/hf_dataset"  # from convert_to_hf_dataset.py
test_dataset_path = "test_dataset/hf_dataset"          # separate dataset folder

train_hf = load_from_disk(train_dataset_path)
test_hf = load_from_disk(test_dataset_path)

# 4) Preprocess function (integrating image processing and text tokenization)
def preprocess_data(example):
    # Load and transform the image
    image = Image.open(example["image_path"]).convert("RGB")
    example["pixel_values"] = image_transforms(image)
    
    # Tokenize the ground truth text for generation
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    example["labels"] = tokenized.input_ids
    
    # Optionally include bounding box information if needed
    example["bbox"] = example.get("bbox", None)
    return example

# 5) Map preprocessing over both datasets
train_hf = train_hf.map(preprocess_data)
test_hf = test_hf.map(preprocess_data)

# 6) Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="surya_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_steps=10
)

# 7) Create Seq2Seq Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf
)

# 8) Train the Model
trainer.train()

# 9) Save the Fine-Tuned Model
model.save_pretrained("surya_finetuned")
print("Training complete; model saved to 'surya_finetuned'.")
