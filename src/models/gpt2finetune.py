import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import logging
import os

# Function to load the dataset
from src.data.train_data import get_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load and preprocess the dataset
train_dataset, val_dataset = get_dataset()  # Assumes get_dataset returns a tuple of (train, val) datasets

# Create DataLoaders for training and validation with reduced batch size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size
val_loader = DataLoader(val_dataset, batch_size=16)  # Reduced batch size

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Function to train the model
def train(dataloader, model, optimizer, device):
    model.train()
    total_train_loss = 0
    for i, batch in enumerate(dataloader):
        logging.info(f"Training batch {i+1}/{len(dataloader)}")
        # Tokenize the batch
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)  # Reduced max_length
        inputs = {k: v.to(device) for k, v in inputs.items()}
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return total_train_loss / len(dataloader)

# Function to evaluate the model
def evaluate(dataloader, model, device):
    model.eval()
    total_eval_loss = 0
    for i, batch in enumerate(dataloader):
        logging.info(f"Evaluating batch {i+1}/{len(dataloader)}")
        # Tokenize the batch
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)  # Reduced max_length
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_eval_loss += loss.item()
    return total_eval_loss / len(dataloader)

# Training loop
epochs = 3
for epoch in range(epochs):
    try:
        total_train_loss = train(train_loader, model, optimizer, device)
        total_eval_loss = evaluate(val_loader, model, device)

        logging.info(f"Epoch: {epoch + 1}, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}")

        # Save the model every epoch
        model_save_path = os.path.join("model_checkpoint_epoch_{}.pt".format(epoch + 1))
        torch.save(model.state_dict(), model_save_path)

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        break
