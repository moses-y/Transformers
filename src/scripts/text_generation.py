import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

def generate_text(model, tokenizer, prompt, max_length=1000, temp=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, temperature=temp, do_sample=True, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Path to the fine-tuned model
model_path = r'C:\Users\moses_y\OneDrive\Desktop\ML Projects\Transformers\src\models\model_checkpoint_epoch_3.pt'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# User input for initial task
user_task = input("Please enter a task for the model: ")

#superprompt = (
    #f"Task: Write Python code to {user_task}.\n"
    #"Step 1: Start by defining the main function and its required parameters.\n"
    #"Step 2: Implement the core logic or algorithm inside the function.\n"
    #"Step 3: Ensure proper data handling and error checking within the code.\n"
    #"Step 4: Add necessary comments for code clarity and documentation.\n"
    #"Step 5: Provide a few examples at the end to demonstrate how to use the function.")

# Generate text using the superprompt
generated_text = generate_text(model, tokenizer, user_task, max_length=1000)
print(generated_text)
