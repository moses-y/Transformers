from datasets import load_dataset
from torch.utils.data import Dataset, random_split

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return self.encodings[idx]

    def __len__(self):
        return len(self.encodings)

def get_dataset():
    dataset = load_dataset("Arjun-G-Ravi/Python-codes")
    print("Dataset keys:", dataset.keys())
    print("Train dataset keys:", dataset['train'].features.keys())

    full_dataset = dataset['train']
    
    # Concatenate 'question' and 'code', handling None values
    full_texts = [f"{example['question'] or ''} {example['code'] or ''}" for example in full_dataset]

    full_dataset = CustomDataset(full_texts)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset

# ... (rest of your train_data.py code)

if __name__ == "__main__":
    train_dataset, val_dataset = get_dataset()

    # Print the size of the datasets
    print("Size of training dataset:", len(train_dataset))
    print("Size of validation dataset:", len(val_dataset))

    # Print a few examples from the datasets
    print("Some examples from training dataset:")
    for i in range(min(3, len(train_dataset))):
        print(train_dataset[i])

    print("Some examples from validation dataset:")
    for i in range(min(3, len(val_dataset))):
        print(val_dataset[i])


