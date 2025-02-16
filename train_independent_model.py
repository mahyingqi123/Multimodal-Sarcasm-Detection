import torch.nn as nn
import torch
from torch import optim
from torch import optim
from sklearn.metrics import accuracy_score
import pickle


training = True
batch_size = 64
text_model_file = 'text/text_model'
visual_model_file = 'visual/visual_model'
audio_model_file = 'audio/audio_model'
early_stop = 20
max_epochs = 200
learning_rate = 0.0001

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
import json
mapping = json.load(open("mapping.json"))
current_visual = mapping['current_visual']
current_audio = mapping['current_audio']
current_text = mapping['current_text']

audio_map = mapping[current_audio]
text_map = mapping[current_text]
video_map = mapping[current_visual]

text_data = json.load(open(text_map['filename'], 'r'))
visual_data = json.load(open(video_map['filename'], 'r'))
audio_data = json.load(open(audio_map['filename'], 'r'))
label_data = json.load(open('sarcasm_data.json', 'r'))

import torch
from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dict, label_dict):
        """
        Args:
            embedding_dict: A dictionary mapping IDs to embeddings (numpy arrays or lists).
            label_dict: A dictionary mapping IDs to labels (integers).
        """
        self.ids = list(embedding_dict.keys())
        self.embeddings = [torch.tensor(embedding_dict[id], dtype=torch.float32) for id in self.ids]
        self.labels = [torch.tensor(label_dict[id]['sarcasm'], dtype=torch.long) for id in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

indices_file = "split_indices.p"
def pickle_loader(filename):
    return pickle.load(open(filename, 'rb'), encoding="latin1")
split_indices = pickle_loader(indices_file)
dataset = EmbeddingDataset(audio_data, label_data)
device = 'cuda'

from sklearn.metrics import precision_recall_fscore_support


def get_dataloader(dataset, indices, batch_size, shuffle):
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

# Train 5 models for each fold
for fold, (train_indices, val_indices) in enumerate(split_indices):
    print(f"Starting fold {fold}")
    
    train_loader = get_dataloader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(dataset, val_indices, batch_size=1, shuffle=False)
    
    model = SimpleNN(audio_map['size'], 256, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_epoch = 0
    best_loss = 100000000
    epochs = 0
    precision, recall, f1 = 0, 0, 0
    accuracy = 0
    
    while True:
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            embeddings, labels = embeddings.to(device), labels.to(device)
            model.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                predictions = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        if training:
            if total_loss/batch_size <= best_loss:
                best_loss = total_loss/batch_size
                best_epoch = epochs
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                torch.save(model.cpu().state_dict(), f'model/{audio_model_file}_fold_{fold}.pt')
                best_acc = accuracy

        if epochs - best_epoch > early_stop or epochs == max_epochs:
            break
        epochs += 1

    print(f"Fold {fold} complete. Best accuracy: {best_acc} at epoch {best_epoch}")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

dataset = EmbeddingDataset(text_data, label_data)


# Train 5 models for each fold
for fold, (train_indices, val_indices) in enumerate(split_indices):
    print(f"Starting fold {fold}")
    
    train_loader = get_dataloader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(dataset, val_indices, batch_size=1, shuffle=False)
    
    model = SimpleNN(text_map['size'], 256, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_epoch = 0
    epochs = 0
    best_loss = 100000000
    precision, recall, f1 = 0, 0, 0
    accuracy = 0
    
    while True:
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            embeddings, labels = embeddings.to(device), labels.to(device)
            if next(model.parameters()).device != embeddings.device:
                model.to(embeddings.device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                predictions = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        if total_loss/batch_size <= best_loss:
            best_loss = total_loss/batch_size
            best_epoch = epochs
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            torch.save(model.cpu().state_dict(), f'model/{text_model_file}_fold_{fold}.pt')
            best_acc = accuracy

        
        if epochs - best_epoch > early_stop or epochs == max_epochs:
            break
        epochs += 1

    print(f"Fold {fold} complete. Best accuracy: {best_acc} at epoch {best_epoch}")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


dataset = EmbeddingDataset(visual_data, label_data)
from sklearn.metrics import precision_recall_fscore_support


def get_dataloader(dataset, indices, batch_size, shuffle):
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

# Train 5 models for each fold
for fold, (train_indices, val_indices) in enumerate(split_indices):
    print(f"Starting fold {fold}")
    
    train_loader = get_dataloader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(dataset, val_indices, batch_size=1, shuffle=False)
    
    model = SimpleNN(video_map['size'], 256, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_loss = 100000000
    best_epoch = 0
    epochs = 0
    precision, recall, f1 = 0, 0, 0
    accuracy = 0

    while True:
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            embeddings, labels = embeddings.to(device), labels.to(device)
            if next(model.parameters()).device != embeddings.device:
                model.to(embeddings.device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                predictions = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        if total_loss/batch_size <= best_loss:
            best_loss = total_loss/batch_size
            best_epoch = epochs
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            torch.save(model.cpu().state_dict(), f'model/{visual_model_file}_fold_{fold}.pt')
            best_acc = accuracy


        
        if epochs - best_epoch > early_stop or epochs == max_epochs:
            break
        epochs += 1

    print(f"Fold {fold} complete. Best accuracy: {best_acc} at epoch {best_epoch}")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")