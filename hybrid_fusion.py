import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from torch import nn, optim
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Precision: 85.7 ± 4.8 (variance: 0.2322)
# Recall: 85.6 ± 4.8 (variance: 0.2347)
# F1: 85.6 ± 4.8 (variance: 0.2352)

TEXT_INDEX = 0
VIDEO_INDEX = 1
AUDIO_INDEX = 2
SHOW_INDEX = 3
SPEAKER_INDEX = 4
RAW_TEXT_INDEX = 5
RAW_VIDEO_INDEX = 6
RAW_AUDIO_INDEX = 7

splits = 5
batch_size = 32
learning_rate = 25e-5
weight_decay = 0.0
early_stop = 20
model_path = 'model/hybrid/hybrid_fusion_'
device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')

result_file = "result/hybrid_fusion/{}.json"

def oneHot(data, size=None):
    '''
    Returns one hot label version of data
    '''
    result = np.zeros((len(data), size))
    result[range(len(data)), data] = 1
    return result

def pickle_loader(filename):
    return pickle.load(open(filename, 'rb'), encoding="latin1")

class MultiModalDataset(Dataset): #MMDataset
    def __init__(self, text_feature, video_feature, audio_feature, raw_text, raw_video, raw_audio, label):
        self.vision = video_feature
        self.text = text_feature
        self.audio = audio_feature
        self.raw_text = raw_text
        self.raw_video = raw_video
        self.raw_audio = raw_audio
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]),
            'vision': torch.Tensor(self.vision[index]),
            'audio': torch.Tensor(self.audio[index]),
            'raw_text': torch.Tensor(self.raw_text[index]),
            'raw_video': torch.Tensor(self.raw_video[index]),
            'raw_audio': torch.Tensor(self.raw_audio[index]),
            'labels': torch.Tensor(self.label[index]).type(torch.LongTensor)

        }

        return sample
    
def group_data(dataset, text, video, audio, raw):
    data_input, data_output = [], []
    
    for id in dataset.keys():
        data_input.append(
            (text[id],  # 0 TEXT_ID
             video[id],  # 1 VIDEO_ID
             audio[id],           # 2
             dataset[id]["show"],      # 3 SHOW_ID
             dataset[id]["speaker"],           # 4
             raw[id]["text"],  # 5 TEXT_RAW
            raw[id]["visual"],  # 6
            raw[id]["audio"],  # 7
             ))
        data_output.append(int(dataset[id]["sarcasm"]))
    return data_input, data_output

def load_data(data_input, data_output,train_ind_SI, author_ind):
    def getData(ID=None):
        return [instance[ID] for instance in train_input]
    
    train_input = [data_input[ind] for ind in train_ind_SI]
    train_out = np.array([data_output[ind] for ind in train_ind_SI])
    train_out = np.expand_dims(train_out, axis=1)

    train_text_feature = getData(TEXT_INDEX)
    train_video_feature = getData(VIDEO_INDEX)
    train_audio_feature = getData(AUDIO_INDEX)
    train_raw_text = getData(RAW_TEXT_INDEX)
    train_raw_video = getData(RAW_VIDEO_INDEX)
    train_raw_audio = getData(RAW_AUDIO_INDEX)
    authors = getData(SPEAKER_INDEX)

    UNK_AUTHOR_ID = author_ind["PERSON"]
    authors = [author_ind.get(author.strip(), UNK_AUTHOR_ID) for author in authors]
    authors = oneHot(authors, len(author_ind))
    train_dataset = MultiModalDataset(train_text_feature, train_video_feature, train_audio_feature, train_raw_text, train_raw_video, train_raw_audio,train_out)

    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    return train_dataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class IntermediateFusionNetwork(nn.Module):
    def __init__(self, text_dim=1024, vision_dim=1024, audio_dim=1024, hidden_dim=256, num_classes=2):
        super(IntermediateFusionNetwork, self).__init__()
        
        total_dim = text_dim + vision_dim + audio_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )


    def forward(self, text, vision, audio):
        combined = torch.cat((text, vision, audio), dim=1)
        return self.fusion(combined)
    
class EarlyFusionNetwork(nn.Module):
    def __init__(self, text_dim=1024, vision_dim=1024, audio_dim=1024, hidden_dim=256, num_classes=2):
        super(EarlyFusionNetwork, self).__init__()
        
        total_dim = text_dim + vision_dim + audio_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )


    def forward(self, text, vision, audio):
        combined = torch.cat((text, vision, audio), dim=1)
        return self.fusion(combined)

class LateFusionNetwork(nn.Module):
    def __init__(self, fold = 0, vision_dim=512, text_dim=512, audio_dim=1024):
        super(LateFusionNetwork, self).__init__()
        # Deeper processing branches for each modality
        self.text_branch = SimpleNN(text_dim, 256, 2)
        self.text_branch.load_state_dict(torch.load('model/text/text_model_fold_'+str(fold)+'.pt'))
        self.text_branch.eval()
        
        self.vision_branch = SimpleNN(vision_dim, 256, 2)
        self.vision_branch.load_state_dict(torch.load('model/visual/visual_model_fold_'+str(fold)+'.pt'))
        self.vision_branch.eval()

        self.audio_branch = SimpleNN(audio_dim, 256, 2)
        self.audio_branch.load_state_dict(torch.load('model/audio/audio_model_fold_'+str(fold)+'.pt'))
        self.audio_branch.eval()
        # Fusion layer

        self.attention = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

        self.classifier = nn.Linear(2, 2)
    
    def forward(self, text, vision, audio):
        # Process each modality independently
        text_features = self.text_branch(text)
        vision_features = self.vision_branch(vision)
        audio_features = self.audio_branch(audio)
        
        # Concatenate and apply attention
        combined = torch.cat([text_features, vision_features, audio_features], dim=1)

        attn_scores = self.attention(combined)

        attn_weights = torch.softmax(attn_scores, dim=1)

        stacked = torch.stack([text_features, vision_features, audio_features], dim=1)
        
        attended_features = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # Final classification
        return self.classifier(attended_features)


class HybridFusionNetwork(nn.Module):
    def __init__(self, text_dim=512, vision_dim=512, audio_dim=1024, hidden_dim=128, num_classes=2, fold=0):
        super(HybridFusionNetwork, self).__init__()
        
        # Load pretrained models
        self.late_fusion = LateFusionNetwork(fold,vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size'])
        self.late_fusion.load_state_dict(torch.load(f"model/late/late_fusion_{fold}.pt"))
        self.late_fusion.eval()
        
        self.early_fusion = EarlyFusionNetwork()
        self.early_fusion.load_state_dict(torch.load(f"model/early/early_fusion_{fold}.pt"))
        self.early_fusion.eval()

        self.intermediate_fusion = IntermediateFusionNetwork(vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size'])
        self.intermediate_fusion.load_state_dict(torch.load(f"model/intermediate/intermediate_fusion_{fold}.pt"))
        self.intermediate_fusion.eval()

        self.intermodal_inconsistency = IntermodalInconsistencyNetwork(vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size'])
        self.intermodal_inconsistency.load_state_dict(torch.load(f"model/intermodality/intermodality_model_{fold}.pt"))
        self.intermodal_inconsistency.eval()  

        # Projection layers for each modality to match attention dimensions
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Attention mechanisms
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)

        self.late_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.early_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.intermediate_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.intermodal_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Cross-modal fusion layer with residual connection
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = ResidualClassifier(hidden_dim, hidden_dim, num_classes)
        
    def forward(self, text, vision, audio, raw_text, raw_vision, raw_audio):
        
        # Project each modality to same dimension
        text_proj = self.text_proj(text)      # [batch_size, hidden_dim]
        vision_proj = self.vision_proj(vision) # [batch_size, hidden_dim]
        audio_proj = self.audio_proj(audio)    # [batch_size, hidden_dim]
        
        # Get predictions from pretrained models
        with torch.no_grad():
            late_fusion_out = self.late_fusion(text, vision, audio)
            intermediate_fusion_out = self.intermediate_fusion(text, vision, audio)
            early_fusion_out = self.raw_fusion(raw_text, raw_vision, raw_audio)
            intermodal_out = self.intermodal_inconsistency(text, vision, audio)

        late_proj = self.late_proj(late_fusion_out)    # [batch_size, hidden_dim]
        intermediate_proj = self.intermediate_proj(intermediate_fusion_out)  # [batch_size, hidden_dim]
        early_proj = self.early_proj(early_fusion_out)       # [batch_size, hidden_dim]

        intermodal_proj = self.intermodal_proj(intermodal_out)  # [batch_size, hidden_dim]
        
        
        fusion_stack = torch.stack([late_proj, early_proj, intermediate_proj, intermodal_proj, text_proj, vision_proj, audio_proj], dim=1)  # [batch_size, 3, hidden_dim]


        attended_fusion, attention_weights  = self.fusion_attention(
            fusion_stack, fusion_stack, fusion_stack
        )

        attended_fusion = self.layer_norm(attended_fusion + fusion_stack)

        attention_scores = F.softmax(attention_weights.mean(dim=1), dim=-1).unsqueeze(-1)
        weighted_fusion = (attended_fusion * attention_scores).sum(dim=1)

        # Final classification
        output = self.classifier(weighted_fusion)
        
        return output
    
class IntermodalInconsistencyNetwork(nn.Module):
    def __init__(self, text_dim, vision_dim, audio_dim, hidden_dim=256, num_classes=2):
        super(IntermodalInconsistencyNetwork, self).__init__()
        
        # Project each modality to same hidden dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Inconsistency detection networks for each modality pair
        pair_input_dim = hidden_dim * 2
        self.text_vision_compare = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        self.text_audio_compare = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        self.vision_audio_compare = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final classifier that combines inconsistency scores
        self.final_classifier = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def get_inconsistency_score(self, mod1, mod2, compare_net):
        """Calculate inconsistency score between two modalities"""
        combined = torch.cat([mod1, mod2], dim=1)
        return compare_net(combined)
        
    def forward(self, text, vision, audio):
        # Project each modality to common space
        text_h = self.text_proj(text)
        vision_h = self.vision_proj(vision)
        audio_h = self.audio_proj(audio)
        
        # Calculate inconsistency scores between modality pairs
        text_vision_score = self.get_inconsistency_score(
            text_h, vision_h, self.text_vision_compare
        )
        text_audio_score = self.get_inconsistency_score(
            text_h, audio_h, self.text_audio_compare
        )
        vision_audio_score = self.get_inconsistency_score(
            vision_h, audio_h, self.vision_audio_compare
        )
        
        # Combine inconsistency scores
        all_scores = torch.cat(
            [text_vision_score, text_audio_score, vision_audio_score], 
            dim=1
        )
        
        # Final classification
        output = self.final_classifier(all_scores)
        
        return output


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ResidualClassifier, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.residual_block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Residual block
        identity = x
        x = self.residual_block(x)
        x = x + identity
        
        # Final classification
        x = self.classifier(x)
        
        return x
    
def fit(model, train_data, val_data, fold):
    best_acc = 0
    epochs, best_epoch = 0, 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    while True:
        epochs += 1
        y_pred, y_true = [], []
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch_data in train_data:
            vision = batch_data['vision'].to(device)
            audio = batch_data['audio'].to(device)
            text = batch_data['text'].to(device)
            raw_text = batch_data['raw_text'].to(device)
            raw_video = batch_data['raw_video'].to(device)
            raw_audio = batch_data['raw_audio'].to(device)

            labels = batch_data['labels'].to(device)

            optimizer.zero_grad()
            # forward
            outputs = model(text, vision, audio, raw_text, raw_video, raw_audio)

            loss = criterion(outputs, labels.squeeze())
            # backward
            loss.backward()
            # update
            optimizer.step()
            train_loss += loss.item()

            train_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()

            y_pred.append(outputs.argmax(1).cpu())
            y_true.append(labels.squeeze().long().cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)

        train_loss = train_loss / len(pred)

        train_acc = train_acc / len(pred)
        val_acc, _, _ = test(model, val_data, mode="VAL")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epochs
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(model.cpu().state_dict(),  model_path+str(fold)+'.pt')
            model.to(device)

        # early stop
        if epochs - best_epoch >= early_stop:
            return

def test(model, test_data , mode="VAL"):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0.0
    eval_acc = 0.0
    with torch.no_grad():
        for batch_data in test_data:
            vision = batch_data['vision'].to(device)
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            labels = batch_data['labels'].to(device)
            raw_text = batch_data['raw_text'].to(device)
            raw_video = batch_data['raw_video'].to(device)
            raw_audio = batch_data['raw_audio'].to(device)

            outputs = model(text, vision, audio, raw_text, raw_video, raw_audio)

            loss = criterion(outputs, labels.squeeze())


            eval_loss += loss.item()
            eval_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()

            y_pred.append(outputs.argmax(1).cpu())
            y_true.append(labels.squeeze().long().cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)


    eval_loss = eval_loss / len(pred)
    eval_acc = eval_acc / len(pred)

    return eval_acc, pred, true

def get_author_index(train_ind_SI,data_input):
    # (text,video,AUDIO)
    train_input = [data_input[ind] for ind in train_ind_SI]

    def getData(ID=None):
        return [instance[ID] for instance in train_input]

    authors = getData(SPEAKER_INDEX)
    author_list = set()
    author_list.add("PERSON")

    for author in authors:
        author = author.strip()
        if "PERSON" not in author:  # PERSON3 PERSON1 all --> PERSON haha
            author_list.add(author)

    author_ind = {author: ind for ind, author in enumerate(author_list)}
    return  author_ind

def result_formatter(model_name):
    results = json.load(open(result_file.format(model_name), "rb"))
    weighted_precision, weighted_recall = [], []
    weighted_fscores = []
    print("#" * 20)
    for fold, result in enumerate(results):
        weighted_fscores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])
        print("Fold {}:".format(fold ))
        print("Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}".format(
            result["weighted avg"]["precision"],
            result["weighted avg"]["recall"],
            result["weighted avg"]["f1-score"]))
    print("#" * 20)
    print("Avg :")
    print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(
        np.mean(weighted_precision),
        np.mean(weighted_recall),
        np.mean(weighted_fscores)))

    tmp_dict = {}
    tmp_dict['precision'] = np.mean(weighted_precision)
    tmp_dict['recall'] = np.mean(weighted_recall)
    tmp_dict['f1'] = np.mean(weighted_fscores)

    return tmp_dict

def five_fold(index, split_indices, model_path, result_file, device, data_input, data_output):
    results = []
    for fold, (train_index, test_index) in enumerate(split_indices):
        train_ind_SI, val_ind_SI, test_ind_SI = train_index, test_index, test_index

        author_ind = get_author_index(train_ind_SI,data_input)
        train_dataLoader = load_data(data_input, data_output, train_ind_SI,author_ind)
        val_dataLoader = load_data(data_input, data_output, val_ind_SI,author_ind)
        test_dataLoader = load_data(data_input, data_output, test_ind_SI,author_ind)

        model = HybridFusionNetwork(fold=fold,vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size']) #todo: put actual values
        model = model.to(device)

        fit(model, train_dataLoader, val_dataLoader, fold)
        print()
        print(f'load:{model_path}')
        model.load_state_dict(torch.load(model_path+str(fold)+'.pt'))
        model.to(device)
        
        val_acc, y_pred, y_true = test(model, test_dataLoader, mode="TEST")
        print('Test: ', val_acc)
        
        print('confusion_matrix(y_true, y_pred)')
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=3))

        result_dict = classification_report(y_true, y_pred, digits=3, output_dict=True)
        results.append(result_dict)

    model_name = 'hybrid_fusion_model_'
    model_name = model_name + str(index)
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    with open(result_file.format(model_name), 'w') as file:
        json.dump(results, file)
    print('dump results  into ', result_file.format(model_name))

def calculate_overall_stats(results_list):
    # Extract all metrics from all folds
    all_metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Collect metrics from each fold's results
    for fold_results in results_list:
        weighted_metrics = fold_results["weighted avg"]
        all_metrics['precision'].append(weighted_metrics['precision'])
        all_metrics['recall'].append(weighted_metrics['recall'])
        all_metrics['f1'].append(weighted_metrics['f1-score'])

    # Calculate statistics
    stats = {}
    for metric, values in all_metrics.items():
        values = np.array(values)
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'variance': np.var(values)
        }
        print(f"{metric.capitalize()}: {np.mean(values)*100:.1f} ± {np.std(values)*100:.1f} (variance: {np.var(values)*100:.4f})")
    
    return stats

mapping = json.load(open("mapping.json"))
current_visual = mapping['current_visual']
current_audio = mapping['current_audio']
current_text = mapping['current_text']

audio_map = mapping[current_audio]
text_map = mapping[current_text]
video_map = mapping[current_visual]
def main():
    
    video_embeddings = video_map['filename']    
    text_embeddings = text_map['filename']
    audio_embeddings = audio_map['filename']
    sarcasm_data = "sarcasm_data.json" # DATA_PATH_JSON 
    indices_file = "split_indices.p"
    raw_data = "raw_features.json"

    dataset_json = json.load(open(sarcasm_data))
    text_features = json.load(open(text_embeddings, 'r'))
    video_features = json.load(open(video_embeddings, 'r'))
    audio_features = json.load(open(audio_embeddings, 'r'))
    raw_features = json.load(open(raw_data, 'r'))

    data_input, data_output = group_data(dataset_json, text_features, video_features,audio_features, raw_features)
    split_indices = pickle_loader(indices_file)


    model = HybridFusionNetwork(vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size']).to(device) #todo: put actual values
    model = model.to(device)
    summary(model, [(text_map['size'], ), (video_map['size'], ), (audio_map['size'], ), (1024, ), (1024, ), (1024, )])

    five_results = []
    
    results_per_fold = []

    for i in range(5):
        five_fold(i, split_indices, model_path, result_file, device, data_input, data_output)
        model_name = 'hybrid_fusion_model_'
        model_name = model_name + str(i)
                
        fold_results = json.load(open(result_file.format(model_name), "rb"))
        results_per_fold.extend(fold_results)

        tmp_dict = result_formatter(model_name=model_name)
        five_results.append(tmp_dict)

    file_name = 'five_results'
    with open(result_file.format(file_name), 'w') as file:
        json.dump(five_results, file)
    print('dump results  into ', result_file.format(file_name))

    print("\nOverall Performance Statistics:")
    stats = calculate_overall_stats(results_per_fold)


    results = json.load(open(result_file.format(file_name), "rb"))
    precisions, recalls, f1s = [], [], []
    for _, result in enumerate(results):
        tmp1 = result['precision']
        tmp2 = result['recall']
        tmp3 = result['f1']
        precisions.append(tmp1)
        recalls.append(tmp2)
        f1s.append(tmp3)

    print('five average: precision recall f1')
    print(round(np.mean(precisions) * 100, 1), round(np.mean(recalls) * 100, 1), round(np.mean(f1s) * 100, 1))

    tmp = {
        'precision:': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

    file_name = 'five_results_average'
    with open(result_file.format(file_name), 'w') as file:
        json.dump(tmp, file)



if __name__ == "__main__":
    main()