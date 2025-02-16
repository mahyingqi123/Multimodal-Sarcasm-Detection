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

training = True

TEXT_INDEX = 0
VIDEO_INDEX = 1
AUDIO_INDEX = 2
SHOW_INDEX = 3
SPEAKER_INDEX = 4

splits = 5
batch_size = 32
learning_rate = 5e-4
weight_decay = 0.0
early_stop = 20
model_path = 'model/intermodality/intermodality_model_'
device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')

result_file = "result/intermodality_fusion/{}.json"

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
    def __init__(self, text_feature, video_feature, audio_feature, label):
        self.vision = video_feature
        self.text = text_feature
        self.audio = audio_feature
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]),
            'vision': torch.Tensor(self.vision[index]),
            'audio': torch.Tensor(self.audio[index]),
            'labels': torch.Tensor(self.label[index]).type(torch.LongTensor)
        }

        return sample
    
def group_data(dataset, text, video, audio):
    data_input, data_output = [], []
    
    for id in dataset.keys():
        data_input.append(
            (text[id],  # 0 TEXT_ID
             video[id],  # 1 VIDEO_ID
             audio[id],           # 2
             dataset[id]["show"],      # 3 SHOW_ID
             dataset[id]["speaker"],           # 4
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
    authors = getData(SPEAKER_INDEX)
    UNK_AUTHOR_ID = author_ind["PERSON"]
    authors = [author_ind.get(author.strip(), UNK_AUTHOR_ID) for author in authors]
    authors = oneHot(authors, len(author_ind))
    train_dataset = MultiModalDataset(train_text_feature, train_video_feature, train_audio_feature, train_out)

    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    return train_dataLoader

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

    
def fit(model, train_data, val_data,fold):
    best_acc = 0
    epochs, best_epoch = 0, 0
    criterion = nn.CrossEntropyLoss()
    best_loss = 100000000
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

            labels = batch_data['labels'].to(device)

            optimizer.zero_grad()
            # forward
            outputs = model(text, vision, audio)

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
        if training:
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epochs
                torch.save(model.cpu().state_dict(), model_path+str(fold)+'.pt')
                model.to(device)
        else:
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epochs
                if os.path.exists(model_path):
                    os.remove(model_path)
                torch.save(model.cpu().state_dict(), model_path+str(fold)+'.pt')
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

            outputs = model(text, vision, audio)

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
        print("Fold {}:".format(fold + 1))
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

        model = IntermodalInconsistencyNetwork(vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size']) #todo: put actual values
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

    model_name = 'intermodality_fusion_model_'
    model_name = model_name + str(index)
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    with open(result_file.format(model_name), 'w') as file:
        json.dump(results, file, indent=4)
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

    dataset_json = json.load(open(sarcasm_data))
    text_features = json.load(open(text_embeddings, 'r'))
    video_features = json.load(open(video_embeddings, 'r'))
    audio_features = json.load(open(audio_embeddings, 'r'))

    data_input, data_output = group_data(dataset_json, text_features, video_features,audio_features)

    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    split_indices = [(train_index, test_index) for train_index, test_index in skf.split(data_input, data_output)]
    split_indices = pickle_loader(indices_file)


    model = IntermodalInconsistencyNetwork(vision_dim=video_map['size'], text_dim=text_map['size'], audio_dim=audio_map['size']).to(device) #todo: put actual values
    model = model.to(device)
    summary(model, [(text_map['size'], ), (video_map['size'], ), (audio_map['size'], )])

    five_results = []
    results_per_fold = []

    for i in range(5):
        five_fold(i, split_indices, model_path, result_file, device, data_input, data_output)
        model_name = 'intermodality_fusion_model_'
        model_name = model_name + str(i)
        tmp_dict = result_formatter(model_name=model_name)

        fold_results = json.load(open(result_file.format(model_name), "rb"))
        results_per_fold.extend(fold_results)

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