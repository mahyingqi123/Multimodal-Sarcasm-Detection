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
model_path = 'model/late/late_fusion_'
device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')

result_file = "result/late_fusion/{}.json"

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
    

class LateFusionNetwork(nn.Module):
    def __init__(self, fold = 0):
        super(LateFusionNetwork, self).__init__()
        # Deeper processing branches for each modality
        self.text_branch = SimpleNN(512, 256, 2)
        self.text_branch.load_state_dict(torch.load('model/text/text_model_fold_'+str(fold)+'.pt'))
        self.text_branch.eval()
        
        self.vision_branch = SimpleNN(512, 256, 2)
        self.vision_branch.load_state_dict(torch.load('model/visual/visual_model_fold_'+str(fold)+'.pt'))
        self.vision_branch.eval()

        self.audio_branch = SimpleNN(1024, 256, 2)
        self.audio_branch.load_state_dict(torch.load('model/audio/audio_model_fold_'+str(fold)+'.pt'))
        self.audio_branch.eval()
        # Fusion layer

        self.classifier = nn.Linear(2 * 3, 2)
    
    def forward(self, text, vision, audio):
        # Process each modality independently
        text_features = self.text_branch(text)
        vision_features = self.vision_branch(vision)
        audio_features = self.audio_branch(audio)
        
        # Concatenate and apply attention
        combined = torch.cat([text_features, vision_features, audio_features], dim=1)
        
        # Final classification
        return self.classifier(combined)


    
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

        model = LateFusionNetwork(fold+1) #todo: put actual values
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

    model_name = 'late_fusion_model_'
    model_name = model_name + str(index)
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    with open(result_file.format(model_name), 'w') as file:
        json.dump(results, file)
    print('dump results  into ', result_file.format(model_name))


    
def main():
    video_embeddings = "video_embeddings_clip.json"
    text_embeddings = "text_features_clip.json"
    audio_embeddings = "audio_features_wav2vec2_bert.json"
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


    model = LateFusionNetwork(1).to(device)
    model = model.to(device)
    summary(model, [(512, ), (512, ), (1024, )])

    five_results = []

    for i in range(5):
        five_fold(i, split_indices, model_path, result_file, device, data_input, data_output)
        model_name = 'late_fusion_model_'
        model_name = model_name + str(i)
        tmp_dict = result_formatter(model_name=model_name)
        five_results.append(tmp_dict)

    file_name = 'five_results'
    with open(result_file.format(file_name), 'w') as file:
        json.dump(five_results, file)
    print('dump results  into ', result_file.format(file_name))

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