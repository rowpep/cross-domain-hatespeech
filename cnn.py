#CNN

#LIBRARIES
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

################################# DOWNLOADING THE DATA #################################
#Download spacy tokenised data

base_path = Path(r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\json')

def load_json(path):
    with path.open('r', encoding ='utf-8') as f:
        return json.load(f)

#zero shot datasets
davidson_class_list = load_json(base_path / 'davidson_class_list.json')
davidson_text_list = load_json(base_path / 'davidson_text_list_spcy.json')
reddit_class_list = load_json(base_path / 'reddit_class_list.json')
reddit_text_list = load_json(base_path / 'reddit_text_list_spcy.json')
gab_class_list = load_json(base_path / 'gab_class_list.json')
gab_text_list = load_json(base_path / 'gab_text_list_spcy.json')
wikipedia_class_list = load_json(base_path / 'wikipedia_class_list.json')
wikipedia_text_list = load_json(base_path / 'wikipedia_text_list_spcy.json')
hatexplain_class_list = load_json(base_path / 'hatexplain_class_list.json')
hatexplain_text_list = load_json(base_path / 'hatexplain_text_list_spcy.json')

#few shot data

#Gab few shot

davidson_fs_gab_labels = load_json(base_path / 'davidson_fs_gab_labels.json')
davidson_fs_gab_text = load_json(base_path / 'davidson_fs_gab_text.json')
hatexplain_fs_gab_labels = load_json(base_path / 'hatexplain_fs_gab_labels.json')
hatexplain_fs_gab_text = load_json(base_path / 'hatexplain_fs_gab_text.json')
fs_gab_text_list = load_json(base_path / 'fs_gab_text_list.json')
fs_gab_class_list = load_json(base_path / 'fs_gab_class_list.json')

#Reddit few shot

davidson_fs_reddit_labels = load_json(base_path / 'davidson_fs_reddit_labels.json')
davidson_fs_reddit_text = load_json(base_path / 'davidson_fs_reddit_text.json')
hatexplain_fs_reddit_labels = load_json(base_path / 'hatexplain_fs_reddit_labels.json')
hatexplain_fs_reddit_text = load_json(base_path / 'hatexplain_fs_reddit_text.json')
fs_reddit_text_list = load_json(base_path / 'fs_reddit_text_list.json')
fs_reddit_class_list = load_json(base_path / 'fs_reddit_class_list.json')

#Wikipedia few shot

davidson_fs_wikipedia_labels = load_json(base_path / 'davidson_fs_wikipedia_labels.json')
davidson_fs_wikipedia_text = load_json(base_path / 'davidson_fs_wikipedia_text.json')
hatexplain_fs_wikipedia_labels = load_json(base_path / 'hatexplain_fs_wikipedia_labels.json')
hatexplain_fs_wikipedia_text = load_json(base_path / 'hatexplain_fs_wikipedia_text.json')
fs_wikipedia_text_list = load_json(base_path / 'fs_wikipedia_text_list.json')
fs_wikipedia_class_list = load_json(base_path / 'fs_wikipedia_class_list.json')



################################# EXPERIMENT DATASET CONFIGURATIONS #################################

#IN-DOMAIN
davidson_xtrain, davidson_xtest, davidson_ytrain, davidson_ytest = train_test_split(davidson_text_list, davidson_class_list, test_size=0.2, stratify=davidson_class_list, random_state=42)

hatexplain_xtrain, hatexplain_xtest, hatexplain_ytrain, hatexplain_ytest = train_test_split(hatexplain_text_list, hatexplain_class_list, test_size=0.2, stratify=hatexplain_class_list, random_state=42)

baseline_configs = {
    "davidson_to_davidson":(
        davidson_xtrain, davidson_ytrain,
        davidson_xtest, davidson_ytest),
    
    "hatexplain_to_hatexplain":(
        hatexplain_xtrain, hatexplain_ytrain,
        hatexplain_xtest, hatexplain_ytest
    )}


#ZEROSHOT
zero_shot_configs = {
    "davidson_to_reddit":(
        davidson_text_list, davidson_class_list,
        reddit_text_list, reddit_class_list),
    
    "davidson_to_gab":(
        davidson_text_list, davidson_class_list,
        gab_text_list, gab_class_list),
    
    "davidson_to_wikipedia":(
        davidson_text_list, davidson_class_list,
        wikipedia_text_list, wikipedia_class_list),
    
    "hatexplain_to_reddit":(
        hatexplain_text_list, hatexplain_class_list,
        reddit_text_list, reddit_class_list),
    
    "hatexplain_to_gab":(
        hatexplain_text_list, hatexplain_class_list,
        gab_text_list, gab_class_list),
    
    "hatexplain_wikipedia":(
        hatexplain_text_list, hatexplain_class_list,
        wikipedia_text_list, wikipedia_class_list)
    }


#FEWSHOT
few_shot_configs = {
    "davidson_fs_reddit":(
        davidson_fs_reddit_text, davidson_fs_reddit_labels,
        fs_reddit_text_list, fs_reddit_class_list),

    "davidson_fs_gab":(
        davidson_fs_gab_text, davidson_fs_gab_labels,
        fs_gab_text_list, fs_gab_class_list),

    "davidson_fs_wikipedia":(
        davidson_fs_wikipedia_text, davidson_fs_wikipedia_labels,
        fs_wikipedia_text_list, fs_wikipedia_class_list),

    "hatexplain_fs_reddit":(
        hatexplain_fs_reddit_text, hatexplain_fs_reddit_labels,
        fs_reddit_text_list, fs_reddit_class_list),

    "hatexplain_fs_gab":(
        hatexplain_fs_gab_text, hatexplain_fs_gab_labels,
        fs_gab_text_list, fs_gab_class_list),

    "hatexplain_fs_wikipedia":(
        hatexplain_fs_wikipedia_text, hatexplain_fs_wikipedia_labels,
        fs_wikipedia_text_list, fs_wikipedia_class_list)
}


#ALL SETS
all_config_sets = {
    "baseline": baseline_configs,
    "zero_shot": zero_shot_configs,
    "few_shot": few_shot_configs}




################################# VOCAB BUILDER #################################

from collections import Counter
def build_vocab(token_lists, min_freq=1):
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = {'<PAD>':0, '<UNK>':1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab
    

################################# CLASSES & FUNCTIONS #################################

#DATASET
class HateSpeechDataset(Dataset):
    def __init__(self, token_lists, labels, vocab, max_length=100):
        self.token_lists = token_lists
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.pad_idx = vocab['<PAD>']
        self.unk_idx = vocab.get('<UNK>', 1)

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx]
        label = self.labels[idx]
        ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]

        if len(ids) < self.max_length:
            ids += [self.pad_idx] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


#TRAINING
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


#EVALUATION
def evaluate(model, loader, device, num_classes=2):
    model.eval()
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            softmax_probs = torch.softmax(outputs, dim=1)

            preds.extend(predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            probs.extend(softmax_probs[:, 1].cpu().tolist())  # Only class 1 for AUC

    # Compute metrics
    f1 = f1_score(targets, preds, average="macro")
    mcc = matthews_corrcoef(targets, preds)

    if num_classes == 2:
        auc = roc_auc_score(targets, probs)
    else:
        try:
            auc = roc_auc_score(targets, softmax_probs.cpu().numpy(), multi_class='ovr')
        except ValueError:
            auc = None  # AUC may fail if not all classes are present

    return {
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
        "predictions": preds,
        "targets": targets
    }



#RUNEXPERIMENT

def run_experiment(train_tokens, train_labels, test_tokens, test_labels,
                   config_name="unnamed", num_classes=2, max_length=100):
    
    vocab = build_vocab(train_tokens)
    
    train_dataset = HateSpeechDataset(train_tokens, train_labels, vocab, max_length)
    test_dataset = HateSpeechDataset(test_tokens, test_labels, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TextCNN(vocab_size=len(vocab), embed_dim=100, num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"[{config_name}] Epoch {epoch+1}, Train loss: {train_loss:.4f}")

    results = evaluate(model, test_loader, device, num_classes)

    output = {
        "config": config_name,
        "f1": results["f1"],
        "mcc": results["mcc"],
        "auc": results["auc"],
        "predictions": results["predictions"],
        "targets": results["targets"]
    }

    return output


#MODEL RNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):

        #call the parent constructor
        super(TextCNN, self).__init__()

        #embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        #conv layers w diff ketnel sizes. each layer looks at dif n-gram patterns
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_sizes[2])

        #activation function relu
        self.relu = nn.ReLU()

        #dropout
        self.dropout = nn.Dropout(0.5)

        #final linear layer - classifier
        self.fc = nn.Linear(in_features=num_filters*len(kernel_sizes), out_features=num_classes)


    def forward(self, x):
        #x shape = [batch_size, sequence_length]

        #embed the input words
        x = self.embedding(x) #x shape [batch_size, sequence_length, embed_dim]

        #rearrange for conv1d
        x = x.permute(0, 2, 1) #x shape [batch_size, sequence_length, embed_dim]

        #apply convolutions and activations
        x1 = self.relu(self.conv1(x)) #x shape [batch_size, num_filters, L1]
        x2 = self.relu(self.conv2(x)) #x shape [batch_size, num_filters, L2]
        x3 = self.relu(self.conv3(x)) #x shape [batch_size, num_filters, L3]

        #apply max pooling to each
        x1 = torch.max(x1, dim=2).values
        x2 = torch.max(x2, dim=2).values
        x3 = torch.max(x3, dim=2).values

        #concatenate the pooled outputs
        x= torch.cat((x1, x2, x3),dim=1)

        #dropout
        x = self.dropout(x)

        #final classification layer
        logits = self.fc(x)

        return logits
    




################################# TRAINING & TESTING #################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_path = Path(r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\results')
cnn_output = results_path / "cnn_results.json"

cnn_results = {}

for config_group_name, config_dict in all_config_sets.items():

    for config_name, (x_train, y_train, x_test, y_test) in config_dict.items():
        
        full_name = f"{config_group_name}/{config_name}"
        
        result = run_experiment(x_train, y_train, x_test, y_test, config_name=full_name)
        cnn_results[full_name] = result





#SAVE RESULTS
with cnn_output.open("w", encoding="utf-8") as f:
    json.dump(cnn_results, f, indent=2)



