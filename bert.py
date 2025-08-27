#BERT

#LIBRARIES
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertModel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################# DOWNLOADING THE DATA #################################
#using the raw data instead of spacy data

base_path = Path(r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\json\bert-input')

def load_json(path):
    with path.open('r', encoding ='utf-8') as f:
        return json.load(f)

#zero shot datasets
davidson_class_list = load_json(base_path / 'bert_davidson_class_list.json')
davidson_text_list = load_json(base_path / 'bert_davidson_text_list.json')
reddit_class_list = load_json(base_path / 'bert_reddit_class_list.json')
reddit_text_list = load_json(base_path / 'bert_reddit_text_list.json')
gab_class_list = load_json(base_path / 'bert_gab_class_list.json')
gab_text_list = load_json(base_path / 'bert_gab_text_list.json')
wikipedia_class_list = load_json(base_path / 'bert_wikipedia_class_list.json')
wikipedia_text_list = load_json(base_path / 'bert_wikipedia_text_list.json')
hatexplain_class_list = load_json(base_path / 'bert_hatexplain_class_list.json')
hatexplain_text_list = load_json(base_path / 'bert_hatexplain_text_list.json')

#few shot data

#Gab few shot

davidson_fs_gab_labels = load_json(base_path / 'bert_davidson_fs_gab_labels.json')
davidson_fs_gab_text = load_json(base_path / 'bert_davidson_fs_gab_text.json')
hatexplain_fs_gab_labels = load_json(base_path / 'bert_hatexplain_fs_gab_labels.json')
hatexplain_fs_gab_text = load_json(base_path / 'bert_hatexplain_fs_gab_text.json')
fs_gab_text_list = load_json(base_path / 'bert_fs_gab_text_list.json')
fs_gab_class_list = load_json(base_path / 'bert_fs_gab_class_list.json')

#Reddit few shot

davidson_fs_reddit_labels = load_json(base_path / 'bert_davidson_fs_reddit_labels.json')
davidson_fs_reddit_text = load_json(base_path / 'bert_davidson_fs_reddit_text.json')
hatexplain_fs_reddit_labels = load_json(base_path / 'bert_hatexplain_fs_reddit_labels.json')
hatexplain_fs_reddit_text = load_json(base_path / 'bert_hatexplain_fs_reddit_text.json')
fs_reddit_text_list = load_json(base_path / 'bert_fs_reddit_text_list.json')
fs_reddit_class_list = load_json(base_path / 'bert_fs_reddit_class_list.json')

#Wikipedia few shot

davidson_fs_wikipedia_labels = load_json(base_path / 'bert_davidson_fs_wikipedia_labels.json')
davidson_fs_wikipedia_text = load_json(base_path / 'bert_davidson_fs_wikipedia_text.json')
hatexplain_fs_wikipedia_labels = load_json(base_path / 'bert_hatexplain_fs_wikipedia_labels.json')
hatexplain_fs_wikipedia_text = load_json(base_path / 'bert_hatexplain_fs_wikipedia_text.json')
fs_wikipedia_text_list = load_json(base_path / 'bert_fs_wikipedia_text_list.json')
fs_wikipedia_class_list = load_json(base_path / 'bert_fs_wikipedia_class_list.json')



################################# EXPERIMENT DATASET CONFIGURATIONS #################################


#IN DOMAIN
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


#ZERO SHOT
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


#FEW SHOT

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



################################# TOKENISER #################################

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

################################# CLASSES & FUNCTIONS #################################


#BERT DATASER
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return input_ids, attention_mask, label


#BERT MODEL
class BertClassifier(nn.Module):
    def __init__(self, num_classes=2):

        super().__init__()
        self.bert =BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output= outputs.pooler_output
        dropped = self.dropout(pooled_output)
        output = self.fc(dropped)

        return output


#TRAINING
def train(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    for input_ids, attention_mask, labels in loader:
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        
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
        for input_ids, attention_mask, labels in loader:
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            predictions = outputs.argmax(dim=1)
            softmax_probs = torch.softmax(outputs, dim=1)

            preds.extend(predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            probs.extend(softmax_probs[:, 1].cpu().tolist())  # Only class 1 for AUC

    # calculate metrics
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



#RUN EXPERIMENT
def run_experiment_bert(train_texts, train_labels, test_texts, test_labels, config_name="unnamed", num_classes=2, max_length=100):

    train_dataset = BertDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = BertDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = BertClassifier(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(4):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"[{config_name}] Epoch {epoch+1}, Train loss: {train_loss:.4f}")

    results = evaluate(model, test_loader, device, num_classes)

    return {"config": config_name,
            "f1": results["f1"],
            "mcc": results["mcc"],
            "auc": results["auc"],
            "predictions": results["predictions"],
            "targets": results["targets"]}


################################# TRAINING & TESTING #################################

results_path = Path(r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\results')
bert_output = results_path / "bert_results.json"

# loading previous results if they already exist 
if bert_output.exists():
    with bert_output.open("r", encoding="utf-8") as f:
        bert_results = json.load(f)
else:
    bert_results = {}

#training loop 
for config_group_name, config_dict in all_config_sets.items():
    print(f"running {config_group_name.upper()}")

    for config_name, (x_train, y_train, x_test, y_test) in config_dict.items():
        full_name = f"{config_group_name}/{config_name}"

        # skip config if its already done
        if full_name in bert_results:
            print(f"skipping as already completed config: {full_name}")
            continue

        print(f"running config: {full_name}")

        try:
            result = run_experiment_bert(x_train, y_train, x_test, y_test, config_name=full_name)
            bert_results[full_name] = result

            # save results per configuration
            with bert_output.open("w", encoding="utf-8") as f:
                json.dump(bert_results, f, indent=2)

        except Exception:
            print(f"runtime error")
