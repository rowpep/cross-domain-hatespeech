#LIBRARIES

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

################################# DOWNLOADING THE DATA #################################
#Download spacy tokenised data


base_path = Path(r'C:\Users\rowan\OneDrive\Documents\Dissertation\data\json')

def load_json(path):
    with path.open('r', encoding ='utf-8') as f:
        return json.load(f)

#Zero shot datasets
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

#Few shot datasets

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

#join the zero shot models

davidson_text_list = [' '.join(tokens) for tokens in davidson_text_list]
hatexplain_text_list = [' '.join(tokens) for tokens in hatexplain_text_list]
reddit_text_list = [' '.join(tokens) for tokens in reddit_text_list]
wikipedia_text_list = [' '.join(tokens) for tokens in wikipedia_text_list]
gab_text_list = [' '.join(tokens) for tokens in gab_text_list]



################################# MODELS #################################


#kNN Model
def get_knn_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer
         (lowercase=True, stop_words=None, max_features=10000)), 
         ('knn',KNeighborsClassifier(n_neighbors=5, metric='euclidean'))])

#RF Model
def get_rf_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer
         (lowercase=True, stop_words='english', max_features=10000)),
         ('rf', RandomForestClassifier(random_state =42))])

#SVM Model 
def get_svm_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer
         (lowercase=True, stop_words='english', max_features = 10000)), 
         ('svc', SVC(C=2.0, kernel='rbf', probability=True, random_state=42))])


################################# EXPERIMENT DATASET CONFIGURATIONS #################################

#IN-DOMAIN CONFIGURATONS
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


#ZERO SHOT CONFIGURATONS
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



#FEW SHOT CONFIGURATIONS
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



#Model and Configuration sets
model_constructors = {"knn": get_knn_model,"rf": get_rf_model,"svm": get_svm_model}

all_config_sets = [("baseline", baseline_configs),
                   ("zero-shot", zero_shot_configs),
                   ("few-shot", few_shot_configs)]



################################# TRAINING & TESTING #################################


#Empty results
results = []

#Iterating through every model, configuration type, and experiment configurations and evaluation metric
for setting_name, config_dict in all_config_sets:
    for config_name, (x_train, y_train, x_test, y_test) in config_dict.items():
        for model_name, get_model in model_constructors.items():

            model = get_model()
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            results.append({
                "setting": setting_name,
                "experiment": config_name,
                "model": model_name,
                "f1": f1,
                "mcc": mcc,
                "auc": auc,
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist(),
                "y_test":list(y_test)})



#SAVE THE RESULTS
with open("baseline_model_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

