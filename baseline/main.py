import collections
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support,precision_recall_curve
import numpy as np

'''
CSV_HEADERS of sapsii_feature_scores.csv
"hadm_id","icustay_id","uo_score","wbc_score","bicarbonate_score","sodium_score",
"potassium_score","pao2fio2_score","hr_score","sysbp_score","temp_score","gcs_score","comorbidity_score","admissiontype_score","age_score"
'''

'''
CSV_HEADERS OF sapsii_logistic_regression_label.csv
"hadm_id","label"
'''


#FILE RELATED VARS
TRAIN_SET_FILE = "../data/train_hadmids.csv"
EVAL_SET_FILE = "../data/val_hadmids.csv"
TEST_SET_FILE = "../data/test_hadmids.csv"

#SAPSII_FEATURES_FILE = "../raw_data/sapsii_feature_scores.csv"
SAPSII_FEATURES_FILE = "../raw_data/sapsii_feature_scores_maxed.csv"
SAPSII_LABEL_FILE = "../raw_data/sapsii_logistic_regression_label.csv"




#DATASET VARS
HADMID_LABELS_DICT = {}  #{"hadmId": 0/1, 1 means death}
DATASET_FEATURES_DICT = {} #

# "hadm_id","icustay_id"
FEATURE_LIST = ["uo_score","wbc_score","bicarbonate_score","sodium_score",
"potassium_score","pao2fio2_score","hr_score","sysbp_score","temp_score","gcs_score","comorbidity_score","admissiontype_score","age_score"]
FEATURE_SCORE_AVERAGE = {
    "uo_score": 1.6738,
    "wbc_score": 0.4984,
    "bicarbonate_score": 0.7240,
    "sodium_score": 0.2222,
    "potassium_score": 0.9910,
    "pao2fio2_score": 7.6079,
    "hr_score": 1.4015,
    "sysbp_score": 3.2434,
    "temp_score": 0.0032,
    "gcs_score": 2.4341,
    "comorbidity_score": 1.0714,
    "admissiontype_score": 5.8387,
    "age_score": 11.4332
}

# Not Scored
FEATURE_RAW_AVERAGE= {
    "uo_score": -1,
    "wbc_score": -1,
    "bicarbonate_score": -1,
    "sodium_score": -1,
    "potassium_score": -1,
    "pao2fio2_score": 236.8955,
    "hr_score": -1,
    "sysbp_score": -1,
    "temp_score": -1,
    "gcs_score": -1,
    "comorbidity_score": -1,
    "admissiontype_score": -1,
    "age_score": -1
}


#MODEL VARS
MODEL_RANDOM_STATE= 0
POSITIVE_LABEL = 0
MAX_ITER = 1000
PENALTY = 'l2'
CLASS_WEIGHT = 'balanced'

def processLabelsToDict():
    with open(SAPSII_LABEL_FILE, "r") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            HADMID_LABELS_DICT[row['hadm_id']] = int(row["label"])

# Perform mean imputation as well...
def processFeatureToDict():
    with open(SAPSII_FEATURES_FILE, "r") as fp:
        reader = csv.DictReader(fp)
        # TODO mean imputation
        for row in reader:
            data_dict = {}
            for key in FEATURE_LIST:
                value = row[key]
                if not value.strip():
                    value = FEATURE_SCORE_AVERAGE[key]
                data_dict[key] = float(value)
            DATASET_FEATURES_DICT[row['hadm_id']] = data_dict

#Rebalance training set
def resampleTrainingSet(hadm_index, features, labels):
    resample_count = np.sum(labels) # sum of positives(deaths)
    #TODO randomize randomize
    new_hadm_index, new_features, new_labels = [], [] ,[]
    positive_count, negative_count = 0, 0
    for idx, _ in enumerate(hadm_index):
        if labels[idx] == 1:
            new_hadm_index.append(hadm_index[idx])
            new_features.append(features[idx])
            new_labels.append(labels[idx])
            positive_count+=1
        else:
            if negative_count < resample_count:
                new_hadm_index.append(hadm_index[idx])
                new_features.append(features[idx])
                new_labels.append(labels[idx])
                negative_count+=1
    return new_hadm_index, new_features, new_labels

def generateDataSet(file_path):
    hadm_index, features, labels = [], [], []
    with open(file_path, "r") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            hadm_id = row['HADM_ID']
            hadm_index.append(hadm_id)
            labels.append(HADMID_LABELS_DICT[hadm_id])
            dict_data = DATASET_FEATURES_DICT[hadm_id]
            record_feature_row = []
            for key in FEATURE_LIST:
                record_feature_row.append(dict_data[key])
            features.append(record_feature_row)
    return hadm_index, features, labels
 

def trainModel(train_features, train_label):
    clf = LogisticRegression(random_state=0, max_iter = MAX_ITER, penalty=PENALTY, class_weight=CLASS_WEIGHT).fit(train_features, train_label)
    return clf


def evaluate(model, features, label, dataset_string="EVAL"):
    prediction = model.predict(features)
    acc = model.score(features, label)
    predict_proba = model.predict_proba(features)
    aurocscore = roc_auc_score(label, predict_proba[:,1]) #TODO first label or second label which one is positive
    apscore = average_precision_score(label, predict_proba[:,1], average='weighted')    
    auprcscore = precision_recall_curve(label, predict_proba[:,1], pos_label=1) #TODO first label or second label which one is positive
    precision, recall, f1, support = precision_recall_fscore_support(label, prediction, labels=[POSITIVE_LABEL])
    print(f"--------------------------------------------------------------------------")
    #print(f"Dataset: {dataset_string} AUROC: {aurocscore} APSCORE: {apscore} AUPRC: {auprcscore}")
    print(f"Dataset: {dataset_string} AUROC: {aurocscore} AUPRC: {apscore}")
    print(f"Precision: {precision[0]} Recall:{recall[0]} F1:{f1[0]} Accuracy:{acc}")
    print(f"Records Predicted:{len(prediction)}")
    print(f"Predicted as deaths:{np.sum(prediction)} Predicted as non deaths:{len(prediction)-np.sum(prediction)}")
    print(f"True Record Deaths:{np.sum(label)} True Non Deaths:{len(label)-np.sum(label)}")
    print(f"--------------------------------------------------------------------------")

def important_weights(model):
    coefficients = model.coef_[0]
    np_coef = np.array(coefficients)
    coefficient_index = np_coef.argsort()[-3:][::-1]
    print(f"--------------------------------------------------------------------------")
    print(f"Top 3 features are.... ")
    for idx in coefficient_index:
        print(f"feature key {FEATURE_LIST[idx]} feature coef {coefficients[idx]}")
    print(f"--------------------------------------------------------------------------")

    print(f"All features key {FEATURE_LIST}")
    print(f"All features coef {coefficients}")



def main():
    processLabelsToDict()
    processFeatureToDict()

    # training
    train_index, train_features, train_label = generateDataSet(TRAIN_SET_FILE)
    train_index, train_features, train_label = resampleTrainingSet(train_index, train_features, train_label)

    model = trainModel(train_features, train_label)
    evaluate(model,train_features, train_label, "Train Dataset")

    # eval
    eval_index, eval_features, eval_label = generateDataSet(EVAL_SET_FILE)
    evaluate(model,eval_features, eval_label, "Eval Dataset")

    # test dataset
    test_index, test_features, test_label = generateDataSet(TEST_SET_FILE)
    evaluate(model,test_features, test_label, "Test Dataset")


    important_weights(model)




main()