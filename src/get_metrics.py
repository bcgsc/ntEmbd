from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import pickle
import torch
import pandas as pd
import sys

def sum_metrics(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp/(tp+fn)
    spec = tn/(fp+tn)
    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print("\n")
    print("|sens|spec|prec|f1|roc_auc|acc|")
    print("|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|".format(sens,spec,prec,f1,roc_auc,acc))
    print()


def load_embeddings(emb_file, label_file):
    embs = pd.read_csv(emb_file, delim_whitespace=True, header=None)
    labels = pd.read_csv(label_file, header=None, names=["label"])
    #seqs = pd.read_csv("{}_seqs".format(emb_file), header=None, names=["seq"])
    #df = pd.concat([embs, labels, seqs], axis=1)
    df = pd.concat([embs, labels], axis=1)
    return df


emb_path = sys.argv[1]
label_path = sys.argv[2]
model_path = sys.argv[3]


test_set = load_embeddings(emb_path, label_path)
#X_test = torch.from_numpy(test_set.iloc[:, 0:test_set.shape[1] - 2].values)
X_test = torch.from_numpy(test_set.iloc[:, 0:test_set.shape[1] - 1].values)
y_test = test_set[['label']].values.flatten()

clf = pickle.load(open(model_path, "rb"))

sum_metrics(y_test, clf.predict(X_test))
