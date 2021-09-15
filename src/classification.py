import sys
import pickle
import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier



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
model = sys.argv[4]


train_set = load_embeddings(emb_path, label_path)
#X_train = torch.from_numpy(train_set.iloc[:, 0:train_set.shape[1] - 2].values)
X_train = torch.from_numpy(train_set.iloc[:, 0:train_set.shape[1] - 1].values)
y_train = train_set[['label']].values.flatten()


if model == "mlp":
    #MLP
    clf_mlp = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(1000, 500, 800),solver='adam', random_state=42)
    clf_mlp.fit(X_train, y_train)
    pickle.dump(clf_mlp, open("{}_mlp".format(model_path), 'wb'), protocol=4)


elif model == "rf":
    #RandomForest
    clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_rf.fit(X_train, y_train)
    pickle.dump(clf_rf, open("{}_rf".format(model_path), 'wb'), protocol=4)


elif model == "gb":
    #GradientBoosting
    clf_gb = GradientBoostingClassifier(random_state=42)
    clf_gb.fit(X_train, y_train)
    pickle.dump(clf_gb, open("{}_gb".format(model_path), 'wb'), protocol=4)


elif model == "knn":
    #KNN
    clf_knn = KNeighborsClassifier()
    clf_knn.fit(X_train, y_train)
    pickle.dump(clf_knn, open("{}_knn".format(model_path), 'wb'), protocol=4)


else:
    #stacking models
    clf_mlp = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(1000, 500, 800),solver='adam', random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_gb = GradientBoostingClassifier(random_state=42)
    clf_knn = KNeighborsClassifier()

    estimators = [('mlp', clf_mlp),
                  ('rf', clf_rf),
                  ('gb', clf_gb),
                  ('knn', clf_knn)]

    clf_stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf_stacking.fit(X_train, y_train)
    pickle.dump(clf_stacking, open("{}_stacking".format(model_path), 'wb'), protocol=4)

