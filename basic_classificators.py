from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from tools import get_data


train_epochs, y_train, test_epochs, y_test = get_data(validation = False, resample=True,
                                                      segment_length = 2.0, step = 1.0)

X_train = train_epochs.get_data()
X_test = test_epochs.get_data()

print(X_train.shape)
print(X_test.shape)

clf = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear", probability=True))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print("Covariances + Tangent Space + SVM F1 weighted:", f1)

clf = Pipeline([
    ('vectorizer', Vectorizer()),
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print("Initial SVM F1 weighted:", f1)

svm = SVC(kernel = 'rbf', class_weight='balanced', gamma = "scale",
          decision_function_shape='ovo', shrinking=True)
clf = Pipeline([
    ('vectorizer', Vectorizer()),
    ('scaler', StandardScaler()),
    ('svm', svm)
])
param_grid = {
    'svm__C': [0.5, 0.8, 1, 1.2, 1.5],
}
grid_search = GridSearchCV(
    clf,
    param_grid,
    cv=2,
    scoring='f1_micro',
    n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


knn_pipe = Pipeline([
    ("vectorizer", Vectorizer()),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2, algorithm='auto'))
])
knn_param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
}
knn_gs = GridSearchCV(
    knn_pipe,
    knn_param_grid,
    cv=2,
    scoring="f1_micro",
    n_jobs=-1
)
knn_gs.fit(X_train, y_train)
print("KNN best params:", knn_gs.best_params_)
print("KNN best CV:", knn_gs.best_score_)


mlp_pipe = Pipeline([
    ("vectorizer", Vectorizer()),
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42
    ))
])
mlp_param_grid = {
    "mlp__hidden_layer_sizes": [(128, 64), (256, 128), (512, 256)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__solver": ["adam"],
    "mlp__alpha": [1e-5, 1e-4, 1e-3],
    "mlp__learning_rate_init": [1e-4, 1e-3, 1e-2]
}
mlp_gs = GridSearchCV(
    mlp_pipe,
    mlp_param_grid,
    cv=2,
    scoring="f1_weighted",
    n_jobs=-1
)
mlp_gs.fit(X_train, y_train)
print("MLP best params:", mlp_gs.best_params_)
print("MLP best CV:", mlp_gs.best_score_)