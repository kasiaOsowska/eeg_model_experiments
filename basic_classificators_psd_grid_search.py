from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer
from sklearn.svm import SVC

from tools import get_data


train_epochs, y_train, test_epochs, y_test = get_data(validation = False, resample=True,
                                                      segment_length = 1.0, step = 0.2)

def get_freq(epochs, fmin=1., fmax=38.):
    psd = epochs.compute_psd(fmin=fmin, fmax=fmax)
    X = psd.get_data()
    freqs = psd.freqs

    print("X shape:", X.shape)
    print("freqs shape:", freqs.shape)

    return X, freqs

X_train, freqs = get_freq(train_epochs)
X_test, _ = get_freq(test_epochs)

print(X_train.shape)
print(X_test.shape)


svm = SVC(probability=True, class_weight='balanced', decision_function_shape='ovr', gamma = "scale")
clf = Pipeline([
    ('vectorizer', Vectorizer()),
    ('scaler', StandardScaler()),
    ('svm', svm)
])
param_grid = {
    'svm__kernel': ['rbf'],
    'svm__C': [0.8, 1, 1.2],
    'svm__gamma': ['scale', 'auto'],
    'svm__shrinking': [True, False],
    'svm__decision_function_shape': ['ovo', 'ovr']
}
grid_search = GridSearchCV(clf, param_grid, cv=2, scoring='f1_micro', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


knn_pipe = Pipeline([
    ("vectorizer", Vectorizer()),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn_param_grid = {
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
    "knn__algorithm": ["auto", "ball_tree", "kd_tree"]
}
knn_gs = GridSearchCV(
    knn_pipe,
    knn_param_grid,
    cv=5,
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
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)
mlp_gs.fit(X_train, y_train)
print("MLP best params:", mlp_gs.best_params_)
print("MLP best CV:", mlp_gs.best_score_)

