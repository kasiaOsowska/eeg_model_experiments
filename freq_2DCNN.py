import numpy as np
from keras.src.metrics.metrics_utils import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from tools import get_data
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


train_epochs, y_train, test_epochs, y_test, valid_epochs, y_valid = get_data(resample = False, segment_length = 2.0,
                                                                             step = 1.0)


def get_freq(epochs, fmin=8., fmax=38.):
    psd = epochs.compute_psd(fmin=fmin, fmax=fmax)
    X = psd.get_data()
    freqs = psd.freqs

    print("X shape:", X.shape)
    print("freqs shape:", freqs.shape)

    return X, freqs

X_train, freqs = get_freq(train_epochs)

X_valid, _ = get_freq(valid_epochs)
X_test, _ = get_freq(test_epochs)


le = LabelEncoder()
y_train_dl = le.fit_transform(y_train)
y_valid_dl = le.transform(y_valid)
y_test_dl  = le.transform(y_test)

n_classes = len(np.unique(y_train_dl))

X_train_dl = np.transpose(X_train, (0, 2, 1))
X_valid_dl = np.transpose(X_valid, (0, 2, 1))
X_test_dl  = np.transpose(X_test,  (0, 2, 1))

mean = X_train_dl.mean(axis=0, keepdims=True)          # (1, F, Ch)
std  = X_train_dl.std(axis=0, keepdims=True) + 1e-8    # (1, F, Ch)


X_train_dl = (X_train_dl - mean) / std
X_valid_dl = (X_valid_dl - mean) / std
X_test_dl  = (X_test_dl  - mean) / std


X_train_2d = X_train_dl[..., np.newaxis]  # (N, F, C, 1)
X_valid_2d = X_valid_dl[..., np.newaxis]
X_test_2d  = X_test_dl[...,  np.newaxis]

F_dim = X_train_dl.shape[1]
C_dim = X_train_dl.shape[2]

inputs = keras.Input(shape=(F_dim, C_dim, 1))

x = layers.Conv2D(16, kernel_size = (3, 3), padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(32, kernel_size = (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.2)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.1)(x)

outputs = layers.Dense(n_classes, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer=keras.optimizers.Adam(5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
]


history = model.fit(
    X_train_dl, y_train_dl,
    validation_data=(X_valid_dl, y_valid_dl),
    epochs=120,
    batch_size=64,
    shuffle=True,
    callbacks = callbacks
)

test_loss, test_acc = model.evaluate(X_test_dl, y_test_dl)
print("Test loss:", test_loss)
print("Test acc:", test_acc)

y_pred_proba = model.predict(X_test_dl)
y_pred = np.argmax(y_pred_proba, axis=1)

print(confusion_matrix(y_test_dl, y_pred, num_classes=n_classes))
print(classification_report(y_test_dl, y_pred))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
