import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from utils import load_libs_data
from preprocessing.normalize import minmax_normalize

# Muat data
spectra, labels = load_libs_data('data/raw')
spectra = np.array([minmax_normalize(s) for s in spectra])

# Encode label
label_to_id = {label: i for i, label in enumerate(np.unique(labels))}
y = np.array([label_to_id[label] for label in labels])

# Split data
X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.2)

# Bangun model 1D CNN
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.Conv1D(32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(len(label_to_id), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Pelatihan
history = model.fit(
    X_train[..., np.newaxis],  # Tambah dimensi channel
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test[..., np.newaxis], y_test)
)

# Simpan model
model.save('models/saved_models/lib_cnn_1d.h5')