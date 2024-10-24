import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Veri klasör yolunu belirleyin
data_dir = "/kaggle/input/a-large-scale-fish-dataset/NA_Fish_Dataset"  # Kaggle üzerinde dataset yolu.

# 1. Resim dosyalarını ve etiketleri listeleme
filepaths = []
labels = []

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, img_file))
            labels.append(folder)  # Klasör adı etiketi temsil eder

## Resim yolları ve etiketlerden oluşan bir DataFrame oluşturarak verimizi organize ediyoruz.
data = pd.DataFrame({"filepath": filepaths, "label": labels})
print(data.head())

#  Veriyi daha iyi anlamak için ilk birkaç görseli çizdiriyoruz.
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    img = load_img(data["filepath"][i], target_size=(128, 128))
    ax.imshow(img)
    ax.set_title(data["label"][i])
    ax.axis('off')
plt.show()

# Modelimizi eğitmek ve doğrulamak için veriyi %80 eğitim %20 test olarak ayırıyoruz.
train_data, test_data = train_test_split(
    data, test_size=0.2, stratify=data["label"], random_state=42
)

def preprocess_image(filepath):
    """Görüntüyü numpy array'e çevirip normalize eder."""
    img = load_img(filepath, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return img_array

# Modelin eğitebileceği forma getirmek için veriyi işliyoruz.
X_train = np.array([preprocess_image(fp) for fp in train_data["filepath"]])
X_test = np.array([preprocess_image(fp) for fp in test_data["filepath"]])
y_train = pd.get_dummies(train_data["label"]).values
y_test = pd.get_dummies(test_data["label"]).values

# 4. Modelin tanımlanması
model = Sequential([
    Input(shape=(128, 128, 3)),  # İlk katmanda Input kullanımı
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(y_train[0]), activation='softmax')  # Çıkış katmanı
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Class weights hesaplama (Veri dengesizliği için)
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_data["label"]), y=train_data["label"]
)
class_weights_dict = dict(enumerate(class_weights))

# 6. Modelin eğitilmesi
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    validation_data=(X_test, y_test), 
    class_weight=class_weights_dict  # Class weights ile eğitim
)

# 7. Eğitim sürecinin görselleştirilmesi
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.show()

# 8. Model değerlendirme ve metrikler
#Test verisi üzerinde tahmin yapıp, performansı ölçüyoruz.
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, zero_division=1))  # Uyarıyı giderir
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
