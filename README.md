# 📰 Fake News Detection with LSTM

Ce projet vise à détecter les fake news à partir de leur contenu textuel en utilisant des réseaux de neurones (LSTM). Il s'inscrit dans une approche d’apprentissage supervisé avec une phase expérimentale de génération de données adversariales via GPT-2.

## 🚀 Objectifs

- Construire un modèle LSTM pour la détection de fake news.
- Utiliser un ensemble de données de textes labellisés (vrais vs faux).
- Évaluer la performance du modèle à l'aide de métriques comme l'accuracy, le F1-score, etc.
- Tester la robustesse du modèle avec des données synthétiques générées par GPT-2 .

## 🛠️ Technologies

- Python 3
- Keras / TensorFlow
- Numpy, Pandas
- Sklearn
- Matplotlib / Seaborn

## 📁 Structure du projet

📦fake-news-detection ┣ 📂data ┃ ┗ 📄 fake_news.csv ┣ 📂models ┃ ┗ 📄 lstm_model.h5 ┣ 📄 train.py ┣ 📄 evaluate.py ┣ 📄 README.md ┗ 📄 requirements.txt


## 🧠 Modèle LSTM

```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=200))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
📈 Résultats

Métrique	Score
Accuracy	0.982
Precision	0.98
Recall	0.98
F1-Score	0.98

🧪 Robustesse
Le modèle a été testé avec des données générées automatiquement par GPT-2. Résultat : Baisse considérable de précision, montrant la sensibilité du modèle aux perturbations sémantiques.

Precision de detection des fake news générées après le fine-tuning	 0.92

👨‍💻 Auteur
Franklin — Data Science & Intelligence Artificielle
Université du Québec en Outaouais
GitHub: [Platini20]