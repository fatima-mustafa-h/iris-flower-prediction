# Iris Flower Prediction

<div align="center">

![Iris Flower Prediction](https://i.postimg.cc/mkD0H1r2/iris-flower-prediction.jpg)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iris-flower-prediction-classify.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A beautiful, dark-themed ML web app that classifies Iris flowers in real time.**

[Live Demo](https://iris-flower-prediction-classify.streamlit.app/) · [Author](#author)

</div>

---

## Features

- **Real-time prediction** using a trained Random Forest Classifier
- **Animated flower confetti** — unique per predicted species
- **Confidence scores** with visual probability bars for all 3 classes
- **Species info cards** with habitat, traits & fun facts
- **Dark glassmorphism UI** with Cormorant Garamond + DM Sans typography
- **Input summary** displayed after each prediction

---

## Species Covered

| Species | Emoji | Description |
|---|---|---|
| *Iris Setosa* | 💜 | Small & delicate · Arctic regions · Broad sepals |
| *Iris Versicolor* | 💙 | Medium · North America · Blue Flag Iris |
| *Iris Virginica* | 🩷 | Large & elegant · Eastern USA · Virginia Blue Flag |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `Streamlit` | Web app framework |
| `scikit-learn` | Random Forest model |
| `joblib` | Model serialization |
| `numpy` | Feature array handling |
| `pandas` | Input DataFrame formatting |

---

## Project Structure

```
iris-flower-prediction/
│
├── .streamlit/            # Custom Configurations
  ├── config.toml          # Custom Slider Theme
├── iris_app.py            # Main Streamlit app
├── iris_model.pkl         # Trained Random Forest model
├── lable_encoder.pkl      # Label encoder for species names
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | UCI Iris (150 samples) |
| Features | Sepal length, Sepal width, Petal length, Petal width |
| Classes | Setosa · Versicolor · Virginica |
| Published | Ronald Fisher, 1936 |

---

## Author

**Fatima Mustafa H**

Built using Python & Streamlit.

---

<div align="center">
<sub>Iris Prediction App &nbsp;·&nbsp; Powered by Streamlit</sub>
</div>
