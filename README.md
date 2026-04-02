# 🌾 Crop Recommendation System

An intelligent machine learning system that recommends the best crops based on soil and climate parameters. Built with scikit-learn, Streamlit, and pandas.

## 🎯 Features

- **Multi-Model Training**: Compares 6 different ML algorithms
- **Hyperparameter Tuning**: GridSearchCV optimization for Random Forest
- **Interactive Web UI**: Streamlit app for real-time predictions
- **Data Validation**: Robust input validation and handling
- **Crop Recommendations**: Top-N recommendations with confidence scores
- **Agronomic Reasoning**: Explains why each crop is recommended

## 📊 Supported Crops

Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation-system.git
cd crop-recommendation-system
```

2. **Create virtual environment**
```bash
python -m venv cvenv
cvenv\Scripts\activate  # Windows
# OR
source cvenv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Train the Model
```bash
python main.py
```

This will:
- Load and validate data from `Crop_recommendation.csv`
- Train and compare all models
- Perform hyperparameter tuning
- Save the best model to `models/crop_model.pkl`
- Generate reports in `reports/`

#### Launch Streamlit App
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
crop_recommendation/
├── data_preprocessing.py      # Data loading, validation, preprocessing
├── model_training.py          # Model training & evaluation
├── predict.py                 # Inference & recommendations
├── main.py                    # Training pipeline entry point
├── app.py                     # Streamlit web app
├── requirements.txt           # Python dependencies
├── Crop_recommendation.csv    # Training dataset
├── models/
│   ├── crop_model.pkl         # Trained model pipeline
│   └── label_encoder.pkl      # Label encoder
└── reports/
    ├── model_comparison.png
    ├── feature_importance.png
    └── confusion_matrix.png
```

## 📈 Input Features

The system takes 7 agronomic features:

| Feature | Range | Unit |
|---------|-------|------|
| Nitrogen (N) | 0-140 | kg/hectare |
| Phosphorus (P) | 5-145 | kg/hectare |
| Potassium (K) | 5-205 | kg/hectare |
| Temperature | 8-44 | °C |
| Humidity | 14-100 | % |
| Soil pH | 3.5-10.0 | - |
| Rainfall | 20-300 | mm |

## 🤖 Models

The system trains and compares:
1. Logistic Regression
2. Gaussian Naive Bayes
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. Gradient Boosting

## 📊 Performance

Models are evaluated using:
- Accuracy
- Precision & Recall
- F1-Score
- Confusion Matrix
- Cross-Validation (5-fold)

## 🛠️ Technologies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Web UI**: Streamlit
- **Model Serialization**: joblib

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

[Your Name](https://github.com/Preethi-567)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Have questions? Open an issue or contact me at [preethib1011@gmail.com]

---

⭐ If you find this project helpful, please give it a star!
