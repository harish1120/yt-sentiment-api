# YouTube Comment Sentiment Analysis

![YouTube Sentiment Analysis](https://img.shields.io/badge/YouTube-Sentiment%20Analysis-blue?style=for-the-badge&logo=youtube)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-orange?style=for-the-badge&logo=fastapi)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-yellow?style=for-the-badge)
![DVC](https://img.shields.io/badge/DVC-3.0-purple?style=for-the-badge)

A comprehensive YouTube comment sentiment analysis system that uses machine learning to classify comments as positive, neutral, or negative. The project includes a complete ML pipeline, REST API service, and a Chrome extension for seamless integration with YouTube.

## 📊 Project Overview

This project provides an end-to-end solution for analyzing YouTube video comments:

- **Data Pipeline**: Automated data ingestion, preprocessing, and feature engineering using DVC
- **ML Model**: LightGBM classifier with TF-IDF vectorization for sentiment classification
- **API Service**: FastAPI-based REST API for real-time sentiment predictions
- **Chrome Extension**: Browser extension to analyze YouTube comments directly
- **MLOps**: Experiment tracking with MLflow and model versioning

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         YouTube Sentiment Analysis                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────┐    ┌───────────────┐     ┌─────────────┐    ┌─────────────┐   │
│  │   Data      │───▶│  Data         │───▶ │   Model     │───▶│    Model    │   │
│  │ Ingestion   │    │ Preprocessing │     │  Building   │    │ Evaluation  │   │
│  └─────────────┘    └───────────────┘     └─────────────┘    └─────────────┘   │
│         │                  │                   │                  │            │
│         ▼                  ▼                   ▼                  ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   DVC       │    │    NLP      │    │   LightGBM  │    │   MLflow    │      │
│  │   Pipeline  │    │  Pipeline   │    │  + TF-IDF   │    │  Tracking   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                        FastAPI Backend                          │           │
│  │  /predict  /predict_with_timestamps  /generate_chart            │           │
│  │  /generate_wordcloud  /generate_trend_graph                     │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                   Chrome Extension (Frontend)                   │           │
│  │     YouTube Comment Analysis • Sentiment Charts • Word Cloud    │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

### Core Functionality
- **Sentiment Classification**: Classifies YouTube comments into 3 categories:
  - 😊 Positive (1)
  - 😐 Neutral (0)
  - 😞 Negative (-1)

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check endpoint |
| `/predict` | POST | Predict sentiment for list of comments |
| `/predict_with_timestamps` | POST | Predict sentiment with timestamps |
| `/generate_chart` | POST | Generate sentiment distribution pie chart |
| `/generate_wordcloud` | POST | Generate word cloud from comments |
| `/generate_trend_graph` | POST | Generate sentiment trend over time |

### Chrome Extension Features
- Extracts comments from YouTube videos
- Real-time sentiment analysis
- Visual sentiment distribution charts
- Word cloud generation
- Sentiment trend analysis over time
- Top comments with sentiment scores
- Comment metrics (total, unique commenters, avg length, avg sentiment)

## 📁 Project Structure

```
yt-sentiment-api/
├── .dvcignore                 # DVC ignore file
├── .gitignore                 # Git ignore file
├── LICENSE                    # CC0 1.0 Universal License
├── README.md                  # This file
├── dvc.lock                   # DVC lock file
├── dvc.yaml                   # DVC pipeline configuration
├── params.yaml                # Model and pipeline parameters
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
│
├── data/                      # Data directory (created by DVC)
│   ├── raw/                   # Raw train/test data
│   └── interim/               # Processed data
│
├── fastapi/                   # FastAPI application
│   ├── main.py               # API endpoints and logic
│   └── schema.py             # Pydantic schemas
│
├── main/                      # Jupyter notebooks
│   ├── baseline.ipynb        # Baseline model experiments
│   └── data.ipynb            # Data exploration
│
├── model_notebooks/           # Model experiments
│   ├── baseline.ipynb        # Baseline experiments
│   ├── bow_tfidf.ipynb       # TF-IDF experiments
│   ├── feature_num.ipynb     # Feature selection
│   ├── imbalance.ipynb       # Class imbalance handling
│   ├── lightgbm.ipynb        # LightGBM experiments
│   └── xgboost.ipynb         # XGBoost experiments
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   │
│   ├── data/                  # Data processing modules
│   │   ├── data_ingestion.py # Data loading and splitting
│   │   └── data_preprocessing.py # Text preprocessing
│   │
│   └── model/                 # ML model modules
│       ├── __init__.py
│       ├── model_builder.py   # Model training
│       ├── model_evaluation.py # Model evaluation with MLflow
│       └── register_model.py  # Model registration to MLflow
│
└── yt-chrome-pluging-fe/      # Chrome extension
    ├── manifest.json          # Extension manifest
    ├── popup.html             # Extension UI
    └── popup.js               # Extension logic
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda
- YouTube Data API key (for Chrome extension)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/harish1120/yt-sentiment-api.git
   cd yt-sentiment-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package**
   ```bash
   pip install -e .
   ```

5. **Download NLTK data** (for text preprocessing)
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

## 🏃 Running the Pipeline

### Using DVC

```bash
# Run the entire pipeline
dvc repro

# Run individual stages
dvc repro data_ingestion
dvc repro data_preprocessing
dvc repro model_building
dvc repro model_evaluation
dvc repro register_model
```

### Pipeline Stages

| Stage | Command | Output |
|-------|---------|--------|
| Data Ingestion | `python src/data/data_ingestion.py` | `data/raw/train.csv`, `data/raw/test.csv` |
| Preprocessing | `python src/data/data_preprocessing.py` | `data/interim/train_processed.csv`, `data/interim/test_processed.csv` |
| Model Building | `python src/model/model_builder.py` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| Evaluation | `python src/model/model_evaluation.py` | `experiment_info.json`, confusion matrix plots |
| Registration | `python src/model/register_model.py` | Model registered in MLflow |

## 🚀 Running the API Server

```bash
cd fastapi
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

FastAPI provides automatic API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🌐 Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `yt-chrome-pluging-fe` directory
5. Get a YouTube Data API key from [Google Cloud Console](https://console.cloud.google.com/)
6. Update the API key in `popup.js`:
   ```javascript
   const API_KEY = 'YOUR_YOUTUBE_API_KEY';
   ```
7. Configure the API URL in `popup.js`:
   ```javascript
   const API_URL = 'http://localhost:8000';
   ```

## 📊 Model Parameters

Configuration is managed through `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.20

model_building:
  ngram_range: [1, 3]        # Unigrams to trigrams
  max_features: 1000         # TF-IDF features
  learning_rate: 0.09        # LightGBM learning rate
  max_depth: 20              # Tree depth
  n_estimators: 367          # Number of trees
```

## 🔧 Text Preprocessing

The preprocessing pipeline includes:

1. **Lowercase conversion**: Normalizes text
2. **Whitespace handling**: Removes extra spaces and newlines
3. **Special character removal**: Keeps alphanumeric and basic punctuation
4. **Stopword removal**: Removes common words (keeps sentiment-relevant words)
5. **Lemmatization**: Reduces words to base forms

## 📈 Model Training

The model uses:
- **Vectorizer**: TF-IDF with n-grams (1-3)
- **Classifier**: LightGBM with:
  - Multiclass objective
  - Balanced class weights
  - L1/L2 regularization
  - Early stopping capability

## 🧪 Evaluation

Model evaluation includes:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- MLflow experiment tracking
- Model signature inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the CC0 1.0 Universal License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LightGBM](https://lightgbm.readthedocs.io/) for the gradient boosting framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [DVC](https://dvc.org/) for data version control
- [MLflow](https://mlflow.org/) for experiment tracking
- [NLTK](https://www.nltk.org/) for natural language processing
- [scikit-learn](https://scikit-learn.org/) for ML utilities

## 📧 Contact

For questions or support, please open an issue in the repository.

---

