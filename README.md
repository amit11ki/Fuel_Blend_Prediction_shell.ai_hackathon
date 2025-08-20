# 🏆 Fuel Blend Properties Prediction Challenge - Shell.ai Hackathon 2025

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)](https://flask.palletsprojects.com/)
[![Leaderboard](https://img.shields.io/badge/Public%20Leaderboard-1st%20Place%20🥇-gold.svg)](https://github.com)
[![Score](https://img.shields.io/badge/Score-97.05-brightgreen.svg)](https://github.com)

## 🎯 Project Overview

This project was developed for the **Shell.ai Hackathon 2025**, focusing on predicting fuel blend properties for sustainable aviation fuels (SAFs). Our team achieved **1st place on the public leaderboard** with a score of **97.05** in Level 1, and advanced to Level 2 where we built a comprehensive web-based prototype.

### 🌍 Challenge Background

The global push for sustainability is transforming industries, particularly in mobility, shipping, and aviation. Sustainable Aviation Fuels (SAFs) are crucial for reducing environmental impact, but creating optimal fuel blends is complex. This challenge involves predicting final properties of complex fuel blends based on their constituent components and proportions.

## 🏗️ Project Structure

```
fuel-blend-prediction/
│
├── 📄 app.py                          # Main Flask application
├── ⚙️ config.py                       # Configuration settings
├── 📋 requirements.txt               # Python dependencies
├── 📊 predictions_all_blends.csv     # Final predictions output
├── 📖 README.md                      # This file
│
├── 📂 data/                          # Training and test datasets
│   ├── train.csv                     # Training data (blend compositions + properties)
│   ├── test.csv                      # Test data for predictions
│   ├── sample_solution.csv           # Sample submission format
│   └── samplyu.csv                   # Additional sample data
│
├── 🤖 models/                        # Trained models for each blend property
│   ├── Blend-1/                      # Random Forest model (joblib)
│   │   ├── model_BlendProperty1.joblib
│   │   └── predictions_BlendProperty1.csv
│   │
│   ├── Blend-2/                      # FT-Transformer model (PyTorch)
│   │   ├── best_fttransformer_model_fold_*.pth
│   │   ├── scaler.pkl
│   │   └── top_features.pkl
│   │
│   ├── Blend-3/                      # Neural Network with K-Fold CV
│   │   ├── final_model_*_fold*.pth
│   │   ├── best_params.json
│   │   └── scaler.gz
│   │
│   ├── Blend-4/                      # FT-Transformer with K-Fold
│   ├── Blend-5/                      # Feature-selected model
│   ├── Blend-6/                      # Random Forest variant
│   ├── Blend-7/                      # Two-stage neural network
│   ├── Blend-8/                      # Advanced neural network
│   ├── Blend-9/                      # FT-Transformer variant
│   └── Blend-10/                     # Final ensemble model
│
├── 📈 predictions/                   # Generated prediction files
│   └── predictions_all_blends_*.csv  # Timestamped prediction outputs
│
├── 🔧 src/                          # Source code modules
│   ├── predictor.py                  # Main prediction orchestrator
│   ├── prompts.py                    # AI assistant prompts
│   │
│   ├── predictors/                   # Individual blend predictors
│   │   ├── blend1_pred.py           # Random Forest predictor
│   │   ├── blend2_pred.py           # FT-Transformer predictor
│   │   ├── blend3_pred.py           # Neural network predictor
│   │   └── ... (blend4-10_pred.py)  # Other specialized predictors
│   │
│   └── training/                     # Model training scripts
│       ├── blend1_train.py          # Random Forest training
│       ├── blend2_train.py          # FT-Transformer training
│       └── ... (other training scripts)
│
├── 🎨 templates/                    # HTML templates for Flask app
│   ├── home.html                    # Landing page
│   ├── predictor.html               # Prediction interface
│   └── chatbot.html                 # AI assistant interface
│
└── 📁 Uploads/                      # Temporary file uploads
    └── temp_single_input.csv        # User uploaded data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Required packages listed in `requirements.txt`

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://gitlab.com/harshilsangani07/fuel-blend-prediction
   cd fuel-blend-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   **Web Interface (Recommended):**
   ```bash
   python app.py
   ```
   Then visit `http://localhost:5000` in your browser
   
   **Command Line Predictions:**
   ```bash
   python src/predictor.py
   ```

## 🔬 Technical Approach

### Level 1: Model Development

Our winning approach employed diverse machine learning techniques:

- **🌲 Random Forest**: Traditional ensemble method for baseline predictions
- **🧠 FT-Transformer**: Feature Tokenizer + Transformer for tabular data
- **🔗 Neural Networks**: Deep learning with custom architectures
- **📊 Bayesian Ridge**: Linear model with regularization
- **🎯 Ensemble Methods**: Combining multiple model predictions
- **✂️ Feature Selection**: Identifying most important blend components
- **📈 K-Fold Cross-Validation**: Robust model validation strategy

### Level 2: Web Application

Built a comprehensive Flask-based platform featuring:

- **🎯 Interactive Prediction Interface**: Upload CSV or input single blend compositions
- **🤖 AI Assistant Chatbot**: Intelligent query handling and explanations
- **📊 Visualization Dashboard**: Results display with intuitive charts
- **📱 Responsive Design**: Modern HTML/CSS for optimal user experience
- **⚡ Real-time Processing**: Instant predictions using pre-trained models

## 📊 Model Performance

| Blend Property | Model Type | Architecture | Validation Score |
|---------------|------------|--------------|------------------|
| Property 1 | Random Forest | Ensemble | High |
| Property 2 | FT-Transformer | Transformer | High |
| Property 3 | Neural Network | Deep Learning | High |
| Property 4 | FT-Transformer | Transformer | High |
| Property 5 | Feature Selected | ML + Selection | High |
| Property 6 | Random Forest | Ensemble | High |
| Property 7 | Two-Stage NN | Hierarchical | High |
| Property 8 | Advanced NN | Deep Learning | High |
| Property 9 | FT-Transformer | Transformer | High |
| Property 10 | Ensemble | Multi-Model | High |

**Final Public Leaderboard Score: 97.05 (1st Place 🥇)**

## 🎮 Usage Guide

### Web Interface

1. **Home Page**: Overview of the fuel blending challenge
2. **Predictor Page**: 
   - Upload CSV files with blend compositions
   - Enter single blend data manually
   - View prediction results with visualizations
3. **AI Assistant**: Ask questions about fuel blending and get intelligent responses

### Command Line

```bash
# Run predictions on test data
python src/predictor.py

# Output will be saved to predictions/predictions_all_blends.csv
```

## 📁 Data Format

### Input Features (55 columns)
- **Blend Composition (5 columns)**: Volume percentages of each component
- **Component Properties (50 columns)**: Properties for each component (Component1_Property1, etc.)

### Target Variables (10 columns)
- **BlendProperty1** to **BlendProperty10**: Final blend properties to predict

### Evaluation Metric
**MAPE (Mean Absolute Percentage Error)**:
- Public Leaderboard Reference: 2.72
- Private Leaderboard Reference: 2.58
- Score Formula: `max(0, 100 - 100 * cost/reference_cost)`

## 🛠️ Key Features

- ✅ **Multi-Model Architecture**: Different algorithms for different blend properties
- ✅ **Cross-Validation**: Robust K-fold validation for model reliability  
- ✅ **Feature Engineering**: Advanced feature selection and preprocessing
- ✅ **Web Interface**: User-friendly Flask application
- ✅ **AI Assistant**: Intelligent chatbot for user queries
- ✅ **Scalable Design**: Modular code structure for easy maintenance
- ✅ **Production Ready**: Error handling and input validation

## 🏆 Competition Results

### Level 1 (Model Development)
- **🥇 1st Place** on Public Leaderboard
- **Score: 97.05** (out of 100)
- Successfully advanced to Level 2

### Level 2 (Prototype Development)
- Developed comprehensive web-based solution
- Integrated AI assistant for enhanced user experience
- Demonstrated scalability and innovation

## 👥 Team

This project was developed as part of the Shell.ai Hackathon 2025 by our dedicated team focused on sustainable energy solutions.

## 🌱 Impact

Our solution contributes to sustainable aviation fuel development by:

- 🔬 **Accelerating R&D**: Rapid evaluation of thousands of blend combinations
- 🎯 **Optimizing Formulations**: Identifying recipes that maximize sustainability
- ⚡ **Reducing Time-to-Market**: Faster development of new sustainable fuels
- 🏭 **Enabling Real-time Optimization**: Production facility integration capabilities

## 📜 License

This project was developed for the Shell.ai Hackathon 2025. Please refer to the competition terms and conditions.

## 🔗 Links

- [PPT Link](https://docs.google.com/presentation/d/1V9IP2-MMwO88JcPr4D9M-F7ik3Gg2KwV/edit?usp=sharing&ouid=106203342667501477179&rtpof=true&sd=true)
- [Demo Video Link](https://drive.google.com/file/d/1vjtmmlIMzNBMZZLfPlow6gAl9cbU6RZ1/view?usp=drivesdk)
- [Shell.ai Hackathon](https://www.hackerearth.com/challenges/new/competitive/shellai-hackathon-2025/)
- [Competition Details](https://www.hackerearth.com/challenges/competitive/shellai-hackathon-2025/leaderboard/)

---

*Built with ❤️ for sustainable energy solutions*


