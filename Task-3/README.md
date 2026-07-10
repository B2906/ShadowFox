# \# ⌨ AI Keyboard Assistant: LSTM-Based Next Word Prediction with Autocorrect

# 

# !\[Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)

# !\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)

# !\[Streamlit](https://img.shields.io/badge/Streamlit-Web\_App-red?logo=streamlit)

# !\[License](https://img.shields.io/badge/License-MIT-green)

# 

# \## 📌 Project Overview

# 

# The \*\*AI Keyboard Assistant\*\* is a Deep Learning based Natural Language Processing (NLP) application that predicts the \*\*next word\*\* in a sentence using an \*\*LSTM (Long Short-Term Memory)\*\* neural network.

# 

# The application also includes an optional \*\*Autocorrect\*\* feature that corrects spelling mistakes before generating predictions, providing a more intuitive typing experience.

# 

# The model has been trained using the \*\*Tiny Shakespeare\*\* dataset and is deployed through an interactive \*\*Streamlit\*\* web application.

# 

# \---

# 

# \## ✨ Features

# 

# \- 🔤 LSTM-based Next Word Prediction

# \- ✍ Optional Autocorrect using TextBlob

# \- 🎯 Top-3 Word Predictions

# \- 📊 Confidence Scores

# \- 📈 Training Accuracy \& Loss Visualization

# \- 🌐 Interactive Streamlit Web Application

# \- 💾 Saved Model and Tokenizer

# \- ⚡ Real-time Prediction

# 

# \---

# 

# \## 🛠 Technologies Used

# 

# | Category | Technology |

# |----------|------------|

# | Language | Python |

# | Deep Learning | TensorFlow / Keras |

# | NLP | TextBlob |

# | Frontend | Streamlit |

# | Data Processing | NumPy |

# | Visualization | Matplotlib |

# | Dataset | Tiny Shakespeare |

# 

# \---

# 

# \## 📂 Project Structure

# 

# ```text

# Task-3

# │

# ├── app/

# │   ├── helper.py

# │   └── streamlit\_app.py

# │

# ├── data/

# │   ├── raw/

# │   └── processed/

# │

# ├── models/

# │   ├── config.json

# │   ├── next\_word\_model.keras

# │   └── tokenizer.pkl

# │

# ├── outputs/

# │   ├── accuracy.png

# │   └── loss.png

# │

# ├── src/

# │   ├── autocorrect.py

# │   ├── train.py

# │   ├── predict.py

# │   ├── model.py

# │   ├── preprocessing.py

# │   ├── tokenizer\_utils.py

# │   └── ...

# │

# ├── requirements.txt

# ├── main.py

# └── README.md

# ```

# 

# \---

# 

# \## 🧠 Model Architecture

# 

# The application uses the following neural network architecture:

# 

# Input Layer

# 

# ↓

# 

# Embedding Layer

# 

# ↓

# 

# LSTM Layer

# 

# ↓

# 

# Dropout Layer

# 

# ↓

# 

# Dense Layer (ReLU)

# 

# ↓

# 

# Output Layer (Softmax)

# 

# \---

# 

# \## 📊 Dataset

# 

# Dataset Used:

# 

# \*\*Tiny Shakespeare Dataset\*\*

# 

# The dataset contains thousands of Shakespearean sentences used for training an LSTM language model.

# 

# \---

# 

# \## ⚙ Installation

# 

# Clone the repository

# 

# ```bash

# git clone https://github.com/B2906/ShadowFox.git

# ```

# 

# Move to Task-3

# 

# ```bash

# cd ShadowFox/Task-3

# ```

# 

# Create Virtual Environment

# 

# ```bash

# python -m venv venv

# ```

# 

# Activate Environment

# 

# Windows

# 

# ```bash

# venv\\Scripts\\activate

# ```

# 

# Install Dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# Run Streamlit

# 

# ```bash

# streamlit run app/streamlit\_app.py

# ```

# 

# \---

# 

# \## 🚀 How It Works

# 

# User Input

# 

# ↓

# 

# (Optional) Autocorrect

# 

# ↓

# 

# Text Preprocessing

# 

# ↓

# 

# Tokenization

# 

# ↓

# 

# Sequence Padding

# 

# ↓

# 

# LSTM Model

# 

# ↓

# 

# Top-3 Predictions

# 

# ↓

# 

# Display Results

# 

# \---

# 

# \## 📈 Results

# 

# The trained model successfully predicts the most probable next words based on user input.

# 

# Example:

# 

# Input:

# 

# ```

# to be

# ```

# 

# Prediction:

# 

# ```

# the

# ```

# 

# The application also displays the Top-3 predictions along with confidence scores.

# 

# \---

# 

# \## 📷 Screenshots

# 

# Add the following screenshots:

# 

# \- Home Page

# \- Prediction Example

# \- Autocorrect Example

# \- Accuracy Graph

# \- Loss Graph

# 

# \---

# 

# \## 🔮 Future Scope

# 

# \- Transformer-based Language Models

# \- Larger Training Dataset

# \- Voice Input Support

# \- Multi-language Prediction

# \- Mobile Keyboard Integration

# \- Personalized User Suggestions

# 

# \---

# 

# \## 👨‍💻 Developer

# 

# \*\*Bhuvanesh Gupta\*\*

# 

# Machine Learning \& Deep Learning Enthusiast

# 

# GitHub:

# https://github.com/B2906

# 

# \---

# 

# \## 📜 License

# 

# This project was developed as part of the \*\*ShadowFox Machine Learning Internship\*\* for educational purposes.

