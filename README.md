# AutoJudge: Predicting Programming Problem Difficulty 🧠

**AutoJudge** is an intelligent system designed to automatically predict the difficulty level (**Easy / Medium / Hard**) and a numerical difficulty score for competitive programming problems. By analyzing the textual description, input constraints, and mathematical density of a problem statement, AutoJudge predicts its difficulty as percieved by a human.

---

## 📂 Project Overview

Online coding platforms (like Codeforces, LeetCode, CodeChef) host thousands of problems. Classifying them usually requires manual tagging or user feedback. This project automates that process using **Machine Learning** and **Natural Language Processing (NLP)**.

### Key Objectives
* **Classification:** Categorize a problem as *Easy*, *Medium*, or *Hard*.
* **Regression:** Predict a precise numerical difficulty score (e.g., `800`, `1200`, `2100`).
* **Web Interface:** A simple, user-friendly UI where anyone can paste a problem and get an instant prediction.

---

## 📊 Dataset Used

The model was trained on a dataset of competitive programming problems containing:
* **Title:** Name of the problem.
* **Description:** The main story and task statement.
* **Input Description:** Constraints on variables (e.g., $1 \le N \le 10^5$).
* **Output Description:** What the program should print.
* **Target Labels:**
    * `problem_class`: The categorical difficulty (Easy/Medium/Hard).
    * `problem_score`: The numerical difficulty rating.

**Link to the dataset**: [Link](https://github.com/AREEG94FAHAD/TaskComplexityEval-24)

---

## 🛠 Approach & Methodology

We treat this as a **Hybrid NLP + Feature Engineering** task. Pure text analysis isn't enough to give a proper judgement. Specific mathematical signals define difficulty in coding.

### 1. Feature Engineering 
We extracted specific signals that act as proxies for algorithmic complexity:
* **Constraint Extraction:** Regex is used to find time complexity hints (e.g., $N \le 10^5$ usually implies $O(N \log N)$, whereas $N \le 20$ implies Exponential complexity).
* **Keyword Clusters:** Counting frequencies of words belonging to specific difficulty tiers:
    * *Hard:* "graph", "tree", "dynamic programming", "shortest path".
    * *Easy:* "brute force", "implementation", "print".
* **Math Density:** The ratio of mathematical symbols to total words.

### 2. Text Vectorization
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Used with **Bi-grams** to capture phrases like "binary search" or "segment tree".

### 3. Models Used
* **Classification (Random Forest Classifier):** Chosen for its robustness in handling high-dimensional text data mixed with dense numerical features. It achieved an accuracy of **~52%** and a macro-F1 score of **~48**(significantly better than the random baseline of 33%). These values are to be expected due to the smaller size of the dataset. 
* **Regression (Random Forest Regressor):** Used to predict the difficulty score on a scale of 1.1 to 9.7. For this model, MAE gave **1.69**, RMSE **2.01** and R2 **0.1856**, comparatively better than Linear Regression and XGBoost Regressor, although the latter had similar metrics, but Random Forest performed consistently better.

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### Prerequisites
* Python 3.8 or higher(preferable python 3.10)
* pip (Python package installer)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/autojudge.git](https://github.com/your-username/autojudge.git)
cd autojudge
```

### 2. Install Dependencies
```bash
pip install pip install pandas numpy scikit-learn streamlit scipy joblib
```

### 3. Train the Model (First Run Only)
To generate the model files (`.pkl`), run the following command if there is any issue with the (`.pkl`) files already uploaded. Otherwise there is no need.
```bash
python train_models.py
```
*This script will process the data, train the Random Forest models, and save them to disk.*

### 4. Run the Web Interface
Start the Streamlit application:
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Web Interface Guide

The web UI is built with **Streamlit** for simplicity and speed.

1.  **Input Fields:**
    * **Problem Description:** Paste the main body of the problem.
    * **Input Description:** Paste the constraints (e.g., "The first line contains an integer T...").
    * **Output Description:** Paste the required output format.
2.  **Predict Button:** Click to run the model.
3.  **Results:**
    * **Difficulty Class:** Displays whether the problem is Easy, Medium, or Hard.
    * **Difficulty Score:** Displays the estimated numerical rating (e.g., 4.5).

---

## 📈 Evaluation Metrics

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Classification Accuracy** | **52%** | The model correctly identifies the difficulty tier over half the time, which is strong for a subjective 3-class problem. |
| **Regression R²** | **0.1856** | Indicates the difficulty of predicting precise scores from text alone, highlighting the need for deeper semantic analysis. |
| **Regression MAE** | **1.69** | On average, the predicted score deviates by ~1.6 points from the actual user rating. |

---

## 🎥 Demo

**[Link to Demo Video]** *: blank 

**Developed by:**
* **Name:** Ishika Chikate
