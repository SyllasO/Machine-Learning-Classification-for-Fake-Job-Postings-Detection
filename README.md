# 🧠 Fake Job Postings Detection using Machine Learning & Apache Spark  
Machine Learning–based detection of fake job postings using text and structured data, powered by Apache Spark distributed computing for faster execution.


## 🚀 Project Overview  
The goal of this project is to use **statistical machine learning (ML)** and **Apache Spark** to detect **fraudulent job postings**.  
The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), includes job descriptions, titles, company profiles, and metadata such as company logos and screening questions.

After cleaning and preprocessing, the dataset was reduced to **5,000 stratified samples** (balanced between real and fake job postings).  
The combined `full_text` column merges all textual fields for Natural Language Processing (NLP) analysis.

---

## 🧩 Dataset Description  
| Feature | Description |
|----------|-------------|
| `title` | Job title |
| `company_profile` | Company background |
| `description` | Job posting details |
| `requirements` | Qualifications or skills |
| `benefits` | Offered job benefits |
| `has_company_logo` | 1 if posting includes logo |
| `has_questions` | 1 if posting includes screening questions |
| `full_text` | Concatenated text column |
| `Ground_Truth` | 0 = Real job, 1 = Fake job |

---

## ⚙️ Spark Execution Setup  

Before running experiments, Spark was **manually powered and configured** via command-line scripts across two virtual machines (VMs).

```bash
# Start master node on Hadoop1
/opt/spark/sbin/start-master.sh

# Start worker node(s)
/opt/spark/sbin/start-worker.sh spark://hadoop1:7077

# Start both master and workers
/opt/spark/sbin/start-all.sh

# Submit Spark job
/opt/spark/bin/spark-submit --master spark://hadoop1:7077 /opt/preprocessing.py
```

Spark ran in **true cluster mode**, where the master coordinated and distributed tasks to worker nodes. The Spark UI was monitored to confirm job execution and resource utilization.

---

## 🧠 Methods and Model Implementation  

**Programming Environment:**  
- Python 3.11  
- Jupyter Notebook  
- scikit-learn, pandas, numpy, matplotlib, seaborn  

**Data Preprocessing Steps:**  
- Replaced missing values with `"No information provided"`  
- Normalized text (lowercase, punctuation removal, etc.)  
- Combined all text columns into `full_text`  
- Applied **TF-IDF vectorization** (`max_features = 5000`)  
- Included numeric predictors (`has_company_logo`, `has_questions`)  

**Train/Test Split:**  
80/20 with stratification to preserve class balance.

**Machine Learning Models Used:**  
1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Support Vector Machine (Linear SVM)  
4. Random Forest  
5. Gradient Boosting  

---

## 📊 Model Performance  

| Model | Accuracy | AUC | Recall | Precision | Specificity |
|:------|:---------:|:---:|:-------:|:-----------:|:-------------:|
| Logistic Regression | 0.94 | 0.97 | 0.82 | 0.45 | 0.95 |
| KNN (k=5) | 0.96 | 0.90 | 0.43 | 0.70 | 0.99 |
| Linear SVM | 0.97 | 0.97 | 0.69 | 0.81 | 0.99 |
| Random Forest | 0.96 | 0.93 | 0.33 | 1.00 | 1.00 |
| Gradient Boosting | **0.97** | **0.96** | **0.41** | **1.00** | **1.00** |

🟢 **Gradient Boosting** achieved the best performance overall, with an **accuracy of 97%** and **AUC of 0.99**.

---

## ⚡ Spark Cluster Performance  

To measure the benefit of distributed computing, the project was executed on one VM (single-node mode) and then on two VMs (cluster mode).  

| Configuration | Description | Runtime (minutes) |
|---------------|--------------|------------------|
| Single VM (Worker only) | Spark Master and Worker on a single VM | 35 |
| Two VMs (Master + Worker) | Master on Hadoop1, Worker on Hadoop2 | 17 |

### 🔹 Performance Interpretation  
Using two VMs reduced runtime by **51.4%**, providing a **2.06× speedup**:

\[
\text{Speedup} = \frac{35}{17} \approx 2.06
\]

This demonstrates that distributed Spark clusters significantly improve computation speed and scalability for big-data processing tasks.

---

## 💬 Discussion and Conclusion  
The integration of **textual and structured data** proved powerful for detecting fake job postings.  
Ensemble models (Random Forest and Gradient Boosting) captured complex feature interactions more effectively than baseline methods.  

From a computational perspective, the Spark-powered multi-VM configuration achieved more than double the processing speed of a single-VM setup.  
This project highlights the combined advantage of **machine learning** and **distributed computing** in addressing real-world data challenges such as online fraud detection.  

### 🔮 Future Improvements  
- Deploy deep learning models (LSTM or BERT) for contextual NLP.  
- Use explainable AI tools (SHAP/LIME) for model interpretation.  
- Optimize via hyperparameter tuning and cross-validation.  
- Expand Spark cluster to additional worker nodes for scalability testing.  

---

## 🧾 Repository Contents  
```
📁 FakeJobPostingsDetection
 ┣ 📄 Fake_Job_Postings_Cleaned.csv
 ┣ 📄 Fake_Job_Postings_Report.docx
 ┣ 📄 notebook.ipynb
 ┣ 📄 README.md
 ┗ 📁 screenshots/
     ┣ Spark_Single_VM.png
     ┗ Spark_Two_VMs.png
```
