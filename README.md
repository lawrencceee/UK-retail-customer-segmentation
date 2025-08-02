# 🛍️ UK Retail Customer Segmentation using KMeans Clustering

This project performs **unsupervised learning** on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) to segment UK-based retail customers based on their purchasing behavior. It helps identify key customer groups such as **VIPs, casual buyers, and at-risk customers**, providing actionable insights for marketing and retention strategies.

---

## 🚀 Interactive Clustering App (Gradio)

👉 Try the live demo here:  
https://your-gradio-app-link.com)](https://huggingface.co/spaces/astimepassesby/UK-retail-customer-segmentation

<img width="1494" height="614" alt="image" src="https://github.com/user-attachments/assets/b52791f6-8269-4971-8669-bee47015c63a" />


Enter basic customer attributes like **quantity purchased**, **total price**, **recency**, and more — and get an instant prediction of which customer segment they belong to.

---

## 📊 Project Overview

### ✨ Objective
To use **KMeans clustering** to segment customers into behavior-based groups using:
- Purchase volume
- Total spend
- Frequency
- Recency
- Unit price

### 📁 Dataset
- Source: [UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
- UK-based retail transactions from 2010 to 2011
- Key columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

---

## 🔍 Features Used for Clustering

- `TotalQuantity`: Total items bought per customer
- `TotalPrice`: Total spend
- `InvoiceCount`: Number of purchases
- `AverageRecency`: Days since last purchase
- `AverageUnitPrice`: Average price per unit

---

## 🧠 Model Pipeline

1. **Data Cleaning**  
   - Removed nulls, zero quantity, and outliers  
2. **Feature Engineering**  
   - Aggregated transactions per customer per day  
   - Calculated recency and average price  
3. **Feature Scaling**  
   - Used `StandardScaler` for normalization  
4. **KMeans Clustering**  
   - Evaluated using **Elbow Method** and **Silhouette Score**  
   - Optimal clusters: **K = 4**
5. **Model Deployment**  
   - Interactive Gradio app for real-time prediction

---

## 📈 Visualizations

### Elbow Method & Silhouette Scores
<img src="./assets/elbow_silhouette.png" width="600"/>

### Cluster Distribution with PCA
<img src="./assets/cluster_pca_plot.png" width="600"/>

### Spending Profile per Cluster
<img src="./assets/cluster_profile_barplot.png" width="600"/>

---

## 🧩 Cluster Interpretation

| Cluster | Spending Label              | Description                                       |
|---------|-----------------------------|--------------------------------------------------|
| 0       | Casual Shoppers             | Infrequent buyers, low spend, high recency       |
| 1       | VIP Customers               | Frequent, recent, and high spenders              |
| 2       | Engaged Mid-Tier Customers  | Moderate purchase volume and frequency           |
| 3       | One-Time Big Spenders       | Very high unit price, single high-value purchase |

---

## 💾 Model Files

- `scaler.pkl`: StandardScaler object for consistent input scaling
- `kmeans_model.pkl`: Trained KMeans clustering model

---

## 📦 Tech Stack

- **Python** (Pandas, NumPy, scikit-learn, Gradio, Matplotlib, Seaborn)
- **Jupyter Notebook** for analysis and visualization
- **Gradio** for app deployment
