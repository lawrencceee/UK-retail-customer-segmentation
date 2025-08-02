import gradio as gr
import numpy as np
import joblib

# Load saved scaler and model
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')

def predict_cluster(total_qty, total_price, invoice_count, avg_recency, avg_unit_price):
    X = np.array([[total_qty, total_price, invoice_count, avg_recency, avg_unit_price]])
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]
    
    return f"Predicted Cluster: {cluster}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Number(label="Total Quantity"),
        gr.Number(label="Total Price"),
        gr.Number(label="Invoice Count"),
        gr.Number(label="Average Recency"),
        gr.Number(label="Average Unit Price")
    ],
    outputs="text",
    title="Customer Segment Predictor"
)

iface.launch()
