# E-Commerce Customer Analytics Dashboard (Streamlit)

Interactive dashboard built with **Streamlit** and **Plotly** to explore
e-commerce transaction data and customer segmentation using the **RFM (Recency, Frequency, Monetary)** model.

> This project is part of my learning/journey in data analytics & dashboarding.
> The app is designed to mimic a Power BI-style dashboard but fully in Python.

---

## 📌 Main Features

The app consists of **two main pages**:

### 1. Overview Dashboard

High-level view of **sales performance** with filters and charts.

**Filters:**

- Date range  
- Country  
- Product  

**KPIs:**

- Total Customers  
- Total Transactions  
- Total Sales  
- Average Monetary per Customer  
- Average Frequency (transactions per customer)  
- Average Recency (days since last transaction – joined from RFM table)

**Tabs & Visuals:**

1. **Trends**
   - Line chart:
     - X: Month  
     - Y (left): Total Customers & Total Transactions  
     - Y (right): Total Sales (in $M)

2. **Products**
   - Bar + line chart:
     - Top 5 products by Total Sales  
     - Bars: Total Sales  
     - Line: Total Transactions  

3. **Customers**
   - Bar + line chart:
     - Top 5 customers by Total Sales  
     - Bars: Total Sales  
     - Line: Total Transactions  

4. **Countries**
   - Pie chart: Total Sales by Country (United Kingdom vs Others)  
   - Pie chart: Total Customers by Country (United Kingdom vs Others)

---

### 2. Customer Segmentation Dashboard

Focus on **RFM-based customer segments** (e.g. Champions, At Risk, Casual Shoppers, etc.).

Uses a join between **transaction data** and **RFM output**.

**Filters:**

- Date range  
- Country  
- Product  
- Customer Segment  

**Tabs & Visuals:**

1. **Trend**
   - Line chart:
     - X: Month  
     - Y: Total Sales  
     - Series: Segment  

2. **Segments**
   - **KPI Table per Segment** (with formatting):
     - Total Customers & % Customers  
     - Total Sales & % Total Sales  
     - Total Transactions & % Total Transactions  
     - Average Days Since Last Transaction (Recency)  
   - **Bar Chart – Distribution of Avg R, F, M Scores per Segment**:
     - X: Customer Segment  
     - Y: Average Score  
     - Series: R Score, F Score, M Score  

3. **Products**
   - Horizontal **stacked bar chart**:
     - X: Number of Sales / Transactions  
     - Y: Product (Top N products)  
     - Color: Customer Segment  

---

## 🗂️ Project Structure

```bash
AS41/
├── AS41.py                    # Main Streamlit app
├── data finpro manipulated.csv# Cleaned transaction data
├── data rfm finpro.csv        # RFM segmentation output per customer
├── requirements.txt           # Python dependencies (Streamlit, Plotly, Pandas, etc.)
└── README.md                  # This file
