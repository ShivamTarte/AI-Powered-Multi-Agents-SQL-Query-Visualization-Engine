# 🚀 Multi-Agent SQL Query & Visualization Engine

![Multi-Agent AI](https://img.shields.io/badge/Multi--Agent-AI-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg) ![SQL](https://img.shields.io/badge/SQL-MySQL-red.svg) ![Visualization](https://img.shields.io/badge/Visualization-Plotly%2FMatplotlib-green.svg)

## 🎯 Overview
**Multi-Agent SQL Query & Visualization Engine** is an AI-powered chatbot system that seamlessly generates, executes, and visualizes SQL queries. It enables users to interact with databases using natural language, returning structured responses along with dynamic visualizations. This system is built using **Gradio, CrewAI, LangChain, and Python REPL**, making data analysis more accessible and intuitive.

## ✨ Features
- **AI-Powered SQL Generation** – No SQL expertise required
- **Real-Time Query Validation** – Prevents SQL injection & errors
- **Automated Data Visualization** – Converts raw data into insights
- **Python REPL Execution** – Dynamically runs visualization scripts
- **Interactive Gradio Chatbot Interface** – User-friendly & intuitive
- **Supports Bar Graphs, Scatter Plots, Tables, and More**

## 🏗 Tech Stack & Frameworks Used
- **Gradio** – AI-powered chatbot interface
- **CrewAI** – Multi-agent orchestration
- **LangChain** – Natural language to SQL conversion
- **MySQL + SQLAlchemy** – Database interaction
- **Matplotlib, Plotly, Seaborn** – Stunning visualizations
- **Python REPL Tool** – Secure script execution

## 🔧 Installation & Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/multi-agent-sql-visualization.git
cd multi-agent-sql-visualization
pip install -r requirements.txt
```

## 🚀 Usage
Run the AI-powered chatbot via Gradio:
```bash
python src/chatbot_interface.py
```

**Example Interaction:**
```bash
User: Show me the top 10 most populated countries.
Chatbot: Fetching data from MySQL...
```

📊 **Result:**  
A **bar chart** displaying the top 10 most populated countries is automatically generated, alongside a table of insights.

## 🔍 Example Visualization (Python REPL Tool)
```python
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Country": ["China", "India", "USA"],
    "Population": [1400000000, 1380000000, 331000000]
}

df = pd.DataFrame(data)
plt.figure(figsize=(8,5))
plt.bar(df["Country"], df["Population"], color='skyblue')
plt.xlabel("Country")
plt.ylabel("Population")
plt.title("Top 3 Most Populated Countries")
plt.show()
```

## 🙌 Contributing
Pull requests are welcome! Feel free to open an issue if you have suggestions or encounter any bugs.

🔥 **Transform SQL Queries into Insights, Effortlessly!** 🔥

## ✨ Chatbot Examples

![AI Powered SQL+Visualizing Chatbot 1](https://github.com/user-attachments/assets/0912a268-dcc7-42ad-a01f-62785ffb2eb5)
![AI Powered SQL+Visualizing Chatbot 2](https://github.com/user-attachments/assets/5758a778-f905-41b5-a68c-5abb108b3277)



