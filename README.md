# 🚀 Multi-Agent SQL Query & Visualization Engine

![Multi-Agent AI](https://img.shields.io/badge/Multi--Agent-AI-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg) ![SQL](https://img.shields.io/badge/SQL-MySQL-red.svg) ![Visualization](https://img.shields.io/badge/Visualization-Plotly%2FMatplotlib-green.svg)

## 🎯 Overview
**Multi-Agent SQL Query & Visualization Engine** is an AI-powered system that generates, executes, and visualizes SQL queries seamlessly. Using **CrewAI**, **LangChain**, and **Python REPL**, this system turns natural language queries into SQL results and dynamic charts—automatically! 🤖📊

## ✨ Features
✅ **AI-Powered SQL Generation** – No SQL expertise required! 🤯  
✅ **Real-Time Query Validation** – Prevents SQL injection & errors 🛡️  
✅ **Automated Data Visualization** – Converts raw data into insights 📈  
✅ **Python REPL Execution** – Dynamically runs visualization scripts 🚀  
✅ **Interactive CLI Interface** – Simple, efficient, and user-friendly 💡  

## 🏗 Tech Stack & Frameworks Used
🔹 **CrewAI** – Multi-agent orchestration 🔄  
🔹 **LangChain** – AI-powered query generation 🤖  
🔹 **MySQL + SQLAlchemy** – Database interaction 🗄️  
🔹 **Matplotlib, Plotly, Seaborn** – Stunning visualizations 🎨  
🔹 **Python REPL Tool** – Secure script execution 🖥️  

## 🔧 Installation & Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/multi-agent-sql-visualization.git
cd multi-agent-sql-visualization
pip install -r requirements.txt
```

## 🚀 Usage
Run the system via CLI:
```bash
python src/ai_powered_agent_for_data_annalysis.py
```
**Example:**
```bash
Enter your SQL Query in Natural Language: Show me the top 10 most populated countries.
```
📊 **Result:**
A bar chart will automatically pop up displaying the top 10 most populated countries! 🎉

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
Pull requests are welcome! Feel free to open an issue if you have suggestions or encounter any bugs. 🚀

🔥 **Transform SQL Queries into Insights, Effortlessly!** 🚀
