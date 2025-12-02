# ⚽ Agentic Football Analytics AI

### Natural Language Match Insights using API-Football + LangChain + Streamlit

This project is an end-to-end **Agentic AI system** that allows users to query football matches using simple natural language.  
Example:

> “Give me the result and stats between Manchester United vs Liverpool, season 2022”

The system automatically:

- Parses the query
- Fetches data from **API-Football**
- Cleans + structures the match events and statistics
- Generates insights using **GPT-4o + LangChain agents**
- Displays results in a clean **Streamlit UI**

---

## 🚀 Features

### 🧠 **Agentic AI (LLM-powered)**

- Intelligent parsing of natural-language football queries
- Handles any teams + any season supported by the free API
- Generates human-like match summaries

### 📊 **Live Match Data Retrieval**

- Uses **API-Football** for fixtures, events, and statistics
- Supports 200+ teams and multiple seasons
- Automatic caching for fast responses

### 🖥️ **Interactive Streamlit Web App**

- Single text-input interface
- Dynamic match summary cards
- Clean color-coded UI inspired by modern analytics dashboards

### 📁 **Efficient Backend Pipeline**

- Custom utilities for stats normalization, event parsing, and caching
- Modularized `football_agent.py` backend
- Fully integrated LangChain agent with tools

---

## 🧩 Tech Stack

| Layer                     | Tools Used                   |
| ------------------------- | ---------------------------- |
| **Agentic AI**            | LangChain, OpenAI GPT-4o     |
| **Data Source**           | API-Football (APISports)     |
| **Frontend UI**           | Streamlit                    |
| **Backend Data Pipeline** | Python, Requests, Pandas     |
| **Caching**               | Local JSON cache per fixture |
| **Deployment**            | GitHub + Streamlit web app   |

---

## 🏗 Project Structure

project/
│── football_agent.py # Core logic: APIs, stats parser, agent tool
│── app.py # Streamlit frontend UI
│── charts/ # Generated charts (optional)
│── cache/ # Cached API results
│── requirements.txt # Dependencies
│── README.md # Documentation

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add your API keys

Create a .env file:

APISPORTS_FOOTBALL_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here

4️⃣ Run the Streamlit App
streamlit run app.py

🎮 How to Use

Open the Streamlit UI
Type a natural-language query, e.g.:
Give me result and stats between Arsenal vs Tottenham, season 2021

The agent will:
Parse the prompt
Fetch stats
Generate insights
Display the results instantly

🖼 Sample Output
Example Query

“Give me result and stats between Manchester United vs Liverpool, season 2022”

Output

✔ Match results
✔ Shot statistics
✔ Possession comparison
✔ LLM-generated summary
✔ (Optional) visual charts

⭐ Future Enhancements

Player-level insights (xG, passes, defensive actions)
Compare multiple seasons
Add team form analysis
Deploy on Streamlit Cloud

📬 Contact
If you have ideas or want to collaborate, feel free to reach out!
Najmus Sakib
📧 snajmus16@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/najmussakib97/
