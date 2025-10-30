# 🎬 GFlix – AI-Powered Movie Knowledge Graph

<img width="1919" height="912" alt="Screenshot 2025-10-31 013110" src="https://github.com/user-attachments/assets/7fb9fb87-913d-4bff-b347-cf2cc22d21a7" />


> **GFlix** is an AI-driven semantic movie search engine that combines **Neo4j knowledge graphs** with **Large Language Models (LLMs)** to deliver context-aware movie discovery.  
> Instead of relying on plain keyword matches, ReelSense *understands* movie plots, directors, and genres — enabling deeper, relationship-based search.

---

## 🧩 Table of Contents
- [✨ Inspiration](#-inspiration)
- [🧠 What It Does](#-what-it-does)
- [⚙️ How It Works](#️-how-it-works)
- [🧹 Data Refinement Pipeline](#-data-refinement-pipeline)
- [💻 Tech Stack](#-tech-stack)
- [☁️ Deployment](#-deployment)
- [🧩 Architecture](#-architecture)
- [🎯 Future Scope](#-future-scope)
- [🚀 Getting Started](#-getting-started)
- [📽️ Demo](#️-demo)
- [👥 Team](#-team)

---

## ✨ Inspiration

Search engines today treat movies as plain text.  
We wanted to build an engine that *understands* stories — not just keywords.  
By connecting entities like **directors, genres, and actors** in a **Neo4j knowledge graph**, and enriching them with **semantic embeddings**, we aimed to build a smarter, more intuitive way to explore cinema.

---

## 🧠 What It Does

ReelSense lets users:
- 🔍 **Search movies semantically** by concept or plot (“space travel and time dilation” → *Interstellar*)
- 🧩 **Visualize knowledge relationships** between movies, directors, and genres
- 🧠 **Understand thematic similarities** between films via NLP embeddings
- ☁️ **Run entirely in the cloud** using free-tier services

---

## ⚙️ How It Works

1. **Data Cleaning & Enrichment**
   - A raw dataset of 34,000 movies is cleaned using `pandas` and refined using an **LLM** (GPT / local model).
   - Missing plots, genres, and directors are completed intelligently.

2. **Embedding Generation**
   - Plot summaries are embedded using `sentence-transformers (all-MiniLM-L6-v2)` to capture semantic meaning.

3. **Graph Construction**
   - Cleaned and enriched data is stored in **Neo4j AuraDB**, forming a knowledge graph with nodes:
     ```
     (Movie) -[:DIRECTED_BY]-> (Director)
             -[:BELONGS_TO]-> (Genre)
             -[:ACTED_IN]-> (Actor)
     ```

4. **Semantic Search API**
   - A **FastAPI backend** receives user queries, encodes them as vectors, and compares similarity with stored embeddings to find relevant movies.

5. **Visualization**
   - Uses **NetworkX** + **Matplotlib** (and optionally Neo4j Bloom) to visualize relationships and clusters.

---

## 🧹 Data Refinement Pipeline

```mermaid
graph TD
A[Raw Movie CSV (34k rows)] --> B[pandas Cleaning]
B --> C[LLM Enrichment (Fill Missing Data)]
C --> D[SentenceTransformer Embeddings]
D --> E[Neo4j Graph Upload]
E --> F[Semantic Search API + Visualization]
````

---

## 💻 Tech Stack

| Layer               | Tool / Framework             | Purpose                                 |
| ------------------- | ---------------------------- | --------------------------------------- |
| **Database**        | Neo4j AuraDB                 | Graph-based movie knowledge storage     |
| **LLM Integration** | GPT / Mistral / Ollama       | Data enrichment & refinement            |
| **NLP Model**       | Sentence Transformers        | Semantic embeddings of plot text        |
| **Backend**         | FastAPI                      | REST API for search & recommendations   |
| **Visualization**   | NetworkX, Matplotlib         | Graph and relationship display          |
| **Frontend / UI**   | Streamlit / Gradio           | Interactive semantic search demo        |
| **Deployment**      | Hugging Face Spaces / Render | Free cloud deployment of backend and UI |

---

## ☁️ Deployment

* **Neo4j AuraDB (Free Tier)** hosts the knowledge graph
* **FastAPI Semantic Search Service** is deployed on Render / Hugging Face
* Optional **Streamlit UI** lets users run semantic queries live

### Example Query:

> “Find movies similar to *Inception* with time-travel or dream sequences.”

---

## 🧩 Architecture

```mermaid
graph LR
U[User] -->|Query| A[FastAPI / Streamlit App]
A -->|Bolt URI| B[Neo4j AuraDB]
A -->|Embeddings| C[SentenceTransformer]
B -->|Results| A
A -->|Response| U
```

---

## 🎯 Future Scope

* 🧬 Integrate **LLM-based question answering** directly on graph (RAG + Cypher)
* 🎞️ Add trailer and metadata retrieval via IMDb / TMDB APIs
* 🗣️ Implement **voice-based semantic search**
* 📊 Enhance visualization with interactive front-end graph explorer

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/reelsense.git
cd reelsense
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Environment Variables

```
NEO4J_URI=bolt+s://<your-db-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_api_key
```

### 4️⃣ Run Locally

```bash
python app.py
```

### 5️⃣ Access App

Go to `http://localhost:8000` or Hugging Face Space link.

---

## 📽️ Demo

🎥 **Live Demo:** [https://huggingface.co/spaces/yourusername/reelsense](#)
📸 **Preview Screenshot:**
*(Insert an image or short gif of your app here)*

---

## 👥 Team

| **Caleb Chandrasekar** | 
| **Devraj Singh**       | 
| **Vedant Srivastava**  | 
| **Ankita Gupta**       | 

---

## 🏁 Hackathon Highlights

* 🧩 Combines **graph intelligence** and **semantic embeddings**
* 🧠 Uses **LLM for intelligent data repair and enrichment**
* ☁️ Fully **deployable using free-tier cloud tools**
* 🔍 Enables **meaningful, story-level movie search**

---

> “From keywords to context — GFlix redefines how we explore cinema.”

```
