import streamlit as st
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
from sentence_transformers import SentenceTransformer

# Keep a local helper to avoid import issues with hyphenated filename
from neo4j import GraphDatabase

# -------------------------
# Embedding Model (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    # Small, fast, and semantically strong for search
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Neo4j Helper Class
# -------------------------
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]


# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="🎬 Movie Knowledge Graph Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🎞️ Navigation")
page = st.sidebar.radio("Go to", ["📤 Upload Data", "🔎 Explore Graph (Semantic)"])

# -------------------------
# PAGE 1: Upload Data
# -------------------------
if page == "📤 Upload Data":
    st.title("📤 Upload Movie Dataset")
    st.markdown("""
    Upload your **movie dataset CSV** here to prepare for import into Neo4j.

    **Example Columns:**
    - Title  
    - Release Year  
    - Director  
    - Cast  
    - Genre  
    - Wiki Page  
    - Plot
    """)

    uploaded_file = st.file_uploader("Choose your movie CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(10), use_container_width=True)
        st.info("ℹ️ Data upload demo only — not yet connected to Neo4j import logic.")

# -------------------------
# PAGE 2: Explore Graph
# -------------------------
elif page == "🔎 Explore Graph (Semantic)":
    st.title("🌐 Semantic Movie Search over the Knowledge Graph")

    # --- Neo4j Connection Section ---
    with st.expander("⚙️ Neo4j Connection", expanded=True):
        uri = st.text_input("Bolt URI", "neo4j+s://85d0b38b.databases.neo4j.io")
        user = st.text_input("Username", "neo4j")
        password = st.text_input("Password", type="password", value="")

        if st.button("🔌 Connect to Neo4j"):
            try:
                st.session_state.conn = Neo4jConnection(uri, user, password)
                st.success("✅ Connected to Neo4j successfully!")
            except Exception as e:
                st.error(f"❌ Failed to connect: {e}")

    # --- If connected ---
    if "conn" in st.session_state:
        st.divider()
        st.subheader("🧠 Natural Language Search")

        query_text = st.text_input(
            "Describe the movies you're looking for",
            placeholder="e.g., thought-provoking sci-fi about space exploration with a strong emotional story",
        )

        with st.expander("Advanced options"):
            top_k = st.slider("Number of results", min_value=5, max_value=50, value=25, step=5)
            candidate_limit = st.select_slider(
                "Candidate pool size from Neo4j (larger = better recall, slower)",
                options=[100, 200, 300, 500, 800, 1000],
                value=500,
            )

        run_semantic = st.button("🔍 Search Semantically")

        if run_semantic and query_text.strip():
            conn = st.session_state.conn

            # 1) Fetch candidate movies and lightweight metadata from Neo4j
            cypher = """
            MATCH (m:Movie)
            OPTIONAL MATCH (m)-[:OF_GENRE|BELONGS_TO|IN_GENRE|HAS_GENRE]->(g:Genre)
            OPTIONAL MATCH (m)-[:DIRECTED_BY|DIRECTED]->(d:Director)
            WITH m,
                 collect(DISTINCT coalesce(g.name, g.genre, g.title, g.id)) AS genres_g,
                 collect(DISTINCT coalesce(d.name, d.fullName, d.title)) AS directors
            RETURN m.title AS title,
                   coalesce(m.plot, m.description, m.Plot, m.Description, m.overview, m.Summary, "") AS synopsis,
                   genres_g AS genres_g,
                   m.genre AS genreProp,
                   directors AS directors
            LIMIT $limit
            """
            movies = conn.query(cypher, {"limit": int(candidate_limit)})

            if not movies:
                st.warning("No movies available in the graph to search.")
            else:
                # 2) Build texts and compute embeddings
                model = get_embedding_model()

                def combine_genres(m):
                    # Merge relationship genres and movie property genres (string or list)
                    genres = list({g for g in (m.get("genres_g") or []) if g})
                    gp = m.get("genreProp")
                    if isinstance(gp, list):
                        genres.extend([g for g in gp if g])
                    elif isinstance(gp, str) and gp.strip():
                        genres.append(gp.strip())
                    # Deduplicate and clean
                    genres = [str(g).strip() for g in genres if str(g).strip()]
                    # Keep unique order
                    seen = set()
                    uniq = []
                    for g in genres:
                        if g not in seen:
                            uniq.append(g)
                            seen.add(g)
                    return uniq

                def to_text(m):
                    title = m.get("title") or ""
                    genres_list = combine_genres(m)
                    genres = ", ".join(genres_list)
                    directors = ", ".join([d for d in (m.get("directors") or []) if d])
                    synopsis = (m.get("synopsis") or "").strip()
                    parts = [title]
                    if directors:
                        parts.append(f"Directors: {directors}")
                    if genres:
                        parts.append(f"Genres: {genres}")
                    if synopsis:
                        parts.append(synopsis)
                    return ". ".join([p for p in parts if p])

                texts = [to_text(m) for m in movies]
                titles = [m.get("title") for m in movies]

                # Normalize for cosine via dot-product
                movie_embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                query_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]

                sims = movie_embs @ query_emb  # cosine similarity due to normalization
                order = np.argsort(-sims)
                top_idx = order[: int(top_k)]

                top_rows = []
                top_titles = []
                for i in top_idx:
                    row = movies[i]
                    row_score = float(sims[i])
                    top_rows.append(
                        {
                            "Title": row.get("title"),
                            "Directors": ", ".join(row.get("directors") or []),
                            "Genres": ", ".join(combine_genres(row)),
                            "Score": round(row_score, 4),
                        }
                    )
                    top_titles.append(row.get("title"))

                st.success(f"✅ Found {len(top_rows)} semantically similar movie(s)")
                st.dataframe(pd.DataFrame(top_rows), use_container_width=True)

                # 3) Build Interactive Graph for top results
                rel_query = """
                UNWIND $titles AS t
                MATCH (m:Movie {title: t})
                OPTIONAL MATCH (m)-[:DIRECTED_BY|DIRECTED]->(d:Director)
                OPTIONAL MATCH (m)-[:OF_GENRE|BELONGS_TO|IN_GENRE|HAS_GENRE]->(g:Genre)
                RETURN m.title AS Movie,
                       collect(DISTINCT coalesce(d.name, d.fullName, d.title)) AS Directors,
                       collect(DISTINCT coalesce(g.name, g.genre, g.title, g.id)) AS Genres,
                       m.genre AS genreProp
                """
                rels = conn.query(rel_query, {"titles": top_titles})

                net = Network(
                    height="600px",
                    width="100%",
                    bgcolor="#0E1117",
                    font_color="white",
                    notebook=False,
                    directed=True,
                )

                # Add nodes and edges for each top movie
                for rec in rels:
                    movie = rec.get("Movie")
                    net.add_node(movie, label=movie, color="#fd8d3c")

                    for d in rec.get("Directors") or []:
                        if not d:
                            continue
                        net.add_node(d, label=d, color="#6baed6", shape="box")
                        net.add_edge(d, movie, title="DIRECTED_BY")

                    # Merge Genres from relationship and potential property
                    genres_rel = [g for g in (rec.get("Genres") or []) if g]
                    gp = rec.get("genreProp")
                    if isinstance(gp, list):
                        genres_rel.extend([g for g in gp if g])
                    elif isinstance(gp, str) and gp.strip():
                        genres_rel.append(gp.strip())

                    seen_g = set()
                    for g in genres_rel:
                        g = str(g).strip()
                        if not g or g in seen_g:
                            continue
                        seen_g.add(g)
                        net.add_node(g, label=g, color="#74c476", shape="ellipse")
                        net.add_edge(movie, g, title="OF_GENRE")

                # Save to temp HTML file instead of .show()
                tmp_dir = tempfile.gettempdir()
                html_path = os.path.join(tmp_dir, "graph.html")
                net.write_html(html_path, open_browser=False)

                # Render inside Streamlit
                with open(html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=600, scrolling=True)

        st.divider()
        if st.button("🔒 Disconnect"):
            st.session_state.conn.close()
            del st.session_state.conn
            st.info("Disconnected from Neo4j.")
