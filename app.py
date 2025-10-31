import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from pyvis.network import Network
import tempfile
import os

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
page = st.sidebar.radio("Go to", ["📤 Upload Data", "🔎 Explore Graph"])

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
elif page == "🔎 Explore Graph":
    st.title("🌐 Explore Movie Knowledge Graph")

    # --- Neo4j Connection Section ---
    with st.expander("⚙️ Neo4j Connection", expanded=True):
        uri = st.text_input("Bolt URI", "neo4j+s://85d0b38b.databases.neo4j.io")
        user = st.text_input("Username", "neo4j")
        password = st.text_input("3qkT1hM62Nngu_nW4ms3jxpwMDloICr598Zb9BotGOU", type="password", value="test")

        if st.button("🔌 Connect to Neo4j"):
            try:
                st.session_state.conn = Neo4jConnection(uri, user, password)
                st.success("✅ Connected to Neo4j successfully!")
            except Exception as e:
                st.error(f"❌ Failed to connect: {e}")

    # --- If connected ---
    if "conn" in st.session_state:
        st.divider()
        st.subheader("🎯 Search Movies by Attributes")

        col1, col2 = st.columns([2, 1])
        with col1:
            search_type = st.selectbox("Search by", ["Director", "Genre"], index=0)
            search_value = st.text_input(
                f"Enter {search_type.lower()} name", "Christopher Nolan" if search_type == "Director" else "Action"
            )

        with col2:
            st.write("")
            run_query = st.button("🔍 Search")

        if run_query:
            conn = st.session_state.conn
            if search_type == "Director":
                query = """
                MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)
                WHERE toLower(d.name) CONTAINS toLower($name)
                RETURN m.title AS Movie, d.name AS Director, m.genre AS Genre
                LIMIT 25
                """
            else:
                query = """
                MATCH (m:Movie)-[:OF_GENRE]->(g:Genre)
                WHERE toLower(g.name) CONTAINS toLower($name)
                RETURN m.title AS Movie, g.name AS Genre, m.director AS Director
                LIMIT 25
                """

            results = conn.query(query, {"name": search_value})

            if not results:
                st.warning("No movies found for your search.")
            else:
                st.success(f"✅ Found {len(results)} movie(s)!")
                st.dataframe(results, use_container_width=True)

                # -------------------------
                # Build Interactive Graph
                # -------------------------
                net = Network(
                    height="600px",
                    width="100%",
                    bgcolor="#0E1117",
                    font_color="white",
                    notebook=False,
                    directed=True,
                )

                # Add nodes and edges
                for record in results:
                    movie = record["Movie"]
                    if search_type == "Director":
                        person = record["Director"]
                        net.add_node(person, label=person, color="#6baed6", shape="box")
                        net.add_node(movie, label=movie, color="#fd8d3c")
                        net.add_edge(person, movie, title="DIRECTED_BY")
                    else:
                        genre = record["Genre"]
                        net.add_node(genre, label=genre, color="#74c476", shape="ellipse")
                        net.add_node(movie, label=movie, color="#fd8d3c")
                        net.add_edge(movie, genre, title="OF_GENRE")

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
