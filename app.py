import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from rdflib import Graph, Literal, Namespace, RDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Gemini SDK for enhanced conversational responses
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


st.set_page_config(
    page_title="HeritageMind X — Digital Museum Guidance Engine",
    page_icon="🏛️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Curated Museum Dataset (Artifacts, Galleries & Cultural Knowledge)
# ---------------------------------------------------------------------------
@st.cache_data
def load_museum_data() -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    artifacts = [
        {
            "id": "ART-01",
            "name": "Chola Nataraja Bronze",
            "gallery": "Bronze & Sacred Art",
            "period": "Chola Dynasty (10th–11th Century)",
            "culture": "Tamil / South Indian",
            "origin": "Thanjavur, Tamil Nadu",
            "material": "Lost-Wax Bronze",
            "tradition": "Shaivism & Cosmic Dance",
            "visit_mins": 10,
            "tags": "chola bronze sculpture dance shaivism south india temple",
            "summary": (
                "Cast using the lost-wax (cire perdue) process, depicting Shiva as "
                "Lord of the Dance within a ring of flame symbolizing cosmic creation and renewal."
            ),
        },
        {
            "id": "ART-02",
            "name": "Thanjavur Copper Plate Inscription",
            "gallery": "Epigraphy & Manuscripts",
            "period": "Chola Dynasty (11th Century)",
            "culture": "Tamil / Royal Administration",
            "origin": "Thanjavur, Tamil Nadu",
            "material": "Engraved Copper",
            "tradition": "Temple Grants & Governance",
            "visit_mins": 8,
            "tags": "chola inscription copper temple grant governance tamil history",
            "summary": (
                "An administrative record detailing royal endowments, land rights, and "
                "temple management practices during the height of Chola rule."
            ),
        },
        {
            "id": "ART-03",
            "name": "Pallava Shore Relief Fragment",
            "gallery": "Ancient Stone & Architecture",
            "period": "Pallava Dynasty (7th–8th Century)",
            "culture": "Tamil / Rock-Cut Heritage",
            "origin": "Mamallapuram, Tamil Nadu",
            "material": "Granite",
            "tradition": "Rock-Cut & Structural Temples",
            "visit_mins": 12,
            "tags": "pallava stone sculpture architecture temple mamallapuram ancient",
            "summary": (
                "Carved granite relief illustrating early South Indian narrative sculpture "
                "and architectural transition from rock-cut caves to structural shrines."
            ),
        },
        {
            "id": "ART-04",
            "name": "Terracotta Mother Goddess Figurine",
            "gallery": "Prehistory & Folk Heritage",
            "period": "Early Historic / Folk",
            "culture": "Indigenous / Rural Craft",
            "origin": "Deccan & South India",
            "material": "Fired Terracotta Clay",
            "tradition": "Votive & Domestic Rituals",
            "visit_mins": 7,
            "tags": "terracotta clay folk ritual sculpture ancient craft",
            "summary": (
                "Hand-modeled clay figurine reflecting enduring folk traditions, "
                "protective symbolism, and accessible artisanal craft."
            ),
        },
        {
            "id": "ART-05",
            "name": "Palm-Leaf Sangam Poetry Folio",
            "gallery": "Epigraphy & Manuscripts",
            "period": "Classical / Medieval Transmission",
            "culture": "Tamil Literary Tradition",
            "origin": "Madurai Region, Tamil Nadu",
            "material": "Processed Palmyra Palm Leaf",
            "tradition": "Classical Literature & Scribal Craft",
            "visit_mins": 9,
            "tags": "manuscript palm leaf literature poetry script ancient text",
            "summary": (
                "Incised palm-leaf manuscript preserving poetic landscapes (akam and puram) "
                "and scribal preservation techniques."
            ),
        },
        {
            "id": "ART-06",
            "name": "Temple Pillar Yaksha Carving",
            "gallery": "Ancient Stone & Architecture",
            "period": "Vijayanagara / Nayak Era",
            "culture": "South Indian Temple Art",
            "origin": "Madurai / Hampi Tradition",
            "material": "Carved Granite",
            "tradition": "Mandapa Architecture & Guardianship",
            "visit_mins": 10,
            "tags": "stone pillar carving architecture temple sculpture guardian",
            "summary": (
                "Elaborate stone bracket and pillar relief showcasing monumental temple "
                "architecture and mythological motifs."
            ),
        },
    ]

    # Weighted gallery graph: (Gallery A, Gallery B, walking_minutes)
    gallery_edges = [
        ("Entrance Lobby", "Ancient Stone & Architecture", 3.0),
        ("Entrance Lobby", "Bronze & Sacred Art", 4.0),
        ("Entrance Lobby", "Prehistory & Folk Heritage", 3.5),
        ("Ancient Stone & Architecture", "Bronze & Sacred Art", 2.5),
        ("Ancient Stone & Architecture", "Epigraphy & Manuscripts", 4.0),
        ("Bronze & Sacred Art", "Epigraphy & Manuscripts", 3.0),
        ("Prehistory & Folk Heritage", "Bronze & Sacred Art", 4.5),
        ("Prehistory & Folk Heritage", "Epigraphy & Manuscripts", 5.0),
    ]

    return pd.DataFrame(artifacts), gallery_edges


# ---------------------------------------------------------------------------
# Module 1: Gallery Graph Navigation & Tour Planner
# ---------------------------------------------------------------------------
def build_gallery_graph(edges: List[Tuple[str, str, float]]) -> nx.Graph:
    g = nx.Graph()
    for u, v, weight in edges:
        g.add_edge(u, v, weight=weight)
    return g


def plan_guided_tour(
    graph: nx.Graph,
    df: pd.DataFrame,
    start_gallery: str,
    selected_ids: List[str],
    available_minutes: int,
) -> Dict:
    """Computes shortest gallery paths and builds a time-aware itinerary."""
    if not selected_ids:
        selected_df = df.head(2)
    else:
        selected_df = df[df["id"].isin(selected_ids)].copy()

    current_node = start_gallery
    elapsed = 0.0
    stops = []
    full_path = [start_gallery]
    skipped = []

    for _, row in selected_df.iterrows():
        target_gallery = row["gallery"]
        visit_time = float(row["visit_mins"])

        try:
            path = nx.shortest_path(graph, source=current_node, target=target_gallery, weight="weight")
            walk_time = float(nx.shortest_path_length(graph, source=current_node, target=target_gallery, weight="weight"))
        except nx.NetworkXNoPath:
            skipped.append((row["name"], "No connecting gallery path"))
            continue

        if elapsed + walk_time + visit_time <= available_minutes:
            elapsed += walk_time + visit_time
            if full_path[-1] == path[0]:
                full_path.extend(path[1:])
            else:
                full_path.extend(path)
            stops.append(
                {
                    "Artifact ID": row["id"],
                    "Artifact": row["name"],
                    "Gallery": target_gallery,
                    "Walk (min)": round(walk_time, 1),
                    "Viewing (min)": int(visit_time),
                    "Cumulative Time (min)": round(elapsed, 1),
                }
            )
            current_node = target_gallery
        else:
            skipped.append((row["name"], f"Exceeds {available_minutes} min budget"))

    return {
        "stops": pd.DataFrame(stops),
        "gallery_sequence": full_path,
        "total_minutes": round(elapsed, 1),
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Module 2: RDF Cultural Knowledge Base & Logical Reasoning
# ---------------------------------------------------------------------------
@st.cache_resource
def build_knowledge_graph(df: pd.DataFrame) -> Tuple[Graph, Namespace]:
    kg = Graph()
    HM = Namespace("http://heritagemind.org/ontology#")
    kg.bind("hm", HM)

    for _, row in df.iterrows():
        art_uri = HM[row["id"]]
        kg.add((art_uri, RDF.type, HM.Artifact))
        kg.add((art_uri, HM.title, Literal(row["name"])))
        kg.add((art_uri, HM.belongsToPeriod, Literal(row["period"])))
        kg.add((art_uri, HM.associatedWithCulture, Literal(row["culture"])))
        kg.add((art_uri, HM.originatedIn, Literal(row["origin"])))
        kg.add((art_uri, HM.madeFromMaterial, Literal(row["material"])))
        kg.add((art_uri, HM.relatesToTradition, Literal(row["tradition"])))
        kg.add((art_uri, HM.displayedInGallery, Literal(row["gallery"])))

    # Rule-based logical inference:
    # Rule: If an artifact is associated with "Tamil" culture or "Chola" period,
    # infer thematic connection to "South Indian Heritage Continuum".
    for s, _, obj in list(kg.triples((None, HM.associatedWithCulture, None))):
        if "Tamil" in str(obj) or "South Indian" in str(obj):
            kg.add((s, HM.inferredTheme, Literal("South Indian Heritage Continuum")))

    for s, _, obj in list(kg.triples((None, HM.belongsToPeriod, None))):
        if "Chola" in str(obj) or "Pallava" in str(obj):
            kg.add((s, HM.inferredTheme, Literal("Classical Dynastic Art")))

    return kg, HM


def query_knowledge_base(kg: Graph, HM: Namespace, artifact_id: str) -> List[Dict[str, str]]:
    art_uri = HM[artifact_id]
    rows = []
    predicate_labels = {
        HM.title: "Artifact Title",
        HM.belongsToPeriod: "Belongs to Period",
        HM.associatedWithCulture: "Associated with Culture",
        HM.originatedIn: "Originated In",
        HM.madeFromMaterial: "Made from Material",
        HM.relatesToTradition: "Related Tradition",
        HM.displayedInGallery: "Displayed in Gallery",
        HM.inferredTheme: "Inferred Theme (Rule)",
    }
    for pred, label in predicate_labels.items():
        for _, _, val in kg.triples((art_uri, pred, None)):
            rows.append({"Relation": label, "Value": str(val)})
    return rows


# ---------------------------------------------------------------------------
# Module 3: ML Visitor Interest & Recommendation Engine
# ---------------------------------------------------------------------------
def recommend_artifacts(
    df: pd.DataFrame,
    interest_query: str,
    interactions: Dict[str, int],
    top_k: int = 4,
) -> pd.DataFrame:
    corpus = (
        df["name"] + " " + df["period"] + " " + df["culture"] + " "
        + df["material"] + " " + df["tradition"] + " " + df["tags"] + " " + df["summary"]
    ).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Build user preference vector from text query + interaction-weighted items
    weighted_texts = []
    if interest_query.strip():
        weighted_texts.append(interest_query.strip())

    for art_id, weight in interactions.items():
        if weight > 0:
            match = df[df["id"] == art_id]
            if not match.empty:
                row = match.iloc[0]
                boost_text = f"{row['tags']} {row['period']} {row['material']} {row['tradition']}"
                weighted_texts.extend([boost_text] * int(weight))

    if not weighted_texts:
        weighted_texts.append("chola bronze stone sculpture heritage")

    user_profile_text = " ".join(weighted_texts)
    user_vec = vectorizer.transform([user_profile_text])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    result = df.copy()
    result["Relevance Score"] = np.round(scores, 4)
    result = result.sort_values(by="Relevance Score", ascending=False).head(top_k)
    return result[["id", "name", "gallery", "period", "material", "Relevance Score"]]


def compute_ranking_metrics(recommended_ids: List[str], relevant_ids: List[str], k: int = 3) -> Dict[str, float]:
    top_k_recs = recommended_ids[:k]
    relevant_set = set(relevant_ids)
    if not top_k_recs or not relevant_set:
        return {"Precision@K": 0.0, "Recall@K": 0.0}
    hits = sum(1 for item in top_k_recs if item in relevant_set)
    return {
        "Precision@K": round(hits / len(top_k_recs), 3),
        "Recall@K": round(hits / len(relevant_set), 3),
    }


# ---------------------------------------------------------------------------
# Module 4: Grounded Museum Chatbot
# ---------------------------------------------------------------------------
def build_grounded_context(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"- [{r['id']}] {r['name']} | Gallery: {r['gallery']} | Period: {r['period']} | "
            f"Origin: {r['origin']} | Material: {r['material']} | Tradition: {r['tradition']} | "
            f"Summary: {r['summary']}"
        )
    return "\n".join(lines)


def generate_museum_answer(question: str, df: pd.DataFrame, language: str = "English") -> Tuple[str, str]:
    context = build_grounded_context(df)
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    if HAS_GENAI and api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are HeritageMind X, a helpful museum guide assistant. "
                "Answer ONLY using the curated museum catalog below. If a detail is not in the catalog, "
                "say you don't have that verified information in the museum records.\n"
                f"Respond in {language}.\n\n"
                f"MUSEUM CATALOG:\n{context}\n\n"
                f"VISITOR QUESTION: {question}"
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if resp.text:
                return resp.text, "Gemini (Grounded in Museum KB)"
        except Exception as exc:
            fallback_ans, _ = _local_grounded_answer(question, df)
            return f"{fallback_ans}\n\n*(Note: Live LLM fallback used due to API error: {exc})*", "Local Grounded Retriever"

    ans, src = _local_grounded_answer(question, df)
    return ans, src


def _local_grounded_answer(question: str, df: pd.DataFrame) -> Tuple[str, str]:
    q_lower = question.lower()
    matched_rows = []
    for _, r in df.iterrows():
        haystack = f"{r['name']} {r['period']} {r['culture']} {r['origin']} {r['material']} {r['tradition']} {r['gallery']} {r['summary']}".lower()
        keywords = [w for w in q_lower.replace("?", "").split() if len(w) > 3]
        if any(kw in haystack for kw in keywords):
            matched_rows.append(r)

    if not matched_rows:
        available_names = ", ".join(df["name"].tolist())
        return (
            f"I could not find an exact match in the current museum records for **\"{question}\"**. "
            f"You can ask about these cataloged exhibits: {available_names}."
        ), "Local Grounded Retriever"

    parts = [f"Based on the **HeritageMind X** museum knowledge base, here is what matches your query:\n"]
    for r in matched_rows[:3]:
        parts.append(
            f"* **{r['name']}** (*{r['id']}*) — Located in **{r['gallery']}**, dating to **{r['period']}** "
            f"({r['origin']}), crafted from **{r['material']}** ({r['tradition']}). {r['summary']}"
        )
    return "\n".join(parts), "Local Grounded Retriever"


# ---------------------------------------------------------------------------
# Sidebar: Visitor Profile & Session State
# ---------------------------------------------------------------------------
df_artifacts, gallery_edges = load_museum_data()
gallery_graph = build_gallery_graph(gallery_edges)
kg, HM_NS = build_knowledge_graph(df_artifacts)

if "interactions" not in st.session_state:
    st.session_state.interactions = {aid: 0 for aid in df_artifacts["id"]}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Welcome to **HeritageMind X**! Ask me about our Chola bronzes, Pallava stone carvings, inscriptions, palm-leaf manuscripts, or gallery locations.",
            "source": "System",
        }
    ]

with st.sidebar:
    st.title("🏛️ Visitor Profile")
    visitor_name = st.text_input("Visitor Name", value="Guest Explorer")
    preferred_lang = st.selectbox("Preferred Language", ["English", "Tamil", "Hindi"])
    available_time = st.slider("Available Visit Time (minutes)", min_value=15, max_value=120, value=45, step=5)
    interest_input = st.text_input(
        "Interests / Keywords",
        value="chola bronze sculpture temple architecture",
        help="Used for personalized recommendations and tour prioritization.",
    )

    st.divider()
    st.caption("Recorded Interactions (Views / Saves)")
    for _, row in df_artifacts.iterrows():
        aid = row["id"]
        count = st.session_state.interactions.get(aid, 0)
        st.write(f"• {row['name']}: **{count}**")


# ---------------------------------------------------------------------------
# Main Header & Module Tabs
# ---------------------------------------------------------------------------
st.title("HeritageMind X")
st.subheader("Digital Museum Exploration and Personalized Visitor Guidance Engine")
st.write(f"Hello **{visitor_name}** — explore guided routes, cultural knowledge reasoning, personalized recommendations, and grounded Q&A.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Guide Agent & Navigation",
    "2. Cultural Knowledge & Reasoning",
    "3. ML Recommendations",
    "4. Museum Chatbot",
    "5. Evaluation Metrics",
])


# --- TAB 1: Navigation & Tour Planner ---
with tab1:
    st.header("Intelligent Museum Guide Agent & Gallery Navigation")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        start_gal = st.selectbox(
            "Starting Point",
            options=["Entrance Lobby"] + sorted(df_artifacts["gallery"].unique().tolist()),
        )
        artifact_options = {f"{r['id']} — {r['name']} ({r['gallery']})": r["id"] for _, r in df_artifacts.iterrows()}
        chosen_labels = st.multiselect(
            "Select Exhibits to Visit (Ordered by Priority)",
            options=list(artifact_options.keys()),
            default=list(artifact_options.keys())[:3],
        )
        chosen_ids = [artifact_options[lbl] for lbl in chosen_labels]

    with col_b:
        tour = plan_guided_tour(
            graph=gallery_graph,
            df=df_artifacts,
            start_gallery=start_gal,
            selected_ids=chosen_ids,
            available_minutes=available_time,
        )
        st.metric("Estimated Tour Duration", f"{tour['total_minutes']} / {available_time} min")
        st.write("**Gallery Traversal Sequence:**")
        st.info(" → ".join(tour["gallery_sequence"]))

    if not tour["stops"].empty:
        st.dataframe(tour["stops"], use_container_width=True, hide_index=True)
        for aid in tour["stops"]["Artifact ID"].tolist():
            st.session_state.interactions[aid] = st.session_state.interactions.get(aid, 0) + 1
    else:
        st.warning("No exhibits fit within the selected time budget. Increase available time or choose fewer exhibits.")

    if tour["skipped"]:
        with st.expander("Skipped Exhibits (Time / Path Constraints)"):
            for name, reason in tour["skipped"]:
                st.write(f"- **{name}**: {reason}")


# --- TAB 2: Cultural Knowledge Base & Logical Reasoning ---
with tab2:
    st.header("Cultural Knowledge Representation & Logical Reasoning")
    selected_art_label = st.selectbox(
        "Inspect Artifact in RDF Knowledge Graph",
        options=[f"{r['id']} — {r['name']}" for _, r in df_artifacts.iterrows()],
        key="kb_select",
    )
    sel_id = selected_art_label.split(" — ")[0]
    st.session_state.interactions[sel_id] = st.session_state.interactions.get(sel_id, 0) + 1

    kg_rows = query_knowledge_base(kg, HM_NS, sel_id)
    st.table(pd.DataFrame(kg_rows))

    with st.expander("View RDF Triples (Turtle Format)"):
        st.code(kg.serialize(format="turtle"), language="turtle")


# --- TAB 3: ML Visitor Interest & Recommendations ---
with tab3:
    st.header("ML-Based Visitor Interest Prediction & Personalized Recommendations")
    top_k = st.slider("Number of Recommendations (Top-K)", min_value=1, max_value=len(df_artifacts), value=4)

    recs = recommend_artifacts(
        df=df_artifacts,
        interest_query=interest_input,
        interactions=st.session_state.interactions,
        top_k=top_k,
    )
    st.dataframe(recs, use_container_width=True, hide_index=True)

    st.caption("Tip: Interacting with exhibits in Tabs 1, 2, or 4 boosts your interest profile dynamically.")


# --- TAB 4: Grounded Museum Chatbot ---
with tab4:
    st.header("Conversational Museum Guide (Grounded in Curated Records)")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.caption(f"Source: {msg['source']}")

    user_prompt = st.chat_input("Ask about an artifact, dynasty, material, or gallery...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        reply_text, src_label = generate_museum_answer(user_prompt, df_artifacts, language=preferred_lang)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply_text,
            "source": src_label,
        })
        with st.chat_message("assistant"):
            st.markdown(reply_text)
            st.caption(f"Source: {src_label}")


# --- TAB 5: Testing & Evaluation ---
with tab5:
    st.header("Module Evaluation & Recommendation Metrics")
    st.write("Test recommendation ranking quality against a reference set of relevant artifact IDs.")

    default_relevant = st.multiselect(
        "Ground-Truth Relevant Artifacts for Current Interest",
        options=df_artifacts["id"].tolist(),
        default=["ART-01", "ART-02", "ART-03"],
    )
    current_recs = recommend_artifacts(
        df=df_artifacts,
        interest_query=interest_input,
        interactions=st.session_state.interactions,
        top_k=len(df_artifacts),
    )
    rec_ids = current_recs["id"].tolist()
    k_eval = st.slider("Evaluation Cutoff K", min_value=1, max_value=5, value=3)
    metrics = compute_ranking_metrics(rec_ids, default_relevant, k=k_eval)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Precision@{k_eval}", metrics["Precision@K"])
    c2.metric(f"Recall@{k_eval}", metrics["Recall@K"])
    c3.metric("Total RDF Triples", len(kg))

    st.markdown(
        """
        ### Capstone Evaluation Checklist
        * **Module 1 (Navigation):** Verify valid shortest paths, walking + viewing time within `available_minutes`, and graceful handling of skipped stops.
        * **Module 2 (Knowledge Base):** Verify RDF relations (`belongsToPeriod`, `madeFromMaterial`, `originatedIn`) and rule-inferred themes (`inferredTheme`).
        * **Module 3 (Recommendations):** Track `Precision@K` and `Recall@K` before and after visitor interactions.
        * **Module 4 (Chatbot):** Check factual grounding against the museum catalog and response latency.
        """
    )
