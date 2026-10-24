
import sys
from pathlib import Path

import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Allow Python to import our Hybrid RAG module
sys.path.insert(0, str(SRC_DIR))

from hybrid_rag import ask_hybrid_rag

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Heading
st.title("🧠 Hybrid RAG Assistant")

st.caption(
    "Spotify-like Web Application Architecture | "
    "FAISS + Knowledge Graph + OpenAI"
)

st.info(
    "Answers are based on the uploaded architecture PDF. "
    "This document describes a proposed Spotify-like "
    "application, not Spotify's verified production systems."
)

st.divider()

# Question input
question = st.text_input(
    "Ask a question about the architecture",
    placeholder="What technologies does the Backend use?"
)

ask_button = st.button(
    "Generate Answer",
    type="primary"
)

# Run RAG when button is clicked
if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Searching FAISS and Knowledge Graph..."
            ):
                result = ask_hybrid_rag(question)

            st.subheader("AI Answer")
            st.markdown(result["answer"])

            st.divider()

            # Retrieval statistics
            st.subheader("Retrieval Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "FAISS Chunks",
                len(result["vector_results"])
            )

            col2.metric(
                "Graph Relationships",
                len(result["graph_results"])
            )

            col3.metric(
                "Source Pages",
                len(result["pages"])
            )

            st.write(
                "Retrieved PDF pages:",
                result["pages"]
            )

            # Show retrieved evidence
            with st.expander(
                "View FAISS Retrieved Evidence"
            ):
                for item in result["vector_results"]:

                    st.markdown(
                        f"**PDF Page {item['page']}**"
                    )

                    st.write(
                        "Similarity:",
                        round(item["score"], 4)
                    )

                    st.write(item["text"])
                    st.divider()

            # Show graph relationships
            with st.expander(
                "View Knowledge Graph Relationships"
            ):
                if not result["graph_results"]:
                    st.write(
                        "No matching graph relationships."
                    )

                for item in result["graph_results"]:

                    st.markdown(
                        f"**{item['source']}** "
                        f"→ {item['relation']} → "
                        f"**{item['target']}**"
                    )

                    st.caption(
                        f"PDF Page {item['page']}"
                    )

                    st.write(item["evidence"])
                    st.divider()

        except Exception as error:
            st.error(
                "Unable to generate an answer. "
                "Check your API configuration and terminal logs."
            )
            # Display diagnostic details during local development.
            st.exception(error)

st.divider()

st.caption(
    "Hybrid RAG Project | "
    "Python • FAISS • NetworkX • GPT-4o mini • Streamlit"
)
