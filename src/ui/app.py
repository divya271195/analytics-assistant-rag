from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.utils.config import load_environment

DEFAULT_INDEX_DIR = "data/index"


@st.cache_resource
def load_pipeline(index_dir: str):
    load_environment()
    from src.pipeline import RagPipeline
    pipeline = RagPipeline()
    pipeline.load_index(index_dir)
    return pipeline


def main() -> None:
    st.set_page_config(page_title="Analytics Assistant", layout="wide")

    st.title("Analytics Assistant")
    st.caption("Ask a question in natural language and get an answer from indexed documents.")

    with st.sidebar:
        st.header("Settings")
        index_dir = st.text_input("Index directory", value=DEFAULT_INDEX_DIR)
        mode = st.radio("Response type", ["Business answer", "Generate code"], index=0)
        show_sources = st.checkbox("Show retrieved sources", value=True)

    query = st.text_area(
        "What's your query?",
        height=150,
        placeholder="Example: What data quality rules are defined for hotel profile ETL?",
    )

    if st.button("Submit", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        if not Path(index_dir).exists():
            st.error(f"Index directory not found: {index_dir}")
            return

        try:
            with st.spinner("Loading pipeline..."):
                pipeline = load_pipeline(index_dir)

            with st.spinner("Generating response..."):
                if mode == "Business answer":
                    answer = pipeline.discovery(query)
                else:
                    answer = pipeline.codegen(query)

            st.subheader("Answer")
            st.write(answer)

            if show_sources:
                st.subheader("Retrieved sources")
                results = pipeline.retrieve(query)
                for i, row in enumerate(results, start=1):
                    metadata = row.get("metadata", {}) or {}
                    score = row.get("score", 0.0)
                    text = row.get("text", "")

                    title = metadata.get("title", "Unknown")
                    path = metadata.get("source_path", metadata.get("path", "Unknown"))
                    chunk_id = metadata.get("chunk_id", metadata.get("chunk_index", "N/A"))

                    with st.expander(f"[{i}] {title} | score={score:.4f}"):
                        st.write(f"**Path:** {path}")
                        st.write(f"**Chunk:** {chunk_id}")
                        st.write(text)

        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()