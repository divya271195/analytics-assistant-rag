from src.generation.prompt_builder import build_codegen_prompt


def test_prompt_contains_required_sections():
    results = [{"metadata": {"title": "Doc", "source_path": "a", "chunk_id": "1"}, "text": "hello", "score": 0.9}]
    prompt = build_codegen_prompt("Generate SQL", results)
    assert "SQL Code" in prompt
    assert "PySpark Code" in prompt
