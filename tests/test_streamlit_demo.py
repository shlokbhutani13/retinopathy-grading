from pathlib import Path


def test_streamlit_demo_keeps_the_research_warning_visible() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert "Research use only" in source
    assert "not clinically validated" in source
    assert "OrdinalRetinopathyPredictor" in source
