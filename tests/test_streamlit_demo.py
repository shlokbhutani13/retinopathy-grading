from pathlib import Path


def test_streamlit_demo_keeps_the_research_warning_visible() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert "Research use only" in source
    assert "not clinically validated" in source
    assert "OrdinalRetinopathyPredictor" in source


def test_streamlit_demo_adds_the_src_package_directory_for_hosted_runs() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert 'SRC_DIR = Path(__file__).resolve().parent / "src"' in source
    assert "sys.path.insert(0, str(SRC_DIR))" in source
