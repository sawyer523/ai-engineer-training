try:
    from work.tools import load_course_catalog
except Exception:
    from tools import load_course_catalog

def test_course_catalog():
    data = load_course_catalog()
    assert isinstance(data, dict)
    assert "sections" in data and "items" in data
    assert len(data["items"]) >= 1
    first = data["items"][0]
    assert "q" in first
    assert "a" in first