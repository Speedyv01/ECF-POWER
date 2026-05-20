import importlib
import sys
import types
from pathlib import Path

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def make_dummy_streamlit():
    st = types.ModuleType("streamlit")

    class DummySessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    st.session_state = DummySessionState()

    def no_op(*args, **kwargs):
        return None

    st.title = no_op
    st.subheader = no_op
    st.space = no_op
    st.dataframe = no_op
    st.success = no_op
    st.error = no_op
    st.write = no_op
    st.selectbox = lambda *args, **kwargs: args[1][0] if len(args) > 1 else None
    st.slider = lambda *args, **kwargs: kwargs.get("value", 0)
    st.number_input = lambda *args, **kwargs: kwargs.get("value", 0.0)
    st.text_input = lambda *args, **kwargs: kwargs.get("value", "")
    st.button = lambda *args, **kwargs: False

    return st


def test_streamlit_app_imports_without_error(monkeypatch):
    dummy_streamlit = make_dummy_streamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_streamlit)

    module = importlib.import_module("api.frontend.streamlit_app")

    assert hasattr(module, "zipcode")
    assert module.zipcode == "75014"
    assert hasattr(module, "api_key")
    assert module.api_key == ""
    assert hasattr(module, "conso_annuelle")
    assert module.conso_annuelle == 0.0
