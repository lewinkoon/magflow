set windows-shell := ["powershell.exe", "-c"]

run:
    uv run streamlit run hemoflow/run.py

docs:
    uv run mkdocs serve