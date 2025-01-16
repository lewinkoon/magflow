set windows-shell := ["powershell.exe", "-c"]

run command:
    uv run magflow {{command}}

ui:
    uv run streamlit run magflow/run.py

docs:
    uv run mkdocs serve