set windows-shell := ["powershell.exe", "-c"]

run +COMMAND:
    uv run magflow {{COMMAND}}

ui:
    uv run streamlit run magflow/run.py

docs:
    uv run mkdocs serve