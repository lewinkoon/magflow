set windows-shell := ["powershell.exe", "-c"]

run +COMMAND:
    uv run magflow {{COMMAND}}

ui:
    uv run streamlit run magflow/ui/run.py

docs:
    uv run mkdocs serve

update:
    uv lock --upgrade
    uv sync