// Configure justfile to use PowerShell on Windows
set windows-shell := ["powershell.exe", "-c"]

// Generate visualizations using magflow
build:
    uv run magflow visualize

// Extract data using magflow
extract:
    uv run magflow extract

// Update all dependencies to their latest versions
update:
    uv lock --upgrade
    uv sync

// Run any magflow command with arguments
run +COMMAND:
    uv run magflow {{COMMAND}}