# Setup

!!! info "Prerequisites"

    - Python 3.8 or higher
    - Git
    - pip or uv package manager

## Repository Setup

Clone the repository from GitHub:

```shell
git clone https://github.com/lewinkoon/magflow.git
cd magflow
```

## Environment Configuration

Create and activate a Python virtual environment to isolate dependencies:

```shell
python -m venv .venv

# Activate virtual environment:
# On Windows:
.venv\Scripts\activate  
# On Unix or MacOS:
source .venv/bin/activate
```

Install project dependencies using one of the following methods:

```shell
# Using pip (standard approach):
pip install -r requirements.txt

# Using uv (recommended for faster installation):
uv sync
```
## Troubleshooting

For additional support, please open an issue on the GitHub repository.