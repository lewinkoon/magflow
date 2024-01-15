## Update dependencies

Project dependencies can be updated easily with `pur`. First we need to install it from `pip`.

```bash
python -m pip install pur
```

Then to update the `requirements.txt` file just run.

```bash
pur -r requirements.txt
```

From here it's still needed to actually install the packages.

```bash
python -m pip install -r requirements.txt
```