import subprocess
from pathlib import Path


def main():
    # print(Path(__file__).parent / "app.py")
    subprocess.run(["streamlit", "run", Path(__file__).parent / "automl_app.py"])
