#!/bin/bash
python3 -m streamlit run automl_app.py --server.address=0.0.0.0 $@ 2>&1
