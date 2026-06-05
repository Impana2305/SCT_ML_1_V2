import sys
import time

print("Starting imports test...", flush=True)

try:
    print("Importing numpy...", flush=True)
    import numpy
    print("numpy imported successfully", flush=True)
except Exception as e:
    print(f"numpy error: {e}", flush=True)

try:
    print("Importing pandas...", flush=True)
    import pandas
    print("pandas imported successfully", flush=True)
except Exception as e:
    print(f"pandas error: {e}", flush=True)

try:
    print("Importing sklearn...", flush=True)
    import sklearn
    print("sklearn imported successfully", flush=True)
except Exception as e:
    print(f"sklearn error: {e}", flush=True)

try:
    print("Importing streamlit...", flush=True)
    import streamlit
    print("streamlit imported successfully", flush=True)
except Exception as e:
    print(f"streamlit error: {e}", flush=True)

try:
    print("Importing plotly...", flush=True)
    import plotly
    print("plotly imported successfully", flush=True)
except Exception as e:
    print(f"plotly error: {e}", flush=True)

print("Finished all imports test.", flush=True)
