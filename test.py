import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import requests
import matplotlib.pyplot as plt
import seaborn as sns

print('Done')

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": "say hello",
        "stream": False
    }
)
print(response.json())