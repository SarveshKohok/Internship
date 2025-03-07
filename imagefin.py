import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.preprocessing import image
import gradio as gr
from PIL import Image

# File path setup
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
cat_dog_file_path = os.path.join(os.getcwd(), 'cats_and_dogs_filtered.zip')

# Create Data directory
os.makedirs('Data', exist_ok=True)
cat_dog_file_path = os.path.join(os.getcwd(), 'Data', 'cats_and_dogs_filtered.zip')

print(f"Dataset path: {cat_dog_file_path}")