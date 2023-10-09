import base64
#import cv2
import io
import numpy as np
import re
from datetime import datetime
from PIL import Image

import os
import subprocess
import torch
#from pdf2image import convert_from_path
from unidecode import unidecode
import pandas as pd
import itertools    
os.sys.path
from io import StringIO
from google.cloud import translate_v2 as translate

def is_pdf(pdf_bytes):
    try:
        # Check if the first 4 bytes match the PDF file header
        return pdf_bytes[:4] == b'%PDF'
    except Exception as e:
        print("Error in is_pdf_bytes:", e)
        return False


def gcloud_translate(text, src='ar', dest='en'):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "streamlit-connection-b1a38b694505 (1).json"
    translate_client = translate.Client()
    results = translate_client.translate(text, source_language=src, target_language=dest)
    translated_texts = [result['translatedText'] for result in results]
    return translated_texts
