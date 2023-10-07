import streamlit as st
import pandas as pd
from google.cloud import vision
import re
from datetime import datetime
from hijri_converter import convert
from googletrans import Translator
import pycountry
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "streamlit-connection-b1a38b694505 (1).json"

def display_details_in_table(details, id_number):
    df = pd.DataFrame(details, index=[f"ID{id_number}"])
    st.table(df)

def detect_text(uploaded_id_content):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=uploaded_id_content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    return texts[0].description.split('\n')

def get_country_code(name_in_foreign_language):
    translator = Translator()
    translated_country_name = translator.translate(name_in_foreign_language, src='auto', dest='en').text
    
    try:
        country = pycountry.countries.get(name=translated_country_name)
        return country.alpha_3
    except AttributeError:
        return "Country Not Found"
    
def eastern_arabic_to_english(eastern_numeral):
    arabic_to_english_map = {
        '٠': '0', '۰': '0',
        '١': '1', '۱': '1',
        '٢': '2', '۲': '2',
        '٣': '3', '۳': '3',
        '٤': '4', '۴': '4',
        '٥': '5', '۵': '5',
        '٦': '6', '۶': '6',
        '٧': '7', '۷': '7',
        '٨': '8', '۸': '8',
        '٩': '9', '۹': '9',
        '/': '/'
    }

    # If the character is an Eastern Arabic numeral, convert it to English; otherwise, keep it unchanged.
    english_numeral = ''.join([arabic_to_english_map[char] if char in arabic_to_english_map else char for char in eastern_numeral])
    
    return english_numeral

def distinguish_dates(date_list):
    today = datetime.now().date()
    
    # Calculate the difference between each date and today's date
    differences = [(abs((today - datetime.strptime(date, '%Y/%m/%d').date()).days), date) for date in date_list]
    
    # Sort by difference
    differences.sort(key=lambda x: x[0])
    
    # The date with the smallest difference is considered Gregorian, and the one with the largest difference is considered Hijri
    gregorian_date = differences[0][1]
    hijri_date = differences[-1][1]

    return hijri_date, gregorian_date

def hijri_to_gregorian(hijri_date):
    # Split the hijri date
    year, month, day = map(int, hijri_date.split('/'))
    
    # Convert the hijri date to Gregorian
    gregorian_date = convert.Hijri(year, month, day).to_gregorian()
    
    # Format the result as a string
    return f"{gregorian_date.year}/{gregorian_date.month:02}/{gregorian_date.day:02}"

def detect_script(word):
    arabic_chars = range(0x0600, 0x06FF)  # Arabic Unicode Block
    english_chars = range(0x0041, 0x007A)  # English uppercase Unicode Block
    english_chars_lower = range(0x0061, 0x007A)  # English lowercase Unicode Block
    
    has_arabic = any(ord(char) in arabic_chars for char in word)
    has_english = any(ord(char) in english_chars or ord(char) in english_chars_lower for char in word)
    
    if has_arabic and has_english:
        return "Mixed"
    elif has_arabic:
        return "Arabic"
    elif has_english:
        return "English"
    else:
        return "Other"
    
def extract_id_details(uploaded_id):
    result = detect_text(uploaded_id)

    pattern = r'(\d{4}/\d{1,2}/\d{1,2}|[۰-۹]{4}/[۰-۹]{1,2}/[۰-۹]{1,2})'

    try:
        nationality=[ele for ele in [ele for ele in result if 'الجنسية' in ele ][0].split('الجنسية') if ele!=''][0].strip()

        nationality=get_country_code(nationality)
    except:
        nationality=''
    try:
        ## employer
        employer_ar=[ele for ele in [ele for ele in result if 'صاحب العمل' in ele ]][0]
        employer=[ele for ele in employer_ar.split('صاحب العمل') if ele!=''][0].strip()

    except:
        employer=''
    try:
        ### issuing place
        issuing_place_ar=[ele for ele in result if 'مكان الإصدار' in ele][0]
        issuing_place=issuing_place_ar.split('مكان الإصدار')[-1].strip()
    except:
        issuing_place=''
    try:
        comon_pattern=[ele for ele in [ele for ele in result if (('الإصدار' in ele ) and('مكان' not in ele))][0].split('الإصدار') if ele!=''][0].strip()
        matches = re.findall(pattern, comon_pattern)

        matches=[eastern_arabic_to_english(ele) for ele in matches]

        issuing_date, dob=distinguish_dates(matches)

        issuing_date = hijri_to_gregorian(issuing_date)

    except:
        issuing_date,dob='',''

    try: 

        id_number=[item for item in result if re.fullmatch(r'\d{10}', item)][0]

    except:
        id_number=''

    try:
        profession_Ar=[ele for ele in [ele for ele in result if 'المهنة' in ele ]][0]

        profession=[ele for ele in profession_Ar.split('المهنة') if ele!=''][-1]

    except:
        profession=''
    try:
        Name_Index=[result.index(ele) for ele in result if 'وزارة' in ele][0]
        Name_1=result[Name_Index+1]
        Name_length=len(Name_1.split(' '))
        Name_2=[ele for ele in result if (len(ele.split(' '))==Name_length) and (ele!=Name_1)][0]


        Name_en=[ele for ele in [Name_1,Name_2] if detect_script(ele)=='English'][0]
        Name_ar=[ele for ele in [Name_1,Name_2] if ele!=Name_en][0]  

    except:

        Name_en,Name_ar='',''

    return {"Name": Name_ar, "DOB": dob, "ID Number": id_number}


