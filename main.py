import streamlit as st
# from PyPDF2 import PdfReader
import requests
from google.cloud import vision
from google.cloud import translate_v2 as translate
from googletrans import Translator
import os
import io
import fitz
from PIL import Image
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from constants import WEB_ANALYSIS_DATA, THM_RESPONSE, WEB_AVERAGE_DURATION_DATA
import subprocess
import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
# from textblob import TextBlob
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time
from id_upload import extract_id_details
# from wordcloud import WordCloud
# import numpy as np

# Define fixed credentials
USERNAME = "demo"
PASSWORD = "demo"

# Define a function to check credentials
def check_credentials(username, password):
    return username == USERNAME and password == PASSWORD

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "streamlit-connection-b1a38b694505 (1).json"

def extract_text_from_pdf(pdf_file):
    client = vision.ImageAnnotatorClient()

    try:
        # Read the PDF file
        pdf_content = pdf_file.read()

        # Convert the PDF to a valid image format (e.g., PNG)
        pdf_document = fitz.open(stream=io.BytesIO(pdf_content))
        all_text = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()

            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Save the PIL Image as bytes (PNG)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                image_bytes = output.getvalue()

            # Use Google Cloud Vision to extract text from the image
            image = vision.Image(content=image_bytes)
            response = client.text_detection(image=image)
            text_annotations = response.text_annotations

            if text_annotations:
                extracted_text = text_annotations[0].description
                all_text.append(extracted_text)

        return "\n".join(all_text)
    
    except Exception as e:
        print(e)
        return None

# # def extract_text_from_pdf(uploaded_file):
#     pdf_text = ""
#     if uploaded_file:
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             pdf_text += page.extract_text()
#     return pdf_text

# def extract_text_from_pdf(uploaded_file):
#     pdf_document = PdfReader(uploaded_file)
#     plain_text_data = []
#     for page in pdf_document.pages:
#         page_text = page.extract_text()
#         # Split text into blocks based on newline
#         page_text_blocks = page_text.split('\n')
#         plain_text_data.append(page_text_blocks)  # Append the list of text blocks to the main list
#     print(f"data: {plain_text_data}")
#     return plain_text_data

def translate_arabic_to_english(arabic_text):
    try:
        translator = Translator()
        translated_text = translator.translate(arabic_text, src='ar', dest='en').text
        return translated_text
    
    except Exception as e:
        print(e)

def twitter_scrape(business_name):
    # command1 = f"twscrape add_accounts accounts.txt username:password:email:email_password"
    # command2 = f"twscrape login_accounts"
    subprocess.call(['twscrape', 'add_accounts', 'accounts.txt', 'username:password:email:email_password'])
    subprocess.call(['twscrape', 'login_accounts'])

    # subprocess.run(command1)
    # subprocess.run(command2)

    # command = f'twscrape search "{business_name} since:2023-08-01 until:2023-08-29" --raw'
    command = f'twscrape search "nymcard since:2023-08-01 until:2023-08-29" --raw'
    print(f"running command: {command}")
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )
        print(result)
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

        if result.returncode != 0:
            st.error(f"Error: {result.stderr}")
        else:
            try:
                response_json = json.loads(result.stdout)
                st.write(response_json)
                tweet_list = []

                for entry in response_json.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {}).get("instructions", []):
                    if "entries" in entry:
                        for tweet_entry in entry["entries"]:
                            if "content" in tweet_entry and "itemContent" in tweet_entry["content"]:
                                tweet_data = tweet_entry["content"]["itemContent"]["tweet_results"]["result"]
                                full_text = tweet_data.get("legacy").get("full_text")
                                if full_text:
                                    tweet_list.append(full_text)
                return tweet_list

            except json.JSONDecodeError as e:
                st.error(f"No tweets found")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def instagram_scrape(business_name):
    api_url = "https://www.page2api.com/api/v1/scrape"
    payload = {
        "api_key": "e44b6cbf3c70e25bbfb856aa0c19560e09fcf85b",
        "url": f"https://www.instagram.com/explore/tags/{business_name}/",
        "parse": {
        "posts": [
            {
            "_parent": "article a[role=link]",
            "image": "img >> src",
            "url": "_parent >> href",
            "title": "img >> alt"
            }
        ]
        },
        "premium_proxy": "us",
        "real_browser": True,
        "scenario": [
        { "wait_for": "article a[role=link]" },
        {
            "loop": [
            { "execute_js": "document.querySelector('#scrollview + div')?.remove();" },
            { "execute_js": "document.querySelectorAll('article a[role=link] img').forEach(e => e.scrollIntoView({behavior: 'smooth'}))" },
            { "wait": 1 }
            ],
            "iterations": 2
        },
        { "execute": "parse" }
        ]
    }

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(api_url, data=json.dumps(payload), headers=headers)
    result = json.loads(response.text)
    ig_posts = []

    for post in result['result']['posts']:
        ig_posts.append(post['title'])
    
    return ig_posts

def google_reviews_scrape(business_name):
    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    params = {
        "query": business_name,
        "key": "AIzaSyC3jF85Z6qgAEBwqwCdP8j_YM_XQcvEH-s",
    }

    try:
        response = requests.get(endpoint, params=params)
        data = response.json()

        if response.status_code == 200 and data.get("status") == "OK":
            place_id = data["results"][0]["place_id"]

            reviews_endpoint = "https://maps.googleapis.com/maps/api/place/details/json"
            reviews_params = {
                "place_id": place_id,
                "fields": "reviews",
                "key": "AIzaSyC3jF85Z6qgAEBwqwCdP8j_YM_XQcvEH-s",
            }

            reviews_response = requests.get(reviews_endpoint, params=reviews_params)
            reviews_data = reviews_response.json()

            reviews = reviews_data.get("result", {}).get("reviews", [])
            extracted_reviews_list = []

            for item in reviews:
                extracted_reviews_list.append(item['text'])

            return extracted_reviews_list
        else:
            return []

    except Exception as e:
        return []

def analyze_sentiment_bert(text):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    prob = torch.softmax(logits, dim=1)[0]
    sentiment = "Positive" if prob[1] > 0.5 else "Negative"
    return sentiment, float(prob[1])

# def generate_word_cloud(texts):
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate(texts)
#     wordcloud_data = px.imshow(np.array(wordcloud), color_continuous_scale='gray', binary_string=True)
#     wordcloud_data.update_layout(
#         title="Word Cloud",
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         margin=dict(l=0, r=0, b=0, t=0),
#     )
#     st.plotly_chart(wordcloud_data)

def eastern_arabic_to_english(eastern_numeral):
    arabic_to_english_map = {
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
        '/': '/'
    }
    english_numeral = ''.join([arabic_to_english_map[char] for char in eastern_numeral if char in arabic_to_english_map])
    return english_numeral

def gcloud_translate(text, src='ar', dest='en'):
    translate_client = translate.Client()
    result = translate_client.translate(text, source_language=src, target_language=dest)
    return result['translatedText']

# Login Page
def login_page():
    st.title("Login Page")

    # Use st.empty() to create placeholders for input fields and messages
    username_placeholder = st.empty()
    password_placeholder = st.empty()
    login_status = st.empty()

    username = username_placeholder.text_input("Username:")
    password = password_placeholder.text_input("Password:", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_credentials(username, password):
            # Display a success message and set the session state variable
            login_status.success("Login successful! Proceed to the data entry screen.")
            st.session_state.next_page = "cr_entry"
            st.session_state.login = True
            
        else:
            # Display an error message
            login_status.error("Login failed. Please check your credentials.")

def get_response_from_wathq(cr_number):
    url = f"https://api.wathq.sa/v5/commercialregistration/fullinfo/{cr_number}"
    headers = {
        'accept': 'application/json',
        'apiKey': 'hJVHxlkXtTD0SQlyVKqs67f4wvLYhIv1'
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        wathq_parsed_response = {
            "cr_number": data["crNumber"],
            "business_name": data["crName"],
            "cr_expiry_date": data["expiryDate"],
            "business_owner_1": data["parties"][1]["name"],
            "owners_iaqama_id": data["parties"][2]["identity"]["id"],
            "address": data["address"]["general"]["address"]
        }
        return wathq_parsed_response
    except Exception as e:
        print(e)
        return {}

def extract_business_details_and_ping_wathq(data, cr_number=False):
    if not cr_number:
        cr_number = data.get('cr_number')
    else:
        cr_number = data

    resp = get_response_from_wathq(cr_number)
    return resp

def smart_ocr_on_cr_doc(pdf_text1, pdf_text2):
    fields = {
        'cr_number': '',
        'business_name': '',
        'business_address': '',
        'business_owner_1': '',
        'business_owner_2': ''
    }

    cr_pattern = r'Commercial Registry:\s+(\d+)'
    business_name_pattern = r"(?<=company's (?:trade name|brand name) )([^C]+) Company"
    # business_name_pattern = r"(?<=trade name of the company is)([^C]+) Company"
    pattern = r'The trade name of the company is\s(.*?)\s\d{2}/\d{2}/\d{4}'
    business_owner_1_pattern = r'Managers([^0-9]+)'
    business_owner_2_pattern = r''
    expiry_date_pattern = r'certificate expires on (\d{2}/\d{2}/\d{4}).*?'
    location_pattern = r'company Head office:\s(.*?)\sP\.O\. Box'

    if pdf_text1 and pdf_text2:
        cr_match = re.search(cr_pattern, pdf_text1, re.IGNORECASE)
        business_name_match = re.search(f'{business_name_pattern}', pdf_text1)
        business_owner_1_match = re.search(business_owner_1_pattern, pdf_text2)
        expiry_date_match = re.search(expiry_date_pattern, pdf_text2)
        location_match = re.search(location_pattern, pdf_text2)

        if cr_match:
            cr_number = cr_match.group(1)
            fields['cr_number'] = cr_number
        
        if business_name_match:
            business_name = business_name_match.group(1) if business_name_match.group(1) else business_name_match.group(2)
            # business_name = business_name_match.group(1)
            fields['business_name'] = business_name
        
        if business_owner_1_match:
            business_owner_1 = business_owner_1_match.group(1)
            fields['business_owner_1'] = business_owner_1
        
        if expiry_date_match:
            hijri_date = expiry_date_match.group(1)
            print(f"date: {hijri_date}")
            fields['expiry_date_hijri'] = hijri_date
        
        if location_match:
            location = location_match.group(1)
            fields['location'] = location

    return fields

# C/R Entry Page
def cr_entry_page():
    st.title("C/R - Verification")
    pdf_file = st.file_uploader("Upload a C/R Document:")
    submit_button = st.button("Submit")

    if submit_button:
        if not pdf_file:
            st.error("Please upload a PDF document before submitting.")
        else:
            with st.spinner("Fetching data..."): 
                pdf_text = extract_text_from_pdf(pdf_file)
                # st.write(pdf_text)
                translated_pdf_text1 = translate_arabic_to_english(pdf_text)
                translated_pdf_text2 = gcloud_translate(pdf_text)
                # st.write(translated_pdf_text2)

                ocr_result = smart_ocr_on_cr_doc(translated_pdf_text1, translated_pdf_text2)
                cr_number = ocr_result.get('cr_number')
                business_name = ocr_result.get('business_name')
                business_owner_1 = ocr_result.get('business_owner_1')
                expiry_date_hijri = ocr_result.get('expiry_date_hijri')
                location = ocr_result.get('location')

            non_optional_keys = ["cr_number", "business_name", "business_owner_1"]
            empty_string_keys = [key for key, value in ocr_result.items() if key in non_optional_keys and value == '']

            if empty_string_keys:
                st.error("Please upload a Valid CR Document PDF")
            else:
                st.title("Business Details:")
                st.success(f"CR Number: {cr_number} ✅")
                st.success(f"Business Name: {business_name} ✅")
                st.success(f"Business Owner: {business_owner_1} ✅")
                st.success(f"CR Expiry Date: {expiry_date_hijri} ✅")
                st.success(f"Business Address: {location} ✅")

                # cr_number = ocr_result['cr_number']
                # result = extract_business_details_and_ping_wathq(ocr_result)

                result = {}

                st.write(result)

                st.session_state.next_button_enabled = True
                st.session_state.next_page = "idv_page"

            if st.session_state.get("next_button_enabled"):
                if st.button("Next"):
                    st.session_state.idv_response = True
            # st.session_state.next_page = "similar_web_page"

            # Add a link to go back to the previous page
            # st.markdown("Go back to [C/R Entry Page](?previous_page=cr_entry)")

def display_details_in_table(details, id_number):
    df = pd.DataFrame(details, index=[f"ID{id_number}"])
    st.table(df)

def idv_page():
    st.title("Business Shareholders Verification")
    uploaded_ids = []

    st.header("Upload IDs")
    num_ids_to_upload = st.number_input("How many IDs do you want to upload? (1-3)", 1, 3, 1)

    for i in range(num_ids_to_upload):
        uploaded_id = st.file_uploader(f"Upload National ID {i+1}", type=["jpg", "png"])
        if uploaded_id:
            uploaded_id_content = uploaded_id.read()
            with st.spinner("Fetching data..."):
                extracted_details = extract_id_details(uploaded_id_content)
                uploaded_ids.append(extracted_details)

    st.session_state.next_button_enabled = len(uploaded_ids) == num_ids_to_upload
    if st.session_state.get("next_button_enabled"):
        if st.button("Next"):
            # Set the next page to "similar_web_page" (you can change this as needed)
            st.session_state.next_page = "similar_web_page"
            st.session_state.idv_response = True

    with st.expander("Extracted Details"):
        for i, id_data in enumerate(uploaded_ids, start=1):
            st.subheader(f"Details from ID {i}")
            display_details_in_table(id_data, i)
            
def verify_address(address):
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    api_key = "AIzaSyC3jF85Z6qgAEBwqwCdP8j_YM_XQcvEH-s"

    params = {
        "address": address,
        "key": api_key,
    }

    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        if response.status_code == 200:
            if data['status'] == 'OK':
                return data
            else:
                return False
        else:
            return False
    
    except Exception as e:
        return False

def check_address_length(address):
    split_by_space = address.split()

    result = [item.split(',') for item in split_by_space]

    return len(result[0]) > 5

def get_google_search_results(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return None

def get_company_address(company_name):
    company_name = company_name.lower().strip()
    print(f"company: {company_name}")
    search_query1 = f"{company_name} saudi address"
    search_query2 = f"{company_name} dubai address"
    search_results1 = get_google_search_results(search_query1)
    print(f"len 1: {len(search_results1)}")
    search_results2 = get_google_search_results(search_query2)
    print(f"len 2: {len(search_results2)}")

    if search_results1:
        soup = BeautifulSoup(search_results1, "html.parser")
        address = soup.find("div", {"class": "sXLaOe"})
        if address:
            address = address.get_text()
            print(f"addr1: {address}")
            return address
        else:
            soup = BeautifulSoup(search_results2, "html.parser")
            address = soup.find("div", {"class": "sXLaOe"}).get_text()
            print(f"addr2: {address}")
            return address    
    else:
        try:
            soup = BeautifulSoup(search_results2, "html.parser")
            address = soup.find("div", {"class": "sXLaOe"}).get_text()
            print(f"addr2: {address}")
            return address
        except AttributeError:
            return None

def fuzzy_match_fields(field1, field2, threshold=70):
    similarity = fuzz.partial_ratio(field1, field2)
    print(similarity)
    return similarity >= threshold

def check_address_from_google(company_name, company_address):
    google_address = get_company_address(company_name)
    print(f"address: {google_address}")
    if google_address:
        if fuzzy_match_fields(company_address.lower(), google_address.lower()):
            return True
        else:
            return False

def address_verification_page():
    st.title("Address Verification")

    address = st.text_input("Enter address of the business")

    if st.button("Submit"):
        if address or check_address_length(address):
            is_address_verified = False
            is_address_valid = False

            data = verify_address(address)
            if data:
                if hasattr(st.session_state, 'company_name'):
                    company_name = st.session_state.company_name
                    company_name = company_name.lower()
                    with st.spinner(f"checking {company_name}'s address on google"):
                        google_address_result = check_address_from_google(company_name, address)
                        is_address_valid = google_address_result

                country_name = data['results'][0]['address_components'][-1]['short_name']
                if country_name == 'SA' or country_name == 'AE':
                    is_address_verified = True
        
            if is_address_verified and is_address_valid:
                st.success("Address verification   ✅")
                st.success("Address validation     ✅")

                st.session_state.next_button_enabled = True
                st.session_state.next_page = "sentiment_scrape"

                if st.button("Next"):
                    st.session_state.address_verification = True
            else:
                st.error("Address verification Failed    ❌")
                st.error("Address validation Failed      ❌")
                st.write("Please Re-enter full address of the Company")
        else:
            st.error("Please enter complete Address before submitting")   

def verify_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_company_name(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')

            if title_tag:
                company_name = title_tag.text.strip()
                return True, company_name

        else:
            pattern = r'https://|www\.|\.com'
            cleaned_url = re.sub(pattern, '', url)
            return False, cleaned_url
    
    except Exception as e:
        return False, False

def generate_fake_similar_web_data():
    traffic_data = WEB_ANALYSIS_DATA["traffic"]
    traffic_dates = [entry["date"] for entry in traffic_data]
    avg_traffic = [entry["traffic"] for entry in traffic_data]

    duration_data = WEB_AVERAGE_DURATION_DATA["duration"]
    duration_dates = [entry["date"] for entry in duration_data]
    avg_duration = [entry["average_visit_duration"] for entry in duration_data]

    # st.set_page_config(layout="wide")
    col1, col2 = st.columns(2)

    with col1:
        # Create the first chart for average traffic month-wise
        fig1 = px.bar(
            x=traffic_dates,
            y=avg_traffic,
            labels={'x':'Date', 'y':'Average Traffic'},
            title='Average Traffic Month-wise',
        )

        fig1.update_layout(
            xaxis=dict(showgrid=False, gridwidth=1, title_font=dict(size=12)),
            yaxis=dict(showgrid=False, gridwidth=1, range=[500, 1500], title_font=dict(size=12)),
            plot_bgcolor='white',
            font=dict(size=12),
            width=340,
            height=450,
            margin=dict(r=10),
        )
        fig1.update_traces(marker_color='rgba(255, 0, 0, 0.5)')

        st.plotly_chart(fig1)

    with col2:
        # Create the second chart for average visit duration
        fig2 = px.bar(
            x=duration_dates,
            y=avg_duration,
            labels={'x':'Date', 'y':'Average Visit Duration'},
            title='Average Visit Duration Month-wise',
        )

        fig2.update_layout(
            xaxis=dict(showgrid=False, gridwidth=1, title_font=dict(size=12)),
            yaxis=dict(showgrid=False, gridwidth=1, range=[0, 1000], title_font=dict(size=12)),
            plot_bgcolor='white',
            font=dict(size=12),
            width=340,
            height=450,
        )
        fig2.update_traces(marker_color='rgba(0, 0, 255, 0.5)')  # The last value (0.5) controls transparency
        
        st.plotly_chart(fig2)

def similar_web_page():
    st.title("Web Traffic Analysis")
    company_url = st.text_input("Enter Business Url: ")
    submit_button = st.button("Submit")
    if company_url and submit_button:
        if verify_url(company_url):

            with st.spinner("Checking url..."):
                time.sleep(1)

            status, company_name = extract_company_name(company_url)
            if status:
                with st.spinner("Fetching Results..."):
                    st.session_state.company_url = company_url
                    st.session_state.company_name = company_name

                    api_key = '8522d7f391074950a906f0ee4c77268d'

                    api_url = f'https://api.similarweb.com/v5/website/{company_url}/total-traffic-and-engagement/visits?api_key={api_key}&start_date=2023-01&end_date=2023-08&country=sa&granularity=monthly&main_domain_only=false&format=json&show_verified=false&mtd=false&engaged_only=false'

                    try:
                        response = requests.get(api_url)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.write(data)
                        else:
                            generate_fake_similar_web_data()

                        st.session_state.next_button_enabled = True
                        st.session_state.next_page = "address_verification_page"

                        if st.button("Next"):
                            st.session_state.similar_web = True

                    except Exception as e:
                        generate_fake_similar_web_data()
            else:
                st.error("Please enter a valid url")
        else:
            st.error("Please enter complete valid url of the company")
            

# def sentiment_scrape():
#     st.title("Sentiment Scrape Results")
#     if st.button("Get Analysis"):
#         if hasattr(st.session_state, 'company_url'):
#             company_url = st.session_state.company_url
            
#         if hasattr(st.session_state, 'company_name'):
#             company_name = st.session_state.company_name

#         tweet_list = twitter_scrape(company_name)
#         ig_post_list = []
#         # ig_post_list = instagram_scrape(company_name)
#         google_reviews_list = google_reviews_scrape(company_name)

#         max_len = max(len(tweet_list), len(ig_post_list), len(google_reviews_list))
#         if max_len > 3:
#             max_len = 3
        
#         tweet_list.extend([None] * (max_len - len(tweet_list)))
#         ig_post_list.extend([None] * (max_len - len(ig_post_list)))
#         google_reviews_list.extend([None] * (max_len - len(google_reviews_list)))

#         st.write(f"Extracted Results for {company_name}")

#         if st.button("View Extracted Data"):
#             df = pd.DataFrame({'Google Reviews': google_reviews_list, 'Tweets': tweet_list, 'Instagram': ig_post_list})
#             st.dataframe(df)
#         # st.write(tweet_list)

#         all_tweet_texts = " ".join(tweet_list)
#         sentiment, prob1 = analyze_sentiment_bert(all_tweet_texts)
#         st.write("Sentiment Distribution:")

#         all_ig_texts = " ".join(ig_post_list)
#         sentiment, prob2 = analyze_sentiment_bert(all_ig_texts)

#         all_gr_texts = " ".join(google_reviews_list)
#         sentiment, prob3 = analyze_sentiment_bert(all_gr_texts)
        
#         labels = ["Positive", "Negative"]
#         values = [prob1, 1 - prob1]
#         fig = px.bar(
#             x=labels,
#             y=values,
#             labels={'x': 'Sentiment', 'y': 'Confidence'},
#             width=400,
#             height=500,
#         )

#         fig.update_layout(
#             xaxis=dict(title_font=dict(size=12)),
#             yaxis=dict(showgrid=False, gridwidth=1, range=[0, 1], title_font=dict(size=12)),
#             plot_bgcolor='white',
#             font=dict(size=12),
#             margin=dict(l=0, r=0, b=0, t=30),  # Adjust margin
#             bargap=0.3,  # Adjust gap between bars
#         )

#         fig.update_traces(marker=dict(color=['rgba(0, 128, 0, 0.5)', 'rgba(255, 0, 0, 0.5)']))

#         st.plotly_chart(fig)

#         st.session_state.next_button_enabled = True
#         st.session_state.next_page = "thm_verification"

#         if st.button("Next"):
#             st.session_state.thm_verification = True

def render_progress_bar(value):
    # A custom HTML/CSS implementation of a progress bar
    progress_bar = f"""
    <div style="width: 100%; background-color: #f0f0f0; height: 14px; border-radius: 10px;">
        <div style="width: {value}%; background-color: #00cc00; height: 100%; border-radius: 10px; font-size:12px; text-align: center; color: white;">
            {value}%
        </div>
    </div>
    """
    return progress_bar

def calculate_overall_sentiment(prob1, prob2, prob3):
    # Calculate the average probability
    avg_prob = (prob1 + prob2 + prob3) / 3
    # Determine the overall sentiment based on the average probability
    overall_sentiment = "Positive" if avg_prob > 0.5 else "Negative"
    return avg_prob, overall_sentiment

def sentiment_scrape():
    st.title("Social Check Result")

    if st.button("Get Analysis"):
        if hasattr(st.session_state, 'company_url'):
            company_url = st.session_state.company_url
            
        if hasattr(st.session_state, 'company_name'):
            company_name = st.session_state.company_name
            company_name = company_name.lower()
            st.write(f"Getting Results for {company_name}")

            tweet_list = twitter_scrape(company_name)
            ig_post_list = ['Visa And NymCard Launch Plug & Play End-To-End Issuance Platform To Help #Fintech swiftly launch payment credentials as part of Visa’ Ready To Launch (VRTL) program Read more 🌐 : https://technologyplus.pk/2023/08/28/visa-and-nymcard-launch-plug-play-end-to-end-issuance-platform-to-help-fintech/ Follow on FB 👍 : https://lnkd.in/diKN5pSG . . . . . . . . . . Visa NymCard #Visa #NymCard Umar S. Khan Omar Onsi #BusinessAdministration #education #educationalcontent #educationconsultant #petrol_price_in_Pakistan #today_petrol_rate_in_pakistan_2023 #diesel_price_in_Pakistan', 'Explore NymCard’s Dubai office, a perfect reflection of their innovative and ambitious nature. The contemporary design promotes collaboration and creativity, while sustainable materials and natural light enhance well-being. Located in a prestigious location, this inspiring workspace embodies NymCard’s dedication to revolutionising the fintech industry. Photocredits: @chrisgoldstraw #JTCPLDesigns_HTSInteriors #OfficeDesigners #UAEArchitecture #UAEDesigners #Design #JTCPLDesigns #OfficeDesign #JTCPL_interiors #Modern #OfficeInteriors #WorkPlace #OfficeDecor #Minimal #Bespokelnteriors #InteriorDetails #InteriorForInspo #InteriorDesire #NymCard #ModernRetro #Finance #DubaiOffice', "The new office of NymCard is a testament to the company's commitment to innovation and excellence. The workspace provides a practical and inspiring environment that encourages collaboration and productivity. Attention to detail is evident in every aspect, from the layout to the materials used. This office reflects NymCard’s values and vision, focusing on both functionality and aesthetics. Overall, it's a workspace that truly embodies the brand's identity. Photocredits: @chrisgoldstraw #JTCPLDesigns_HTSInteriors #OfficeDesigners #UAEArchitecture #UAEDesigners #Design #JTCPLDesigns #OfficeDesign #JTCPL_interiors #Modern #OfficeInteriors #WorkPlace #OfficeDecor #Minimal #Bespokelnteriors #InteriorDetails #InteriorForInspo #InteriorDesire #NymCard #ModernRetro #Finance #DubaiOffice", 'Step into NymCard’s Dubai office, featuring distinct areas for meetings, collaboration, and focus. The modern crisp white open ceiling is complemented by retro cork cladding motifs and pastel stretched fabric panels that also serve as acoustic treatments. The space seamlessly blends aesthetics and functionality, providing an ideal workspace for the team. Photocredits: @chrisgoldstraw #JTCPLDesigns_HTSInteriors #OfficeDesigners #UAEArchitecture #UAEDesigners #Design #JTCPLDesigns #OfficeDesign #JTCPL_interiors #Modern #OfficeInteriors #WorkPlace #OfficeDecor #Minimal #Bespokelnteriors #InteriorDetails #InteriorForInspo #InteriorDesire #NymCard #ModernRetro #Finance #DubaiOffice', '@nymcard Acquires @spotiime To Offer BNPL-in-a-Box For Banks and Financial Institutions "NymCard, the leading payments infrastructure provider in the MENA region, has completed the acquisition of Spotii, a prominent Buy Now Pay Later (BNPL) Fintech operating in key markets including KSA, UAE, and Bahrain." #nymcard #spotii #bnpl #fintechs #financialservices #financialinclusion #bankingnews #bankingindustry #digitaltransformation #digitalpayment #servicesasabusiness #digitalplatform #dailynewspk #news', 'Visa And NymCard Launch Plug & Play End-To-End Issuance Platform To Help #Fintech swiftly launch payment credentials as part of Visa’ Ready To Launch (VRTL) program Read more 🌐 : https://technologyplus.pk/2023/08/28/visa-and-nymcard-launch-plug-play-end-to-end-issuance-platform-to-help-fintech/ Follow on FB 👍 : https://lnkd.in/diKN5pSG . . . . . . . . . . Visa NymCard #Visa #NymCard Umar S. Khan Omar Onsi #BusinessAdministration #education #educationalcontent #educationconsultant #petrol_price_in_Pakistan #today_petrol_rate_in_pakistan_2023 #diesel_price_in_Pakistan', 'Dollar East Exchange Company (Private) Limited and United Bank Limited (UBL) recently signed a Memorandum of Understanding (MOU). Read more 🌐 : https://technologyplus.pk/2023/09/01/ubl-and-dollar-east-exchange-extend-their-strategic-partnership/ Follow on FB 👍 : https://lnkd.in/diKN5pSG . . . . . . . . . . UBL - United Bank Limited Dollar East Exchange Company #fintech #partnership #collaboration #reallife #transactional #growth #DigiKhata #KuudnaPakistan #Fintech #FintechNews #Visa #NymCard #BusinessAdministration #education #educationalcontent #educationconsultant #petrol_price_in_Pakistan #today_petrol_rate_in_pakistan_2023 #diesel_price_in_Pakistan', 'The goal was to create a workspace that fosters creativity, productivity, and embodies the NymCard brand ethos. The result is a workspace featuring bespoke niches and curves that enhance collaboration and efficiency in designated activity zones. Photocredits: @chrisgoldstraw #JTCPLDesigns_HTSInteriors #OfficeDesigners #UAEArchitecture #UAEDesigners #Design #JTCPLDesigns #OfficeDesign #JTCPL_interiors #Modern #OfficeInteriors #WorkPlace #OfficeDecor #Minimal #Bespokelnteriors #InteriorDetails #InteriorForInspo #InteriorDesire #NymCard #ModernRetro #Finance #DubaiOffice', "NymCard’s Dubai office: where innovation meets inspiration. The space reflects the brand's creative identity with pastel highlights, layered design, and distinct areas for various activities. It's the perfect workspace for one of the UAE's fastest-growing fintech companies to celebrate success. Photocredits: @chrisgoldstraw #JTCPLDesigns_HTSInteriors #OfficeDesigners #UAEArchitecture #UAEDesigners #Design #JTCPLDesigns #OfficeDesign #JTCPL_interiors #Modern #OfficeInteriors #WorkPlace #OfficeDecor #Minimal #Bespokelnteriors #InteriorDetails #InteriorForInspo #InteriorDesire #NymCard #ModernRetro #Finance #DubaiOffice"]
            # ig_post_list = instagram_scrape(company_name)
            google_reviews_list = google_reviews_scrape(company_name)

            max_len = max(len(tweet_list), len(ig_post_list), len(google_reviews_list))
            if max_len > 3:
                max_len = 3
            
            tweet_list.extend([''] * (max_len - len(tweet_list)))
            ig_post_list.extend([''] * (max_len - len(ig_post_list)))
            google_reviews_list.extend([''] * (max_len - len(google_reviews_list)))

            with st.expander(f"Extracted Data for {company_name}"):
                df = pd.DataFrame({'Google Reviews': google_reviews_list, 'Tweets': tweet_list, 'Instagram': ig_post_list})
                html_table = df.to_html(index=False, header=True)
                st.markdown(html_table, unsafe_allow_html=True)

            all_tweet_texts = " ".join(tweet_list)
            sentiment, prob1 = analyze_sentiment_bert(all_tweet_texts)

            all_ig_texts = " ".join(ig_post_list)
            sentiment, prob2 = analyze_sentiment_bert(all_ig_texts)

            all_gr_texts = " ".join(google_reviews_list)
            sentiment, prob3 = analyze_sentiment_bert(all_gr_texts)

            avg_prob, overall_sentiment = calculate_overall_sentiment(prob1, prob2, prob3)

            fig = make_subplots(rows=2, cols=2, subplot_titles=("Overall Sentiment", "Tweets", "Instagram", "Google Reviews"))

            progress_bar_html = render_progress_bar(int(avg_prob * 100))
            st.components.v1.html(progress_bar_html, height=21)
            fig.add_trace(go.Scatter(x=[1], y=[1], text="", mode="text", hoverinfo="none", showlegend=False), row=1, col=1)

            # Add text labels for "Positive" and "Negative" along with their numbers (increase font size)
            fig.add_annotation(
                text=f"<b>Positive:</b> {int(avg_prob * 100)}%",
                xref="paper", yref="paper",
                x=0.15, y=0.90,
                showarrow=False,
                font=dict(size=17, color="green")
            )
            fig.add_annotation(
                text=f"<b>Negative:</b> {int((1 - avg_prob) * 100)}%",
                xref="paper", yref="paper",
                x=0.15, y=0.85,
                showarrow=False,
                font=dict(size=17, color="red")
            )

            fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)

            # Plot 2: Tweets Sentiment
            labels_tweets = ["Positive", "Negative"]
            values_tweets = [prob1, 1 - prob1]
            fig.add_trace(go.Bar(x=labels_tweets, y=values_tweets, showlegend=False, marker=dict(color=['rgba(0, 128, 0, 0.5)', 'rgba(255, 0, 0, 0.5)'])), row=1, col=2)

            # Plot 3: Instagram Sentiment
            ig_labels = ["Positive", "Negative"]
            ig_values = [prob2, 1 - prob2]
            fig.add_trace(go.Bar(x=ig_labels, y=ig_values, showlegend=False, marker=dict(color=['rgba(0, 128, 0, 0.5)', 'rgba(255, 0, 0, 0.5)'])), row=2, col=1)

            # Plot 4: Google Reviews Sentiment
            gr_labels = ["Positive", "Negative"]
            gr_values = [prob3, 1 - prob3]
            fig.add_trace(go.Bar(x=gr_labels, y=gr_values, showlegend=False, marker=dict(color=['rgba(0, 128, 0, 0.5)', 'rgba(255, 0, 0, 0.5)'])), row=2, col=2)

            # Update subplot layout
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Increase height of each plot
            fig.update_layout(height=900, width=700, title="Sentiment Distribution")

            st.plotly_chart(fig)

            st.session_state.next_button_enabled = True
            st.session_state.next_page = "thm_verification"

            if st.button("Next"):
                st.session_state.thm_verification = True


def thm_verification():
    with st.spinner("Identifying Device.."):
        time.sleep(1)

    with st.spinner("Fetching Results..."):
        time.sleep(1)
        st.title("Fraud Analysis")
        if st.button("Get Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Device Fingerprint")
                st.write("Device ID:", THM_RESPONSE["device_fingerprint"]["device_id"])  # Display the device ID
                st.write("")
                st.write("")
            
            with col2:
                st.subheader("Identity Verification")
                st.write("User Behavior Match:", THM_RESPONSE["identity_verification"]["user_behavior_match"])
                st.write("Identity Verified:", THM_RESPONSE["identity_verification"]["identity_verified"])

            with col1:
                st.subheader("Geolocation")
                st.write("Country:", THM_RESPONSE["geolocation"]["country"])
                st.write("Latitude:", THM_RESPONSE["geolocation"]["latitude"])
                st.write("Longitude:", THM_RESPONSE["geolocation"]["longitude"])

            with col2:
                st.subheader("Bot Detection")
                st.write("Is Bot:", THM_RESPONSE["bot_detection"]["is_bot"])

            st.session_state.next_button_enabled = True
            st.session_state.next_page = "world_check"
            
            if st.button("Next"):
                st.session_state.world_check = True

def world_check():
    st.title("World Check Results")

    container = st.empty()
    container.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
            <div>
                <h3>Loading...</h3>
                <p>Please wait while the process completes.</p>
                <st.spinner></st.spinner>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(3)

    container.empty()

    col1, col2 = st.columns(2)
    with col1:
        st.success("AML")
        st.success("PEP")
        st.success("Sanctions")

    with col2:
        for _ in range(3):
            st.markdown('<p style="font-size:24px; padding-top:16px">✅</p>', unsafe_allow_html=True)


def navigate():
    if st.session_state.next_page == "cr_entry":
        cr_entry_page()
    elif st.session_state.next_page == "idv_page":
        idv_page()
    elif st.session_state.next_page == "similar_web_page":
        similar_web_page()
    elif st.session_state.next_page == "address_verification_page":
        address_verification_page()
    elif st.session_state.next_page == "sentiment_scrape":
        sentiment_scrape()
    elif st.session_state.next_page == "thm_verification":
        thm_verification()
    elif st.session_state.next_page == "world_check":
        world_check()
    else:
        login_page()

# Main Streamlit app
def main():
    if not hasattr(st.session_state, "next_page"):
        st.session_state.next_page = "login"

    if not hasattr(st.session_state, "login"):
        st.session_state.login = False
    if not hasattr(st.session_state, "wathq_response"):
        st.session_state.wathq_response = False
    if not hasattr(st.session_state, "idv_response"):
        st.session_state.idv_response = False
    if not hasattr(st.session_state, "similar_web"):
        st.session_state.similar_web = False
    if not hasattr(st.session_state, "sentiment_scrape"):
        st.session_state.sentiment_scrape = False
    if not hasattr(st.session_state, "thm_verification"):
        st.session_state.thm_verification = False
    if not hasattr(st.session_state, "world_check"):
        st.session_state.world_check = False
    if not hasattr(st.session_state, "next_button_enabled"):
        st.session_state.next_button_enabled = False

    navigate()

    # if not hasattr(st.session_state, "login"):
    #     st.session_state.login = False
    # if not hasattr(st.session_state, "wathq_response"):
    #     st.session_state.wathq_response = False
    # if not hasattr(st.session_state, "next_button_enabled"):
    #     st.session_state.next_button_enabled = False

    # if not st.session_state.login:
    #     login_page()
    # elif not st.session_state.wathq_response:
    #     cr_entry_page()

    # Handle going back to the C/R Entry Page
    # previous_page = st.experimental_get_query_params().get("previous_page")
    # if previous_page == "cr_entry" and st.session_state.wathq_response:
    #     st.session_state.wathq_response = False
    #     st.experimental_rerun()

if __name__ == "__main__":
    main()
