import base64
#import cv2
import io
import numpy as np
import re
from datetime import datetime
from PIL import Image
import pdfplumber
from bankstatementextractor_sau.banks_utils import *
import json
from PyPDF2 import * # PyPDF2 library
import os
import subprocess
import torch
#from pdf2image import convert_from_path
from unidecode import unidecode
import pandas as pd
import itertools    
os.sys.path
from io import StringIO

class Banks:

    def __init__(self):
        pass
    # adcb_1
    def alrajhi_1(self,pdf_file_path):
        # print('alrajhi_1')
        try:
            # Read file
            pdf_document = PdfReader(io.BytesIO(pdf_file_path))
            # print('after reading bytes')
            plain_text_data = []

            for page in pdf_document.pages:
                page_text = page.extract_text()
                page_text_blocks = page_text.split('\n')
                plain_text_data.append(page_text_blocks)

            # Account info extraction
            customer_name = None
            account_number = None
            iban_number = None
            opening_balance = None
            closing_balance = None
            num_deposits = None
            num_withdrawals = None
            total_deposits = None
            total_withdrawals = None

            name_pattern = r'Customer Name ([A-Za-z\s]+)'
            account_pattern = r'Account Number (\d+)'
            iban_pattern = r'IBAN Number ([A-Za-z\d]+)'
            balance_pattern = r'([,\d.]+) SAR'
            deposits_pattern = r'Number Of Deposits (\d+)'
            withdrawals_pattern = r'Number Of Withdrawals (\d+)'
            total_deposits_pattern = r'Total Deposits ([,\d.]+) SAR'
            total_withdrawals_pattern = r'Total Withdrawals ([,\d.]+) SAR'

            for sublist in plain_text_data:
                for item in sublist:
                    if 'Customer Name' in item:
                        match = re.search(name_pattern, item)
                        if match:
                            customer_name = match.group(1)
                    elif 'Account Number' in item:
                        match = re.search(account_pattern, item)
                        if match:
                            account_number = match.group(1)
                    elif 'IBAN Number' in item:
                        match = re.search(iban_pattern, item)
                        if match:
                            iban_number = match.group(1)
                    # Extract opening balance
                    elif 'Opening Balance' in item:
                        match = re.search(balance_pattern, item)
                        if match:
                            opening_balance = match.group(1)
                    # Extract closing balance
                    elif 'Closing Balance' in item:
                        match = re.search(balance_pattern, item)
                        if match:
                            closing_balance = match.group(1)
                    # Extract number of deposits
                    elif 'Number Of Deposits' in item:
                        match = re.search(deposits_pattern, item)
                        if match:
                            num_deposits = match.group(1)
                    # Extract number of withdrawals
                    elif 'Number Of Withdrawals' in item:
                        match = re.search(withdrawals_pattern, item)
                        if match:
                            num_withdrawals = match.group(1)
                    # Extract total deposits
                    elif 'Total Deposits' in item:
                        match = re.search(total_deposits_pattern, item)
                        if match:
                            total_deposits = match.group(1)
                    # Extract total withdrawals
                    elif 'Total Withdrawals' in item:
                        match = re.search(total_withdrawals_pattern, item)
                        if match:
                            total_withdrawals = match.group(1)

            obj = {
                'opening balance': opening_balance,
                'closing balance': closing_balance,
                'number of deposits': num_deposits,
                'number of withdrawals': num_withdrawals,
                'revenues': total_deposits,
                'expenses': total_withdrawals
            }

            # Transaction extraction
            lst_1 = [sublist[i+1:] for sublist in plain_text_data for i, element in enumerate(
                sublist) if element.startswith('Date Transaction Details Debit Credit Balance')]

            new_lst = []
            for sublist in lst_1:
                confidential_index = -1
                for i, item in enumerate(sublist):
                    if item.startswith('This document is confidential and under the'):
                        confidential_index = i
                        break
                if confidential_index >= 1:
                    new_sublist = sublist[:confidential_index - 1]
                    new_lst.append(new_sublist)
                else:
                    new_lst.append(sublist)

            # Flatten after removing the elements that are single digits i.e. page numbers
            filtered_lst = [
                item for sublist in new_lst for item in sublist if not item.isdigit()]

            # Define regex patterns
            date_pattern = r'(\d{4}/\d{2}/\d{2})'
            balance_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2}|\.\d{1})|\d{1,3}(?:,\d{3})*)\sSAR'

            # Initialize lists for DataFrame columns
            dates = []
            running_balances = []
            credits = []
            debits = []
            descriptions = []

            # Extract information from each element in the list
            for item in filtered_lst:
                date_match = re.search(date_pattern, item)
                balance_matches = re.findall(balance_pattern, item)

                if date_match:
                    dates.append(date_match.group(1))
                else:
                    dates.append('')

                if len(balance_matches) >= 3:
                    running_balances.append(float(balance_matches[-1].replace(',', '')))
                    credits.append(float(balance_matches[-2].replace(',', '')))
                    debits.append(-float(balance_matches[-3].replace(',', '')))
                else:
                    running_balances.append('')
                    credits.append('')
                    debits.append('')

                # Extract description
                description_match = re.search(date_pattern + r'(.*?)' + balance_pattern, item)
                if description_match:
                    descriptions.append(description_match.group(2))
                else:
                    descriptions.append('')

            # Create a DataFrame
            data = {'date': dates, 'description': descriptions, 'credit': credits, 'debit': debits, 'running_balance': running_balances}
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['amount'] = df['credit']+df['debit']

            result = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index()

            rev_month = df[df['amount'] > 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)
            exp_month = df[df['amount'] < 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)

            final_dict = {
                'free cash flows': dict(zip(result['date'].dt.strftime('%Y-%m'), result['amount'])),
                'rev_by_month': dict(zip(rev_month['date'].dt.strftime('%Y-%m'), rev_month['amount'])),
                'exp_by_month': dict(zip(exp_month['date'].dt.strftime('%Y-%m'), exp_month['amount']))
            }

            # Add the extra string values to the final dictionary
            final_dict.update(obj)

            return final_dict
        except Exception as e:
            return {'error': str(e)}
    
    def riyadh_1(self,pdf_file_path):
        try:
            # Read the PDF file using pdfplumber
            with pdfplumber.open(pdf_file_path) as pdf:
                plain_text_data = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    page_text_blocks = page_text.split('\n')
                    plain_text_data.append(page_text_blocks)

            # Account information
            if plain_text_data[-1][0] == 'CUSTOMER STATEMENT':
                # Initialize variables
                opening_balance = 0
                number_of_deposits = None
                number_of_withdrawals = None
                closing_balance = 0
                revenues = None
                expenses = None

                # Iterate through elements in the last sublist
                for item in plain_text_data[-1]:
                    if item.startswith('Beginning Balance'):
                        # Extract the opening balance
                        opening_balance = float(item.split()[-1].replace(',', ''))
                    elif item.startswith('Ending Balance'):
                        # Extract the closing balance
                        closing_balance = float(item.split()[-1].replace(',', ''))
                    elif 'Deposits' in item:
                        # Extract the number of deposits and revenues
                        parts = item.split()
                        number_of_deposits = int(parts[0].replace(',', ''))
                        revenues = float(parts[2].replace(',', ''))
                    elif 'Withdrawals' in item:
                        # Extract the number of withdrawals and expenses
                        parts = item.split()
                        number_of_withdrawals = int(parts[0].replace(',', ''))
                        expenses = float(parts[2].replace(',', ''))

            # Transaction information
            lst_1 = []

            # Iterate through all sublists
            for sublist in plain_text_data:
                found_beginning_balance = False
                modified_sublist = []

                # Iterate through elements in the sublist
                for item in sublist:
                    if found_beginning_balance:
                        modified_sublist.append(item)
                    elif item.startswith('(G/H) Transaction Detail Debit Credit Balance', ):
                        found_beginning_balance = True

                # Append the modified sublist or the original sublist if no 'Beginning Balance' was found
                if found_beginning_balance:
                    lst_1.append(modified_sublist)
                else:
                    lst_1.append(sublist)

            lst_2 = [[item for item in sublist if not item.startswith(('Public shareholding', 'Beginning Balance'))] for sublist in lst_1]
            lst_3 = [item for sublist in lst_2 for item in sublist]

            # Initialize lists to store extracted data
            dates = []
            amounts = []
            running_balances = []
            descriptions = []

            # Regular expression pattern for the specified format
            pattern = r'(\d{2}/\d{2}) (.+) (\d+\.\d{2}) (\d+\.\d{2})'

            # Iterate through the list and extract relevant information
            for item in lst_3:
                match = re.match(pattern, item)
                if match:
                    date, description, amount, running_balance = match.groups()
                    dates.append(date)
                    amounts.append(float(amount.replace(',', '')))
                    running_balances.append(float(running_balance.replace(',', '')))
                    descriptions.append(description)

            # Create a DataFrame
            df = pd.DataFrame({'date': dates, 'description': descriptions, 'amount': amounts, 'running_balance': running_balances})

            # Convert running_balance to numeric
            df['running_balance'] = pd.to_numeric(df['running_balance'], errors='coerce')

            # Calculate the difference in running balance compared to the previous row
            df['running_balance_diff'] = df['running_balance'].diff()

            df['date'] = pd.to_datetime(df['date'], format='%d/%m')

            # Set the amount to negative if running_balance_diff is negative
            df.loc[df['running_balance_diff'] < 0, 'amount'] = -df['amount']

            # Drop the running_balance_diff column
            df = df.drop(columns=['running_balance_diff'])

            if not df.empty and opening_balance is not None and df.iloc[0]['amount'] < opening_balance:
                df.at[0, 'amount'] = -df.iloc[0]['amount']

            rev_month = df[df['amount'] > 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)
            exp_month = df[df['amount'] < 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)    
            result = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index()

            final_dict = {
                'free cash flows': dict(zip(result['date'].dt.strftime('%Y-%m'), result['amount'])),
                'rev_by_month': dict(zip(rev_month['date'].dt.strftime('%Y-%m'), rev_month['amount'])),
                'exp_by_month': dict(zip(exp_month['date'].dt.strftime('%Y-%m'), exp_month['amount']))
            }


            obj = {
                'opening balance':opening_balance,
                'closing balance':closing_balance,
                'number of deposits':number_of_deposits,
                'number of withdrawals':number_of_withdrawals,
                'revenues':revenues,
                'expenses':expenses,    
            }     
            final_dict.update(obj)
            
            return final_dict

        except Exception as e:
            return {'error': str(e)}
    
    def aljazira_1(self,pdf_file_path):
        try:
            # Open the PDF file using PyPDF2
            pdf_document = PdfReader(io.BytesIO(pdf_file_path))

            plain_text_data = []

            for page in pdf_document.pages:
                page_text = page.extract_text()

                # Split text into blocks based on newline
                page_text_blocks = page_text.split('\n')

                plain_text_data.append(page_text_blocks)  # Extend the list with text blocks
            
            # Google Cloud Translation
            # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds/streamlit-connection-b1a38b694505.json"
            # translate_client = translate.Client()

            # def gcloud_translate(text, src='ar', dest='en'):
            #     results = translate_client.translate(text, source_language=src, target_language=dest)
            #     translated_texts = [result['translatedText'] for result in results]
            #     return translated_texts
            
            res = [gcloud_translate(ele, src='ar', dest='en') for ele in plain_text_data]

            lst_1 = [sublist[i + 1:] for sublist in res for i, element in enumerate(
                sublist) if element.startswith('Balance Date Statement Due Date Debit Credit')]

            new_lst = []
            for sublist in lst_1:
                confidential_index = -1

                # Find the index of the element starting with 'Note'
                for i, item in enumerate(sublist):
                    if item.startswith('Note: This document contains financial confidentiality'):
                        confidential_index = i
                        break

                # Append elements up to the confidential index (inclusive)
                if confidential_index >= 0:
                    new_sublist = sublist[:confidential_index]
                    new_lst.append(new_sublist)
                else:
                    new_lst.append(sublist)

            new_lst_2 = []
            for sublist in new_lst:
                confidential_index = -1

                # Find the index of the element starting with 'Note'
                for i, item in enumerate(sublist):
                    if item.startswith('Electronic statement of account summary'):
                        confidential_index = i
                        break

                # Append elements up to the confidential index (inclusive)
                if confidential_index >= 0:
                    new_sublist_1 = sublist[:confidential_index]
                    new_lst_2.append(new_sublist_1)
                else:
                    new_lst_2.append(sublist)

            flattened_lst = [item for sublist in new_lst_2 for item in sublist]

            # Find indices of elements starting with a date of format 'MM/DD/YY'
            date_pattern = r'\d{2}/\d{2}/\d{2}'
            indices = [i for i, item in enumerate(flattened_lst) if re.match(date_pattern, item)]

            # Create a new list by concatenating elements between indices and adding to the element in the first index
            new_lst_3 = []
            for i in range(len(indices) - 1):
                start_index = indices[i]
                end_index = indices[i + 1]
                concatenated_element = flattened_lst[start_index] + ' ' + ' '.join(flattened_lst[start_index + 1:end_index])
                new_lst_3.append(concatenated_element)

            # Append the remaining elements to the last concatenated element
            if indices:
                last_index = indices[-1]
                concatenated_element = flattened_lst[last_index] + ' '.join(flattened_lst[last_index + 1:])
                new_lst_3.append(concatenated_element)

            # Initialize lists for DataFrame columns
            dates = []
            running_balances = []
            amounts = []

            # Define regex patterns
            date_pattern = r'\d{2}/\d{2}/\d{2}'
            balance_pattern = r'(\d{1,3}(?:,\d{3})*\.\d{2})\s(\d{2}/\d{2}/\d{2})\s(\d{1,3}(?:,\d{3})*\.\d{2})'

            # Iterate through each element in the list
            for item in new_lst_3:
                # Extract date, running balance, and amount
                match = re.search(balance_pattern, item)
                if match:
                    date = pd.to_datetime(match.group(2), format='%m/%d/%y')
                    running_balance = float(match.group(1).replace(',', ''))
                    amount = float(match.group(3).replace(',', ''))
                    dates.append(date)
                    running_balances.append(running_balance)
                    amounts.append(amount)
                else:
                    dates.append(None)
                    running_balances.append(None)
                    amounts.append(None)

            # Create a DataFrame
            df = pd.DataFrame({'date': dates, 'running_balance': running_balances, 'amount': amounts})

            # Initialize variables for financial summary
            withdrawals = df[df['amount'] < 0]['amount'].sum()
            deposits = df[df['amount'] > 0]['amount'].sum()
            withdrawal_count = df[df['amount'] < 0]['amount'].count()
            deposit_count = df[df['amount'] > 0]['amount'].count()

            # Calculate opening and closing balances
            opening_balance = (deposits + withdrawals).round(2)
            closing_balance = opening_balance

            # Calculate revenues and expenses
            revenues = deposits.round(2)
            expenses = withdrawals.round(2)

            # Group by date and calculate monthly cash flows
            result = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index()
            result_1 = result[result.amount >0] #remove in future
            rev_month = df[df['amount'] > 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)
            exp_month = df[df['amount'] < 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)



            final_dict = {
                'free cash flows': dict(zip(result_1['date'].dt.strftime('%Y-%m'), result_1['amount'])),
                'rev_by_month': dict(zip(rev_month['date'].dt.strftime('%Y-%m'), rev_month['amount'])),
                'exp_by_month': dict(zip(exp_month['date'].dt.strftime('%Y-%m'), exp_month['amount']))
            }



            obj = {
            #     'name':customer_name,
            #     'account':account_number,
            #     'iban':iban_number,
                'opening balance':'',
                'closing balance':(deposits+withdrawals).round(2),
                'number of deposits':deposit_count,
                'number of withdrawals':withdrawal_count,
                'revenues':deposits.round(2),
                'expenses':withdrawals.round(2) 
            }

            # Add the extra string values to the final dictionary
            final_dict.update(obj)

            return final_dict

        except Exception as e:
            return {'error': str(e)}
    
    def anb_1(pdf_file_path):
        try:
            # Open the PDF file using PyPDF2
            pdf_document = PdfReader(io.BytesIO(pdf_file_path))

            plain_text_data = []

            for page in pdf_document.pages:
                page_text = page.extract_text()
                page_text_blocks = page_text.split('\n')
                plain_text_data.append(page_text_blocks)

            # Extract data as previously done
            lst_1 = [sublist[i+2:] for sublist in plain_text_data for i, element in enumerate(sublist) if element.startswith(' Value Date Description  Amount  Balance')]
            lst_2 = [sublist[:next((i for i, element in enumerate(sublist) if element.startswith('Arab National Bank-a Saudi joint stock')), len(sublist))] for sublist in lst_1]
            flattened_lst = [item for sublist in lst_2 for item in sublist if not re.match(r'^\d{4}-\d{2}-\d{2}$', item)]  # remove value dates and flatten the sublist

            date_pattern = r'\d{4}-\d{2}-\d{2}'
            balance_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2}|\.\d{1})|\d{1,3}(?:,\d{3})*)$'
            amount_pattern = r'([-+]?\d+(?:,\d{3})*(?:\.\d{2}|\.\d{1})?)\s+[-+]?\d+(?:,\d{3})*(?:\.\d{2}|\.\d{1})?$'
            string_pattern = r'(\d{4}-\d{2}-\d{2})(.*?)(?=\s[-+]?\d+(?:,\d{3})*(?:\.\d{2}|\.\d{1})?$)'

            date_indices = [index for index, element in enumerate(flattened_lst) if re.match(date_pattern, element)]

            result = []
            for i in range(len(date_indices)):
                start_idx = date_indices[i]
                end_idx = date_indices[i + 1] if i + 1 < len(date_indices) else len(flattened_lst)
                concatenated = ' '.join(flattened_lst[start_idx:end_idx])
                result.append(concatenated)

            # Extract dates and running balances
            dates = [re.search(date_pattern, elem).group() for elem in result]
            balances = [re.search(balance_pattern, elem).group(1) if re.search(balance_pattern, elem) else None for elem in result]
            amounts = [re.search(amount_pattern, elem).group(1) if re.search(amount_pattern, elem) else None for elem in result]
            description = [re.search(string_pattern, elem).group(2) if re.search(string_pattern, elem) else None for elem in result]

            # Create a DataFrame
            df = pd.DataFrame({'date': dates,'description':description, 'running_balance': balances, 'amount': amounts})
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

            # Remove commas and convert 'running_balance' and 'amount' columns to float
            df['running_balance'] = df['running_balance'].str.replace(',', '').astype(float)
            df['amount'] = df['amount'].str.replace(',', '').astype(float)

            rev_month = df[df['amount'] > 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)
            exp_month = df[df['amount'] < 0].groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index().round(2)

            result = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().round(2).reset_index()

            final_dict = {
            'free cash flows': dict(zip(result['date'].dt.strftime('%Y-%m'), result['amount'])),
            'rev_by_month': dict(zip(rev_month['date'].dt.strftime('%Y-%m'), rev_month['amount'])),
            'exp_by_month': dict(zip(exp_month['date'].dt.strftime('%Y-%m'), exp_month['amount']))
        }

            obj = {
                'opening balance': '',
                'closing balance': df[df.amount>0]['amount'].sum() +  df[df.amount<0]['amount'].sum(),
                'number of deposits': df[df.amount>0]['amount'].count(),
                'number of withdrawals': df[df.amount<0]['amount'].count(),
                'revenues': df[df.amount>0]['amount'].sum(),
                'expenses': df[df.amount<0]['amount'].sum()
            }

            # Add the extra string values to the final dictionary
            final_dict.update(obj)

            return final_dict
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
