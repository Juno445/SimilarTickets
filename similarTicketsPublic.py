# === Imports ===
from urllib.parse import quote
import requests
from requests.auth import HTTPBasicAuth
import json
import csv
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import logging

# === Configuration ===
API_KEY = "X"
DOMAIN = "X"
TICKETS_CSV = "tickets.csv"

DUPLICATE_CONFIG = {
    'input_file': TICKETS_CSV,
    'output_file': 'potential_duplicates.csv',
    'embedding_model': 'all-MiniLM-L6-v2',
    'similarity_threshold': 0.75,
    'filter_by_requester': True,
    'max_date_diff_days': 60  # Optional: set to None to disable
}

# === Utility Functions ===

def sanitize_html(html_string):
    if html_string is None:
        return ""
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text(strip=True)

# --------------------- Ticket Fetching ---------------------
def fetch_dispatch_tickets():
    query = "X"
    encoded_query = quote(query)

    # Base URL
    base_url = f"https://{DOMAIN}.freshservice.com/api/v2/tickets/filter?query=\"{encoded_query}\""

    # Parameters for pagination
    page = 1
    per_page = 100  # Max allowed per page (may vary by API version)
    all_tickets = []

    while True:
        url = f"{base_url}&page={page}&per_page={per_page}"
        resp = requests.get(url, auth=HTTPBasicAuth(API_KEY, 'X'))

        if resp.status_code != 200:
            print(f"Error: HTTP {resp.status_code}\n{resp.text}")
            break

        data = resp.json()
        tickets = data.get("tickets", [])
        all_tickets.extend(tickets)

        if not tickets:
            break

        page += 1

    slim_tickets = [
        {
            "Ticket ID": t.get("id"),
            "Subject": t.get("subject"),
            "Requester Email": t.get("requester_id"),
            "Date Created": t.get("created_at"),
            "Description": sanitize_html(t.get("description"))
        }
        for t in all_tickets
    ]

    print(json.dumps(slim_tickets, indent=2))

    with open(TICKETS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Ticket ID', 'Subject', 'Requester Email', 'Date Created', 'Description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(slim_tickets)

    return slim_tickets

# --------------------- Duplicate Detection ---------------------
def load_and_prepare_data(CONFIG):
    df = pd.read_csv(CONFIG['input_file'])
    required_cols = [
        'Ticket ID', 'Subject', 'Requester Email', 'Date Created', 'Description'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=['Subject', 'Description'])
    return df

def find_duplicates(df, CONFIG):
    model = SentenceTransformer(CONFIG['embedding_model'])
    pairs = []

    texts = df['Subject'] + " " + df['Description']
    embeddings = model.encode(texts.tolist(), convert_to_tensor=True)
    total = len(df)
    for i in tqdm(range(total), desc="Finding duplicates"):
        for j in range(i + 1, total):
            sim_score = util.cos_sim(embeddings[i], embeddings[j]).item()

            if sim_score < CONFIG['similarity_threshold']:
                continue

            row_i = df.iloc[i]
            row_j = df.iloc[j]

            if CONFIG['filter_by_requester'] and row_i['Requester Email'] != row_j['Requester Email']:
                continue

            if CONFIG['max_date_diff_days']:
                date1 = pd.to_datetime(row_i['Date Created'])
                date2 = pd.to_datetime(row_j['Date Created'])
                if abs((date1 - date2).days) > CONFIG['max_date_diff_days']:
                    continue

            pair = {
                'Ticket ID 1': row_i['Ticket ID'],
                'Subject 1': row_i['Subject'],
                'Requester Email 1': row_i['Requester Email'],
                'Date Created 1': row_i['Date Created'],
                'Description 1': row_i['Description'],
                'Ticket ID 2': row_j['Ticket ID'],
                'Subject 2': row_j['Subject'],
                'Requester Email 2': row_j['Requester Email'],
                'Date Created 2': row_j['Date Created'],
                'Description 2': row_j['Description'],
                'Similarity Score': sim_score
            }
            pairs.append(pair)

    return pd.DataFrame(pairs)

def save_duplicates(df_duplicates, output_file):
    if not df_duplicates.empty:
        df_duplicates.to_csv(output_file, index=False)
        logging.info(f"Saved {len(df_duplicates)} potential duplicates to {output_file}")
    else:
        logging.info("No potential duplicates found.")

# --------------------- GUI for File Selection ---------------------
def get_save_file_location(defaultname="potential_duplicates.csv"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(
        title="Save potential duplicates as...",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile=defaultname
    )
    root.destroy()
    return file_path
# --------------------- Master Main ---------------------
def main():
    logging.basicConfig(level=logging.INFO)
    # Step 1: Fetch tickets and save to CSV
    logging.info("Fetching tickets...")
    tickets = fetch_dispatch_tickets()
    if not tickets:
        logging.error("No tickets found or fetch failed.")
        return

    logging.info(f"Successfully fetched {len(tickets)} tickets.")

    # Step 2: Detect and save duplicates
    logging.info("Running duplicate detection...")
    df = load_and_prepare_data(DUPLICATE_CONFIG)
    duplicates = find_duplicates(df, DUPLICATE_CONFIG)
    # Ask user where to save final CSV
    output_csv = get_save_file_location(defaultname="potential_duplicates.csv")
    if output_csv:  # User did NOT cancel
        save_duplicates(duplicates, output_csv)
    else:
        logging.info("Save cancelled by user; duplicates not written.")

if __name__ == "__main__":
    main()