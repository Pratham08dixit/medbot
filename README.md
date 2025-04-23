# medbot
MedBot is a Streamlit-based AI medical assistant that provides pharmacy-related insights including inventory management, usage statistics, forecasts, reorder recommendations, and expiry/manufacturing date checks. It leverages LangChain with Google Gemini 2.0, CSV and SQLite data sources, and FAISS for retrieval-augmented QA.
Inventory Management: Check current stock levels from current_inventory.csv.

Usage Queries: Retrieve drug usage by specific day or date from csv file.

Expiry & Manufacturing Dates: List medicines expiring before or manufactured before a given date.
