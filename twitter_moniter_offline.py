import os
import sys
import psycopg2
from psycopg2 import extras
import json
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
JSON_FILE_NAME = 'vellore_flood_tweets.json'

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    """Establishes a connection to the Neon Postgres database."""
    try:
        conn = psycopg2.connect(NEON_DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to the database: {e}")
        return None

def setup_database(conn):
    """Creates/updates the 'tweets' table to match the JSON file structure."""
    with conn.cursor() as cur:
        # NOTE: author_id is now TEXT and location column is added.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id BIGINT PRIMARY KEY,
                text TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                author_id TEXT,
                location TEXT,
                geo_data JSONB
            );
        """)
        conn.commit()
    print("✅ Database table ensured to exist.")

def insert_tweets(conn, tweets):
    """Inserts a list of tweets from the JSON file into the database."""
    if not tweets:
        print("ℹ️ No tweets to insert.")
        return 0

    # Updated query to include the 'location' column
    insert_query = """
        INSERT INTO tweets (id, text, created_at, author_id, location, geo_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
    """
    
    with conn.cursor() as cur:
        # Map the data from the JSON file to the query parameters
        data_to_insert = [
            (
                int(tweet['id']),
                tweet['text'],
                tweet['created_at'],
                tweet['author_id'],
                tweet['location'],
                json.dumps(tweet.get('geo_data')) # Safely handle null geo_data
            )
            for tweet in tweets
        ]
        
        extras.execute_batch(cur, insert_query, data_to_insert)
        inserted_count = cur.rowcount
        conn.commit()

    print(f"✅ Successfully inserted {inserted_count} new tweets into the database.")
    if inserted_count < len(tweets):
        print(f"ℹ️ {len(tweets) - inserted_count} tweets were already in the database and were skipped.")
    return inserted_count

# --- DATA LOADING FUNCTION ---

def read_local_json_file(filename):
    """Reads tweet data from a local JSON file."""
    print(f"🔎 Reading data from '{filename}'...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{filename}' not found. Make sure it's in the same directory as the script.")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{filename}'. Please check for formatting errors.")
        return None

# --- MAIN EXECUTION ---

def main():
    """Main function to load data from the file to the database."""
    print("--- Starting JSON Data Load to Database ---")
    
    # 1. Load tweets from the local JSON file
    tweets = read_local_json_file(JSON_FILE_NAME)
    
    if tweets is None:
        print("--- Halting execution due to file error. ---")
        return
        
    print(f"📊 Found {len(tweets)} tweets in the JSON file.")

    # 2. Connect to the database
    conn = get_db_connection()
    if conn:
        try:
            # 3. Ensure table exists and insert data
            setup_database(conn)
            insert_tweets(conn, tweets)
        finally:
            # 4. Close the connection
            conn.close()
            print("🔒 Database connection closed.")
    
    print("--- Script finished. ---")

if __name__ == "__main__":
    main()