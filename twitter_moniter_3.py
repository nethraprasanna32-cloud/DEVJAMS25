import os
import sys
import time
import requests
import psycopg2
from psycopg2 import extras
import json
from dotenv import load_dotenv

# --- CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()

# Get credentials from environment variables
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

# --- HIGH-FIDELITY SEARCH QUERY WITH CONTEXT FILTER ---
# This query uses the 'context' operator to ensure results are about
# actual natural disasters, significantly reducing contextual noise.
# 131.1048369021374464 is the ID for the "Natural Disasters" topic.
SEARCH_QUERY = (
    # Disaster & Help Keywords (English)
    '(earthquake OR tremor OR cyclone OR flood OR "heavy rain" OR '
    'wildfire OR "forest fire" OR help OR rescue OR stuck) '
    
    # CONTEXT FILTER: Only include tweets classified as Natural Disasters
    'context:131.1048369021374464 '
    
    # Location Filter for India
    'place_country:IN '
    
    # Language and other filters
    '-is:retweet lang:en'
)

TWEET_FIELDS = "created_at,author_id,geo" # Fields you want to retrieve
SLEEP_INTERVAL_SECONDS = 120 # Pause for 120 seconds (2 minutes) between searches

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
    """Creates the 'tweets' table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id BIGINT PRIMARY KEY,
                text TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                author_id BIGINT,
                geo_data JSONB
            );
        """)
        conn.commit()
    print("✅ Database table ensured to exist.")

def insert_tweets(conn, tweets):
    """Inserts a list of tweets into the database, ignoring duplicates."""
    if not tweets:
        print("ℹ️ No new tweets to insert.")
        return 0

    insert_query = """
        INSERT INTO tweets (id, text, created_at, author_id, geo_data)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
    """
    
    with conn.cursor() as cur:
        data_to_insert = [
            (
                tweet['id'],
                tweet['text'],
                tweet['created_at'],
                tweet['author_id'],
                json.dumps(tweet.get('geo')) # Store geo object as JSONB
            )
            for tweet in tweets
        ]
        
        extras.execute_batch(cur, insert_query, data_to_insert)
        inserted_count = cur.rowcount
        conn.commit()

    print(f"✅ Successfully inserted {inserted_count} new tweets.")
    return inserted_count


# --- TWITTER API FUNCTIONS ---

def search_twitter(query, token, fields):
    """Searches for recent tweets using the Twitter v2 API."""
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "query": query,
        "tweet.fields": fields,
        "max_results": 100
    }
    url = "https://api.twitter.com/2/tweets/search/recent"
    
    print(f"🔎 Searching Twitter with query: '{query}'")
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"❌ Twitter API Error: {response.status_code} {response.text}")
            return None
        return response.json().get('data', [])
    except requests.RequestException as e:
        print(f"❌ Network error fetching tweets: {e}")
        return None

# --- MAIN EXECUTION ---

def main_task():
    """Main function to run one cycle of the data collection process."""
    print("\n--- Starting Tweet Collection Cycle ---")
    
    if not TWITTER_BEARER_TOKEN or not NEON_DATABASE_URL:
        print("❌ Error: Make sure TWITTER_BEARER_TOKEN and NEON_DATABASE_URL are set in your .env file.")
        return

    # 1. Fetch tweets
    tweets = search_twitter(SEARCH_QUERY, TWITTER_BEARER_TOKEN, TWEET_FIELDS)
    
    if tweets is None:
        print("--- Halting cycle due to API error. ---")
        return
        
    print(f"📊 Found {len(tweets)} tweets in the latest search.")

    # 2. Connect to the database
    conn = get_db_connection()
    if conn:
        # 3. Ensure table exists and insert data
        setup_database(conn)
        insert_tweets(conn, tweets)
        
        # 4. Close the connection
        conn.close()
        print("🔒 Database connection closed.")

if __name__ == "__main__":
    try:
        while True:
            main_task()
            print(f"\n--- Pausing for {SLEEP_INTERVAL_SECONDS} seconds... (Press Ctrl+C to stop) ---")
            time.sleep(SLEEP_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n🛑 Script stopped by user. Exiting gracefully.")
        sys.exit(0)