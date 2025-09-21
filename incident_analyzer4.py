import os
import json
import numpy as np
import google.generativeai as genai
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import spacy
from geopy.geocoders import Nominatim
import time # <-- 1. IMPORT THE TIME MODULE

# --- Configuration ---
load_dotenv() 

try:
    api_key = os.getenv('GOOGLE_API_KEY')
    db_url = os.getenv('NEON_DATABASE_URL')
    if not api_key or not db_url:
        raise ValueError("API key or Database URL not found. Make sure your .env file is correct.")
    genai.configure(api_key=api_key)
    NEON_CONN_STRING = db_url
    print("✅ Configuration loaded successfully.")
except (KeyError, ValueError) as e:
    print(f"❌ ERROR: {e}")
    exit()

# --- NLP and Geocoding Setup ---
print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="incident_cluster_app")
print("✅ NLP model loaded.")

def fetch_data_from_neon(conn_string):
    """
    Connects to Neon, fetches tweet data, and enriches it by extracting
    locations from text if geo_data is missing.
    """
    
    def extract_location_from_text(text, context_location="Vellore, Tamil Nadu"):
        """Uses NLP to find a location in text and geocode it."""
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["GPE", "FAC", "LOC"]:
                    location_name = ent.text
                    print(f"  - Found potential location in text: '{location_name}'")
                    full_location_str = f"{location_name}, {context_location}"
                    location_data = geolocator.geocode(full_location_str)
                    if location_data:
                        print(f"    - Geocoded to: ({location_data.latitude:.4f}, {location_data.longitude:.4f})")
                        return location_data.latitude, location_data.longitude
            return None, None
        except Exception as e:
            print(f"    - Error during NLP extraction: {e}")
            return None, None

    print("Connecting to Neon database...")
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        sql_query = "SELECT id AS incident_id, text AS description, created_at AS date_time, author_id, location, geo_data FROM tweets;"
        cursor.execute(sql_query)
        colnames = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        data = []
        for row in rows:
            record = dict(zip(colnames, row))
            lat, lon = None, None
            geo = record.get('geo_data')
            
            try:
                if isinstance(geo, dict) and geo.get('type') == 'Point' and 'coordinates' in geo:
                    coords = geo['coordinates']
                    if isinstance(coords, list) and len(coords) == 2:
                        lon, lat = coords[0], coords[1]
            except (ValueError, TypeError, IndexError):
                pass
            
            if lat is None:
                lat, lon = extract_location_from_text(record['description'], record.get('location', 'Vellore'))

            record['latitude'] = lat
            record['longitude'] = lon
            data.append(record)

        cursor.close()
        conn.close()
        print(f"✅ Successfully fetched and parsed {len(data)} records.")
        return data
    except Exception as e:
        print(f"❌ Database connection or query failed: {e}")
        return None

def create_feature_vectors(incident_data):
    """
    Generates combined feature vectors with location as the dominant factor.
    """
    if not incident_data:
        print("No incident data to process.")
        return None
        
    print("Step 1: Generating text embeddings for all records...")
    try:
        descriptions = [incident['description'] for incident in incident_data]
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=descriptions, task_type="clustering"
        )
        text_embeddings = np.array(result['embedding'])
    except Exception as e:
        print(f"An error occurred during text embedding: {e}")
        return None

    print("Step 2: Processing timestamps and handling locations...")
    timestamps = np.array([incident['date_time'].timestamp() for incident in incident_data]).reshape(-1, 1)
    scaler_time = StandardScaler()
    scaled_timestamps = scaler_time.fit_transform(timestamps)

    valid_locations = np.array(
        [[inc['latitude'], inc['longitude']] for inc in incident_data if inc['latitude'] is not None]
    )

    scaler_location = StandardScaler()
    if len(valid_locations) > 0:
        scaler_location.fit(valid_locations)

    scaled_locations = np.zeros((len(incident_data), 2))
    for i, incident in enumerate(incident_data):
        if incident['latitude'] is not None:
            loc = np.array([[incident['latitude'], incident['longitude']]])
            scaled_locations[i] = scaler_location.transform(loc)

    text_weight = 1.0
    location_weight = 4.0

    combined_features = np.hstack([
        text_embeddings * text_weight,
        scaled_timestamps,
        scaled_locations * location_weight
    ])
    print("Step 3: Combined feature vectors created for all records.")
    return combined_features

def cluster_incidents(features):
    """Clusters incidents using very strict, location-focused parameters."""
    if features is None: return None
    print("Step 4: Clustering incidents...")

    db = DBSCAN(eps=0.9, min_samples=2, metric='euclidean').fit(features)
    
    found_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Clustering complete. Found {found_clusters} distinct groups.")
    return db.labels_

def get_cluster_summary(incident_group):
    """Generates a brief, structured summary for a cluster."""
    if not incident_group: return "No incidents to summarize."
    
    incident_texts = "\n- ".join([incident['description'] for incident in incident_group])
    
    prompt = f"""
    Analyze the following incident reports, which have been grouped together because they occurred in the same location.
    Your summary MUST be in the following format. Keep the summary to a single, concise sentence.

    **Type of Incident:** ...
    **Common Location:** ...
    **Urgency Level:** ...
    **Summary:** ...

    Here are the incident reports:
    ---
    {incident_texts}
    ---
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

if __name__ == "__main__":
    incident_data = fetch_data_from_neon(NEON_CONN_STRING)

    if incident_data:
        feature_vectors = create_feature_vectors(incident_data)

        if feature_vectors is not None:
            cluster_labels = cluster_incidents(feature_vectors)
            
            for incident, label in zip(incident_data, cluster_labels):
                incident['cluster'] = label
            
            grouped_incidents = [inc for inc in incident_data if inc['cluster'] != -1]
            
            clusters = {}
            for incident in grouped_incidents:
                cluster_id = incident['cluster']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(incident)

            summaries = []
            print("\n--- Generating Summaries for Each Incident Group ---")

            if not clusters:
                print("\nNo significant incident groups were found.")
            else:
                for cluster_id in sorted(clusters.keys()):
                    incident_group = clusters[cluster_id]
                    print(f"\nProcessing Group #{cluster_id} with {len(incident_group)} incidents...")
                    summary_text = get_cluster_summary(incident_group)
                    summaries.append({
                        "Incident Group": f"Group {cluster_id}",
                        "Number of Incidents": len(incident_group),
                        "Summary": summary_text
                    })
                    
                    # --- 2. ADD A DELAY TO AVOID HITTING THE RATE LIMIT ---
                    time.sleep(1) # Wait for 1 second before the next API call

                ranked_summaries = sorted(summaries, key=lambda x: x['Number of Incidents'], reverse=True)

                print("\n\n✅ Final Ranked Incident Summary Report ✅")
                print("=" * 50)
                for i, summary in enumerate(ranked_summaries):
                    print(f"Rank {i + 1}: {summary['Incident Group']} ({summary['Number of Incidents']} incidents)")
                    print(summary['Summary'])
                    print("-" * 50)
    else:
        print("\nNo data available to process. The script will now exit.")