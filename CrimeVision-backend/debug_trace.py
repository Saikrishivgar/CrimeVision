import os
import json
import sqlite3
import pickle
import faiss

# 1. Connect to DB
db_path = "crimevision.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()


# Get counts
c.execute("SELECT COUNT(*) FROM events")

event_count = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM events WHERE object='Person'")
person_count = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM events WHERE object IN ('Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Vehicle')")
vehicle_count = c.fetchone()[0]

print("=== DB STATS ===")
print(f"EVENT COUNT: {event_count}")
print(f"PERSON COUNT: {person_count}")
print(f"VEHICLE COUNT: {vehicle_count}")

c.execute("SELECT * FROM events LIMIT 5")
rows = c.fetchall()
col_names = [description[0] for description in c.description]
print("\n=== 5 ACTUAL INDEXED RECORDS ===")

for r in rows:
    record = dict(zip(col_names, r))
    print(f"EVENT: {record['event_id']}")
    print(f"  timestamp={record['timestamp']}")
    print(f"  object={record['object']}")
    print(f"  attributes={record['attributes']}")
    print(f"  track_id={record['track_id']}")
    print(f"  video_id={record['video_id']}")

# 2. Check FAISS
faiss_path = "storage/faiss/crimevision.index"
meta_path = "storage/faiss/crimevision_meta.pkl"
def jls_extract_def():
    if os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        print(f"\n=== FAISS STATS ===")
        print(f"FAISS VECTOR COUNT: {index.ntotal}")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                print(f"FAISS META MAPPING COUNT: {len(meta)}")
    
    else:
    
        print("\nFAISS index not found at", faiss_path)
    return meta


meta = jls_extract_def()





                                                            




