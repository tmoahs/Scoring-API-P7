# check_db.py
import sqlite3

conn = sqlite3.connect('data/feature_store.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables trouvées dans la base de données :")
print(cursor.fetchall())
conn.close()