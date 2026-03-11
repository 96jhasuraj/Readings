

import sqlite3
import time
from faker import Faker
import json
import random

fake = Faker()

conn = sqlite3.connect("local.db")
cursor = conn.cursor()
NUM_USERS = 10000

def drop_tables():
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS addresses")
    cursor.execute("DROP TABLE IF EXISTS preferences")
    cursor.execute("DROP TABLE IF EXISTS user_documents")
 
def create_user_table():
    cursor.execute("""
    CREATE TABLE users(
        id INTEGER PRIMARY KEY,
        name TEXT
    )
    """)

def create_address_table():
    cursor.execute("""
    CREATE TABLE addresses(
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        city TEXT
    )
    """)

def create_pref_table():
    cursor.execute("""
    CREATE TABLE preferences(
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        theme TEXT
    )
    """)

def create_user_doc():
    cursor.execute("""
    CREATE TABLE user_documents(
        id INTEGER PRIMARY KEY,
        data TEXT
    )
    """)

def add_index():
    cursor.execute("""
    CREATE INDEX idx_addresses_user
    ON addresses(user_id)
    """)

    cursor.execute("""
    CREATE INDEX idx_preferences_user
    ON preferences(user_id)
    """)

def remove_address_index():
    cursor.execute("""
    DROP INDEX idx_addresses_user
    """)

def start():
    drop_tables()
    create_user_table()
    create_address_table()
    create_pref_table()
    create_user_doc()

def insert_data(num:int):
    for i in range(num):
        name = fake.name()
        city = fake.city()
        theme = random.choice(["dark", "light"])

        cursor.execute("INSERT INTO users VALUES (?,?)",(i,name))
        cursor.execute("INSERT INTO addresses(user_id,city) VALUES (?,?)",(i,city))
        cursor.execute("INSERT INTO preferences(user_id,theme) VALUES (?,?)",(i,theme))

        doc = {
            "id": i,
            "name": name,
            "city": city,
            "theme": theme
        }

        cursor.execute(
            "INSERT INTO user_documents VALUES (?,?)",
            (i,json.dumps(doc))
        )

    conn.commit()

def normalized_read():
    user_id = random.randint(0, NUM_USERS-1)
    
    cursor.execute("SELECT name FROM users WHERE id=?",(user_id,))
    user = cursor.fetchone()
    
    cursor.execute("SELECT city FROM addresses WHERE user_id=?",(user_id,))
    addr = cursor.fetchone()
    
    cursor.execute("SELECT theme FROM preferences WHERE user_id=?",(user_id,))
    pref = cursor.fetchone()

    return user, addr, pref

def normailized_join_read():
    user_id = random.randint(0, NUM_USERS-1)
    cursor.execute("""
    SELECT u.name, a.city, p.theme
    FROM users u
    JOIN addresses a ON u.id = a.user_id
    JOIN preferences p ON u.id = p.user_id
    WHERE u.id = ?
    """, (user_id,))

    row = cursor.fetchone()

    return row

def document_read():

    user_id = random.randint(0, NUM_USERS-1)
    cursor.execute(
        "SELECT data FROM user_documents WHERE id=?",
        (user_id,)
    )
    doc = json.loads(cursor.fetchone()[0])
    return doc

def benchmark():
    N = 1000

    start = time.time()
    for _ in range(N):
        normalized_read()
    print("Normalized time:", time.time() - start)

    start = time.time()
    for _ in range(N):
        normailized_join_read()
    print("Normalized time with JOINS :", time.time() - start)

    start = time.time()
    for _ in range(N):
        document_read()
    print("Document time:", time.time() - start)

def print_comamnd():
    cursor.execute("""
    EXPLAIN QUERY PLAN
    SELECT u.name, a.city, p.theme
    FROM users u
    JOIN addresses a ON u.id = a.user_id
    JOIN preferences p ON u.id = p.user_id
    WHERE u.id = 10
    """)
    for row in cursor.fetchall():
        print(row)
    print("\n")
    
    

start()
insert_data(NUM_USERS)
benchmark()
print("\nsearch before index")
print_comamnd()
add_index()
print("\nsearch after index")
print_comamnd()
benchmark()
print("\nremoving address index")
print_comamnd()
remove_address_index()
benchmark()