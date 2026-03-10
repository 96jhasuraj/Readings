import sqlite3

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT
)
""")

cursor.execute("""
CREATE TABLE addresses (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    city TEXT
)
""")

conn.commit()

class Address:
    def __init__(self, city):
        self.city = city


class User:
    def __init__(self, name, addresses=None):
        self.id = None
        self.name = name
        self.addresses = addresses or []
        
def save_user(conn, user):
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO users(name) VALUES(?)",
        (user.name,)
    )

    user_id = cursor.lastrowid
    user.id = user_id

    for address in user.addresses:
        cursor.execute(
            "INSERT INTO addresses(user_id, city) VALUES (?, ?)",
            (user_id, address.city)
        )

    conn.commit()
def load_user(conn, user_id):
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name FROM users WHERE id=?",
        (user_id,)
    )

    row = cursor.fetchone()

    user = User(row[1])
    user.id = row[0]

    cursor.execute(
        "SELECT city FROM addresses WHERE user_id=?",
        (user_id,)
    )

    addresses = cursor.fetchall()

    for addr in addresses:
        user.addresses.append(Address(addr[0]))

    return user

def load_all_users(conn):
    cursor = conn.cursor()
    count = 0
    cursor.execute("SELECT id FROM users")
    count+=1
    users = []

    for row in cursor.fetchall():
        users.append(load_user(conn, row[0]))
        count+=1
    print(f"{count} queries called")
    return users

user1 = User(
    "Suraj",
    [Address("Delhi"), Address("Hyd")]
)
user2 = User(
    "Tanuja",
    [Address("Delhi"), Address("Punjab")]
)
save_user(conn, user1)
save_user(conn, user2)
user = load_user(conn, 1)

print(user.name)
for addr in user.addresses:
    print(addr.city)
    
users = load_all_users(conn)
for u in users:
    print(u.name)
    
    
print("eager loading")

def load_users_with_addresses(conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT u.id, u.name, a.city
        FROM users u
        LEFT JOIN addresses a
        ON u.id = a.user_id
    """)

    users = {}

    for uid, name, city in cursor.fetchall():
        if uid not in users:
            users[uid] = User(name)
            users[uid].id = uid

        if city:
            users[uid].addresses.append(Address(city))

    return list(users.values())

users = load_users_with_addresses(conn)
for u in users:
    print(u.name)