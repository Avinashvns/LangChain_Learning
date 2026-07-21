import sqlite3

conn = sqlite3.connect("Module_4/company.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS employees(
    id INTEGER PRIMARY KEY,
    name TEXT,
    salary INTEGER
               )
""")

cursor.execute("DELETE FROM employees")

cursor.executemany(
    "INSERT INTO employees(name,salary) VALUES(?,?)",
    [
        ("Akash",50000),
        ("Rahul",70000),
        ("Priya",60000)
    ]
)

conn.commit()
conn.close()