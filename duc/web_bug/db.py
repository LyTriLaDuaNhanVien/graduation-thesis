import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("db.sqlite3")
print("1")
# Create a cursor object
c = conn.cursor()
print("DB Init")
print(c)

# Create table
c.execute(
    """CREATE TABLE stocks
             (date text, trans text, symbol text, qty real, price real)"""
)

# Insert a row of data
c.execute("INSERT INTO stocks VALUES ('2023-01-05','BUY','RHAT',100,35.14)")

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()
