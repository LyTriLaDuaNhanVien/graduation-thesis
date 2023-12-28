
import os
from fastapi import FastAPI,APIRouter
import sqlite3

app = APIRouter()


@app.get("/startup")
async def startup_event():
    conn = sqlite3.connect("test.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE users
                 (id int primary key, name text)"""
    )

    # Insert a row of data
    c.execute("INSERT INTO users VALUES (1, 'Alice')")

    # Save (commit) the changes
    conn.commit()

    # Close the connection
    conn.close()


# @app.get("/items/")
# async def read_item(user_id: str):
#     conn = sqlite3.connect('test.db')
#     c = conn.cursor()

#     # Safe from SQL Injection
#     c.execute("SELECT * FROM users WHERE id = ?", (user_id,))

#     user = c.fetchone()
#     return {"user": user}


@app.get("/items/")
async def read_item(user_id: str):
    conn = sqlite3.connect("test.db")
    c = conn.cursor()

    # Vulnerable to SQL Injection
    c.execute(f"SELECT * FROM users WHERE id = {user_id}")
    os.system(f"echo {user_id}")
    user = c.fetchone()
    return {"user": user}
