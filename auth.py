import sqlite3
import hashlib
import os
import re
from datetime import datetime

DB_PATH = "users.db"

# ─────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    UNIQUE NOT NULL,
            email     TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            created_at TEXT   NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            symbol     TEXT    NOT NULL,
            searched_at TEXT   NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Password Hashing
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
def is_valid_email(email: str) -> bool:
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

def is_strong_password(password: str) -> bool:
    return len(password) >= 6


# ─────────────────────────────────────────────
# Register
# ─────────────────────────────────────────────
def register_user(username: str, email: str, password: str):
    """Returns (success: bool, message: str)"""
    if not username or not email or not password:
        return False, "All fields are required."
    if not is_valid_email(email):
        return False, "Invalid email address."
    if not is_strong_password(password):
        return False, "Password must be at least 6 characters."

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created successfully! Please log in."
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already taken."
        elif "email" in str(e):
            return False, "Email already registered."
        return False, "Registration failed."
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Login
# ─────────────────────────────────────────────
def login_user(username: str, password: str):
    """Returns (success: bool, message: str, user_dict or None)"""
    if not username or not password:
        return False, "Please enter username and password.", None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, username, email, created_at FROM users WHERE username=? AND password=?",
        (username.strip(), hash_password(password))
    )
    row = c.fetchone()
    conn.close()

    if row:
        user = {"id": row[0], "username": row[1], "email": row[2], "created_at": row[3]}
        return True, f"Welcome back, {row[1]}!", user
    return False, "Invalid username or password.", None


# ─────────────────────────────────────────────
# Search History
# ─────────────────────────────────────────────
def save_search(user_id: int, symbol: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO search_history (user_id, symbol, searched_at) VALUES (?, ?, ?)",
        (user_id, symbol.upper(), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_search_history(user_id: int, limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT symbol, searched_at FROM search_history WHERE user_id=? ORDER BY searched_at DESC LIMIT ?",
        (user_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{"symbol": r[0], "searched_at": r[1][:16].replace("T", " ")} for r in rows]