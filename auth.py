
import sqlite3
import hashlib
import secrets
import hmac
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = BASE_DIR / "users.db"
USER_LOG = BASE_DIR / "registered_users.txt"


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ==========================================
# CREATE DATABASE AND TABLES
# ==========================================

def create_database():

    with get_connection() as conn:

        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                registered_at TEXT
            )
        """)

        # Check existing users table columns
        user_columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        ]

        # Add registered_at to older databases
        if "registered_at" not in user_columns:
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN registered_at TEXT
            """)

        # Fill missing registration dates
        conn.execute("""
            UPDATE users
            SET registered_at = ?
            WHERE registered_at IS NULL
               OR registered_at = ''
        """, (
            datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
        ))

        # Prediction history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                area REAL NOT NULL,
                bedrooms INTEGER NOT NULL,
                bathrooms INTEGER NOT NULL,
                predicted_price REAL NOT NULL,
                prediction_date TEXT NOT NULL,
                user_id INTEGER
            )
        """)

        # Check prediction history columns
        history_columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(prediction_history)"
            ).fetchall()
        ]

        # Add user_id to older prediction history table
        if "user_id" not in history_columns:
            conn.execute("""
                ALTER TABLE prediction_history
                ADD COLUMN user_id INTEGER
            """)

        # Link existing predictions to their users
        conn.execute("""
            UPDATE prediction_history
            SET user_id = (
                SELECT id
                FROM users
                WHERE LOWER(users.email) =
                      LOWER(prediction_history.user_email)
            )
            WHERE user_id IS NULL
        """)

        # User profiles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                full_name TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                bio TEXT DEFAULT '',
                profile_image TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Create empty profiles for existing users
        conn.execute("""
            INSERT OR IGNORE INTO user_profiles (
                user_id,
                full_name,
                phone,
                bio,
                profile_image
            )
            SELECT id, '', '', '', ''
            FROM users
        """)

        # Index for faster user-specific history
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_user_id
            ON prediction_history(user_id)
        """)

        # Remember Me login sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        # Index for faster session verification
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_token_hash
            ON login_sessions(token_hash)
        """)


# ==========================================
# PASSWORD HASHING
# ==========================================

def hash_password(password, salt):

    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000
    ).hex()


# ==========================================
# USER REGISTRATION
# ==========================================

def register_user(email, password):

    email = email.strip().lower()

    salt = secrets.token_hex(16)

    password_hash = hash_password(password, salt)

    registered_at = datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )

    try:

        with get_connection() as conn:

            cursor = conn.execute("""
                INSERT INTO users
                (email, password_hash, salt, registered_at)
                VALUES (?, ?, ?, ?)
            """, (
                email,
                password_hash,
                salt,
                registered_at
            ))

            user_id = cursor.lastrowid

            # Create an empty profile for the new user
            conn.execute("""
                INSERT OR IGNORE INTO user_profiles
                (user_id, full_name, phone, bio, profile_image)
                VALUES (?, '', '', '', '')
            """, (user_id,))

        # Save registration details in text file
        with open(USER_LOG, "a", encoding="utf-8") as file:

            file.write(
                f"User ID: {user_id} | "
                f"Email: {email} | "
                f"Registered: {registered_at}\n"
            )

        return True

    except sqlite3.IntegrityError:

        return False


# ==========================================
# USER LOGIN
# ==========================================

def login_user(email, password):

    email = email.strip().lower()

    with get_connection() as conn:

        user = conn.execute("""
            SELECT password_hash, salt
            FROM users
            WHERE LOWER(email) = LOWER(?)
        """, (email,)).fetchone()

    if user is None:
        return False

    saved_hash, salt = user

    entered_hash = hash_password(password, salt)

    return hmac.compare_digest(
        saved_hash,
        entered_hash
    )


# ==========================================
# GET UNIQUE USER ID
# ==========================================

def get_user_id(email):

    email = email.strip().lower()

    with get_connection() as conn:

        user = conn.execute("""
            SELECT id
            FROM users
            WHERE LOWER(email) = LOWER(?)
        """, (email,)).fetchone()

    return user[0] if user else None


# ==========================================
# GET ALL REGISTERED USERS
# ==========================================

def get_registered_users():

    with get_connection() as conn:

        users = conn.execute("""
            SELECT id, email, registered_at
            FROM users
            ORDER BY id DESC
        """).fetchall()

    return users


# ==========================================
# GET USER PROFILE
# ==========================================

def get_user_profile(user_id):

    with get_connection() as conn:

        profile = conn.execute("""
            SELECT
                full_name,
                phone,
                bio,
                profile_image
            FROM user_profiles
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    return profile


# ==========================================
# SAVE / UPDATE USER PROFILE
# ==========================================

def save_user_profile(
    user_id,
    full_name,
    phone,
    bio,
    profile_image=""
):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO user_profiles
            (
                user_id,
                full_name,
                phone,
                bio,
                profile_image
            )
            VALUES (?, ?, ?, ?, ?)

            ON CONFLICT(user_id) DO UPDATE SET
                full_name = excluded.full_name,
                phone = excluded.phone,
                bio = excluded.bio,
                profile_image = CASE
                    WHEN excluded.profile_image = ''
                    THEN user_profiles.profile_image
                    ELSE excluded.profile_image
                END
        """, (
            user_id,
            full_name,
            phone,
            bio,
            profile_image
        ))


# ==========================================
# DELETE / CLEAR PROFILE PHOTO
# ==========================================

def clear_profile_image(user_id):

    with get_connection() as conn:

        profile = conn.execute("""
            SELECT profile_image
            FROM user_profiles
            WHERE user_id = ?
        """, (user_id,)).fetchone()

        old_image_path = profile[0] if profile else ""

        conn.execute("""
            UPDATE user_profiles
            SET profile_image = ''
            WHERE user_id = ?
        """, (user_id,))

    # Return old image path so app.py can safely delete the file
    return old_image_path or ""


# ==========================================
# SAVE HOUSE PRICE PREDICTION
# ==========================================

def save_prediction(
    user_email,
    area,
    bedrooms,
    bathrooms,
    predicted_price
):

    email = user_email.strip().lower()

    user_id = get_user_id(email)

    if user_id is None:
        raise ValueError("User not found")

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO prediction_history
            (
                user_email,
                user_id,
                area,
                bedrooms,
                bathrooms,
                predicted_price,
                prediction_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            email,
            user_id,
            area,
            bedrooms,
            bathrooms,
            predicted_price,
            datetime.now().strftime("%d-%m-%Y %I:%M %p")
        ))


# ==========================================
# GET USER PREDICTION HISTORY
# ==========================================

def get_prediction_history(user_email):

    email = user_email.strip().lower()

    user_id = get_user_id(email)

    if user_id is None:
        return []

    with get_connection() as conn:

        history = conn.execute("""
            SELECT
                area,
                bedrooms,
                bathrooms,
                predicted_price,
                prediction_date
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()

    return history


# ==========================================
# CREATE REMEMBER ME LOGIN SESSION
# ==========================================

def create_login_session(user_id, days=30):

    # Generate a secure random token
    token = secrets.token_urlsafe(32)

    # Store only token hash in database
    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(days=days)

    with get_connection() as conn:

        # Remove expired sessions
        conn.execute("""
            DELETE FROM login_sessions
            WHERE expires_at <= ?
        """, (now.isoformat(),))

        # Save new login session
        conn.execute("""
            INSERT INTO login_sessions
            (
                user_id,
                token_hash,
                created_at,
                expires_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            user_id,
            token_hash,
            now.isoformat(),
            expires_at.isoformat()
        ))

    # Return original token for browser cookie
    return token


# ==========================================
# VERIFY REMEMBER ME LOGIN TOKEN
# ==========================================

def verify_login_token(token):

    if not token or not isinstance(token, str):
        return None

    # Hash browser token
    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc)

    with get_connection() as conn:

        session = conn.execute("""
            SELECT
                login_sessions.user_id,
                users.email,
                login_sessions.expires_at
            FROM login_sessions

            JOIN users
            ON users.id = login_sessions.user_id

            WHERE login_sessions.token_hash = ?
        """, (token_hash,)).fetchone()

        if session is None:
            return None

        user_id, email, expires_at = session

        try:

            expiry_time = datetime.fromisoformat(expires_at)

            if expiry_time.tzinfo is None:
                expiry_time = expiry_time.replace(
                    tzinfo=timezone.utc
                )

        except (TypeError, ValueError):

            conn.execute("""
                DELETE FROM login_sessions
                WHERE token_hash = ?
            """, (token_hash,))

            return None

        # Check session expiry
        if expiry_time <= now:

            conn.execute("""
                DELETE FROM login_sessions
                WHERE token_hash = ?
            """, (token_hash,))

            return None

    # Return user ID and email
    return user_id, email


# ==========================================
# REVOKE REMEMBER ME LOGIN SESSION
# ==========================================

def revoke_login_session(token):

    if not token or not isinstance(token, str):
        return

    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    with get_connection() as conn:

        conn.execute("""
            DELETE FROM login_sessions
            WHERE token_hash = ?
        """, (token_hash,))


# ==========================================
# INITIALIZE DATABASE
# ==========================================

create_database()
