import os
import json
import bcrypt
import re
import uuid
from datetime import datetime, timedelta

# Constants
USER_DB = "users.json"
SESSIONS_DB = "sessions.json"
MIN_PASSWORD_LENGTH = 8

# Password validation rules
def validate_password(password):
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

# User management functions
def load_db(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_db(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def create_user(email, password, role="medical_pro"):
    users = load_db(USER_DB)
    
    if email in users:
        return False, "Email already registered"
    
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format"
    
    is_valid, msg = validate_password(password)
    if not is_valid:
        return False, msg
    
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    users[email] = {
        "password": hashed,
        "role": role,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "metadata": {}
    }
    
    save_db(users, USER_DB)
    return True, "User created successfully"

def verify_user(email):
    users = load_db(USER_DB)
    if email not in users:
        return False, "User not found"
    
    users[email]["email_verified"] = True
    save_db(users, USER_DB)
    return True, "Email verified successfully"

def authenticate_user(email, password):
    users = load_db(USER_DB)
    if email not in users:
        return False, "Invalid credentials", None
    
    user = users[email]
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return False, "Invalid credentials", None
    
    # Update last login
    user["last_login"] = datetime.now().isoformat()
    save_db(users, USER_DB)
    
    return True, "Login successful", user["role"]

def create_session(email):
    sessions = load_db(SESSIONS_DB)
    session_id = str(uuid.uuid4())
    expiry = datetime.now() + timedelta(hours=24)
    
    sessions[session_id] = {
        "email": email,
        "expiry": expiry.isoformat()
    }
    
    save_db(sessions, SESSIONS_DB)
    return session_id

def validate_session(session_id):
    sessions = load_db(SESSIONS_DB)
    if session_id not in sessions:
        return False, None
    
    session = sessions[session_id]
    if datetime.fromisoformat(session["expiry"]) < datetime.now():
        del sessions[session_id]
        save_db(sessions, SESSIONS_DB)
        return False, None
    
    return True, session["email"]

def delete_session(session_id):
    sessions = load_db(SESSIONS_DB)
    if session_id in sessions:
        del sessions[session_id]
        save_db(sessions, SESSIONS_DB)
        return True
    return False