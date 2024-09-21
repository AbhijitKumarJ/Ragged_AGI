import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

# Set up SQLite database
SQLITE_DB_PATH = "app_data.db"

@st.cache_resource
def init_db():
    """
    Initialize the SQLite database and create necessary tables if they don't exist.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_actions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  action_type TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS indexed_files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_name TEXT,
                  file_type TEXT,
                  file_size INTEGER,
                  indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query_text TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def log_user_action(action_type):
    """
    Log a user action in the database.
    
    Args:
    action_type (str): The type of action performed by the user.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO user_actions (action_type) VALUES (?)", (action_type,))
    conn.commit()
    conn.close()

def log_indexed_file(file_name, file_type, file_size):
    """
    Log information about an indexed file in the database.
    
    Args:
    file_name (str): Name of the indexed file.
    file_type (str): Type of the indexed file.
    file_size (int): Size of the indexed file in bytes.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO indexed_files (file_name, file_type, file_size) VALUES (?, ?, ?)",
              (file_name, file_type, file_size))
    conn.commit()
    conn.close()

def log_query(query_text):
    """
    Log a user query in the database.
    
    Args:
    query_text (str): The text of the query submitted by the user.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO queries (query_text) VALUES (?)", (query_text,))
    conn.commit()
    conn.close()

@st.cache_data
def visualize_user_actions():
    """
    Create a visualization of user actions.
    
    Returns:
    plotly.graph_objs._figure.Figure: A bar chart of user actions.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT action_type, COUNT(*) as count FROM user_actions GROUP BY action_type", conn)
    conn.close()

    fig = px.bar(df, x='action_type', y='count', title='User Actions')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

@st.cache_data
def visualize_indexed_files():
    """
    Create a visualization of indexed files.
    
    Returns:
    plotly.graph_objs._figure.Figure: A pie chart of indexed files by type.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT file_type, COUNT(*) as count FROM indexed_files GROUP BY file_type", conn)
    conn.close()

    fig = px.pie(df, values='count', names='file_type', title='Indexed Files by Type')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

@st.cache_data
def visualize_queries():
    """
    Create a visualization of the most common queries.
    
    Returns:
    plotly.graph_objs._figure.Figure: A bar chart of the top 10 most common queries.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT query_text, COUNT(*) as count FROM queries GROUP BY query_text ORDER BY count DESC LIMIT 10", conn)
    conn.close()

    fig = px.bar(df, x='query_text', y='count', title='Top 10 Most Common Queries')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig
