import mysql.connector
import streamlit as st;
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_connection():
    connection = mysql.connector.connect(
        host=st.secrets["DB_HOST"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_NAME"],
        port=st.secrets["DB_PORT"]
    )
    return connection