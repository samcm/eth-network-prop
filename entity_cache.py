#!/usr/bin/env python3
"""
Entity Cache Module using Streamlit caching
"""

import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import dotenv

# Load environment variables
dotenv.load_dotenv()


def get_database_connection():
    """Create database connection."""
    username = os.getenv('XATU_CLICKHOUSE_USERNAME')
    password = os.getenv('XATU_CLICKHOUSE_PASSWORD')
    host = os.getenv('XATU_CLICKHOUSE_HOST')
    
    if not all([username, password, host]):
        raise ValueError("Missing database credentials. Please check your .env file.")
        
    db_url = f"clickhouse+http://{username}:{password}@{host}:443/default?protocol=https"
    engine = create_engine(db_url)
    return engine.connect()


@st.cache_data(ttl=21600)  # 6 hours
def load_entity_data(network: str = 'mainnet'):
    """Load all entity data from database."""
    conn = get_database_connection()
    
    try:
        query = text("""
            SELECT 
                `index` as proposer_index,
                entity
            FROM ethseer_validator_entity
            WHERE 
                meta_network_name = :network
        """)
        
        df = pd.read_sql(query, conn, params={'network': network})
        
        # Convert to dictionary
        entity_dict = dict(zip(df['proposer_index'], df['entity']))
        
        return entity_dict
        
    finally:
        conn.close()


@st.cache_data(ttl=3600)  # 1 hour cache
def get_slot_proposers(slots, network: str = 'mainnet'):
    """Get proposer indices for a list of slots."""
    conn = get_database_connection()
    
    try:
        # Convert slots to string for query
        slots_str = ','.join(str(s) for s in slots)
        
        query = f"""
            SELECT 
                slot,
                proposer_index
            FROM canonical_beacon_block
            WHERE 
                slot IN ({slots_str})
                AND meta_network_name = '{network}'
        """
        
        df = pd.read_sql(query, conn)
        
        # Convert to dictionary
        slot_to_proposer = dict(zip(df['slot'], df['proposer_index']))
        
        return slot_to_proposer
        
    finally:
        conn.close()


def get_entities_for_slots(slots, network: str = 'mainnet'):
    """Get entities for a list of slots by doing the join in Python."""
    # Load all entities into memory
    all_entities = load_entity_data(network)
    
    # Get slot to proposer mapping
    slot_to_proposer = get_slot_proposers(slots, network)
    
    # Join in Python
    slot_to_entity = {}
    for slot, proposer_index in slot_to_proposer.items():
        if proposer_index in all_entities:
            slot_to_entity[slot] = all_entities[proposer_index]
    
    return slot_to_entity