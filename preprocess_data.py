#!/usr/bin/env python3
"""
Preprocess and cache all data to Parquet files.
Run this script once to fetch all data from ClickHouse and save it locally.
"""

import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text
import dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_beacon_api_data(slots, batch_size=5000):
    """Process beacon API data in batches."""
    print(f"Processing beacon API data for {len(slots):,} slots...")
    
    conn = get_database_connection()
    all_data = []
    
    # Process in batches
    for i in range(0, len(slots), batch_size):
        batch_slots = slots[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(slots)-1)//batch_size + 1} ({len(batch_slots)} slots)...")
        
        # Convert slots to string for query
        slots_str = ','.join(str(s) for s in batch_slots)
        
        # Get min/max slots for date range
        min_slot = min(batch_slots)
        max_slot = max(batch_slots)
        
        # Calculate date range (beacon chain genesis: 2020-12-01 12:00:23 UTC)
        genesis_time = pd.Timestamp('2020-12-01 12:00:23', tz='UTC')
        min_time = genesis_time + pd.Timedelta(seconds=min_slot * 12)
        max_time = genesis_time + pd.Timedelta(seconds=(max_slot + 1) * 12)
        
        # Query 1: Get block arrival times
        query1 = f"""
        SELECT 
            slot,
            block,
            meta_client_name,
            min(propagation_slot_start_diff) as block_arrival_time_ms
        FROM beacon_api_eth_v1_events_block_gossip
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
            AND propagation_slot_start_diff >= 0
            AND propagation_slot_start_diff <= 12000
        GROUP BY slot, block, meta_client_name
        """
        
        # Query 1b: Get blob arrival times
        query1b = f"""
        SELECT 
            slot,
            meta_client_name,
            blob_index,
            min(propagation_slot_start_diff) as blob_arrival_time_ms
        FROM beacon_api_eth_v1_events_blob_sidecar
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
            AND propagation_slot_start_diff >= 0
            AND propagation_slot_start_diff <= 12000
        GROUP BY slot, meta_client_name, blob_index
        """
        
        # Query 2: Get block sizes
        query2 = f"""
        SELECT 
            slot,
            block_root,
            block_total_bytes_compressed / (1024.0 * 1024.0) as block_size_mb
        FROM canonical_beacon_block
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
        """
        
        # Query 3: Get blob counts
        query3 = f"""
        SELECT 
            slot,
            COUNT(DISTINCT blob_index) as blob_count
        FROM canonical_beacon_blob_sidecar
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
        GROUP BY slot
        """
        
        try:
            # Run all queries
            block_df = pd.read_sql(query1, conn)
            blob_arrivals_df = pd.read_sql(query1b, conn)
            size_df = pd.read_sql(query2, conn)
            blob_count_df = pd.read_sql(query3, conn)
            
            # Calculate data availability time for each client
            if not blob_arrivals_df.empty:
                max_blob_time = blob_arrivals_df.groupby(['slot', 'meta_client_name'])['blob_arrival_time_ms'].max().reset_index()
                max_blob_time.rename(columns={'blob_arrival_time_ms': 'max_blob_arrival_time_ms'}, inplace=True)
                
                # Merge with block arrival times
                data_availability = pd.merge(
                    block_df,
                    max_blob_time,
                    on=['slot', 'meta_client_name'],
                    how='left'
                )
                
                # Data is available when we have both block and all blobs
                data_availability['data_available_time_ms'] = data_availability[['block_arrival_time_ms', 'max_blob_arrival_time_ms']].max(axis=1)
            else:
                # No blobs, data is available when block arrives
                data_availability = block_df.copy()
                data_availability['data_available_time_ms'] = data_availability['block_arrival_time_ms']
            
            # Merge block and blob size data
            size_blob_df = pd.merge(
                size_df,
                blob_count_df,
                on='slot',
                how='left'
            )
            
            # Fill missing blob counts with 0
            size_blob_df['blob_count'] = size_blob_df['blob_count'].fillna(0)
            
            # Calculate total size (block + blobs)
            size_blob_df['blob_size_mb'] = size_blob_df['blob_count'] * 0.128
            size_blob_df['size_mb'] = size_blob_df['block_size_mb'] + size_blob_df['blob_size_mb']
            
            # Merge with data availability times
            df = pd.merge(
                data_availability,
                size_blob_df,
                left_on=['slot', 'block'],
                right_on=['slot', 'block_root'],
                how='inner'
            )
            
            # Rename for consistency
            df['propagation_time_ms'] = df['data_available_time_ms']
            
            batch_data = df[['slot', 'block', 'meta_client_name', 'propagation_time_ms', 
                           'size_mb', 'block_size_mb', 'blob_size_mb', 'blob_count']]
            
            # Filter out the problematic client with excessive outliers
            problematic_client = 'ethpandaops/mainnet/latitude-ash-mainnet-lighthouse-besu-001'
            records_before = len(batch_data)
            batch_data = batch_data[batch_data['meta_client_name'] != problematic_client]
            records_after = len(batch_data)
            if records_before > records_after:
                print(f"    Filtered out {records_before - records_after} records from problematic client")
            
            all_data.append(batch_data)
            
        except Exception as e:
            print(f"    Error processing batch: {e}")
    
    conn.close()
    
    # Combine all batches
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def process_gossipsub_data(slots, batch_size=5000):
    """Process gossipsub data in batches."""
    print(f"Processing gossipsub data for {len(slots):,} slots...")
    
    conn = get_database_connection()
    all_data = []
    
    # Process in batches
    for i in range(0, len(slots), batch_size):
        batch_slots = slots[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(slots)-1)//batch_size + 1} ({len(batch_slots)} slots)...")
        
        # Convert slots to string for query
        slots_str = ','.join(str(s) for s in batch_slots)
        
        # Get min/max slots for date range
        min_slot = min(batch_slots)
        max_slot = max(batch_slots)
        
        # Calculate date range
        genesis_time = pd.Timestamp('2020-12-01 12:00:23', tz='UTC')
        min_time = genesis_time + pd.Timedelta(seconds=min_slot * 12)
        max_time = genesis_time + pd.Timedelta(seconds=(max_slot + 1) * 12)
        
        # Similar queries as beacon API but for gossipsub tables
        query1 = f"""
        SELECT 
            slot,
            block,
            meta_client_name,
            min(propagation_slot_start_diff) as block_arrival_time_ms
        FROM libp2p_gossipsub_beacon_block
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
            AND propagation_slot_start_diff >= 0
            AND propagation_slot_start_diff <= 12000
        GROUP BY slot, block, meta_client_name
        """
        
        query1b = f"""
        SELECT 
            slot,
            meta_client_name,
            blob_index,
            min(propagation_slot_start_diff) as blob_arrival_time_ms
        FROM libp2p_gossipsub_blob_sidecar
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
            AND propagation_slot_start_diff >= 0
            AND propagation_slot_start_diff <= 12000
        GROUP BY slot, meta_client_name, blob_index
        """
        
        # Reuse size queries from beacon API processing
        query2 = f"""
        SELECT 
            slot,
            block_root,
            block_total_bytes_compressed / (1024.0 * 1024.0) as block_size_mb
        FROM canonical_beacon_block
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
        """
        
        query3 = f"""
        SELECT 
            slot,
            COUNT(DISTINCT blob_index) as blob_count
        FROM canonical_beacon_blob_sidecar
        WHERE slot_start_date_time >= '{min_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot_start_date_time < '{max_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND slot IN ({slots_str})
            AND meta_network_name = 'mainnet'
        GROUP BY slot
        """
        
        try:
            # Process similar to beacon API
            block_df = pd.read_sql(query1, conn)
            blob_arrivals_df = pd.read_sql(query1b, conn)
            size_df = pd.read_sql(query2, conn)
            blob_count_df = pd.read_sql(query3, conn)
            
            # Same processing logic as beacon API
            if not blob_arrivals_df.empty:
                max_blob_time = blob_arrivals_df.groupby(['slot', 'meta_client_name'])['blob_arrival_time_ms'].max().reset_index()
                max_blob_time.rename(columns={'blob_arrival_time_ms': 'max_blob_arrival_time_ms'}, inplace=True)
                
                data_availability = pd.merge(
                    block_df,
                    max_blob_time,
                    on=['slot', 'meta_client_name'],
                    how='left'
                )
                
                data_availability['data_available_time_ms'] = data_availability[['block_arrival_time_ms', 'max_blob_arrival_time_ms']].max(axis=1)
            else:
                data_availability = block_df.copy()
                data_availability['data_available_time_ms'] = data_availability['block_arrival_time_ms']
            
            size_blob_df = pd.merge(
                size_df,
                blob_count_df,
                on='slot',
                how='left'
            )
            
            size_blob_df['blob_count'] = size_blob_df['blob_count'].fillna(0)
            size_blob_df['blob_size_mb'] = size_blob_df['blob_count'] * 0.128
            size_blob_df['size_mb'] = size_blob_df['block_size_mb'] + size_blob_df['blob_size_mb']
            
            df = pd.merge(
                data_availability,
                size_blob_df,
                left_on=['slot', 'block'],
                right_on=['slot', 'block_root'],
                how='inner'
            )
            
            df['propagation_time_ms'] = df['data_available_time_ms']
            
            batch_data = df[['slot', 'block', 'meta_client_name', 'propagation_time_ms', 
                           'size_mb', 'block_size_mb', 'blob_size_mb', 'blob_count']]
            
            all_data.append(batch_data)
            
        except Exception as e:
            print(f"    Error processing batch: {e}")
    
    conn.close()
    
    # Combine all batches
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def process_entity_data():
    """Load all entity data."""
    print("Loading entity data...")
    
    conn = get_database_connection()
    
    try:
        # Load all entities
        query = text("""
            SELECT 
                `index` as proposer_index,
                entity
            FROM ethseer_validator_entity
            WHERE 
                meta_network_name = 'mainnet'
        """)
        
        entities_df = pd.read_sql(query, conn)
        print(f"  Loaded {len(entities_df):,} validator entities")
        
        return entities_df
        
    finally:
        conn.close()


def process_slot_proposers(slots, batch_size=10000):
    """Get proposer indices for slots."""
    print(f"Loading proposer data for {len(slots):,} slots...")
    
    conn = get_database_connection()
    all_data = []
    
    # Process in batches
    for i in range(0, len(slots), batch_size):
        batch_slots = slots[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(slots)-1)//batch_size + 1}...")
        
        slots_str = ','.join(str(s) for s in batch_slots)
        
        query = f"""
            SELECT 
                slot,
                proposer_index
            FROM canonical_beacon_block
            WHERE 
                slot IN ({slots_str})
                AND meta_network_name = 'mainnet'
        """
        
        try:
            batch_df = pd.read_sql(query, conn)
            all_data.append(batch_df)
        except Exception as e:
            print(f"    Error processing batch: {e}")
    
    conn.close()
    
    # Combine all batches
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def main():
    """Main preprocessing function."""
    print("Data Preprocessing Started")
    print("=" * 60)
    
    # Check for credentials
    if not all([os.getenv('XATU_CLICKHOUSE_USERNAME'), 
                os.getenv('XATU_CLICKHOUSE_PASSWORD'),
                os.getenv('XATU_CLICKHOUSE_HOST')]):
        print("ERROR: Missing database credentials in .env file")
        sys.exit(1)
    
    # Load ultrasound CSV
    csv_path = "ignore/ultrasound_payload_publish_time_07_21_17_32.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading ultrasound CSV: {csv_path}")
    ultrasound_df = pd.read_csv(csv_path)
    print(f"  Found {len(ultrasound_df):,} ultrasound blocks")
    
    # Save ultrasound data
    ultrasound_df.to_parquet("data/ultrasound_blocks.parquet", index=False)
    print("  Saved: data/ultrasound_blocks.parquet")
    
    # Get all slots
    slots = ultrasound_df['slot'].tolist()
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Process all data concurrently
    print("\n" + "=" * 60)
    print("Processing all data sources concurrently...")
    overall_start = time.time()
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_beacon_api_data, slots): 'beacon_api',
            executor.submit(process_gossipsub_data, slots): 'gossipsub',
            executor.submit(process_entity_data): 'entities',
            executor.submit(process_slot_proposers, slots): 'slot_proposers'
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                results[task_name] = result
                
                # Save results immediately
                if task_name == 'beacon_api':
                    if not result.empty:
                        result.to_parquet("data/beacon_api_data.parquet", index=False)
                        print(f"  ✓ Beacon API: {len(result):,} records (Note: Filtered problematic client)")
                    else:
                        print(f"  ✓ Beacon API: No data")
                elif task_name == 'gossipsub':
                    if not result.empty:
                        result.to_parquet("data/gossipsub_data.parquet", index=False)
                        print(f"  ✓ Gossipsub: {len(result):,} records")
                    else:
                        print(f"  ✓ Gossipsub: No data")
                elif task_name == 'entities':
                    result.to_parquet("data/entities.parquet", index=False)
                    print(f"  ✓ Entities: {len(result):,} records")
                elif task_name == 'slot_proposers':
                    if not result.empty:
                        result.to_parquet("data/slot_proposers.parquet", index=False)
                        print(f"  ✓ Slot Proposers: {len(result):,} records")
                    else:
                        print(f"  ✓ Slot Proposers: No data")
                        
            except Exception as e:
                print(f"  ✗ {task_name}: Error - {e}")
                results[task_name] = pd.DataFrame()
    
    print(f"\nTotal processing time: {time.time() - overall_start:.1f} seconds")
    
    # Extract dataframes from results
    beacon_data = results.get('beacon_api', pd.DataFrame())
    gossipsub_data = results.get('gossipsub', pd.DataFrame())
    entities_df = results.get('entities', pd.DataFrame())
    proposers_df = results.get('slot_proposers', pd.DataFrame())
    
    # Create metadata file
    metadata = {
        'created_at': datetime.now().isoformat(),
        'ultrasound_blocks': len(ultrasound_df),
        'beacon_api_records': len(beacon_data) if not beacon_data.empty else 0,
        'gossipsub_records': len(gossipsub_data) if not gossipsub_data.empty else 0,
        'entities': len(entities_df),
        'slot_proposers': len(proposers_df) if not proposers_df.empty else 0
    }
    
    pd.DataFrame([metadata]).to_parquet("data/metadata.parquet", index=False)
    
    print("\n" + "=" * 60)
    print("Data preprocessing complete!")
    print(f"All data saved to ./data/ directory")
    print("\nYou can now run the Streamlit app without database credentials.")


if __name__ == "__main__":
    main()