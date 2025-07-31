#!/usr/bin/env python3
"""
Network Bandwidth Analysis - Offline Version with Polars
Uses pre-processed Parquet files with Polars for fast processing.
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy import stats
import requests
import tempfile
import time
from dotenv import load_dotenv
from io import BytesIO
import sys
import gc
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Load environment variables from .env file
load_dotenv()

# Enable debug logging
print("[DEBUG] App starting...", file=sys.stderr)
print(f"[DEBUG] Python version: {sys.version}", file=sys.stderr)
print(f"[DEBUG] Streamlit version: {st.__version__}", file=sys.stderr)
print(f"[DEBUG] Working directory: {os.getcwd()}", file=sys.stderr)

# Memory usage helper
def log_memory_usage(label):
    if HAS_PSUTIL:
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"[DEBUG] Memory usage ({label}): {mem_info.rss / 1024 / 1024:.1f} MB", file=sys.stderr)
        except:
            pass

# Page config
st.set_page_config(
    page_title="Network Bandwidth Analysis",
    page_icon="📊",
    layout="wide"
)


@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def load_parquet_data():
    """Load all data from Parquet files using Polars.
    
    First checks for URL environment variables, then falls back to local files.
    Environment variables:
    - DATA_BEACON_URL: URL to beacon_api_data.parquet
    - DATA_GOSSIPSUB_URL: URL to gossipsub_data.parquet  
    - DATA_ULTRASOUND_URL: URL to ultrasound_blocks.parquet
    - DATA_ENTITIES_URL: URL to entities.parquet
    - DATA_PROPOSERS_URL: URL to slot_proposers.parquet
    - DATA_METADATA_URL: URL to metadata.parquet
    """
    data_dir = "data"
    
    def load_parquet_from_url_or_file(filename, env_var, use_polars=True, lazy=False):
        """Load parquet from URL if env var is set, otherwise from local file."""
        print(f"[DEBUG] Loading {filename}...", file=sys.stderr)
        
        # Try st.secrets first (for Streamlit Cloud), then fall back to os.getenv (for local)
        url = None
        try:
            if hasattr(st, 'secrets') and env_var in st.secrets:
                url = st.secrets[env_var]
                print(f"[DEBUG] Found {env_var} in st.secrets", file=sys.stderr)
        except Exception as e:
            print(f"[DEBUG] Error checking st.secrets: {e}", file=sys.stderr)
        
        if not url:
            url = os.getenv(env_var)
            if url:
                print(f"[DEBUG] Found {env_var} in environment", file=sys.stderr)
            else:
                print(f"[DEBUG] No URL found for {env_var}", file=sys.stderr)
        
        if url:
            # Use /tmp for Streamlit Cloud compatibility
            if os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
                # On Streamlit Cloud, use /tmp which is writable
                cache_dir = '/tmp/streamlit_data_cache'
            else:
                # For local development, use current directory
                cache_dir = os.path.join(os.getcwd(), '.streamlit_cache')
            
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, filename)
            
            # Check if file already exists in cache
            if os.path.exists(cache_path):
                # Check if file is less than 24 hours old
                file_age = time.time() - os.path.getmtime(cache_path)
                if file_age < 24 * 60 * 60:  # 24 hours in seconds
                    # Use cached file
                    if use_polars:
                        # Always read into memory first
                        df = pl.read_parquet(cache_path)
                        if lazy:
                            return df.lazy()
                        else:
                            return df
                    else:
                        return pd.read_parquet(cache_path)
            
            # Download from URL directly into memory
            try:
                print(f"[DEBUG] Downloading {filename} from URL...", file=sys.stderr)
                # Download entire file into memory
                response = requests.get(url)
                response.raise_for_status()
                print(f"[DEBUG] Downloaded {len(response.content):,} bytes", file=sys.stderr)
                
                # For large files, save to disk first then use scan_parquet
                if len(response.content) > 50_000_000:  # 50MB threshold
                    print(f"[DEBUG] Large file, saving to disk first", file=sys.stderr)
                    # Write directly to cache file
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Now use scan_parquet for lazy loading
                    if use_polars:
                        if lazy:
                            print(f"[DEBUG] Using scan_parquet for lazy loading", file=sys.stderr)
                            return pl.scan_parquet(cache_path)
                        else:
                            df = pl.read_parquet(cache_path)
                            print(f"[DEBUG] Loaded {len(df):,} rows", file=sys.stderr)
                            return df
                    else:
                        return pd.read_parquet(cache_path)
                else:
                    # Small files can be read from memory
                    data_bytes = BytesIO(response.content)
                    
                    if use_polars:
                        print(f"[DEBUG] Reading parquet with Polars...", file=sys.stderr)
                        df = pl.read_parquet(data_bytes)
                        print(f"[DEBUG] Loaded {len(df):,} rows", file=sys.stderr)
                        # Cache for next time
                        try:
                            df.write_parquet(cache_path)
                        except:
                            pass
                        
                        if lazy:
                            return df.lazy()
                        else:
                            return df
                    else:
                        df = pd.read_parquet(data_bytes)
                        try:
                            df.to_parquet(cache_path)
                        except:
                            pass
                        return df
                    
            except Exception as e:
                # If we have a URL but download failed, don't fall back to local
                print(f"[DEBUG] Download failed: {e}", file=sys.stderr)
                raise Exception(f"Failed to download {filename}: {str(e)}")
        
        # Load from local file
        local_path = f"{data_dir}/{filename}"
        print(f"[DEBUG] Checking for local file: {local_path}", file=sys.stderr)
        if os.path.exists(local_path):
            print(f"[DEBUG] Loading from local file", file=sys.stderr)
            if use_polars:
                if lazy:
                    return pl.scan_parquet(local_path)
                else:
                    return pl.read_parquet(local_path)
            else:
                return pd.read_parquet(local_path)
        else:
            # If neither URL nor local file exists, check if we can use sample data
            if not url and not os.path.exists(data_dir):
                st.error("Data not found. Please either:\n1. Set environment variables with data URLs\n2. Run `python preprocess_data.py` to generate local data")
                st.stop()
            raise FileNotFoundError(f"Data file not found: {filename}")
    
    # Check if we're using URLs or local files
    # Try st.secrets first, then os.getenv
    def has_secret(var):
        try:
            if hasattr(st, 'secrets') and var in st.secrets:
                return True
        except:
            pass
        return os.getenv(var) is not None
    
    using_urls = any(has_secret(var) for var in [
        "DATA_ULTRASOUND_URL", "DATA_BEACON_URL", "DATA_GOSSIPSUB_URL",
        "DATA_ENTITIES_URL", "DATA_PROPOSERS_URL", "DATA_METADATA_URL"
    ])
    
    if using_urls:
        # When using URLs, all must be set
        required_vars = [
            ("ultrasound_blocks.parquet", "DATA_ULTRASOUND_URL"),
            ("beacon_api_data.parquet", "DATA_BEACON_URL"),
            ("gossipsub_data.parquet", "DATA_GOSSIPSUB_URL"),
            ("entities.parquet", "DATA_ENTITIES_URL"),
            ("slot_proposers.parquet", "DATA_PROPOSERS_URL"),
            ("metadata.parquet", "DATA_METADATA_URL")
        ]
        
        missing = [name for name, var in required_vars if not has_secret(var)]
        if missing:
            st.error(f"Missing environment variables for: {', '.join(missing)}")
            st.stop()
    
    # Load all data files
    print("[DEBUG] Starting to load all data files...", file=sys.stderr)
    log_memory_usage("Before loading data")
    try:
        # Load data - use lazy loading for large datasets
        print("[DEBUG] Loading ultrasound data...", file=sys.stderr)
        ultrasound_df = load_parquet_from_url_or_file("ultrasound_blocks.parquet", "DATA_ULTRASOUND_URL")
        # Only keep the columns we need to reduce memory
        ultrasound_df = ultrasound_df.select(['slot', 'time_into_slot_before_publish_ms'])
        print(f"[DEBUG] Ultrasound data: {len(ultrasound_df):,} rows", file=sys.stderr)
        
        print("[DEBUG] Loading beacon data (lazy)...", file=sys.stderr)
        beacon_data = load_parquet_from_url_or_file("beacon_api_data.parquet", "DATA_BEACON_URL", lazy=True)  # Lazy load
        
        print("[DEBUG] Loading gossipsub data (lazy)...", file=sys.stderr)
        gossipsub_data = load_parquet_from_url_or_file("gossipsub_data.parquet", "DATA_GOSSIPSUB_URL", lazy=True)  # Lazy load
        
        print("[DEBUG] Loading entities data...", file=sys.stderr)
        entities_df = load_parquet_from_url_or_file("entities.parquet", "DATA_ENTITIES_URL")
        # Only keep necessary columns
        entities_df = entities_df.select(['proposer_index', 'entity'])
        print(f"[DEBUG] Entities data: {len(entities_df):,} rows", file=sys.stderr)
        
        print("[DEBUG] Loading proposers data...", file=sys.stderr)
        proposers_df = load_parquet_from_url_or_file("slot_proposers.parquet", "DATA_PROPOSERS_URL")
        # Only keep necessary columns
        proposers_df = proposers_df.select(['slot', 'proposer_index'])
        print(f"[DEBUG] Proposers data: {len(proposers_df):,} rows", file=sys.stderr)
        
        print("[DEBUG] Loading metadata...", file=sys.stderr)
        metadata = load_parquet_from_url_or_file("metadata.parquet", "DATA_METADATA_URL", use_polars=False)
        
        print("[DEBUG] All data loaded successfully", file=sys.stderr)
        log_memory_usage("After loading all data")
        
        # Force garbage collection to free up memory
        gc.collect()
        log_memory_usage("After garbage collection")
        
        # Return data - beacon and gossipsub should already be lazy frames
        print(f"[DEBUG] Returning data dictionary", file=sys.stderr)
        return {
            'ultrasound': ultrasound_df,
            'beacon': beacon_data,  # Already lazy
            'gossipsub': gossipsub_data,  # Already lazy
            'entities': entities_df,
            'proposers': proposers_df,
            'metadata': metadata
        }
    except Exception as e:
        print(f"[DEBUG] Exception in load_parquet_data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Re-raise with cleaner error message
        if "Failed to download" in str(e):
            # Extract just the filename from the error
            import re
            match = re.search(r'Failed to download ([^:]+):', str(e))
            if match:
                filename = match.group(1)
                raise Exception(f"Unable to download {filename}. Please check your internet connection and try again.")
        raise Exception(f"Error loading data: {str(e)}")


def get_entities_for_slots_polars(slots, entities_df, proposers_df):
    """Get entity mapping for slots using Polars."""
    # Filter proposers for requested slots
    slot_proposers = proposers_df.filter(pl.col('slot').is_in(slots))
    
    # Join with entities
    slot_entities = slot_proposers.join(
        entities_df,
        left_on='proposer_index',
        right_on='proposer_index',
        how='left'
    ).select(['slot', 'entity'])
    
    return slot_entities


@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def process_data_for_analysis(_beacon_lazy, _gossipsub_lazy, slots, _ultrasound_df, _entities_df, _proposers_df):
    """Process data efficiently using Polars lazy evaluation with chunking."""
    print(f"[DEBUG] process_data_for_analysis called with {len(slots):,} slots", file=sys.stderr)
    log_memory_usage("Start of process_data_for_analysis")
    
    # Get entities for slots
    slot_entities = get_entities_for_slots_polars(slots, _entities_df, _proposers_df)
    
    # Create ultrasound timing lookup - convert to lazy
    ultrasound_subset = _ultrasound_df.filter(pl.col('slot').is_in(slots)).select(['slot', 'time_into_slot_before_publish_ms']).lazy()
    slot_entities_lazy = slot_entities.lazy()
    
    # Process beacon data - filter for clients starting with 'pub'
    print(f"[DEBUG] Processing beacon data...", file=sys.stderr)
    # Process in smaller chunks to avoid memory issues
    chunk_size = 5000  # Smaller chunks for better memory management
    beacon_chunks = []
    
    for i in range(0, len(slots), chunk_size):
        chunk_slots = slots[i:i+chunk_size]
        print(f"[DEBUG] Processing beacon chunk {i//chunk_size + 1}/{(len(slots)-1)//chunk_size + 1}...", file=sys.stderr)
        
        chunk_data = (
            _beacon_lazy
            .filter(pl.col('slot').is_in(chunk_slots))
            .filter(pl.col('meta_client_name').str.starts_with('pub'))  # Only keep clients starting with 'pub'
            .join(ultrasound_subset, on='slot', how='left')
            .join(slot_entities_lazy, on='slot', how='left')
            .with_columns(
                (pl.col('propagation_time_ms') - pl.col('time_into_slot_before_publish_ms')).alias('adjusted_propagation_ms')
            )
            .filter(pl.col('adjusted_propagation_ms') >= 0)
            .with_columns(pl.col('adjusted_propagation_ms').alias('propagation_time_ms'))
            .select(['slot', 'meta_client_name', 'propagation_time_ms', 'size_mb', 'blob_count', 'entity'])  # Only keep needed columns
            .collect(streaming=True)  # Use streaming for better memory efficiency
        )
        beacon_chunks.append(chunk_data)
        
        # Write chunk to disk and keep only reference
        if os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
            chunk_path = f'/tmp/beacon_chunk_{i//chunk_size}.parquet'
            chunk_data.write_parquet(chunk_path)
            beacon_chunks[-1] = chunk_path  # Replace with path
        
        gc.collect()  # Force garbage collection after each chunk
        log_memory_usage(f"After beacon chunk {i//chunk_size + 1}")
    
    # Combine chunks - read from disk if needed
    if beacon_chunks and isinstance(beacon_chunks[0], str):
        # Read chunks from disk
        beacon_dfs = []
        for chunk_path in beacon_chunks:
            beacon_dfs.append(pl.read_parquet(chunk_path))
            os.unlink(chunk_path)  # Delete after reading
        beacon_processed = pl.concat(beacon_dfs) if beacon_dfs else pl.DataFrame()
        del beacon_dfs
    else:
        beacon_processed = pl.concat(beacon_chunks) if beacon_chunks else pl.DataFrame()
    
    del beacon_chunks  # Free memory
    gc.collect()
    
    print(f"[DEBUG] Beacon data processed: {len(beacon_processed):,} rows", file=sys.stderr)
    log_memory_usage("After processing beacon data")
    
    # Process gossipsub data similarly
    print(f"[DEBUG] Processing gossipsub data...", file=sys.stderr)
    gossipsub_chunks = []
    
    for i in range(0, len(slots), chunk_size):
        chunk_slots = slots[i:i+chunk_size]
        print(f"[DEBUG] Processing gossipsub chunk {i//chunk_size + 1}/{(len(slots)-1)//chunk_size + 1}...", file=sys.stderr)
        
        chunk_data = (
            _gossipsub_lazy
            .filter(pl.col('slot').is_in(chunk_slots))
            .join(ultrasound_subset, on='slot', how='left')
            .join(slot_entities_lazy, on='slot', how='left')
            .with_columns(
                (pl.col('propagation_time_ms') - pl.col('time_into_slot_before_publish_ms')).alias('adjusted_propagation_ms')
            )
            .filter(pl.col('adjusted_propagation_ms') >= 0)
            .with_columns(pl.col('adjusted_propagation_ms').alias('propagation_time_ms'))
            .select(['slot', 'meta_client_name', 'propagation_time_ms', 'size_mb', 'blob_count', 'entity'])  # Only keep needed columns
            .collect(streaming=True)  # Use streaming for better memory efficiency
        )
        gossipsub_chunks.append(chunk_data)
        
        # Write chunk to disk and keep only reference
        if os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
            chunk_path = f'/tmp/gossipsub_chunk_{i//chunk_size}.parquet'
            chunk_data.write_parquet(chunk_path)
            gossipsub_chunks[-1] = chunk_path  # Replace with path
        
        gc.collect()  # Force garbage collection after each chunk
        log_memory_usage(f"After gossipsub chunk {i//chunk_size + 1}")
    
    # Combine chunks - read from disk if needed
    if gossipsub_chunks and isinstance(gossipsub_chunks[0], str):
        # Read chunks from disk
        gossipsub_dfs = []
        for chunk_path in gossipsub_chunks:
            gossipsub_dfs.append(pl.read_parquet(chunk_path))
            os.unlink(chunk_path)  # Delete after reading
        gossipsub_processed = pl.concat(gossipsub_dfs) if gossipsub_dfs else pl.DataFrame()
        del gossipsub_dfs
    else:
        gossipsub_processed = pl.concat(gossipsub_chunks) if gossipsub_chunks else pl.DataFrame()
    
    del gossipsub_chunks  # Free memory
    gc.collect()
    
    print(f"[DEBUG] Gossipsub data processed: {len(gossipsub_processed):,} rows", file=sys.stderr)
    log_memory_usage("After processing gossipsub data")
    
    return beacon_processed, gossipsub_processed


def calculate_bandwidth_metrics_polars(df):
    """Calculate bandwidth metrics using Polars."""
    if df.height == 0:
        return pl.DataFrame()
    
    # Calculate percentiles per slot
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    metrics = (
        df.group_by('slot')
        .agg([
            pl.col('size_mb').first(),
            *[pl.col('propagation_time_ms').quantile(p).alias(f'p{int(p*100)}') for p in percentiles]
        ])
        .sort('slot')
    )
    
    return metrics


def create_cdf_chart_polars(data_pl, title_prefix):
    """Create a CDF chart for bucketed data by 100KB bins using Polars data."""
    fig = go.Figure()
    
    if data_pl.height == 0:
        return fig
    
    # Sample data if too large to avoid memory issues
    if data_pl.height > 2_000_000:  # If more than 2M rows
        print(f"[DEBUG] Sampling CDF data from {data_pl.height:,} to 2M rows", file=sys.stderr)
        data_pl = data_pl.sample(n=2_000_000, seed=42)
    
    # Calculate total unique contributors/nodes and slots
    # Check if this is beacon data (has contributor format) or libp2p data
    is_beacon_data = "Beacon" in title_prefix
    
    if 'meta_client_name' in data_pl.columns and is_beacon_data:
        # Extract contributor name from format: privacy/username/node-uuid
        contributors = data_pl.with_columns(
            pl.col('meta_client_name').str.split('/').list.get(1).alias('contributor')
        )
        total_contributors = contributors['contributor'].n_unique()
        count_label = "contributors"
    else:
        # For libp2p, just count nodes
        total_contributors = data_pl['meta_client_name'].n_unique() if 'meta_client_name' in data_pl.columns else 0
        count_label = "nodes"
    total_slots = data_pl['slot'].n_unique()
    
    # Create 100KB bins
    max_size = float(data_pl['size_mb'].max())
    max_size = np.ceil(max_size * 10) / 10  # Round up to nearest 0.1
    bin_edges = [round(x, 1) for x in np.arange(0, max_size + 0.1, 0.1)]  # 100KB = 0.1MB bins
    
    # Add size bins using Polars
    data_with_bins = data_pl.with_columns(
        pl.col('size_mb')
        .cut(bin_edges)
        .alias('size_bin')
    )
    
    # Create proper labels for the bins
    bin_labels = {f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} MB" 
                  for i in range(len(bin_edges)-1)}
    
    # Get unique bins and their stats
    bin_stats = (
        data_with_bins
        .filter(pl.col('size_bin').is_not_null())
        .group_by('size_bin')
        .agg([
            pl.len().alias('count'),
            pl.col('blob_count').mean().alias('avg_blob_count') if 'blob_count' in data_with_bins.columns else pl.lit(0).alias('avg_blob_count')
        ])
        .filter(pl.col('count') >= 10)  # Only bins with enough data
        .sort('size_bin')
    )
    
    # Create rainbow colors based on number of bins
    num_bins = bin_stats.height
    colors = []
    for i in range(num_bins):
        # Create rainbow colors from red to violet
        hue = i / max(num_bins - 1, 1) * 270  # 0 to 270 degrees (red to violet)
        # Convert HSL to RGB (using plotly's color utilities)
        colors.append(f'hsl({hue}, 70%, 50%)')
    
    # Process each bin
    for idx, row in enumerate(bin_stats.iter_rows(named=True)):
        size_bin_raw = row['size_bin']
        size_bin_str = str(size_bin_raw)
        size_bin = bin_labels.get(size_bin_str, size_bin_str)
        # Ensure MB is in the label
        if not size_bin.endswith(" MB"):
            # Extract numbers and format properly
            import re
            matches = re.findall(r'[\d.]+', size_bin)
            if len(matches) >= 2:
                size_bin = f"{float(matches[0]):.1f}-{float(matches[1]):.1f} MB"
            else:
                size_bin = size_bin + " MB"
        avg_blob_count = int(round(row['avg_blob_count']))
        
        # Get data for this bin
        bin_data_pl = (
            data_with_bins
            .filter(pl.col('size_bin') == size_bin_raw)
            .select(['propagation_time_ms'])
        )
        
        # Calculate percentiles at 1% intervals using Polars
        percentile_points = list(range(0, 101))  # 0%, 1%, 2%, ..., 100%
        percentile_values = []
        
        for p in percentile_points:
            val = bin_data_pl['propagation_time_ms'].quantile(p / 100.0)
            if val is not None:
                percentile_values.append(val / 1000.0)  # Convert to seconds
            else:
                percentile_values.append(0)
        
        # Count unique slots in this bin
        slot_count = data_with_bins.filter(pl.col('size_bin') == size_bin_raw)['slot'].n_unique()
        sample_count = bin_data_pl.height
        label = f'{size_bin} ({avg_blob_count} blobs avg, {slot_count} slots)'
        
        fig.add_trace(go.Scatter(
            x=percentile_values,
            y=percentile_points,
            mode='lines',
            name=label,
            line=dict(color=colors[idx], width=2),
            hovertemplate=f"Size: {size_bin}<br>" +
                         f"Avg blobs: {avg_blob_count}<br>" +
                         "Time: %{x:.2f} s<br>" +
                         "Percentile: %{y:.1f}%<br>" +
                         f"Samples: {sample_count}<br>" +
                         "<extra></extra>"
        ))
    
    
    fig.update_layout(
        title=f"{title_prefix} - CDF of Data Availability Times by 100KB Bins<br><sub>{total_contributors} {count_label}, {total_slots:,} slots</sub>",
        xaxis_title="Data Availability Time from Ultrasound Broadcast (seconds)",
        yaxis_title="Percentile (%)",
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False,
            ticks='outside',
            layer='above traces',
            range=[0, 3]  # Limit to 3 seconds
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False,
            ticks='outside',
            layer='above traces'
        ),
        images=[dict(
            source="https://ethpandaops.io/img/logo-slim.png",
            xref="paper", yref="paper",
            x=0.02, y=0.98,  # Top left position
            sizex=0.08, sizey=0.08,  # Small logo
            xanchor="left", yanchor="top"
        )]
    )
    
    return fig


def main():
    print("[DEBUG] main() started", file=sys.stderr)
    log_memory_usage("Start of main()")
    st.title("🌐 Network Data Analysis")
    st.markdown("This app analyzes when data becomes available (block + all blobs) across the Ethereum network across two different data sources.")
    st.markdown("Methodology: max(block_arrival_time, all_blob_arrival_time) for each node observing a slot.")
    st.warning("**Note:** This data doesn't cover the full picture. For example, each of these nodes will see the same block/blob multiple times via gossipsub.")
    
    # Load all data first
    try:
        print("[DEBUG] About to call load_parquet_data()", file=sys.stderr)
        with st.spinner("Downloading data..."):
            data = load_parquet_data()
        print("[DEBUG] Data loaded successfully", file=sys.stderr)
        log_memory_usage("After load_parquet_data()")
    except Exception as e:
        print(f"[DEBUG] Error in main: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        st.error(f"""### Error Loading Data
        
{str(e)}
        
If you're running locally, please ensure either:
1. Environment variables are set with valid data URLs
2. Run `python preprocess_data.py` to generate local data files
        """)
        st.stop()
    
    # Now show the content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Beacon Node Event Stream")
        st.markdown("""
        - **Data**: Block and blob arrival events from beacon nodes' event stream API
        - **Coverage**: Public contributors who send ethPandaOps data via Contributoor. Mostly home stakers.
        - **Caveats**: Subject to timing delays from the event stream API.
        """)
    
    with col2:
        st.markdown("#### Libp2p Direct Monitoring")
        st.markdown("""
        - **Data**: Direct p2p network monitoring of gossipsub messages. Custom libp2p client + a fork of Prysm.
        - **Coverage**: All instances run by ethPandaOps in datacenters.
        - **Caveats**: Custom libp2p client can have problems staying peered over long periods of time.
        """)
    
    
    # Display metadata
    metadata = data['metadata'].iloc[0]
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.info(f"""
    **Created:** {pd.to_datetime(metadata['created_at']).strftime('%Y-%m-%d %H:%M')}
    **Total blocks:** {metadata['ultrasound_blocks']:,}
    **Beacon API records:** {metadata['beacon_api_records']:,}
    **Gossipsub records:** {metadata['gossipsub_records']:,}
    """)
    
    # Load all blocks - no sampling
    ultrasound_sample = data['ultrasound']
    slots = ultrasound_sample['slot'].to_list()
    
    st.sidebar.info(f"Analyzing {len(slots):,} blocks from Ultrasound relay")
    
    # Option to focus on solo stakers
    focus_solo = st.sidebar.checkbox(
        "Focus on solo staker blocks",
        value=False,
        help="Pre-filter to only analyze blocks proposed by solo stakers"
    )
    
    # If focusing on solo stakers, pre-filter slots
    if focus_solo:
        with st.spinner("Identifying solo staker blocks..."):
            slot_entities = get_entities_for_slots_polars(slots, data['entities'], data['proposers'])
            solo_slots = slot_entities.filter(pl.col('entity') == 'solo_stakers')['slot'].to_list()
            
            if solo_slots:
                st.sidebar.success(f"Found {len(solo_slots)} solo staker blocks out of {len(slots)}")
                slots = solo_slots
                ultrasound_sample = ultrasound_sample.filter(pl.col('slot').is_in(slots))
            else:
                st.sidebar.warning("No solo staker blocks found in sample. Try a larger sample size.")
    
    # Process data efficiently with Polars
    log_memory_usage("Before processing data")
    print(f"[DEBUG] Processing {len(slots):,} slots", file=sys.stderr)
    with st.spinner(f"Processing {len(slots):,} blocks... This may take a minute on cloud deployment."):
        beacon_data, gossipsub_data = process_data_for_analysis(
            data['beacon'], 
            data['gossipsub'], 
            slots,
            data['ultrasound'],
            data['entities'],
            data['proposers']
        )
    
    if beacon_data.height == 0 and gossipsub_data.height == 0:
        st.error("No propagation data found for selected slots")
        return
    
    # Calculate metrics
    beacon_metrics = calculate_bandwidth_metrics_polars(beacon_data) if beacon_data.height > 0 else pl.DataFrame()
    gossipsub_metrics = calculate_bandwidth_metrics_polars(gossipsub_data) if gossipsub_data.height > 0 else pl.DataFrame()
    
    st.info("📍 All times are measured from when Ultrasound relay broadcasted the block (t=0).")
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Beacon Node Event Stream CDF", "📊 Libp2p Direct Monitoring CDF", "🔄 Combined CDF", "📦 Size Analysis"])
    
    with tab1:
        if beacon_data.height == 0:
            st.warning("No beacon node event stream data available")
        else:
            fig_beacon = create_cdf_chart_polars(beacon_data, "Beacon Node Event Stream")
            st.plotly_chart(fig_beacon, use_container_width=True)
    
    with tab2:
        if gossipsub_data.height == 0:
            st.warning("No libp2p direct monitoring data available")
        else:
            fig_gossip = create_cdf_chart_polars(gossipsub_data, "Libp2p Direct Monitoring")
            st.plotly_chart(fig_gossip, use_container_width=True)
    
    with tab3:
        fig_combined = go.Figure()
        
        # Calculate total contributors and slots for combined view
        if beacon_data.height > 0 and 'meta_client_name' in beacon_data.columns:
            beacon_contributors = beacon_data.with_columns(
                pl.col('meta_client_name').str.split('/').list.get(1).alias('contributor')
            )['contributor'].n_unique()
        else:
            beacon_contributors = 0
        beacon_slots = beacon_data['slot'].n_unique() if beacon_data.height > 0 else 0
        
        # For gossipsub, just count nodes (not contributors)
        gossip_nodes = gossipsub_data['meta_client_name'].n_unique() if gossipsub_data.height > 0 else 0
        gossip_slots = gossipsub_data['slot'].n_unique() if gossipsub_data.height > 0 else 0
        
        # Process both datasets for combined view
        max_size = max(
            float(beacon_data['size_mb'].max()) if beacon_data.height > 0 else 0,
            float(gossipsub_data['size_mb'].max()) if gossipsub_data.height > 0 else 0
        )
        max_size = np.ceil(max_size * 10) / 10  # Round up to nearest 0.1
        bin_edges = [round(x, 1) for x in np.arange(0, max_size + 0.1, 0.1)]
        bin_labels = {f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} MB" 
                      for i in range(len(bin_edges)-1)}
        
        # Add beacon data
        if beacon_data.height > 0:
            beacon_with_bins = beacon_data.with_columns(
                pl.col('size_mb')
                .cut(bin_edges)
                .alias('size_bin')
            )
            
            bin_stats = (
                beacon_with_bins
                .filter(pl.col('size_bin').is_not_null())
                .group_by('size_bin')
                .agg([
                    pl.len().alias('count'),
                    pl.col('blob_count').mean().alias('avg_blob_count') if 'blob_count' in beacon_with_bins.columns else pl.lit(0).alias('avg_blob_count')
                ])
                .filter(pl.col('count') >= 10)
                .sort('size_bin')
            )
            
            # Create rainbow colors for beacon data
            num_bins_beacon = bin_stats.height
            colors_beacon = []
            for i in range(num_bins_beacon):
                hue = i / max(num_bins_beacon - 1, 1) * 270  # 0 to 270 degrees (red to violet)
                colors_beacon.append(f'hsl({hue}, 70%, 50%)')
            
            for idx, row in enumerate(bin_stats.iter_rows(named=True)):
                size_bin_raw = row['size_bin']
                size_bin_str = str(size_bin_raw)
                size_bin = bin_labels.get(size_bin_str, size_bin_str)
                # Ensure MB is in the label
                if not size_bin.endswith(" MB"):
                    # Extract numbers and format properly
                    import re
                    matches = re.findall(r'[\d.]+', size_bin)
                    if len(matches) >= 2:
                        size_bin = f"{float(matches[0]):.1f}-{float(matches[1]):.1f} MB"
                    else:
                        size_bin = size_bin + " MB"
                avg_blob_count = int(round(row['avg_blob_count']))
                
                bin_data_pl = (
                    beacon_with_bins
                    .filter(pl.col('size_bin') == size_bin_raw)
                    .select(['propagation_time_ms'])
                )
                
                # Calculate percentiles at 1% intervals
                percentile_points = list(range(0, 101))
                percentile_values = []
                
                for p in percentile_points:
                    val = bin_data_pl['propagation_time_ms'].quantile(p / 100.0)
                    if val is not None:
                        percentile_values.append(val / 1000.0)  # Convert to seconds
                    else:
                        percentile_values.append(0)
                
                # Count unique slots in this bin
                slot_count = beacon_with_bins.filter(pl.col('size_bin') == size_bin_raw)['slot'].n_unique()
                sample_count = bin_data_pl.height
                
                fig_combined.add_trace(go.Scatter(
                    x=percentile_values,
                    y=percentile_points,
                    mode='lines',
                    name=f'Beacon {size_bin} ({avg_blob_count} blobs, {slot_count} slots)',
                    line=dict(dash='solid', color=colors_beacon[idx], width=2),
                    hovertemplate=f"Beacon Node Event Stream - {size_bin}<br>" +
                                 "Time: %{x:.2f} s<br>" +
                                 "Percentile: %{y:.1f}%<br>" +
                                 f"Samples: {sample_count}<br>" +
                                 "<extra></extra>"
                ))
        
        # Add gossipsub data
        if gossipsub_data.height > 0:
            gossip_with_bins = gossipsub_data.with_columns(
                pl.col('size_mb')
                .cut(bin_edges)
                .alias('size_bin')
            )
            
            bin_stats = (
                gossip_with_bins
                .filter(pl.col('size_bin').is_not_null())
                .group_by('size_bin')
                .agg([
                    pl.len().alias('count'),
                    pl.col('blob_count').mean().alias('avg_blob_count') if 'blob_count' in gossip_with_bins.columns else pl.lit(0).alias('avg_blob_count')
                ])
                .filter(pl.col('count') >= 10)
                .sort('size_bin')
            )
            
            # Create rainbow colors for gossipsub data
            num_bins_gossip = bin_stats.height
            colors_gossip = []
            for i in range(num_bins_gossip):
                hue = i / max(num_bins_gossip - 1, 1) * 270  # 0 to 270 degrees (red to violet)
                colors_gossip.append(f'hsl({hue}, 70%, 50%)')
            
            for idx, row in enumerate(bin_stats.iter_rows(named=True)):
                size_bin_raw = row['size_bin']
                size_bin_str = str(size_bin_raw)
                size_bin = bin_labels.get(size_bin_str, size_bin_str)
                # Ensure MB is in the label
                if not size_bin.endswith(" MB"):
                    # Extract numbers and format properly
                    import re
                    matches = re.findall(r'[\d.]+', size_bin)
                    if len(matches) >= 2:
                        size_bin = f"{float(matches[0]):.1f}-{float(matches[1]):.1f} MB"
                    else:
                        size_bin = size_bin + " MB"
                avg_blob_count = int(round(row['avg_blob_count']))
                
                bin_data_pl = (
                    gossip_with_bins
                    .filter(pl.col('size_bin') == size_bin_raw)
                    .select(['propagation_time_ms'])
                )
                
                # Calculate percentiles at 1% intervals
                percentile_points = list(range(0, 101))
                percentile_values = []
                
                for p in percentile_points:
                    val = bin_data_pl['propagation_time_ms'].quantile(p / 100.0)
                    if val is not None:
                        percentile_values.append(val / 1000.0)  # Convert to seconds
                    else:
                        percentile_values.append(0)
                
                # Count unique slots in this bin
                slot_count = gossip_with_bins.filter(pl.col('size_bin') == size_bin_raw)['slot'].n_unique()
                sample_count = bin_data_pl.height
                
                fig_combined.add_trace(go.Scatter(
                    x=percentile_values,
                    y=percentile_points,
                    mode='lines',
                    name=f'Gossip {size_bin} ({avg_blob_count} blobs, {slot_count} slots)',
                    line=dict(dash='dot', color=colors_gossip[idx], width=2),
                    hovertemplate=f"Libp2p Direct Monitoring - {size_bin}<br>" +
                                 "Time: %{x:.2f} s<br>" +
                                 "Percentile: %{y:.1f}%<br>" +
                                 f"Samples: {sample_count}<br>" +
                                 "<extra></extra>"
                ))
        
        fig_combined.update_layout(
            title=f"Combined CDF - Data Availability Times by 100KB Bins<br><sub>Beacon: {beacon_contributors} contributors, {beacon_slots:,} slots | Libp2p: {gossip_nodes} nodes, {gossip_slots:,} slots</sub>",
            xaxis_title="Data Availability Time from Ultrasound Broadcast (seconds)",
            yaxis_title="Percentile (%)",
            height=600,
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=False,
                ticks='outside',
                layer='above traces',
                range=[0, 3]  # Limit to 3 seconds
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=False,
                ticks='outside',
                layer='above traces'
            ),
            images=[dict(
                source="https://ethpandaops.io/img/logo-slim.png",
                xref="paper", yref="paper",
                x=0.02, y=0.98,  # Top left position
                sizex=0.08, sizey=0.08,  # Small logo
                xanchor="left", yanchor="top"
            )]
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
    
    with tab4:
        # Check if we're focusing on solo stakers
        is_solo_focused = focus_solo
        
        # Create bins for size analysis
        max_size = max(
            float(beacon_data['size_mb'].max()) if beacon_data.height > 0 else 0,
            float(gossipsub_data['size_mb'].max()) if gossipsub_data.height > 0 else 0
        )
        max_size = np.ceil(max_size * 10) / 10  # Round up to nearest 0.1
        bin_edges = [round(x, 1) for x in np.arange(0, max_size + 0.1, 0.1)]
        bin_labels = {f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} MB" 
                      for i in range(len(bin_edges)-1)}
        
        # Beacon API box plots
        if beacon_data.height > 0:
            # Calculate contributor and slot counts for beacon data
            if 'meta_client_name' in beacon_data.columns:
                beacon_box_contributors = beacon_data.with_columns(
                    pl.col('meta_client_name').str.split('/').list.get(1).alias('contributor')
                )['contributor'].n_unique()
            else:
                beacon_box_contributors = 0
            beacon_box_slots = beacon_data['slot'].n_unique()
            
            # Create box plot data
            beacon_with_bins = beacon_data.with_columns(
                pl.col('size_mb')
                .cut(bin_edges)
                .alias('size_bin')
            )
            
            fig_beacon_box = go.Figure()
            
            bin_stats = (
                beacon_with_bins
                .filter(pl.col('size_bin').is_not_null())
                .group_by('size_bin')
                .agg(pl.len().alias('count'))
                .filter(pl.col('count') >= 5)
                .sort('size_bin')
            )
            
            # Create rainbow colors for box plots
            num_bins = bin_stats.height
            box_colors = []
            for i in range(num_bins):
                hue = i / max(num_bins - 1, 1) * 270  # 0 to 270 degrees (red to violet)
                box_colors.append(f'hsl({hue}, 70%, 50%)')
            
            # Collect statistics for table
            beacon_table_data = []
            
            for idx, row in enumerate(bin_stats.iter_rows(named=True)):
                size_bin_raw = row['size_bin']
                size_bin_str = str(size_bin_raw)
                size_bin = bin_labels.get(size_bin_str, size_bin_str)
                # Ensure MB is in the label
                if not size_bin.endswith(" MB"):
                    # Extract numbers and format properly
                    import re
                    matches = re.findall(r'[\d.]+', size_bin)
                    if len(matches) >= 2:
                        size_bin = f"{float(matches[0]):.1f}-{float(matches[1]):.1f} MB"
                    else:
                        size_bin = size_bin + " MB"
                
                # Calculate box plot statistics using Polars
                bin_data_pl = (
                    beacon_with_bins
                    .filter(pl.col('size_bin') == size_bin_raw)
                    .select('propagation_time_ms')
                )
                
                # Calculate all statistics we need
                stats = bin_data_pl.select([
                    pl.col('propagation_time_ms').min().alias('min'),
                    pl.col('propagation_time_ms').quantile(0.25).alias('q1'),
                    pl.col('propagation_time_ms').quantile(0.50).alias('median'),
                    pl.col('propagation_time_ms').quantile(0.75).alias('q3'),
                    pl.col('propagation_time_ms').max().alias('max'),
                    pl.col('propagation_time_ms').count().alias('count')
                ]).row(0, named=True)
                
                # Convert to seconds
                min_val = stats['min'] / 1000
                q1 = stats['q1'] / 1000
                median = stats['median'] / 1000
                q3 = stats['q3'] / 1000
                max_val = stats['max'] / 1000
                
                # Count unique slots in this bin
                slot_count = beacon_with_bins.filter(pl.col('size_bin') == size_bin_raw)['slot'].n_unique()
                
                # Create a minimal dataset that produces the correct box plot
                # We need exactly 5 values: min, q1, median, q3, max
                y_values = [min_val, q1, median, q3, max_val]
                
                fig_beacon_box.add_trace(go.Box(
                    y=y_values,
                    name=f"{size_bin} ({slot_count} slots)",
                    boxpoints=False,  # Don't show individual points
                    marker=dict(color=box_colors[idx], size=4),
                    line=dict(color=box_colors[idx], width=1)
                ))
                
                # Collect data for table
                beacon_table_data.append({
                    'Size': size_bin,
                    'Slots': slot_count,
                    'Count': stats['count'],
                    'Min': f"{min_val:.3f}s",
                    'Q1': f"{q1:.3f}s",
                    'Median': f"{median:.3f}s",
                    'Q3': f"{q3:.3f}s",
                    'Max': f"{max_val:.3f}s"
                })
            
            fig_beacon_box.update_layout(
                title=f"Beacon Node Event Stream - {'Solo Stakers' if is_solo_focused else 'All Proposers'} - Data Availability Time Distribution by 100KB Bins<br><sub>{beacon_box_contributors} contributors, {beacon_box_slots:,} slots</sub>",
                xaxis_title="Combined Size (Block + Blobs)",
                yaxis_title="Data Availability Time (seconds)",
                height=500,
                showlegend=False,
                xaxis=dict(
                    tickangle=-45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=False,
                    ticks='outside',
                    layer='above traces'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=False,
                    ticks='outside',
                    layer='above traces',
                    range=[0, 10]  # Limit to 10 seconds
                ),
                images=[dict(
                    source="https://ethpandaops.io/img/logo-slim.png",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,  # Top left position
                    sizex=0.08, sizey=0.08,  # Small logo
                    xanchor="left", yanchor="top"
                )]
            )
            
            st.plotly_chart(fig_beacon_box, use_container_width=True)
            
            # Display statistics table
            if beacon_table_data:
                df_stats = pd.DataFrame(beacon_table_data)
                # Calculate height based on number of rows (35px per row + header)
                table_height = (len(beacon_table_data) + 1) * 35 + 20
                st.dataframe(df_stats, hide_index=True, use_container_width=True, height=table_height)
        
        # Gossipsub box plots
        if gossipsub_data.height > 0:
            # For gossipsub data, just count unique nodes (not contributors)
            gossip_box_nodes = gossipsub_data['meta_client_name'].n_unique()
            gossip_box_slots = gossipsub_data['slot'].n_unique()
            
            # Create box plot data
            gossip_with_bins = gossipsub_data.with_columns(
                pl.col('size_mb')
                .cut(bin_edges)
                .alias('size_bin')
            )
            
            fig_gossip_box = go.Figure()
            
            bin_stats = (
                gossip_with_bins
                .filter(pl.col('size_bin').is_not_null())
                .group_by('size_bin')
                .agg(pl.len().alias('count'))
                .filter(pl.col('count') >= 5)
                .sort('size_bin')
            )
            
            # Create rainbow colors for box plots
            num_bins = bin_stats.height
            box_colors = []
            for i in range(num_bins):
                hue = i / max(num_bins - 1, 1) * 270  # 0 to 270 degrees (red to violet)
                box_colors.append(f'hsl({hue}, 70%, 50%)')
            
            # Collect statistics for table
            gossip_table_data = []
            
            for idx, row in enumerate(bin_stats.iter_rows(named=True)):
                size_bin_raw = row['size_bin']
                size_bin_str = str(size_bin_raw)
                size_bin = bin_labels.get(size_bin_str, size_bin_str)
                # Ensure MB is in the label
                if not size_bin.endswith(" MB"):
                    # Extract numbers and format properly
                    import re
                    matches = re.findall(r'[\d.]+', size_bin)
                    if len(matches) >= 2:
                        size_bin = f"{float(matches[0]):.1f}-{float(matches[1]):.1f} MB"
                    else:
                        size_bin = size_bin + " MB"
                
                # Calculate box plot statistics using Polars
                bin_data_pl = (
                    gossip_with_bins
                    .filter(pl.col('size_bin') == size_bin_raw)
                    .select('propagation_time_ms')
                )
                
                # Calculate all statistics we need
                stats = bin_data_pl.select([
                    pl.col('propagation_time_ms').min().alias('min'),
                    pl.col('propagation_time_ms').quantile(0.25).alias('q1'),
                    pl.col('propagation_time_ms').quantile(0.50).alias('median'),
                    pl.col('propagation_time_ms').quantile(0.75).alias('q3'),
                    pl.col('propagation_time_ms').max().alias('max'),
                    pl.col('propagation_time_ms').count().alias('count')
                ]).row(0, named=True)
                
                # Convert to seconds
                min_val = stats['min'] / 1000
                q1 = stats['q1'] / 1000
                median = stats['median'] / 1000
                q3 = stats['q3'] / 1000
                max_val = stats['max'] / 1000
                
                # Count unique slots in this bin
                slot_count = gossip_with_bins.filter(pl.col('size_bin') == size_bin_raw)['slot'].n_unique()
                
                # Create a minimal dataset that produces the correct box plot
                # We need exactly 5 values: min, q1, median, q3, max
                y_values = [min_val, q1, median, q3, max_val]
                
                fig_gossip_box.add_trace(go.Box(
                    y=y_values,
                    name=f"{size_bin} ({slot_count} slots)",
                    boxpoints=False,  # Don't show individual points
                    marker=dict(color=box_colors[idx], size=4),
                    line=dict(color=box_colors[idx], width=1)
                ))
                
                # Collect data for table
                gossip_table_data.append({
                    'Size': size_bin,
                    'Slots': slot_count,
                    'Count': stats['count'],
                    'Min': f"{min_val:.3f}s",
                    'Q1': f"{q1:.3f}s",
                    'Median': f"{median:.3f}s",
                    'Q3': f"{q3:.3f}s",
                    'Max': f"{max_val:.3f}s"
                })
            
            fig_gossip_box.update_layout(
                title=f"Libp2p Direct Monitoring - {'Solo Stakers' if is_solo_focused else 'All Proposers'} - Data Availability Time Distribution by 100KB Bins<br><sub>{gossip_box_nodes} nodes, {gossip_box_slots:,} slots</sub>",
                xaxis_title="Combined Size (Block + Blobs)",
                yaxis_title="Data Availability Time (seconds)",
                height=500,
                showlegend=False,
                xaxis=dict(
                    tickangle=-45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=False,
                    ticks='outside',
                    layer='above traces'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=False,
                    ticks='outside',
                    layer='above traces',
                    range=[0, 10]  # Limit to 10 seconds
                ),
                images=[dict(
                    source="https://ethpandaops.io/img/logo-slim.png",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,  # Top left position
                    sizex=0.08, sizey=0.08,  # Small logo
                    xanchor="left", yanchor="top"
                )]
            )
            
            st.plotly_chart(fig_gossip_box, use_container_width=True)
            
            # Display statistics table
            if gossip_table_data:
                df_stats = pd.DataFrame(gossip_table_data)
                # Calculate height based on number of rows (35px per row + header)
                table_height = (len(gossip_table_data) + 1) * 35 + 20
                st.dataframe(df_stats, hide_index=True, use_container_width=True, height=table_height)
    


if __name__ == "__main__":
    main()