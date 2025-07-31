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

# Load environment variables from .env file
load_dotenv()

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
        url = os.getenv(env_var)
        
        if url:
            # Create a cache directory
            cache_dir = os.path.join(tempfile.gettempdir(), 'streamlit_data_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, filename)
            
            # Check if file already exists in cache
            if os.path.exists(cache_path):
                # Check if file is less than 24 hours old
                file_age = time.time() - os.path.getmtime(cache_path)
                if file_age < 24 * 60 * 60:  # 24 hours in seconds
                    # Use cached file
                    if use_polars:
                        if lazy:
                            return pl.scan_parquet(cache_path)
                        else:
                            return pl.read_parquet(cache_path)
                    else:
                        return pd.read_parquet(cache_path)
            
            # Download from URL
            try:
                with st.spinner(f"Downloading {filename}..."):
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Write to cache file
                    with open(cache_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Read from cache file
                    if use_polars:
                        if lazy:
                            df = pl.scan_parquet(cache_path)
                        else:
                            df = pl.read_parquet(cache_path)
                    else:
                        df = pd.read_parquet(cache_path)
                    
                    return df
                    
            except Exception as e:
                st.error(f"Failed to download {filename} from {url}: {e}")
                # Fall back to local file
        
        # Load from local file
        local_path = f"{data_dir}/{filename}"
        if os.path.exists(local_path):
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
            raise FileNotFoundError(f"Neither URL nor local file found for {filename}")
    
    # Load all data files
    try:
        ultrasound_df = load_parquet_from_url_or_file("ultrasound_blocks.parquet", "DATA_ULTRASOUND_URL")
        beacon_data = load_parquet_from_url_or_file("beacon_api_data.parquet", "DATA_BEACON_URL", lazy=True)
        gossipsub_data = load_parquet_from_url_or_file("gossipsub_data.parquet", "DATA_GOSSIPSUB_URL", lazy=True)
        entities_df = load_parquet_from_url_or_file("entities.parquet", "DATA_ENTITIES_URL")
        proposers_df = load_parquet_from_url_or_file("slot_proposers.parquet", "DATA_PROPOSERS_URL")
        metadata = load_parquet_from_url_or_file("metadata.parquet", "DATA_METADATA_URL", use_polars=False)
        
        return {
            'ultrasound': ultrasound_df,
            'beacon': beacon_data,  # LazyFrame
            'gossipsub': gossipsub_data,  # LazyFrame
            'entities': entities_df,
            'proposers': proposers_df,
            'metadata': metadata
        }
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.info("Please run `python preprocess_data.py` to generate the data files.")
        st.stop()


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
    """Process data efficiently using Polars lazy evaluation."""
    # Get entities for slots
    slot_entities = get_entities_for_slots_polars(slots, _entities_df, _proposers_df)
    
    # Create ultrasound timing lookup - convert to lazy
    ultrasound_subset = _ultrasound_df.filter(pl.col('slot').is_in(slots)).select(['slot', 'time_into_slot_before_publish_ms']).lazy()
    slot_entities_lazy = slot_entities.lazy()
    
    # Process beacon data - filter for clients starting with 'pub'
    beacon_processed = (
        _beacon_lazy
        .filter(pl.col('slot').is_in(slots))
        .filter(pl.col('meta_client_name').str.starts_with('pub'))  # Only keep clients starting with 'pub'
        .join(ultrasound_subset, on='slot', how='left')
        .join(slot_entities_lazy, on='slot', how='left')
        .with_columns(
            (pl.col('propagation_time_ms') - pl.col('time_into_slot_before_publish_ms')).alias('adjusted_propagation_ms')
        )
        .filter(pl.col('adjusted_propagation_ms') >= 0)
        .with_columns(pl.col('adjusted_propagation_ms').alias('propagation_time_ms'))
        .collect()  # Execute the lazy query
    )
    
    # Process gossipsub data similarly
    gossipsub_processed = (
        _gossipsub_lazy
        .filter(pl.col('slot').is_in(slots))
        .join(ultrasound_subset, on='slot', how='left')
        .join(slot_entities_lazy, on='slot', how='left')
        .with_columns(
            (pl.col('propagation_time_ms') - pl.col('time_into_slot_before_publish_ms')).alias('adjusted_propagation_ms')
        )
        .filter(pl.col('adjusted_propagation_ms') >= 0)
        .with_columns(pl.col('adjusted_propagation_ms').alias('propagation_time_ms'))
        .collect()  # Execute the lazy query
    )
    
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
    
    # Calculate total unique contributors and slots
    if 'meta_client_name' in data_pl.columns:
        # Extract contributor name from format: privacy/username/node-uuid
        contributors = data_pl.with_columns(
            pl.col('meta_client_name').str.split('/').list.get(1).alias('contributor')
        )
        total_contributors = contributors['contributor'].n_unique()
    else:
        total_contributors = 0
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
        title=f"{title_prefix} - CDF of Data Availability Times by 100KB Bins<br><sub>{total_contributors} contributors, {total_slots:,} slots</sub>",
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
    st.title("🌐 Network Data Availability Analysis")
    st.markdown("Analyzing when data becomes available (block + all blobs) across the Ethereum network.")
    st.markdown("Methodology: max(block_arrival_time, all_blob_arrival_time) for each node observing a slot.")

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
        - **Caveats**: 
        """)
    
    # Load all data
    with st.spinner("Loading pre-processed data"):
        data = load_parquet_data()
    
    # Display metadata
    metadata = data['metadata'].iloc[0]
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.info(f"""
    **Created:** {pd.to_datetime(metadata['created_at']).strftime('%Y-%m-%d %H:%M')}
    **Total blocks:** {metadata['ultrasound_blocks']:,}
    **Beacon API records:** {metadata['beacon_api_records']:,}
    **Gossipsub records:** {metadata['gossipsub_records']:,}
    """)
    
    # Fixed sample size of 100k blocks
    sample_size = 100000
    
    # Random sampling with Polars
    if sample_size < len(data['ultrasound']):
        ultrasound_sample = data['ultrasound'].sample(n=sample_size, seed=42)
    else:
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
    with st.spinner("Processing data with Polars..."):
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
        
        if gossipsub_data.height > 0 and 'meta_client_name' in gossipsub_data.columns:
            gossip_contributors = gossipsub_data.with_columns(
                pl.col('meta_client_name').str.split('/').list.get(1).alias('contributor')
            )['contributor'].n_unique()
        else:
            gossip_contributors = 0
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
            title=f"Combined CDF - Data Availability Times by 100KB Bins<br><sub>Beacon: {beacon_contributors} contributors, {beacon_slots:,} slots | Libp2p: {gossip_contributors} nodes, {gossip_slots:,} slots</sub>",
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
            # Calculate node and slot counts for gossipsub data
            if 'meta_client_name' in gossipsub_data.columns:
                # Check if gossipsub uses same format, if so extract contributors
                sample_name = gossipsub_data.select('meta_client_name').limit(1)['meta_client_name'][0]
                if sample_name and '/' in sample_name:
                    gossip_box_contributors = gossipsub_data.with_columns(
                        pl.col('meta_client_name').str.split('/').list.get(1).alias('contributor')
                    )['contributor'].n_unique()
                    gossip_label = "contributors"
                else:
                    gossip_box_contributors = gossipsub_data['meta_client_name'].n_unique()
                    gossip_label = "nodes"
            else:
                gossip_box_contributors = 0
                gossip_label = "nodes"
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
                title=f"Libp2p Direct Monitoring - {'Solo Stakers' if is_solo_focused else 'All Proposers'} - Data Availability Time Distribution by 100KB Bins<br><sub>{gossip_box_contributors} {gossip_label}, {gossip_box_slots:,} slots</sub>",
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