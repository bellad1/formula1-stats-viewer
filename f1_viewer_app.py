import streamlit as st
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Configure FastF1
# Create cache directory if it doesn't exist
cache_dir = '.fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# Page config
st.set_page_config(
    page_title="F1 Data Viewer",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏎️ F1 Data Viewer</h1>', unsafe_allow_html=True)

# Sidebar for session selection
st.sidebar.header("Select Session")

# Year selection
current_year = datetime.now().year
year = st.sidebar.selectbox(
    "Year",
    options=list(range(current_year, 2017, -1)),
    index=0
)

# Get schedule for selected year
@st.cache_data(ttl=3600)
def get_schedule(year):
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        st.error(f"Error loading schedule: {e}")
        return None

schedule = get_schedule(year)

if schedule is not None:
    # Filter to only past events (and current event if it's started)
    now = pd.Timestamp.now()
    # Convert EventDate to datetime if it's not already
    if 'EventDate' in schedule.columns:
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    available_events = schedule[schedule['EventDate'] <= now]

    if len(available_events) == 0:
        st.warning(f"No completed races available for {year} yet.")
        st.stop()

    # Race selection
    race_names = available_events['EventName'].tolist()
    race_name = st.sidebar.selectbox(
        "Race",
        options=race_names,
        index=len(race_names) - 1  # Default to most recent
    )

    # Session type selection
    session_type = st.sidebar.selectbox(
        "Session",
        options=['Race', 'Qualifying', 'Sprint', 'FP1', 'FP2', 'FP3'],
        index=0
    )

    # Map session names to FastF1 codes
    session_map = {
        'Race': 'R',
        'Qualifying': 'Q',
        'Sprint': 'S',
        'FP1': 'FP1',
        'FP2': 'FP2',
        'FP3': 'FP3'
    }

    # Load session button
    if st.sidebar.button("Load Session", type="primary"):
        st.session_state.load_session = True
        st.session_state.session_loaded = False

    # Initialize session state
    if 'load_session' not in st.session_state:
        st.session_state.load_session = False
    if 'session_loaded' not in st.session_state:
        st.session_state.session_loaded = False

    # Load and cache session data
    @st.cache_data(show_spinner=False)
    def load_session(year, race_name, session_code):
        try:
            session = fastf1.get_session(year, race_name, session_code)
            session.load()
            return session
        except Exception as e:
            return None, str(e)

    if st.session_state.load_session:
        with st.spinner(f"Loading {year} {race_name} {session_type}... This may take a few minutes on first load."):
            result = load_session(year, race_name, session_map[session_type])

            if isinstance(result, tuple):
                st.error(f"Error loading session: {result[1]}")
                st.session_state.load_session = False
            else:
                session = result
                st.session_state.session = session
                st.session_state.session_loaded = True
                st.session_state.load_session = False
                st.success(f"✅ Loaded {year} {race_name} {session_type}")

    # Display visualizations if session is loaded
    if st.session_state.session_loaded and 'session' in st.session_state:
        session = st.session_state.session

        # Get list of drivers
        drivers = session.laps['Driver'].unique().tolist()
        drivers.sort()

        # Get fastest drivers for smart defaults
        def get_fastest_drivers(session, top_n=3):
            """Get the top N fastest drivers based on their fastest lap."""
            try:
                # Get each driver's fastest lap
                fastest_laps = []
                for driver in drivers:
                    driver_laps = session.laps.pick_drivers(driver)
                    if len(driver_laps) > 0:
                        fastest_lap = driver_laps['LapTime'].min()
                        if pd.notna(fastest_lap):
                            fastest_laps.append({
                                'Driver': driver,
                                'LapTime': fastest_lap
                            })

                # Sort by lap time and get top N
                if fastest_laps:
                    fastest_df = pd.DataFrame(fastest_laps).sort_values('LapTime')
                    top_drivers = fastest_df.head(top_n)['Driver'].tolist()
                    return top_drivers
                else:
                    return drivers[:top_n]  # Fallback to first N drivers
            except:
                return drivers[:top_n]  # Fallback on any error

        fastest_drivers = get_fastest_drivers(session, top_n=3)

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Lap Times",
            "⚡ Speed Comparison",
            "🏁 Position Changes",
            "🛞 Tire Strategy",
            "🗺️ Track Map"
        ])

        # Tab 1: Lap Times Comparison
        with tab1:
            st.header("Lap Times Comparison")

            col1, col2 = st.columns([1, 3])

            with col1:
                st.subheader("Select Drivers")
                selected_drivers = st.multiselect(
                    "Choose drivers to compare",
                    options=drivers,
                    default=fastest_drivers
                )

            with col2:
                if selected_drivers:
                    fig = go.Figure()

                    for driver in selected_drivers:
                        driver_laps = session.laps.pick_drivers(driver)
                        driver_laps = driver_laps.pick_quicklaps()  # Only quick laps

                        # Get team color
                        try:
                            color = fastf1.plotting.get_driver_color(driver, session)
                        except:
                            color = '#808080'

                        lap_times = driver_laps['LapTime'].dt.total_seconds()
                        lap_numbers = driver_laps['LapNumber']

                        fig.add_trace(go.Scatter(
                            x=lap_numbers,
                            y=lap_times,
                            mode='lines+markers',
                            name=driver,
                            line=dict(color=color, width=2),
                            marker=dict(size=6)
                        ))

                    fig.update_layout(
                        title=f"Lap Times - {race_name} {year}",
                        xaxis_title="Lap Number",
                        yaxis_title="Lap Time (seconds)",
                        hovermode='x unified',
                        height=500,
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show statistics
                    st.subheader("Statistics")
                    stats_data = []
                    for driver in selected_drivers:
                        driver_laps = session.laps.pick_drivers(driver).pick_quicklaps()
                        if len(driver_laps) > 0:
                            lap_times = driver_laps['LapTime'].dt.total_seconds()
                            stats_data.append({
                                'Driver': driver,
                                'Fastest Lap': f"{lap_times.min():.3f}s",
                                'Average Lap': f"{lap_times.mean():.3f}s",
                                'Total Laps': len(driver_laps)
                            })

                    if stats_data:
                        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
                else:
                    st.info("Please select at least one driver to view lap times.")

        # Tab 2: Speed Comparison
        with tab2:
            st.header("Telemetry Speed Comparison")

            col1, col2 = st.columns([1, 3])

            with col1:
                st.subheader("Select Drivers")
                speed_drivers = st.multiselect(
                    "Choose 2-3 drivers",
                    options=drivers,
                    default=fastest_drivers[:2],
                    key="speed_drivers"
                )

                lap_type = st.radio(
                    "Lap Selection",
                    options=["Fastest Lap", "Specific Lap"],
                    index=0
                )

                if lap_type == "Specific Lap":
                    max_laps = session.laps['LapNumber'].max()
                    lap_number = st.number_input(
                        "Lap Number",
                        min_value=1,
                        max_value=int(max_laps),
                        value=1
                    )

            with col2:
                if len(speed_drivers) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    for driver in speed_drivers[:3]:  # Limit to 3 for readability
                        try:
                            if lap_type == "Fastest Lap":
                                lap = session.laps.pick_drivers(driver).pick_fastest()
                            else:
                                driver_laps = session.laps.pick_drivers(driver)
                                lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]

                            telemetry = lap.get_telemetry()
                            color = fastf1.plotting.get_driver_color(driver, session)

                            ax.plot(
                                telemetry['Distance'],
                                telemetry['Speed'],
                                label=driver,
                                color=color,
                                linewidth=2
                            )
                        except Exception as e:
                            st.warning(f"Could not load telemetry for {driver}: {str(e)}")

                    ax.set_xlabel('Distance (m)', fontsize=12)
                    ax.set_ylabel('Speed (km/h)', fontsize=12)
                    ax.set_title(f'Speed Comparison - {lap_type}', fontsize=14, fontweight='bold')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Please select at least one driver to compare speeds.")

        # Tab 3: Position Changes
        with tab3:
            st.header("Position Changes Throughout Session")

            try:
                # Get position data for all drivers
                fig = go.Figure()

                for driver in drivers:
                    driver_laps = session.laps.pick_drivers(driver)

                    if len(driver_laps) > 0:
                        try:
                            color = fastf1.plotting.get_driver_color(driver, session)
                        except:
                            color = '#808080'

                        fig.add_trace(go.Scatter(
                            x=driver_laps['LapNumber'],
                            y=driver_laps['Position'],
                            mode='lines+markers',
                            name=driver,
                            line=dict(color=color, width=2),
                            marker=dict(size=5)
                        ))

                fig.update_layout(
                    title=f"Position Changes - {race_name} {year}",
                    xaxis_title="Lap Number",
                    yaxis_title="Position",
                    yaxis=dict(autorange='reversed'),  # 1st place at top
                    hovermode='x unified',
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating position chart: {e}")

        # Tab 4: Tire Strategy
        with tab4:
            st.header("Tire Strategy")

            try:
                # Create tire strategy visualization
                fig, ax = plt.subplots(figsize=(14, 10))

                # Get unique drivers and sort by finishing position
                driver_stints = []

                for driver in drivers:
                    driver_laps = session.laps.pick_drivers(driver)
                    if len(driver_laps) > 0:
                        stints = driver_laps[['LapNumber', 'Compound', 'TyreLife']].copy()
                        driver_stints.append({
                            'driver': driver,
                            'laps': driver_laps
                        })

                # Sort by final position if available
                try:
                    driver_stints.sort(key=lambda x: x['laps']['Position'].iloc[-1])
                except:
                    pass

                # Compound colors
                compound_colors = {
                    'SOFT': '#FF0000',
                    'MEDIUM': '#FFF200',
                    'HARD': '#FFFFFF',
                    'INTERMEDIATE': '#00FF00',
                    'WET': '#0000FF'
                }

                y_pos = 0
                for stint_data in driver_stints:
                    driver = stint_data['driver']
                    laps = stint_data['laps']

                    # Group consecutive laps by compound
                    current_compound = None
                    stint_start = None

                    for idx, lap in laps.iterrows():
                        if current_compound != lap['Compound']:
                            # Plot previous stint
                            if current_compound is not None:
                                ax.barh(
                                    y_pos,
                                    lap['LapNumber'] - stint_start,
                                    left=stint_start,
                                    height=0.8,
                                    color=compound_colors.get(current_compound, '#808080'),
                                    edgecolor='black',
                                    linewidth=0.5
                                )

                            # Start new stint
                            current_compound = lap['Compound']
                            stint_start = lap['LapNumber']

                    # Plot final stint
                    if current_compound is not None:
                        ax.barh(
                            y_pos,
                            laps['LapNumber'].max() - stint_start + 1,
                            left=stint_start,
                            height=0.8,
                            color=compound_colors.get(current_compound, '#808080'),
                            edgecolor='black',
                            linewidth=0.5
                        )

                    y_pos += 1

                # Set labels
                ax.set_yticks(range(len(driver_stints)))
                ax.set_yticklabels([d['driver'] for d in driver_stints])
                ax.set_xlabel('Lap Number', fontsize=12)
                ax.set_title(f'Tire Strategy - {race_name} {year}', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=compound_colors['SOFT'], edgecolor='black', label='Soft'),
                    Patch(facecolor=compound_colors['MEDIUM'], edgecolor='black', label='Medium'),
                    Patch(facecolor=compound_colors['HARD'], edgecolor='black', label='Hard'),
                    Patch(facecolor=compound_colors['INTERMEDIATE'], edgecolor='black', label='Intermediate'),
                    Patch(facecolor=compound_colors['WET'], edgecolor='black', label='Wet')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"Error creating tire strategy chart: {e}")

        # Tab 5: Track Map with Speed
        with tab5:
            st.header("Speed on Track Map")

            col1, col2 = st.columns([1, 3])

            with col1:
                st.subheader("Select Driver")
                # Default to fastest driver
                default_index = drivers.index(fastest_drivers[0]) if fastest_drivers[0] in drivers else 0
                map_driver = st.selectbox(
                    "Choose driver",
                    options=drivers,
                    index=default_index,
                    key="map_driver"
                )

                map_lap_type = st.radio(
                    "Lap Selection",
                    options=["Fastest Lap", "Specific Lap"],
                    index=0,
                    key="map_lap_type"
                )

                if map_lap_type == "Specific Lap":
                    max_laps = session.laps['LapNumber'].max()
                    map_lap_number = st.number_input(
                        "Lap Number",
                        min_value=1,
                        max_value=int(max_laps),
                        value=1,
                        key="map_lap_number"
                    )

            with col2:
                try:
                    if map_lap_type == "Fastest Lap":
                        lap = session.laps.pick_drivers(map_driver).pick_fastest()
                    else:
                        driver_laps = session.laps.pick_drivers(map_driver)
                        lap = driver_laps[driver_laps['LapNumber'] == map_lap_number].iloc[0]

                    telemetry = lap.get_telemetry()

                    # Get coordinates and speed
                    x = telemetry['X']
                    y = telemetry['Y']
                    color = telemetry['Speed']

                    # Create line segments
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Create colored line collection
                    norm = plt.Normalize(color.min(), color.max())
                    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=4)
                    lc.set_array(color)

                    ax.add_collection(lc)
                    ax.axis('equal')
                    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
                    ax.set_title(
                        f'{map_driver} - {map_lap_type} - {race_name} {year}',
                        fontsize=14,
                        fontweight='bold'
                    )

                    # Add colorbar
                    cbar = plt.colorbar(lc, ax=ax, pad=0.01, aspect=30)
                    cbar.set_label('Speed (km/h)', fontsize=12)

                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Error creating track map: {e}")

    else:
        st.info("👈 Click 'Load Session' in the sidebar to start exploring F1 data!")

        # Show some info about the app
        st.markdown("""
        ### Welcome to F1 Data Viewer!

        This app lets you explore Formula 1 race data with interactive visualizations:

        - **📊 Lap Times**: Compare lap times between drivers throughout the session
        - **⚡ Speed Comparison**: Overlay speed traces to see where drivers gain/lose time
        - **🏁 Position Changes**: Track how positions evolved during the race
        - **🛞 Tire Strategy**: Visualize pit stops and compound choices
        - **🗺️ Track Map**: See speed visualized on the actual track layout

        **Getting Started:**
        1. Select a year, race, and session type in the sidebar
        2. Click "Load Session" (first load may take 2-3 minutes)
        3. Explore the data in the different tabs!

        **Note**: Data is cached locally, so subsequent loads will be much faster.
        """)

else:
    st.error("Could not load F1 schedule. Please check your internet connection.")
