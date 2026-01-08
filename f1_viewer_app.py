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
import seaborn as sns
from matplotlib.patches import Patch
from timple.timedelta import strftimedelta
from fastf1.core import Laps
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

        # Get race winner for default driver selection
        def get_race_winner(session):
            """Get the race winner's abbreviation."""
            try:
                results = session.results
                if len(results) > 0:
                    winner = results.loc[results['Position'] == 1.0, 'Abbreviation']
                    if len(winner) > 0:
                        return winner.iloc[0]
            except:
                pass
            return fastest_drivers[0] if fastest_drivers else drivers[0]

        race_winner = get_race_winner(session)

        # Sidebar navigation
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Analysis")

        # Determine available sections based on session type
        if session_type == 'Qualifying':
            nav_options = [
                "🏁 Qualifying Results",
                "⚡ Speed Analysis",
                "🗺️ Track Analysis",
                "🏆 Championship Standings"
            ]
        else:  # Race or other sessions
            nav_options = [
                "🏁 Qualifying Results",
                "👤 Driver Performance",
                "🏎️ Team Strategy",
                "🗺️ Track Analysis",
                "📈 Race Overview",
                "🏆 Championship Standings"
            ]

        selected_section = st.sidebar.radio(
            "Select Analysis",
            nav_options,
            label_visibility="collapsed"
        )

        # ===== QUALIFYING RESULTS SECTION =====
        if selected_section == "🏁 Qualifying Results":
            st.header("Qualifying Results")

            try:
                # Enable Matplotlib patches for plotting timedelta values
                fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None)

                # Get all drivers
                drivers_list = pd.unique(session.laps['Driver'])

                # Get each driver's fastest lap
                list_fastest_laps = list()
                for drv in drivers_list:
                    drvs_fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
                    list_fastest_laps.append(drvs_fastest_lap)

                fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

                # Calculate time delta from pole
                pole_lap = fastest_laps.pick_fastest()
                fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

                # Get team colors
                team_colors = []
                for index, lap in fastest_laps.iterlaps():
                    color = fastf1.plotting.get_team_color(lap['Team'], session=session)
                    team_colors.append(color)

                # Create plot
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
                        color=team_colors, edgecolor='grey')
                ax.set_yticks(fastest_laps.index)
                ax.set_yticklabels(fastest_laps['Driver'])
                ax.invert_yaxis()
                ax.set_axisbelow(True)
                ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

                lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')
                plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                            f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

                st.pyplot(fig)
                plt.close()

                # Show data table
                st.subheader("Qualifying Times")
                display_df = fastest_laps[['Driver', 'Team', 'LapTime', 'LapTimeDelta']].copy()
                display_df['LapTime'] = display_df['LapTime'].apply(lambda x: strftimedelta(x, '%m:%s.%ms'))
                display_df['Gap'] = display_df['LapTimeDelta'].apply(lambda x: f"+{strftimedelta(x, '%s.%ms')}" if x.total_seconds() > 0 else "POLE")
                display_df = display_df[['Driver', 'Team', 'LapTime', 'Gap']]
                st.dataframe(display_df, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating qualifying results: {e}")

        # ===== DRIVER PERFORMANCE SECTION =====
        elif selected_section == "👤 Driver Performance":
            st.header("Driver Performance Analysis")

            # Sub-sections for driver performance
            driver_subsection = st.radio(
                "Select Analysis Type",
                ["📊 Lap Times Comparison", "📈 Lap Time Distributions", "🎯 Individual Driver Analysis"],
                horizontal=True
            )

            if driver_subsection == "📊 Lap Times Comparison":
                st.subheader("Lap Times Comparison")

                col1, col2 = st.columns([1, 3])

                with col1:
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

            elif driver_subsection == "📈 Lap Time Distributions":
                st.subheader("Lap Time Distributions - Top 10 Drivers")

                try:
                    # Get top 10 finishers
                    point_finishers = session.drivers[:10]
                    driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps()
                    driver_laps = driver_laps.reset_index()

                    # Get finishing order
                    finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]

                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Convert timedelta to seconds
                    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

                    # Violin plot
                    sns.violinplot(data=driver_laps,
                                x="Driver",
                                y="LapTime(s)",
                                hue="Driver",
                                inner=None,
                                density_norm="area",
                                order=finishing_order,
                                palette=fastf1.plotting.get_driver_color_mapping(session=session))

                    # Swarm plot
                    sns.swarmplot(data=driver_laps,
                                x="Driver",
                                y="LapTime(s)",
                                order=finishing_order,
                                hue="Compound",
                                palette=fastf1.plotting.get_compound_mapping(session=session),
                                hue_order=["SOFT", "MEDIUM", "HARD"],
                                linewidth=0,
                                size=4)

                    ax.set_xlabel("Driver")
                    ax.set_ylabel("Lap Time (s)")
                    plt.suptitle(f"{session.event['EventName']} {session.event.year}\nLap Time Distributions")
                    sns.despine(left=True, bottom=True)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Error creating lap time distribution: {e}")

            elif driver_subsection == "🎯 Individual Driver Analysis":
                st.subheader("Individual Driver Lap Times")

                col1, col2 = st.columns([1, 3])

                with col1:
                    # Driver selection with race winner as default
                    default_idx = drivers.index(race_winner) if race_winner in drivers else 0
                    selected_driver = st.selectbox(
                        "Select Driver",
                        options=drivers,
                        index=default_idx
                    )

                with col2:
                    try:
                        fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
                        driver_laps = session.laps.pick_drivers(selected_driver).pick_quicklaps().reset_index()

                        fig, ax = plt.subplots(figsize=(10, 6))

                        sns.scatterplot(data=driver_laps,
                                      x="LapNumber",
                                      y="LapTime",
                                      ax=ax,
                                      hue="Compound",
                                      palette=fastf1.plotting.get_compound_mapping(session=session),
                                      s=80,
                                      linewidth=0,
                                      legend='auto')

                        ax.set_xlabel("Lap Number")
                        ax.set_ylabel("Lap Time")
                        ax.invert_yaxis()
                        plt.suptitle(f"{selected_driver} Lap Times - {session.event.year} {session.event['EventName']}")
                        plt.grid(color='w', which='major', axis='both')
                        sns.despine(left=True, bottom=True)
                        plt.tight_layout()

                        st.pyplot(fig)
                        plt.close()

                    except Exception as e:
                        st.error(f"Error creating driver lap times: {e}")

        # ===== SPEED ANALYSIS SECTION =====
        elif selected_section == "⚡ Speed Analysis":
            st.header("Speed Analysis with Corner Annotations")

            col1, col2 = st.columns([1, 3])

            with col1:
                st.subheader("Settings")
                speed_drivers = st.multiselect(
                    "Choose drivers (max 3)",
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
                    try:
                        fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
                        fig, ax = plt.subplots(figsize=(14, 7))

                        # Get circuit info for corner annotations
                        circuit_info = session.get_circuit_info()

                        # Plot speed traces for each driver
                        for driver in speed_drivers[:3]:
                            try:
                                if lap_type == "Fastest Lap":
                                    lap = session.laps.pick_drivers(driver).pick_fastest()
                                else:
                                    driver_laps = session.laps.pick_drivers(driver)
                                    lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]

                                car_data = lap.get_car_data().add_distance()
                                color = fastf1.plotting.get_driver_color(driver, session)

                                ax.plot(car_data['Distance'], car_data['Speed'],
                                       color=color, label=driver, linewidth=2)

                            except Exception as e:
                                st.warning(f"Could not load telemetry for {driver}: {str(e)}")

                        # Add corner annotations
                        if len(speed_drivers) > 0:
                            # Get first valid driver's data for y-axis limits
                            try:
                                if lap_type == "Fastest Lap":
                                    ref_lap = session.laps.pick_drivers(speed_drivers[0]).pick_fastest()
                                else:
                                    ref_laps = session.laps.pick_drivers(speed_drivers[0])
                                    ref_lap = ref_laps[ref_laps['LapNumber'] == lap_number].iloc[0]

                                ref_data = ref_lap.get_car_data().add_distance()
                                v_min = ref_data['Speed'].min()
                                v_max = ref_data['Speed'].max()

                                # Draw vertical dotted lines at corners
                                ax.vlines(x=circuit_info.corners['Distance'],
                                        ymin=v_min-20, ymax=v_max+20,
                                        linestyles='dotted', colors='grey', alpha=0.5)

                                # Add corner numbers
                                for _, corner in circuit_info.corners.iterrows():
                                    txt = f"{corner['Number']}{corner['Letter']}"
                                    ax.text(corner['Distance'], v_min-30, txt,
                                           va='center_baseline', ha='center', size='small')

                                ax.set_ylim([v_min - 40, v_max + 20])

                            except:
                                pass  # Skip corner annotations if there's an issue

                        ax.set_xlabel('Distance (m)', fontsize=12)
                        ax.set_ylabel('Speed (km/h)', fontsize=12)
                        ax.set_title(f'Speed Comparison with Corners - {lap_type}', fontsize=14, fontweight='bold')
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)
                        plt.close()

                    except Exception as e:
                        st.error(f"Error creating speed comparison: {e}")
                else:
                    st.info("Please select at least one driver to compare speeds.")

        # ===== TEAM STRATEGY SECTION =====
        elif selected_section == "🏎️ Team Strategy":
            st.header("Team Strategy Analysis")

            team_subsection = st.radio(
                "Select Analysis Type",
                ["🛞 Tire Strategy", "📊 Team Pace Ranking"],
                horizontal=True
            )

            if team_subsection == "🛞 Tire Strategy":
                st.subheader("Tire Strategy")

                try:
                    # Get driver list in finishing order
                    results = session.results
                    drivers_ordered = []

                    # Try to sort by finishing position
                    try:
                        results_sorted = results.sort_values('Position')
                        drivers_ordered = [session.get_driver(d)["Abbreviation"] for d in results_sorted.index]
                    except:
                        drivers_ordered = drivers

                    # Create stint data
                    laps = session.laps
                    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
                    stints = stints.groupby(["Driver", "Stint", "Compound"])
                    stints = stints.count().reset_index()
                    stints = stints.rename(columns={"LapNumber": "StintLength"})

                    # Create plot
                    fig, ax = plt.subplots(figsize=(14, 10))

                    for driver in drivers_ordered:
                        driver_stints = stints.loc[stints["Driver"] == driver]
                        previous_stint_end = 0

                        for idx, row in driver_stints.iterrows():
                            compound_color = fastf1.plotting.get_compound_color(row["Compound"], session=session)
                            plt.barh(
                                y=driver,
                                width=row["StintLength"],
                                left=previous_stint_end,
                                color=compound_color,
                                edgecolor="black",
                                fill=True
                            )
                            previous_stint_end += row["StintLength"]

                    # Add legend
                    compound_colors = {
                        'SOFT': '#FF0000',
                        'MEDIUM': '#FFF200',
                        'HARD': '#FFFFFF',
                        'INTERMEDIATE': '#00FF00',
                        'WET': '#0000FF'
                    }
                    legend_elements = [
                        Patch(facecolor=compound_colors['SOFT'], edgecolor='black', label='Soft'),
                        Patch(facecolor=compound_colors['MEDIUM'], edgecolor='black', label='Medium'),
                        Patch(facecolor=compound_colors['HARD'], edgecolor='black', label='Hard'),
                        Patch(facecolor=compound_colors['INTERMEDIATE'], edgecolor='black', label='Intermediate'),
                        Patch(facecolor=compound_colors['WET'], edgecolor='black', label='Wet')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')

                    plt.title(f"{session.event['EventName']} {session.event.year}\nTire Strategies")
                    plt.xlabel("Lap Number")
                    plt.grid(False)
                    ax.invert_yaxis()  # P1 at top

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Error creating tire strategy chart: {e}")

            elif team_subsection == "📊 Team Pace Ranking":
                st.subheader("Team Pace Comparison")

                try:
                    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')

                    # Pick quick laps
                    laps = session.laps.pick_quicklaps()

                    # Convert lap time to seconds
                    transformed_laps = laps.copy()
                    transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

                    # Order teams by median lap time
                    team_order = (
                        transformed_laps[["Team", "LapTime (s)"]]
                        .groupby("Team")
                        .median()["LapTime (s)"]
                        .sort_values()
                        .index
                    )

                    # Create color palette
                    team_palette = {team: fastf1.plotting.get_team_color(team, session=session)
                                  for team in team_order}

                    # Create plot
                    fig, ax = plt.subplots(figsize=(14, 8))
                    sns.boxplot(
                        data=transformed_laps,
                        x="Team",
                        y="LapTime (s)",
                        hue="Team",
                        order=team_order,
                        palette=team_palette,
                        whiskerprops=dict(color="white"),
                        boxprops=dict(edgecolor="white"),
                        medianprops=dict(color="grey"),
                        capprops=dict(color="white"),
                    )

                    plt.title(f"{session.event['EventName']} {session.event.year}\nTeam Pace Ranking")
                    plt.grid(visible=False)
                    ax.set(xlabel=None)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Error creating team pace ranking: {e}")

        # ===== RACE OVERVIEW SECTION =====
        elif selected_section == "📈 Race Overview":
            st.header("Race Overview")

            st.subheader("Position Changes Throughout Race")

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

        # ===== TRACK ANALYSIS SECTION =====
        elif selected_section == "🗺️ Track Analysis":
            st.header("Track Analysis")

            track_subsection = st.radio(
                "Select Visualization",
                ["🏁 Speed on Track", "⚙️ Gear Shifts on Track"],
                horizontal=True
            )

            if track_subsection == "🏁 Speed on Track":
                st.subheader("Speed Visualization on Track Map")

                col1, col2 = st.columns([1, 3])

                with col1:
                    # Default to race winner
                    default_index = drivers.index(race_winner) if race_winner in drivers else 0
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

                        # Add corner number annotations
                        try:
                            circuit_info = session.get_circuit_info()
                            for _, corner in circuit_info.corners.iterrows():
                                txt = f"{corner['Number']}{corner['Letter']}"
                                # Plot corner marker
                                ax.scatter(corner['X'], corner['Y'], color='white',
                                         edgecolors='black', s=100, zorder=10, linewidths=1.5)
                                # Plot corner number
                                ax.text(corner['X'], corner['Y'], txt,
                                       va='center', ha='center', size='small',
                                       color='black', fontweight='bold', zorder=11)
                        except Exception as e:
                            # Skip corner annotations if they fail
                            pass

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

            elif track_subsection == "⚙️ Gear Shifts on Track":
                st.subheader("Gear Shifts Visualization")

                try:
                    from matplotlib import colormaps

                    # Get fastest lap
                    lap = session.laps.pick_fastest()
                    tel = lap.get_telemetry()

                    # Prepare data
                    x = np.array(tel['X'].values)
                    y = np.array(tel['Y'].values)
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    gear = tel['nGear'].to_numpy().astype(float)

                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    cmap = colormaps['Paired']
                    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
                    lc_comp.set_array(gear)
                    lc_comp.set_linewidth(4)

                    ax.add_collection(lc_comp)

                    # Add corner number annotations
                    try:
                        circuit_info = session.get_circuit_info()
                        for _, corner in circuit_info.corners.iterrows():
                            txt = f"{corner['Number']}{corner['Letter']}"
                            # Plot corner marker
                            ax.scatter(corner['X'], corner['Y'], color='white',
                                     edgecolors='black', s=100, zorder=10, linewidths=1.5)
                            # Plot corner number
                            ax.text(corner['X'], corner['Y'], txt,
                                   va='center', ha='center', size='small',
                                   color='black', fontweight='bold', zorder=11)
                    except Exception as e:
                        # Skip corner annotations if they fail
                        pass

                    ax.axis('equal')
                    ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

                    plt.suptitle(
                        f"Fastest Lap Gear Shift Visualization\n"
                        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
                    )

                    # Add colorbar
                    cbar = plt.colorbar(mappable=lc_comp, label="Gear",
                                      boundaries=np.arange(1, 10))
                    cbar.set_ticks(np.arange(1.5, 9.5))
                    cbar.set_ticklabels(np.arange(1, 9))

                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Error creating gear shifts visualization: {e}")

        # ===== CHAMPIONSHIP STANDINGS SECTION =====
        elif selected_section == "🏆 Championship Standings":
            st.header("Championship Standings Analysis")

            try:
                from fastf1.ergast import Ergast

                # Get current round from session
                current_round = session.event['RoundNumber']

                # Function to get driver standings
                def get_drivers_standings():
                    ergast = Ergast()
                    standings = ergast.get_driver_standings(season=year, round=current_round)
                    return standings.content[0]

                # Function to calculate max points remaining
                def calculate_max_points_for_remaining_season():
                    POINTS_FOR_SPRINT = 8 + 25
                    POINTS_FOR_CONVENTIONAL = 25

                    events = fastf1.events.get_event_schedule(year, backend='ergast')
                    events = events[events['RoundNumber'] > current_round]
                    sprint_events = len(events.loc[events["EventFormat"] == "sprint_qualifying"])
                    conventional_events = len(events.loc[events["EventFormat"] == "conventional"])

                    sprint_points = sprint_events * POINTS_FOR_SPRINT
                    conventional_points = conventional_events * POINTS_FOR_CONVENTIONAL

                    return sprint_points + conventional_points

                # Get standings and max points
                driver_standings = get_drivers_standings()
                max_points = calculate_max_points_for_remaining_season()

                LEADER_POINTS = int(driver_standings.loc[0]['points'])

                # Create data for table
                standings_data = []
                for i, _ in enumerate(driver_standings.iterrows()):
                    driver = driver_standings.loc[i]
                    driver_max_points = int(driver["points"]) + max_points
                    can_win = 'Yes' if driver_max_points >= LEADER_POINTS else 'No'

                    standings_data.append({
                        'Position': int(driver['position']),
                        'Driver': f"{driver['givenName']} {driver['familyName']}",
                        'Current Points': int(driver['points']),
                        'Max Possible Points': driver_max_points,
                        'Can Win WDC': can_win
                    })

                standings_df = pd.DataFrame(standings_data)

                st.subheader(f"World Drivers' Championship - After Round {current_round}")

                # Style the dataframe
                def highlight_can_win(row):
                    if row['Can Win WDC'] == 'Yes':
                        return ['background-color: #90EE90'] * len(row)
                    else:
                        return ['background-color: #FFB6C1'] * len(row)

                styled_df = standings_df.style.apply(highlight_can_win, axis=1)
                st.dataframe(styled_df, hide_index=True, use_container_width=True)

                st.info(f"**Remaining races:** {max_points // 25} conventional + sprint races\n\n"
                       f"**Maximum points available:** {max_points} points")

            except Exception as e:
                st.error(f"Error creating championship standings: {e}\n\nNote: Championship data may not be available for all sessions.")

    else:
        st.info("👈 Click 'Load Session' in the sidebar to start exploring F1 data!")

        # Show some info about the app
        st.markdown("""
        ### Welcome to F1 Data Viewer!

        This app lets you explore Formula 1 race data with comprehensive interactive visualizations organized into intuitive sections:

        **📊 Analysis Sections:**

        - **🏁 Qualifying Results**: View qualifying times, gaps, and performance by team
        - **👤 Driver Performance**: Compare lap times, distributions, and individual driver analysis
        - **⚡ Speed Analysis**: Overlay speed traces with corner annotations to see where drivers gain/lose time
        - **🏎️ Team Strategy**: Analyze tire strategies and compare team pace rankings
        - **🗺️ Track Analysis**: Visualize speed and gear shifts on track map with corner number annotations
        - **📈 Race Overview**: Track how positions evolved lap-by-lap throughout the race
        - **🏆 Championship Standings**: See who can still win the World Drivers' Championship

        **Getting Started:**
        1. Select a year, race, and session type in the left sidebar
        2. Click "Load Session" (first load may take 2-3 minutes)
        3. Use the Analysis navigation menu to explore different visualizations

        **Note**: Data is cached locally, so subsequent loads will be much faster.
        """)

else:
    st.error("Could not load F1 schedule. Please check your internet connection.")
