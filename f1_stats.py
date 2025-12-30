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
import sys
warnings.filterwarnings('ignore')


# Helper/plotting functions
def get_schedule(year):
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        print(f"Error loading schedule: {e}")
        return None


def load_session(year, race_name, session_code):
        try:
            session = fastf1.get_session(year, race_name, session_code)
            session.load()
            return session
        except Exception as e:
            return None, str(e)


def plot_quickest_lap_times(session):
    # Plot lap quickest lap time for selected drivers
    fig, ax = plt.subplots(figsize=(8, 5))

    for driver in ('HAM', 'PIA', 'VER', 'RUS', 'NOR'):
        laps = session.laps.pick_drivers(driver).pick_quicklaps().reset_index()
        style = fastf1.plotting.get_driver_style(identifier=driver,
                                                 style=['color', 'linestyle'],
                                                 session=session)
        ax.plot(laps['LapTime'], **style, label=driver)

    # add axis labels and a legend
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")
    ax.legend()
    fastf1.plotting.add_sorted_driver_legend(ax, session)
    plt.suptitle(f"{session.event['EventName']} {session.event.year}\nQuickest Lap Times")
    plt.show()
    return fig


def draw_track_layout(session):
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()

    circuit_info = session.get_circuit_info()

    ##############################################################################
    # Define a helper function for rotating points around the origin of the
    # coordinate system.
    #
    # The matrix ``[[cos, sin], [-sin, cos]]`` is the rotation matrix.
    #
    # By matrix multiplication of the rotation matrix with a vector [x, y], a new
    # rotated vector [x_rot, y_rot] is obtained.
    # (See also: https://en.wikipedia.org/wiki/Rotation_matrix)
    def rotate(xy, *, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)


    ##############################################################################
    # Get the coordinates of the track map from the telemetry of the lap and
    # rotate the coordinates using the rotation from ``circuit_info`` so that
    # the track map is oriented correctly. After that, plot the rotated track map.

    # Get an array of shape [n, 2] where n is the number of points and the second
    # axis is x and y.
    track = pos.loc[:, ('X', 'Y')].to_numpy()

    # Convert the rotation angle from degrees to radian.
    track_angle = circuit_info.rotation / 180 * np.pi

    # Rotate and plot the track map.
    rotated_track = rotate(track, angle=track_angle)
    plt.plot(rotated_track[:, 0], rotated_track[:, 1])

    ##############################################################################
    # Finally, the corner markers are plotted. To plot the numbers next to the
    # track, an offset vector that points straight up is defined. This offset
    # vector is then rotated by the angle that is given for each corner marker.
    # A line and circular bubble are drawn and the corner marker text is printed
    # inside the bubble.
    offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        # Draw a circle next to the track.
        plt.scatter(text_x, text_y, color='grey', s=140)

        # Draw a line from the track to this circle.
        plt.plot([track_x, text_x], [track_y, text_y], color='grey')

        # Finally, print the corner number inside the circle.
        plt.text(text_x, text_y, txt,
                 va='center_baseline', ha='center', size='small', color='white')

    ##############################################################################
    # Add a title, remove tick labels to clean up the plot, set equal axis ratio,
    # so that the track is not distorted and show the plot.
    plt.title(f"{session.event['Location']} ({session.event['EventName']})")
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.show()


def plot_laptime_distribution(race):
    ###############################################################################
    # Get all the laps for the point finishers only.
    # Filter out slow laps (yellow flag, VSC, pitstops etc.)
    # as they distort the graph axis.
    point_finishers = race.drivers[:10]
    driver_laps = race.laps.pick_drivers(point_finishers).pick_quicklaps()
    driver_laps = driver_laps.reset_index()

    ###############################################################################
    # To plot the drivers by finishing order,
    # we need to get their three-letter abbreviations in the finishing order.
    finishing_order = [race.get_driver(i)["Abbreviation"] for i in point_finishers]

    # First create the violin plots to show the distributions.
    # Then use the swarm plot to show the actual laptimes.
    fig, ax = plt.subplots(figsize=(10, 5))

    # Seaborn doesn't have proper timedelta support,
    # so we have to convert timedelta to float (in seconds)
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    sns.violinplot(data=driver_laps,
                x="Driver",
                y="LapTime(s)",
                hue="Driver",
                inner=None,
                density_norm="area",
                order=finishing_order,
                palette=fastf1.plotting.get_driver_color_mapping(session=race)
                )

    sns.swarmplot(data=driver_laps,
                x="Driver",
                y="LapTime(s)",
                order=finishing_order,
                hue="Compound",
                palette=fastf1.plotting.get_compound_mapping(session=race),
                hue_order=["SOFT", "MEDIUM", "HARD"],
                linewidth=0,
                size=4,
                )
    
    # Make the plot more aesthetic
    ax.set_xlabel("Driver")
    ax.set_ylabel("Lap Time (s)")
    info = race.session_info
    race_name = info['Meeting']['OfficialName']
    plt.suptitle(f"{race.event['EventName']} {race.event.year} \nLap Time Distributions")
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()


def plot_driver_laptimes(race, driver):
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
    driver_laps = race.laps.pick_drivers(driver).pick_quicklaps().reset_index()
    # Make the scattterplot using lap number as x-axis and lap time as y-axis.
    # Marker colors correspond to the compounds used.
    # Note: as LapTime is represented by timedelta, calling setup_mpl earlier
    # is required.
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=driver_laps,
                    x="LapNumber",
                    y="LapTime",
                    ax=ax,
                    hue="Compound",
                    palette=fastf1.plotting.get_compound_mapping(session=race),
                    s=80,
                    linewidth=0,
                    legend='auto')
    
    # Make the plot more aesthetic.
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")

    # The y-axis increases from bottom to top by default
    # Since we are plotting time, it makes sense to invert the axis
    ax.invert_yaxis()
    info = race.session_info
    race_name = info['Meeting']['OfficialName']
    plt.suptitle(f"{driver} Laptimes in the {race.event.year} {race.event['EventName']}")

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()


def plot_position_changes(session):
    # Load FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')

    # Create the figure
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
   
    # For each driver, get their three letter abbreviation (e.g. 'HAM') by simply
    # using the value of the first lap, get their color and then plot their
    # position over the number of laps.
    for drv in session.drivers:
        drv_laps = session.laps.pick_drivers(drv)

        abb = drv_laps['Driver'].iloc[0]
        style = fastf1.plotting.get_driver_style(identifier=abb,
                                                style=['color', 'linestyle'],
                                                session=session)

        ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
                label=abb, **style)

    # Finalize the plot by setting y-limits that invert the y-axis so that position
    # one is at the top, set custom tick positions and axis labels.
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')

    # Because this plot is very crowed, add the legend outside the plot area.
    ax.legend(bbox_to_anchor=(1.0, 1.02))
    plt.suptitle(f"{session.event['EventName']} {session.event.year}\n Position Changes")
    plt.tight_layout()

    plt.show()


def plot_qualifying_results(session):
    """
    Qualifying results overview
    Plot the qualifying result with visualization the fastest times.
    """
    from timple.timedelta import strftimedelta
    from fastf1.core import Laps

    # Enable Matplotlib patches for plotting timedelta values
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None)

    # First, we need to get an array of all drivers.
    drivers = pd.unique(session.laps['Driver'])

    # After that we'll get each driver's fastest lap, create a new laps object
    # from these laps, sort them by lap time and have pandas reindex them to
    # number them nicely by starting position.
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps) \
        .sort_values(by='LapTime') \
        .reset_index(drop=True)

    # The plot is nicer to look at and more easily understandable if we just plot
    # the time differences. Therefore, we subtract the fastest lap time from all
    # other lap times.
    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

    # We can take a quick look at the laps we have to check if everything
    # looks all right. For this, we'll just check the 'Driver', 'LapTime'
    # and 'LapTimeDelta' columns.
    print(fastest_laps[['Driver', 'LapTime', 'LapTimeDelta']])

    # Finally, we'll create a list of team colors per lap to color our plot.
    team_colors = list()
    for index, lap in fastest_laps.iterlaps():
        color = fastf1.plotting.get_team_color(lap['Team'], session=session)
        team_colors.append(color)

    # Now, we can plot all the data
    fig, ax = plt.subplots()
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')
    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    
    # Finally, give the plot a meaningful title
    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

    plt.show()


def plot_strategy(session):
    """
    Tyre strategies during a race
    Plot all drivers' tyre strategies during a race.
    """
    laps = session.laps

    # Get the list of driver numbers
    drivers = session.drivers
   
    # Convert the driver numbers to three letter abbreviations
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    # We need to find the stint length and compound used
    # for every stint by every driver.
    # We do this by first grouping the laps by the driver,
    # the stint number, and the compound.
    # And then counting the number of laps in each group.
    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()

    # The number in the LapNumber column now stands for the number of observations
    # in that group aka the stint length.
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    print(stints)

    # Now we can plot the strategies for each driver
    # fig, ax = plt.subplots(figsize=(5, 10))
    fig, ax = plt.subplots()

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]

        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars
            compound_color = fastf1.plotting.get_compound_color(row["Compound"],
                                                                session=session)
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
    from matplotlib.patches import Patch
    # Compound colors
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
            
    # Make the plot more readable and intuitive
    plt.title(f"{session.event['EventName']} {session.event.year}\n Tire Strategies")
    plt.xlabel("Lap Number")
    plt.grid(False)
    # invert the y-axis so drivers that finish higher are closer to the top
    ax.invert_yaxis()

    # Plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_team_pace_ranking(race):
    """
    Team Pace Comparison
    -Rank team's race pace from the fastest to the slowest.
    """
    # Load FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')

    # Pick all quick laps (within 107% of fastest lap).
    # For races with mixed conditions, pick_wo_box() is better.
    laps = race.laps.pick_quicklaps()

    # Convert the lap time column from timedelta to integer.
    # This is a seaborn-specific modification.
    # If plotting with matplotlib, set mpl_timedelta_support to true
    # with plotting.setup_mpl.
    transformed_laps = laps.copy()
    transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    # order the team from the fastest (lowest median lap time) tp slower
    team_order = (
        transformed_laps[["Team", "LapTime (s)"]]
        .groupby("Team")
        .median()["LapTime (s)"]
        .sort_values()
        .index
    )

    # make a color palette associating team names to hex codes
    team_palette = {team: fastf1.plotting.get_team_color(team, session=race)
                    for team in team_order}

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
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
    # Get plot title based on name
    info = race.session_info
    race_name = info['Meeting']['OfficialName']
    plt.title(f"{race.event['EventName']} {race.event.year}\nFastest Lap Time by Team")
    plt.grid(visible=False)

    # x-label is redundant
    ax.set(xlabel=None)
    plt.tight_layout()
    plt.show()


def plot_season_driver_standing(year):
    """Plot driver standings in a heatmap
    ======================================

    Plot the points for each driven in each race of a given season in a heatmap, as
    https://public.tableau.com/app/profile/mateusz.karmalski/viz/F1ResultsTracker2022
    """

    import plotly.express as px
    from plotly.io import show
    from fastf1.ergast import Ergast

    ##############################################################################
    # First, we load the results for selected year
    ergast = Ergast()
    races = ergast.get_race_schedule(year)  # Races for selected year
    results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():

        # Get results. Note that we use the round no. + 1, because the round no.
        # starts from one (1) instead of zero (0)
        temp = ergast.get_race_results(season=year, round=rnd + 1)
        temp = temp.content[0]

        # If there is a sprint, get the results as well
        sprint = ergast.get_sprint_results(season=year, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            temp = pd.merge(temp, sprint.content[0], on='driverCode', how='left')
            # Add sprint points and race points to get the total
            temp['points'] = temp['points_x'] + temp['points_y']
            temp.drop(columns=['points_x', 'points_y'], inplace=True)

        # Add round no. and grand prix name
        temp['round'] = rnd + 1
        temp['race'] = race.removesuffix(' Grand Prix')
        temp = temp[['round', 'race', 'driverCode', 'points']]  # Keep useful cols.
        results.append(temp)

    # Append all races into a single dataframe
    results = pd.concat(results)
    races = results['race'].drop_duplicates()

    ##############################################################################
    # Then we “reshape” the results to a wide table, where each row represents a
    # driver and each column refers to a race, and the cell value is the points.
    results = results.pivot(index='driverCode', columns='round', values='points')
    # Here we have a 22-by-22 matrix (22 races and 22 drivers, incl. DEV and HUL)

    # Rank the drivers by their total points
    results['total_points'] = results.sum(axis=1)
    results = results.sort_values(by='total_points', ascending=False)
    results.drop(columns='total_points', inplace=True)

    # Use race name, instead of round no., as column names
    results.columns = races

    ##############################################################################
    # The final step is to plot a heatmap using plotly
    fig = px.imshow(
        results,
        text_auto=True,
        aspect='auto',  # Automatically adjust the aspect ratio
        color_continuous_scale=[[0,    'rgb(198, 219, 239)'],  # Blue scale
                                [0.25, 'rgb(107, 174, 214)'],
                                [0.5,  'rgb(33,  113, 181)'],
                                [0.75, 'rgb(8,   81,  156)'],
                                [1,    'rgb(8,   48,  107)']],
        labels={'x': 'Race',
                'y': 'Driver',
                'color': 'Points'}       # Change hover texts
    )
    fig.update_xaxes(title_text='')      # Remove axis titles
    fig.update_yaxes(title_text='')
    fig.update_yaxes(tickmode='linear')  # Show all ticks, i.e. driver names
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                    showline=False,
                    tickson='boundaries')              # Show horizontal grid only
    fig.update_xaxes(showgrid=False, showline=False)    # And remove vertical grid
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')     # White background
    fig.update_layout(coloraxis_showscale=False)        # Remove legend
    fig.update_layout(xaxis=dict(side='top'))           # x-axis on top
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # Remove border margins
    fig
    show(fig)


def plot_season_summary(year):
    """
    Season Summary Visualization
    ==================================
    This example demonstrates how to make an interactive season summarization
    dashboard showing points scored by each driver at each round.
    """

    import plotly.graph_objects as go
    from plotly.io import show
    from plotly.subplots import make_subplots
    import fastf1 as ff1

    ##############################################################################
    season = year
    schedule = ff1.get_event_schedule(season, include_testing=False)

    ##############################################################################
    # Get each driver's finishing positions and points total for each round.
    standings = []
    # Shorten the event names by trimming Grand Prix from the name.
    # This will be used to label our graph.
    short_event_names = []

    for _, event in schedule.iterrows():
        event_name, round_number = event["EventName"], event["RoundNumber"]
        short_event_names.append(event_name.replace("Grand Prix", "").strip())

        # Only need to load the results data
        race = ff1.get_session(season, event_name, "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)

        # Add sprint race points if applicable
        sprint = None
        # F1 has used different names for the sprint race event format
        # From 2024 onwards, it has been "sprint_qualifying"
        # In 2023, you should match on "sprint_shootout"
        # In 2022 and 2021, you should match on "sprint"
        if event["EventFormat"] == "sprint_qualifying":
            sprint = ff1.get_session(season, event_name, "S")
            sprint.load(laps=False, telemetry=False, weather=False, messages=False)

        for _, driver_row in race.results.iterrows():
            abbreviation, race_points, race_position = (
                driver_row["Abbreviation"],
                driver_row["Points"],
                driver_row["Position"],
            )

            sprint_points = 0
            if sprint is not None:
                driver_row = sprint.results[
                    sprint.results["Abbreviation"] == abbreviation
                ]
                if not driver_row.empty:
                    # We need the values[0] accessor because driver_row is actually
                    # returned as a dataframe with a single row
                    sprint_points = driver_row["Points"].values[0]

            standings.append(
                {
                    "EventName": event_name,
                    "RoundNumber": round_number,
                    "Driver": abbreviation,
                    "Points": race_points + sprint_points,
                    "Position": race_position,
                }
            )

    ##############################################################################
    # Now we have a dataframe where each row can be seen as two parts.
    # `["EventName", "RoundNumber", "Driver"]` which act like an index.
    # `["Points", "Position"]` which contain the actual data we want to plot.
    df = pd.DataFrame(standings)

    ##############################################################################
    # We remake it into an easier to use format where the row indices are the
    # drivers, and the columns are the races. This allows us to look up the points
    # scored by a driver at a race more easily.
    heatmap_data = df.pivot(
        index="Driver", columns="RoundNumber", values="Points"
    ).fillna(0)

    # Save the final drivers standing and sort the data such that the lowest-
    # scoring driver is towards the bottom
    heatmap_data["total_points"] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values(by="total_points", ascending=True)
    total_points = heatmap_data["total_points"].values
    heatmap_data = heatmap_data.drop(columns=["total_points"])

    # Do the same for position.
    position_data = df.pivot(
        index="Driver", columns="RoundNumber", values="Position"
    ).fillna("N/A")

    ##############################################################################
    # Prepare the hover text
    hover_info = [
        [
            {
                "position": position_data.at[driver, race],
            }
            for race in schedule["RoundNumber"]
        ]
        for driver in heatmap_data.index
    ]

    ##############################################################################
    # Create the subplots for the two heatmaps
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.85, 0.15],
        subplot_titles=("F1 2024 Season Summary", "Total Points"),
    )
    fig.update_layout(width=900, height=800)

    # Per round summary heatmap
    fig.add_trace(
        go.Heatmap(
            # Use the race names as x labels and the driver abbreviations
            # as the y labels
            x=short_event_names,
            y=heatmap_data.index,
            z=heatmap_data.values,
            # Use the points scored as overlay text
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            customdata=hover_info,
            hovertemplate=(
                "Driver: %{y}<br>"
                "Race Name: %{x}<br>"
                "Points: %{z}<br>"
                "Position: %{customdata.position}<extra></extra>"
            ),
            colorscale="YlGnBu",
            showscale=False,
            zmin=0,
            # We need to set zmax for the two heatmaps separately as the
            # max value in the total points plot is significantly higher.
            zmax=heatmap_data.values.max(),
        ),
        row=1,
        col=1,
    )

    # Heatmap for total points
    fig.add_trace(
        go.Heatmap(
            x=["Total Points"] * len(total_points),
            y=heatmap_data.index,
            z=total_points,
            text=total_points,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="YlGnBu",
            showscale=False,
            zmin=0,
            zmax=total_points.max(),
        ),
        row=1,
        col=2,
    )

    # Plot the updated heatmap
    show(fig)


def who_can_still_win_wdc(year, round):
    """Who can still win the drivers WDC?
    ======================================

    Calculates which drivers still has chance to win the WDC.
    Simplified since it doesn't compare positions if points are equal.

    This example implements 3 functions that it then uses to calculate
    its result.
    """

    import fastf1
    from fastf1.ergast import Ergast

    ##############################################################################
    SEASON = year
    ROUND = round

    ##############################################################################
    # Get the current driver standings from Ergast.
    # Reference https://docs.fastf1.dev/ergast.html#fastf1.ergast.Ergast.get_driver_standings
    def get_drivers_standings():
        ergast = Ergast()
        standings = ergast.get_driver_standings(season=SEASON, round=ROUND)
        return standings.content[0]

    ##############################################################################
    # We need a function to calculates the maximum amount of points possible if a
    # driver wins everything left of the season.
    # https://en.wikipedia.org/wiki/List_of_Formula_One_World_Championship_points_scoring_systems
    def calculate_max_points_for_remaining_season():
        POINTS_FOR_SPRINT = 8 + 25 # Winning the sprint and race
        POINTS_FOR_CONVENTIONAL = 25 # Winning the race

        events = fastf1.events.get_event_schedule(SEASON, backend='ergast')
        events = events[events['RoundNumber'] > ROUND]
        # Count how many sprints and conventional races are left
        sprint_events = len(events.loc[events["EventFormat"] == "sprint_shootout"])
        conventional_events = len(events.loc[events["EventFormat"] == "conventional"])

        # Calculate points for each
        sprint_points = sprint_events * POINTS_FOR_SPRINT
        conventional_points = conventional_events * POINTS_FOR_CONVENTIONAL

        return sprint_points + conventional_points

    ##############################################################################
    # For each driver we will see if there is a chance to get more points than
    # the current leader. We assume the leader gets no more points and the
    # driver gets the theoretical maximum amount of points.
    #
    # We currently don't consider the case of two drivers getting equal points
    # since its more complicated and would require comparing positions.
    def calculate_who_can_win(driver_standings, max_points):
        LEADER_POINTS = int(driver_standings.loc[0]['points'])

        for i, _ in enumerate(driver_standings.iterrows()):
            driver = driver_standings.loc[i]
            driver_max_points = int(driver["points"]) + max_points
            can_win = 'No' if driver_max_points < LEADER_POINTS else 'Yes'

            print(f"{driver['position']}: {driver['givenName'] + ' ' + driver['familyName']}, "
                f"Current points: {driver['points']}, "
                f"Theoretical max points: {driver_max_points}, "
                f"Can win: {can_win}")


    ##############################################################################
    # Now using the 3 functions above we can use them to calculate who
    # can still win.

    # Get the current drivers standings
    driver_standings = get_drivers_standings()

    # Get the maximum amount of points
    points = calculate_max_points_for_remaining_season()

    # Print which drivers can still win
    calculate_who_can_win(driver_standings, points)


def plot_annotate_speed_trace(session):
    """Plot speed traces with corner annotations
    ============================================
    Plot the speed over the course of a lap and add annotations to mark corners.
    Currently this plots the fastest lap, but might be worth changing to fastest
    lap by a specific driver
    """

    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

    ##############################################################################
    # First, select the fastest lap and get the car telemetry data for this lap
    fastest_lap = session.laps.pick_fastest()
    car_data = fastest_lap.get_car_data().add_distance()

    ##############################################################################
    # Next, load the circuit info that includes the information about the location
    # of the corners.
    circuit_info = session.get_circuit_info()

    ##############################################################################
    # Finally, we create a plot and plot the speed trace as well as the corner
    # markers.
    team_color = fastf1.plotting.get_team_color(fastest_lap['Team'],
                                                session=session)

    fig, ax = plt.subplots()
    ax.plot(car_data['Distance'], car_data['Speed'],
            color=team_color, label=fastest_lap['Driver'])

    # Draw vertical dotted lines at each corner that range from slightly below the
    # minimum speed to slightly above the maximum speed.
    v_min = car_data['Speed'].min()
    v_max = car_data['Speed'].max()
    ax.vlines(x=circuit_info.corners['Distance'], ymin=v_min-20, ymax=v_max+20,
            linestyles='dotted', colors='grey')

    # Plot the corner number just below each vertical line.
    # For corners that are very close together, the text may overlap. A more
    # complicated approach would be necessary to reliably prevent this.
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        ax.text(corner['Distance'], v_min-30, txt,
                va='center_baseline', ha='center', size='small')

    ax.set_xlabel('Distance in m')
    ax.set_ylabel('Speed in km/h')
    ax.legend()

    # Manually adjust the y-axis limits to include the corner numbers, because
    # Matplotlib does not automatically account for text that was manually added.
    ax.set_ylim([v_min - 40, v_max + 20])
    plt.title(f"{session.event['EventName']} {session.event.year}\nQuickest Lap Time Speeds ({fastest_lap['Driver']})")
    plt.show()


def plot_gear_shifts_on_track(session):
    """Gear shifts on track
    =======================

    Plot which gear is being used at which point of the track
    """
    ##############################################################################
    # Import FastF1 and load the data
    from matplotlib import colormaps
    from matplotlib.collections import LineCollection


    # Get fastest lap and telemetry
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()

    # Prepare the data for plotting by converting it to the appropriate numpy
    # data types
    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gear = tel['nGear'].to_numpy().astype(float)
  
    # Create a line collection. Set a segmented colormap and normalize the plot
    # to full integer values of the colormap
    cmap = colormaps['Paired']
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    # Create the plot
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    ##############################################################################
    # Add a colorbar to the plot. Shift the colorbar ticks by +0.5 so that they
    # are centered for each color segment.
    cbar = plt.colorbar(mappable=lc_comp, label="Gear",
                        boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))

    plt.show()


def plot_speed_on_track(session, driver):
    """
    Speed visualization on track map
    - currently selects the fastest lap of the provided driver/session
    """

    from matplotlib.collections import LineCollection
    import fastf1 as ff1

    ##############################################################################
    # First, we define some variables that allow us to conveniently control what
    # we want to plot.
    # year = 2021
    # wknd = 9
    # ses = 'R'
    year = session.event.year
    colormap = mpl.cm.plasma

    ##############################################################################
    # Next, we load the session and select the desired data.
    lap = session.laps.pick_drivers(driver).pick_fastest()

    # Get telemetry data
    x = lap.telemetry['X']              # values for x-axis
    y = lap.telemetry['Y']              # values for y-axis
    color = lap.telemetry['Speed']      # value to base color gradient on

    ##############################################################################
    # Now, we create a set of line segments so that we can color them
    # individually. This creates the points as a N x 1 x 2 array so that we can
    # stack points  together easily to get the segments. The segments array for
    # line collection needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ##############################################################################
    # After this, we can actually plot the data.

    # We create a plot with title and adjust some setting to make it look good.
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
    fig.suptitle(f'{session.event['EventName']} {year} - {driver} - Track Speed', size=24, y=0.97)

    # Adjust margins and turn of axis
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis('off')

    # After this, we plot the data itself.
    # Create background track line
    ax.plot(lap.telemetry['X'], lap.telemetry['Y'],
            color='black', linestyle='-', linewidth=16, zorder=0)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm,
                        linestyle='-', linewidth=5)

    # Set the values used for colormapping
    lc.set_array(color)

    # Merge all line segments together
    line = ax.add_collection(lc)

    # Finally, we create a color bar as a legend.
    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap,
                                    orientation="horizontal")

    # Show the plot
    plt.show()


def plot_multi_driver_speed_traces(session, driver1, driver2):
    """Overlaying speed traces of two laps
    ======================================
    -Compare two fastest laps by overlaying their speed traces
    -Should set to qualifying at some point?
    """

    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

    ##############################################################################
    # First, we select the two laps that we want to compare
    ver_lap = session.laps.pick_drivers(driver1).pick_fastest()
    ham_lap = session.laps.pick_drivers(driver2).pick_fastest()

    ##############################################################################
    # Next we get the telemetry data for each lap. We also add a 'Distance' column
    # to the telemetry dataframe as this makes it easier to compare the laps.
    ver_tel = ver_lap.get_car_data().add_distance()
    ham_tel = ham_lap.get_car_data().add_distance()

    ##############################################################################
    # Finally, we create a plot and plot both speed traces.
    # We color the individual lines with the driver's team colors.
    rbr_color = fastf1.plotting.get_team_color(ver_lap['Team'], session=session)
    mer_color = fastf1.plotting.get_team_color(ham_lap['Team'], session=session)

    fig, ax = plt.subplots()
    ax.plot(ver_tel['Distance'], ver_tel['Speed'], color=rbr_color, label='VER')
    ax.plot(ham_tel['Distance'], ham_tel['Speed'], color=mer_color, label='HAM')

    ax.set_xlabel('Distance in m')
    ax.set_ylabel('Speed in km/h')

    ax.legend()
    plt.suptitle(f"Fastest Race Lap Comparison for {driver1} and {driver2}\n "
                f"{session.event['EventName']} {session.event.year}")

    plt.show()



# Configure FastF1
# Create cache directory if it doesn't exist
cache_dir = '.fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

# Year selection
current_year = datetime.now().year
year = current_year

# Get schedule for selected year
schedule = get_schedule(current_year)

if schedule is not None:
    # Filter to only past events (and current event if it's started)
    now = pd.Timestamp.now()
    # Convert EventDate to datetime if it's not already
    if 'EventDate' in schedule.columns:
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    available_events = schedule[schedule['EventDate'] <= now]

    if len(available_events) == 0:
        print(f"No completed races available for {year} yet.")
        sys.exit()

    # Race selection
    race_names = available_events['EventName'].tolist()

    race_name = 'United Arab Emirates Grand Prix'
    session_type = 'Race'
    # Map session names to FastF1 codes
    session_map = {
        'Race': 'R',
        'Qualifying': 'Q',
        'Sprint': 'S',
        'FP1': 'FP1',
        'FP2': 'FP2',
        'FP3': 'FP3'
    }

    # Load and cache session data
    session = load_session(year, race_name, session_map[session_type])

    # Get list of drivers
    drivers = session.laps['Driver'].unique().tolist()
    drivers.sort()


who_can_still_win_wdc(year, 12)
sys.exit()


plot_multi_driver_speed_traces(session,'HAM', 'VER')
plot_speed_on_track(session, 'HAM')
plot_gear_shifts_on_track(session)
draw_track_layout(session)
plot_annotate_speed_trace(session)
who_can_still_win_wdc(year, 12)  # inputs are season year and round

# Call plots
plot_season_summary(year)
plot_season_driver_standing(year)
plot_team_pace_ranking(session)
plot_strategy(session)
plot_qualifying_results(session)
plot_position_changes(session)
plot_quickest_lap_times(session)
plot_laptime_distribution(session)
draw_track_layout(session)
plot_driver_laptimes(session, 'ALO')
