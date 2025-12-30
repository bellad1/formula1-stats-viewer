# F1 Data Viewer

An interactive Streamlit web application for exploring and analyzing Formula 1 race data using FastF1.

## Features

- **Lap Times Analysis**: Compare lap times between drivers throughout the session
- **Speed Comparison**: Overlay telemetry speed traces to see where drivers gain/lose time
- **Position Changes**: Track how positions evolved during the race
- **Tire Strategy**: Visualize pit stops and compound choices
- **Track Map**: See speed visualized on the actual track layout
- **Qualifying Results**: View qualifying results with time deltas (for Qualifying/Sprint sessions)
- **Lap Distribution**: Analyze lap time distributions with tire compounds (for Race sessions)
- **Team Pace**: Compare team performance with box plots (for Race sessions)
- **Track Details**: Gear shifts and speed traces with corner markers

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/f1_stats_viewer.git
cd f1_stats_viewer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run f1_viewer_app.py
```

## Usage

1. Select a year, race, and session type from the sidebar
2. Click "Load Session" (first load may take 2-3 minutes as data is downloaded)
3. Explore the data using the different tabs
4. Data is cached locally, so subsequent loads will be much faster

## Technologies

- **FastF1**: For accessing F1 telemetry and timing data
- **Streamlit**: For the interactive web interface
- **Plotly**: For interactive charts
- **Matplotlib**: For static visualizations
- **Seaborn**: For statistical visualizations
- **Pandas & NumPy**: For data processing

## Data Source

This application uses the [FastF1](https://github.com/theOehrly/Fast-F1) library, which provides access to Formula 1 timing and telemetry data.

## License

MIT License
