"""
Interactive dashboard system for Safe RL analysis and visualization.

This module provides comprehensive dashboard capabilities including
training monitoring, safety analysis, and comparison dashboards.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from pathlib import Path
import json
import threading
import time
from datetime import datetime, timedelta
import queue
import sqlite3
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardManager:
    """
    Central manager for all dashboard instances and data coordination.
    """
    
    def __init__(self, data_dir: Union[str, Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("dashboard_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database for real-time data storage
        self.db_path = self.data_dir / "dashboard_data.db"
        self._init_database()
        
        # Active dashboards registry
        self.active_dashboards = {}
        self.data_queue = queue.Queue()
        self.is_running = False
        
        # Real-time update thread
        self.update_thread = None
        
    def _init_database(self):
        """Initialize SQLite database for dashboard data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Training data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                iteration INTEGER,
                episode_return REAL,
                episode_length INTEGER,
                policy_loss REAL,
                value_loss REAL,
                kl_divergence REAL,
                constraint_violation REAL,
                safety_score REAL,
                metadata TEXT
            )
        ''')
        
        # Safety events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                severity TEXT,
                constraint_name TEXT,
                violation_value REAL,
                state_info TEXT,
                action_info TEXT,
                metadata TEXT
            )
        ''')
        
        # Experiment metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                algorithm TEXT,
                environment TEXT,
                hyperparameters TEXT,
                start_time DATETIME,
                end_time DATETIME,
                status TEXT,
                description TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def register_dashboard(self, dashboard_id: str, dashboard_instance):
        """Register a dashboard instance for management."""
        self.active_dashboards[dashboard_id] = dashboard_instance
        logger.info(f"Registered dashboard: {dashboard_id}")
        
    def start_real_time_updates(self, update_interval: float = 1.0):
        """Start real-time data updates for all dashboards."""
        if self.is_running:
            logger.warning("Real-time updates already running")
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_interval,),
            daemon=True
        )
        self.update_thread.start()
        logger.info("Started real-time dashboard updates")
        
    def stop_real_time_updates(self):
        """Stop real-time data updates."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Stopped real-time dashboard updates")
        
    def _update_loop(self, update_interval: float):
        """Main update loop for real-time data."""
        while self.is_running:
            try:
                # Process queued data updates
                while not self.data_queue.empty():
                    data_update = self.data_queue.get_nowait()
                    self._process_data_update(data_update)
                
                # Trigger dashboard updates
                for dashboard_id, dashboard in self.active_dashboards.items():
                    if hasattr(dashboard, 'update_real_time_data'):
                        dashboard.update_real_time_data()
                        
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {str(e)}")
                time.sleep(update_interval)
    
    def _process_data_update(self, data_update: Dict[str, Any]):
        """Process and store data updates."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            data_type = data_update.get('type')
            
            if data_type == 'training':
                cursor.execute('''
                    INSERT INTO training_data (
                        experiment_id, iteration, episode_return, episode_length,
                        policy_loss, value_loss, kl_divergence, constraint_violation,
                        safety_score, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_update.get('experiment_id'),
                    data_update.get('iteration'),
                    data_update.get('episode_return'),
                    data_update.get('episode_length'),
                    data_update.get('policy_loss'),
                    data_update.get('value_loss'),
                    data_update.get('kl_divergence'),
                    data_update.get('constraint_violation'),
                    data_update.get('safety_score'),
                    json.dumps(data_update.get('metadata', {}))
                ))
                
            elif data_type == 'safety_event':
                cursor.execute('''
                    INSERT INTO safety_events (
                        experiment_id, event_type, severity, constraint_name,
                        violation_value, state_info, action_info, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_update.get('experiment_id'),
                    data_update.get('event_type'),
                    data_update.get('severity'),
                    data_update.get('constraint_name'),
                    data_update.get('violation_value'),
                    json.dumps(data_update.get('state_info', {})),
                    json.dumps(data_update.get('action_info', {})),
                    json.dumps(data_update.get('metadata', {}))
                ))
                
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error processing data update: {str(e)}")
        finally:
            conn.close()
    
    def add_data_point(self, data_point: Dict[str, Any]):
        """Add a data point to the update queue."""
        self.data_queue.put(data_point)
    
    def get_recent_data(self, experiment_id: str, data_type: str = 'training', 
                       limit: int = 1000) -> pd.DataFrame:
        """Get recent data for dashboard updates."""
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            if data_type == 'training':
                query = '''
                    SELECT * FROM training_data 
                    WHERE experiment_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
            elif data_type == 'safety':
                query = '''
                    SELECT * FROM safety_events 
                    WHERE experiment_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
            else:
                raise ValueError(f"Unknown data type: {data_type}")
                
            df = pd.read_sql_query(query, conn, params=(experiment_id, limit))
            return df
            
        except Exception as e:
            logger.error(f"Error getting recent data: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()


class TrainingDashboard:
    """
    Interactive dashboard for real-time training monitoring.
    """
    
    def __init__(self, dashboard_manager: DashboardManager, 
                 experiment_id: str = "default"):
        self.dashboard_manager = dashboard_manager
        self.experiment_id = experiment_id
        self.app = None
        self.port = 8050
        
        # Dashboard data cache
        self.data_cache = {
            'training_data': pd.DataFrame(),
            'last_update': datetime.now(),
            'update_interval': 2.0  # seconds
        }
        
        # Register with dashboard manager
        self.dashboard_manager.register_dashboard(f"training_{experiment_id}", self)
        
    def create_app(self) -> dash.Dash:
        """Create and configure the Dash application."""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        return self.app
    
    def _create_layout(self) -> html.Div:
        """Create the dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Safe RL Training Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#2c3e50'}),
                    html.H3(f"Experiment: {self.experiment_id}",
                           className="text-center mb-4",
                           style={'color': '#7f8c8d'})
                ])
            ]),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Update Interval (seconds)"),
                                    dcc.Slider(
                                        id='update-interval-slider',
                                        min=0.5,
                                        max=10.0,
                                        step=0.5,
                                        value=2.0,
                                        marks={i: str(i) for i in range(1, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Data Points to Show"),
                                    dcc.Slider(
                                        id='data-points-slider',
                                        min=50,
                                        max=2000,
                                        step=50,
                                        value=500,
                                        marks={i: str(i) for i in range(0, 2001, 500)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Pause Updates", id="pause-button", 
                                             color="warning", className="me-2"),
                                    dbc.Button("Reset View", id="reset-button", 
                                             color="secondary", className="me-2"),
                                    dbc.Button("Export Data", id="export-button", 
                                             color="success")
                                ], className="mt-3")
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Main metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Episode Return"),
                        dbc.CardBody([
                            dcc.Graph(id='episode-return-graph',
                                    config={'displayModeBar': True})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Safety Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='safety-metrics-graph',
                                    config={'displayModeBar': True})
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Training progress row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Loss Functions"),
                        dbc.CardBody([
                            dcc.Graph(id='loss-functions-graph',
                                    config={'displayModeBar': True})
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Statistics"),
                        dbc.CardBody([
                            html.Div(id='training-stats-table')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Detailed metrics row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Policy Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='policy-metrics-graph',
                                    config={'displayModeBar': True})
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Auto-update interval store
            dcc.Interval(
                id='interval-component',
                interval=2000,  # 2 seconds
                n_intervals=0
            ),
            
            # Data storage
            dcc.Store(id='training-data-store'),
            dcc.Store(id='dashboard-state-store', data={'paused': False})
            
        ], fluid=True)
    
    def _register_callbacks(self):
        """Register all dashboard callbacks."""
        
        @self.app.callback(
            Output('training-data-store', 'data'),
            Input('interval-component', 'n_intervals'),
            Input('dashboard-state-store', 'data')
        )
        def update_data_store(n_intervals, dashboard_state):
            if dashboard_state.get('paused', False):
                return dash.no_update
                
            # Get recent training data
            df = self.dashboard_manager.get_recent_data(
                self.experiment_id, 'training', limit=1000
            )
            
            if df.empty:
                return {}
                
            return df.to_dict('records')
        
        @self.app.callback(
            [Output('episode-return-graph', 'figure'),
             Output('safety-metrics-graph', 'figure'),
             Output('loss-functions-graph', 'figure'),
             Output('policy-metrics-graph', 'figure'),
             Output('training-stats-table', 'children')],
            [Input('training-data-store', 'data'),
             Input('data-points-slider', 'value')]
        )
        def update_graphs(data, data_points):
            if not data:
                # Return empty figures
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available")
                return empty_fig, empty_fig, empty_fig, empty_fig, "No data available"
            
            df = pd.DataFrame(data).tail(data_points)
            
            # Episode return graph
            return_fig = go.Figure()
            return_fig.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['episode_return'],
                mode='lines',
                name='Episode Return',
                line=dict(color='blue', width=2)
            ))
            
            # Add rolling average
            if len(df) > 10:
                rolling_avg = df['episode_return'].rolling(window=10).mean()
                return_fig.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=rolling_avg,
                    mode='lines',
                    name='Rolling Average (10)',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            return_fig.update_layout(
                title='Episode Return Over Time',
                xaxis_title='Iteration',
                yaxis_title='Episode Return',
                hovermode='x unified'
            )
            
            # Safety metrics graph
            safety_fig = go.Figure()
            
            if 'constraint_violation' in df.columns:
                safety_fig.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=df['constraint_violation'],
                    mode='lines',
                    name='Constraint Violation',
                    line=dict(color='red', width=2)
                ))
            
            if 'safety_score' in df.columns:
                safety_fig.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=df['safety_score'],
                    mode='lines',
                    name='Safety Score',
                    yaxis='y2',
                    line=dict(color='green', width=2)
                ))
                
                safety_fig.update_layout(
                    yaxis2=dict(
                        title='Safety Score',
                        overlaying='y',
                        side='right'
                    )
                )
            
            safety_fig.update_layout(
                title='Safety Metrics Over Time',
                xaxis_title='Iteration',
                yaxis_title='Constraint Violation',
                hovermode='x unified'
            )
            
            # Loss functions graph
            loss_fig = go.Figure()
            
            loss_columns = ['policy_loss', 'value_loss']
            colors = ['blue', 'orange']
            
            for i, col in enumerate(loss_columns):
                if col in df.columns:
                    loss_fig.add_trace(go.Scatter(
                        x=df['iteration'],
                        y=df[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        line=dict(color=colors[i], width=2)
                    ))
            
            loss_fig.update_layout(
                title='Training Losses Over Time',
                xaxis_title='Iteration',
                yaxis_title='Loss Value',
                hovermode='x unified'
            )
            
            # Policy metrics graph
            policy_fig = go.Figure()
            
            if 'kl_divergence' in df.columns:
                policy_fig.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=df['kl_divergence'],
                    mode='lines',
                    name='KL Divergence',
                    line=dict(color='purple', width=2)
                ))
            
            policy_fig.update_layout(
                title='Policy Update Metrics',
                xaxis_title='Iteration',
                yaxis_title='KL Divergence',
                hovermode='x unified'
            )
            
            # Training statistics table
            if not df.empty:
                latest = df.iloc[-1]
                stats_table = dbc.Table([
                    html.Tbody([
                        html.Tr([html.Td("Current Iteration"), html.Td(f"{latest['iteration']}")]),
                        html.Tr([html.Td("Latest Return"), html.Td(f"{latest['episode_return']:.3f}")]),
                        html.Tr([html.Td("Episode Length"), html.Td(f"{latest.get('episode_length', 'N/A')}")]),
                        html.Tr([html.Td("Safety Score"), html.Td(f"{latest.get('safety_score', 'N/A')}")]),
                        html.Tr([html.Td("Last Update"), html.Td(datetime.now().strftime("%H:%M:%S"))])
                    ])
                ], bordered=True, striped=True, size='sm')
            else:
                stats_table = "No statistics available"
            
            return return_fig, safety_fig, loss_fig, policy_fig, stats_table
        
        @self.app.callback(
            Output('interval-component', 'interval'),
            Input('update-interval-slider', 'value')
        )
        def update_interval(interval_seconds):
            return interval_seconds * 1000  # Convert to milliseconds
        
        @self.app.callback(
            Output('dashboard-state-store', 'data'),
            Input('pause-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def toggle_pause(n_clicks):
            if n_clicks:
                return {'paused': True}
            return {'paused': False}
    
    def run(self, host: str = '127.0.0.1', port: int = None, debug: bool = False):
        """Run the dashboard server."""
        if not self.app:
            self.create_app()
        
        if port:
            self.port = port
            
        logger.info(f"Starting training dashboard at http://{host}:{self.port}")
        self.app.run_server(host=host, port=self.port, debug=debug)


class SafetyDashboard:
    """
    Interactive dashboard for safety monitoring and analysis.
    """
    
    def __init__(self, dashboard_manager: DashboardManager, 
                 experiment_id: str = "default"):
        self.dashboard_manager = dashboard_manager
        self.experiment_id = experiment_id
        self.app = None
        self.port = 8051
        
        # Register with dashboard manager
        self.dashboard_manager.register_dashboard(f"safety_{experiment_id}", self)
    
    def create_app(self) -> dash.Dash:
        """Create and configure the safety dashboard."""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        self.app.layout = self._create_safety_layout()
        self._register_safety_callbacks()
        
        return self.app
    
    def _create_safety_layout(self) -> html.Div:
        """Create the safety dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Safety Monitoring Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#c0392b'}),
                    html.H3(f"Experiment: {self.experiment_id}",
                           className="text-center mb-4",
                           style={'color': '#7f8c8d'})
                ])
            ]),
            
            # Safety alert banner
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        [
                            html.H4("Safety Status: ", className="alert-heading"),
                            html.Span(id="safety-status-text")
                        ],
                        id="safety-alert",
                        color="success",
                        className="mb-4"
                    )
                ])
            ]),
            
            # Safety metrics overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Constraint Violations Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id='violations-timeline-graph')
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Violation Severity Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id='severity-distribution-graph')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Detailed safety analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Safety Event Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id='safety-events-graph')
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Heatmap"),
                        dbc.CardBody([
                            dcc.Graph(id='risk-heatmap-graph')
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Safety events table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Safety Events"),
                        dbc.CardBody([
                            html.Div(id='safety-events-table')
                        ])
                    ])
                ])
            ]),
            
            # Update interval
            dcc.Interval(
                id='safety-interval-component',
                interval=1000,  # 1 second
                n_intervals=0
            ),
            
            dcc.Store(id='safety-data-store')
            
        ], fluid=True)
    
    def _register_safety_callbacks(self):
        """Register safety dashboard callbacks."""
        
        @self.app.callback(
            [Output('safety-data-store', 'data'),
             Output('safety-status-text', 'children'),
             Output('safety-alert', 'color')],
            Input('safety-interval-component', 'n_intervals')
        )
        def update_safety_data(n_intervals):
            # Get recent safety events
            safety_df = self.dashboard_manager.get_recent_data(
                self.experiment_id, 'safety', limit=100
            )
            
            # Get recent training data for safety scores
            training_df = self.dashboard_manager.get_recent_data(
                self.experiment_id, 'training', limit=100
            )
            
            # Determine overall safety status
            status_text = "All systems operational"
            alert_color = "success"
            
            if not safety_df.empty:
                recent_events = safety_df.head(10)
                high_severity_events = recent_events[
                    recent_events['severity'].isin(['high', 'critical'])
                ]
                
                if len(high_severity_events) > 0:
                    status_text = f"⚠️ {len(high_severity_events)} high-severity events in last 10"
                    alert_color = "danger"
                elif len(recent_events) > 5:
                    status_text = f"⚡ {len(recent_events)} safety events detected"
                    alert_color = "warning"
            
            return {
                'safety_events': safety_df.to_dict('records') if not safety_df.empty else [],
                'training_data': training_df.to_dict('records') if not training_df.empty else []
            }, status_text, alert_color
        
        @self.app.callback(
            [Output('violations-timeline-graph', 'figure'),
             Output('severity-distribution-graph', 'figure'),
             Output('safety-events-graph', 'figure'),
             Output('risk-heatmap-graph', 'figure'),
             Output('safety-events-table', 'children')],
            Input('safety-data-store', 'data')
        )
        def update_safety_graphs(data):
            if not data or not data.get('safety_events'):
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No safety data available")
                return empty_fig, empty_fig, empty_fig, empty_fig, "No safety events"
            
            safety_df = pd.DataFrame(data['safety_events'])
            
            # Violations timeline
            timeline_fig = go.Figure()
            
            # Group violations by time and count
            safety_df['timestamp'] = pd.to_datetime(safety_df['timestamp'])
            safety_df['hour'] = safety_df['timestamp'].dt.hour
            hourly_counts = safety_df.groupby('hour').size().reset_index(name='count')
            
            timeline_fig.add_trace(go.Bar(
                x=hourly_counts['hour'],
                y=hourly_counts['count'],
                name='Violations per Hour',
                marker_color='red',
                opacity=0.7
            ))
            
            timeline_fig.update_layout(
                title='Constraint Violations Timeline',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Violations'
            )
            
            # Severity distribution
            severity_fig = go.Figure()
            severity_counts = safety_df['severity'].value_counts()
            
            severity_fig.add_trace(go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                hole=0.4
            ))
            
            severity_fig.update_layout(
                title='Violation Severity Distribution'
            )
            
            # Safety events over time
            events_fig = go.Figure()
            
            event_types = safety_df['event_type'].value_counts()
            for event_type in event_types.index[:5]:  # Top 5 event types
                event_data = safety_df[safety_df['event_type'] == event_type]
                events_fig.add_trace(go.Scatter(
                    x=event_data['timestamp'],
                    y=event_data['violation_value'],
                    mode='markers',
                    name=event_type,
                    marker=dict(size=8)
                ))
            
            events_fig.update_layout(
                title='Safety Events Over Time',
                xaxis_title='Timestamp',
                yaxis_title='Violation Value'
            )
            
            # Risk heatmap (simplified)
            heatmap_fig = go.Figure()
            
            # Create risk matrix based on severity and frequency
            risk_matrix = np.random.rand(5, 5)  # Placeholder
            
            heatmap_fig.add_trace(go.Heatmap(
                z=risk_matrix,
                colorscale='Reds',
                showscale=True
            ))
            
            heatmap_fig.update_layout(
                title='Risk Assessment Heatmap',
                xaxis_title='Risk Category',
                yaxis_title='Severity Level'
            )
            
            # Safety events table
            recent_events = safety_df.head(10)
            
            table_rows = []
            for _, event in recent_events.iterrows():
                table_rows.append(
                    html.Tr([
                        html.Td(event['timestamp'].strftime('%H:%M:%S')),
                        html.Td(event['event_type']),
                        html.Td(
                            dbc.Badge(
                                event['severity'].upper(),
                                color='danger' if event['severity'] in ['high', 'critical'] 
                                      else 'warning' if event['severity'] == 'medium' 
                                      else 'secondary'
                            )
                        ),
                        html.Td(event.get('constraint_name', 'N/A')),
                        html.Td(f"{event.get('violation_value', 0):.3f}")
                    ])
                )
            
            events_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th('Time'),
                        html.Th('Event Type'),
                        html.Th('Severity'),
                        html.Th('Constraint'),
                        html.Th('Value')
                    ])
                ]),
                html.Tbody(table_rows)
            ], bordered=True, striped=True, size='sm')
            
            return timeline_fig, severity_fig, events_fig, heatmap_fig, events_table
    
    def run(self, host: str = '127.0.0.1', port: int = None, debug: bool = False):
        """Run the safety dashboard server."""
        if not self.app:
            self.create_app()
        
        if port:
            self.port = port
            
        logger.info(f"Starting safety dashboard at http://{host}:{self.port}")
        self.app.run_server(host=host, port=self.port, debug=debug)


def create_unified_dashboard(dashboard_manager: DashboardManager,
                           experiment_id: str = "default",
                           port: int = 8052) -> dash.Dash:
    """
    Create a unified dashboard combining training and safety monitoring.
    
    Args:
        dashboard_manager: Dashboard manager instance
        experiment_id: Experiment identifier
        port: Port to run the dashboard on
        
    Returns:
        Configured Dash application
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        # Navigation tabs
        dcc.Tabs(id="dashboard-tabs", value='training-tab', children=[
            dcc.Tab(label='Training Monitor', value='training-tab'),
            dcc.Tab(label='Safety Monitor', value='safety-tab'),
            dcc.Tab(label='Combined Analysis', value='combined-tab')
        ]),
        
        html.Div(id='tab-content'),
        
        # Global update interval
        dcc.Interval(
            id='global-interval-component',
            interval=2000,  # 2 seconds
            n_intervals=0
        ),
        
        dcc.Store(id='global-data-store')
        
    ], fluid=True)
    
    @app.callback(
        Output('tab-content', 'children'),
        Input('dashboard-tabs', 'value')
    )
    def render_tab_content(active_tab):
        if active_tab == 'training-tab':
            return html.Div([
                html.H3("Training Monitoring", className="mb-4"),
                # Training dashboard content would go here
                html.P("Training metrics and real-time monitoring...")
            ])
        elif active_tab == 'safety-tab':
            return html.Div([
                html.H3("Safety Monitoring", className="mb-4"),
                # Safety dashboard content would go here
                html.P("Safety events and constraint monitoring...")
            ])
        elif active_tab == 'combined-tab':
            return html.Div([
                html.H3("Combined Analysis", className="mb-4"),
                # Combined analysis content
                html.P("Integrated training and safety analysis...")
            ])
    
    return app


# Example usage and testing
if __name__ == "__main__":
    # Create dashboard manager
    manager = DashboardManager()
    
    # Create training dashboard
    training_dash = TrainingDashboard(manager, "test_experiment")
    
    # Create some test data
    import threading
    import time
    
    def generate_test_data():
        """Generate test data for dashboard demonstration."""
        for i in range(1000):
            # Training data point
            training_point = {
                'type': 'training',
                'experiment_id': 'test_experiment',
                'iteration': i,
                'episode_return': np.random.normal(100 + i * 0.1, 10),
                'episode_length': np.random.poisson(200),
                'policy_loss': np.random.exponential(0.1),
                'value_loss': np.random.exponential(0.05),
                'kl_divergence': np.random.exponential(0.01),
                'constraint_violation': max(0, np.random.normal(0.02, 0.01)),
                'safety_score': np.random.beta(2, 1)
            }
            manager.add_data_point(training_point)
            
            # Occasional safety event
            if np.random.random() < 0.05:
                safety_event = {
                    'type': 'safety_event',
                    'experiment_id': 'test_experiment',
                    'event_type': np.random.choice(['collision', 'boundary_violation', 'speed_limit']),
                    'severity': np.random.choice(['low', 'medium', 'high']),
                    'constraint_name': 'safety_constraint_1',
                    'violation_value': np.random.uniform(0.1, 1.0)
                }
                manager.add_data_point(safety_event)
            
            time.sleep(0.1)  # 10 data points per second
    
    # Start test data generation
    data_thread = threading.Thread(target=generate_test_data, daemon=True)
    data_thread.start()
    
    # Start real-time updates
    manager.start_real_time_updates()
    
    # Run training dashboard
    training_dash.create_app()
    print("Starting dashboard at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop")
    
    try:
        training_dash.run(debug=True)
    except KeyboardInterrupt:
        print("Stopping dashboard...")
        manager.stop_real_time_updates()