import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sustainable Agriculture AI/ML System"

# API base URL
API_BASE_URL = "http://localhost:5000"

# Sample data for demonstration
def generate_sample_data():
    """Generate sample data for visualization."""
    np.random.seed(42)
    n_points = 100
    
    data = {
        'temperature': np.random.normal(25, 8, n_points),
        'rainfall': np.random.exponential(50, n_points),
        'soil_ph': np.random.normal(6.5, 1.0, n_points),
        'nitrogen': np.random.normal(30, 10, n_points),
        'yield': np.random.normal(5.0, 1.0, n_points)
    }
    
    return pd.DataFrame(data)

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🌱 Sustainable Agriculture AI/ML System", 
                    className="text-center text-primary mb-4"),
            html.P("Predict crop yields and get sustainable practice recommendations", 
                   className="text-center text-muted")
        ])
    ]),
    
    # Navigation tabs
    dbc.Tabs([
        # Crop Yield Prediction Tab
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3("Crop Yield Prediction", className="mb-4"),
                    html.P("Enter environmental and soil conditions to predict crop yield.")
                ])
            ]),
            
            # Input form
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Environmental Conditions", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Temperature (°C)"),
                                    dbc.Input(id="temp-input", type="number", value=25, step=0.1)
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Rainfall (mm/month)"),
                                    dbc.Input(id="rainfall-input", type="number", value=80, step=1)
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Humidity (%)"),
                                    dbc.Input(id="humidity-input", type="number", value=65, step=1)
                                ], width=4)
                            ], className="mb-3"),
                            
                            html.H5("Soil Conditions", className="card-title mt-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Soil pH"),
                                    dbc.Input(id="ph-input", type="number", value=6.5, step=0.1, min=4, max=9)
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Nitrogen (mg/kg)"),
                                    dbc.Input(id="nitrogen-input", type="number", value=30, step=1)
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Phosphorus (mg/kg)"),
                                    dbc.Input(id="phosphorus-input", type="number", value=20, step=1)
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Potassium (mg/kg)"),
                                    dbc.Input(id="potassium-input", type="number", value=200, step=1)
                                ], width=3)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Organic Matter (%)"),
                                    dbc.Input(id="om-input", type="number", value=3.0, step=0.1)
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Irrigation Frequency (times/week)"),
                                    dbc.Input(id="irrigation-input", type="number", value=3, step=1)
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Fertilizer Usage (kg/hectare)"),
                                    dbc.Input(id="fertilizer-input", type="number", value=100, step=1)
                                ], width=4)
                            ]),
                            
                            dbc.Button("Predict Yield", id="predict-btn", 
                                      color="primary", className="mt-3 w-100")
                        ])
                    ])
                ], width=6),
                
                # Results display
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Prediction Results", className="card-title"),
                            html.Div(id="prediction-results", className="mt-3")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Feature importance chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Feature Importance", className="card-title"),
                            dcc.Graph(id="feature-importance-chart")
                        ])
                    ])
                ])
            ])
        ], label="🌾 Crop Yield Prediction", tab_id="prediction"),
        
        # Sustainable Practices Tab
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3("Sustainable Practice Recommendations", className="mb-4"),
                    html.P("Get AI-powered recommendations for sustainable farming practices.")
                ])
            ]),
            
            # Input form for recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Current Conditions", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Crop Type"),
                                    dbc.Select(
                                        id="crop-select",
                                        options=[
                                            {"label": "Wheat", "value": "wheat"},
                                            {"label": "Corn", "value": "corn"},
                                            {"label": "Soybeans", "value": "soybeans"},
                                            {"label": "Rice", "value": "rice"}
                                        ],
                                        value="wheat"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Rainfall (mm/month)"),
                                    dbc.Input(id="rec-rainfall-input", type="number", value=60, step=1)
                                ], width=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Organic Matter (%)"),
                                    dbc.Input(id="rec-om-input", type="number", value=2.5, step=0.1)
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Nitrogen (mg/kg)"),
                                    dbc.Input(id="rec-nitrogen-input", type="number", value=25, step=1)
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Phosphorus (mg/kg)"),
                                    dbc.Input(id="rec-phosphorus-input", type="number", value=18, step=1)
                                ], width=4)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Temperature (°C)"),
                                    dbc.Input(id="rec-temp-input", type="number", value=28, step=0.1)
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Humidity (%)"),
                                    dbc.Input(id="rec-humidity-input", type="number", value=45, step=1)
                                ], width=6)
                            ]),
                            
                            dbc.Button("Get Recommendations", id="recommend-btn", 
                                      color="success", className="mt-3 w-100")
                        ])
                    ])
                ], width=6),
                
                # Recommendations display
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("AI Recommendations", className="card-title"),
                            html.Div(id="recommendations-results", className="mt-3")
                        ])
                    ])
                ], width=6)
            ])
        ], label="🌿 Sustainable Practices", tab_id="practices"),
        
        # System Status Tab
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3("System Status", className="mb-4"),
                    html.P("Monitor the health and status of the AI/ML system.")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("API Health", className="card-title"),
                            html.Div(id="api-status", className="mt-3")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Model Status", className="card-title"),
                            html.Div(id="model-status", className="mt-3")
                        ])
                    ])
                ], width=6)
            ])
        ], label="🔧 System Status", tab_id="status")
    ], id="main-tabs"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Sustainable Agriculture AI/ML System - Built with Dash and Flask", 
                   className="text-center text-muted")
        ])
    ], className="mt-5")
], fluid=True, className="py-4")

# Callbacks
@app.callback(
    Output("prediction-results", "children"),
    Input("predict-btn", "n_clicks"),
    State("temp-input", "value"),
    State("rainfall-input", "value"),
    State("humidity-input", "value"),
    State("ph-input", "value"),
    State("nitrogen-input", "value"),
    State("phosphorus-input", "value"),
    State("potassium-input", "value"),
    State("om-input", "value"),
    State("irrigation-input", "value"),
    State("fertilizer-input", "value"),
    prevent_initial_call=True
)
def predict_yield(n_clicks, temp, rainfall, humidity, ph, nitrogen, 
                  phosphorus, potassium, om, irrigation, fertilizer):
    """Handle crop yield prediction."""
    if n_clicks is None:
        return ""
    
    try:
        # Prepare data for API call
        data = {
            "temperature": temp,
            "rainfall": rainfall,
            "humidity": humidity,
            "soil_ph": ph,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "organic_matter": om,
            "irrigation_frequency": irrigation,
            "fertilizer_usage": fertilizer
        }
        
        # Call API
        response = requests.post(f"{API_BASE_URL}/predict/yield", json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            return dbc.Alert([
                html.H4(f"Predicted Yield: {result['predicted_yield']} {result['unit']}"),
                html.P(f"Confidence: {result['confidence']:.1%}"),
                html.Hr(),
                html.P("Input Parameters:"),
                html.Ul([
                    html.Li(f"Temperature: {temp}°C"),
                    html.Li(f"Rainfall: {rainfall} mm/month"),
                    html.Li(f"Humidity: {humidity}%"),
                    html.Li(f"Soil pH: {ph}"),
                    html.Li(f"Nitrogen: {nitrogen} mg/kg"),
                    html.Li(f"Phosphorus: {phosphorus} mg/kg"),
                    html.Li(f"Potassium: {potassium} mg/kg"),
                    html.Li(f"Organic Matter: {om}%"),
                    html.Li(f"Irrigation: {irrigation} times/week"),
                    html.Li(f"Fertilizer: {fertilizer} kg/hectare")
                ])
            ], color="success")
        else:
            return dbc.Alert(f"Error: {response.text}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output("recommendations-results", "children"),
    Input("recommend-btn", "n_clicks"),
    State("crop-select", "value"),
    State("rec-rainfall-input", "value"),
    State("rec-om-input", "value"),
    State("rec-nitrogen-input", "value"),
    State("rec-phosphorus-input", "value"),
    State("rec-temp-input", "value"),
    State("rec-humidity-input", "value"),
    prevent_initial_call=True
)
def get_recommendations(n_clicks, crop_type, rainfall, om, nitrogen, 
                       phosphorus, temp, humidity):
    """Handle sustainable practice recommendations."""
    if n_clicks is None:
        return ""
    
    try:
        # Prepare data for API call
        data = {
            "soil_conditions": {
                "organic_matter": om,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": 180  # Default value
            },
            "weather_conditions": {
                "rainfall": rainfall,
                "temperature": temp,
                "humidity": humidity
            },
            "crop_type": crop_type
        }
        
        # Call API
        response = requests.post(f"{API_BASE_URL}/recommend/practices", json=data)
        
        if response.status_code == 200:
            result = response.json()
            recommendations = result['recommendations']
            
            # Build recommendations display
            rec_cards = []
            for category, rec in recommendations.items():
                priority_color = {
                    'high': 'danger',
                    'medium': 'warning',
                    'low': 'info'
                }.get(rec['priority'], 'secondary')
                
                card = dbc.Card([
                    dbc.CardHeader([
                        html.H6(category.replace('_', ' ').title(), 
                               className="mb-0"),
                        dbc.Badge(rec['priority'], color=priority_color, className="ms-2")
                    ]),
                    dbc.CardBody([
                        html.P(rec['explanation'], className="card-text"),
                        html.Ul([
                            html.Li(practice) for practice in rec['practices']
                        ])
                    ])
                ], className="mb-3")
                
                rec_cards.append(card)
            
            return html.Div(rec_cards)
        else:
            return dbc.Alert(f"Error: {response.text}", color="danger")
            
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

@app.callback(
    Output("feature-importance-chart", "figure"),
    Input("predict-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_feature_importance(n_clicks):
    """Update feature importance chart after prediction."""
    if n_clicks is None:
        return go.Figure()
    
    try:
        # Get feature importance from API (this would need to be stored from prediction)
        # For now, use sample data
        features = ['Temperature', 'Rainfall', 'Humidity', 'Soil pH', 'Nitrogen', 
                   'Phosphorus', 'Potassium', 'Organic Matter', 'Irrigation', 'Fertilizer']
        importance = [0.15, 0.12, 0.08, 0.10, 0.18, 0.12, 0.08, 0.10, 0.05, 0.02]
        
        fig = go.Figure(data=[
            go.Bar(x=features, y=importance, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Feature Importance for Crop Yield Prediction",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return go.Figure()

@app.callback(
    Output("api-status", "children"),
    Output("model-status", "children"),
    Input("main-tabs", "active_tab")
)
def update_system_status(active_tab):
    """Update system status information."""
    if active_tab != "status":
        return "", ""
    
    try:
        # Check API health
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            api_status = dbc.Alert("✅ API is running and healthy", color="success")
        else:
            api_status = dbc.Alert("❌ API is not responding", color="danger")
        
        # Check model status
        model_response = requests.get(f"{API_BASE_URL}/models/status")
        if model_response.status_code == 200:
            model_data = model_response.json()
            crop_model = model_data['crop_yield_predictor']
            
            if crop_model['loaded']:
                model_status = dbc.Alert([
                    "✅ ML Models Loaded",
                    html.Br(),
                    f"Type: {crop_model['type']}",
                    html.Br(),
                    f"Trained: {'Yes' if crop_model['trained'] else 'No'}"
                ], color="success")
            else:
                model_status = dbc.Alert("❌ ML Models Not Loaded", color="warning")
        else:
            model_status = dbc.Alert("❌ Cannot check model status", color="danger")
        
        return api_status, model_status
        
    except Exception as e:
        error_status = dbc.Alert(f"❌ Error: {str(e)}", color="danger")
        return error_status, error_status

if __name__ == '__main__':
    print("Starting Sustainable Agriculture Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8050")
    print(f"Make sure the API server is running at: {API_BASE_URL}")
    
    app.run(debug=True, host='0.0.0.0', port=8050)
