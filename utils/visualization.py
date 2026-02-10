"""
RadioFlow Visualization Utilities
Charts, diagrams, and image overlays for the UI
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
import io
import base64


def create_workflow_diagram(agent_results: List[Dict], current_step: int = -1) -> go.Figure:
    """
    Create an interactive workflow diagram showing agent pipeline status.
    
    Args:
        agent_results: List of results from each agent
        current_step: Index of currently processing agent (-1 if complete)
    
    Returns:
        Plotly figure with workflow visualization
    """
    agents = ["CXR Analyzer", "Finding Interpreter", "Report Generator", "Priority Router"]
    
    # Define positions
    x_positions = [0, 1, 2, 3]
    y_positions = [0, 0, 0, 0]
    
    # Determine colors based on status
    colors = []
    for i, agent in enumerate(agents):
        if i < len(agent_results):
            status = agent_results[i].get("status", "pending")
            if status == "success":
                colors.append("#22c55e")  # Green
            elif status == "error":
                colors.append("#ef4444")  # Red
            else:
                colors.append("#f59e0b")  # Yellow/warning
        elif i == current_step:
            colors.append("#3b82f6")  # Blue (processing)
        else:
            colors.append("#6b7280")  # Gray (pending)
    
    # Create figure
    fig = go.Figure()
    
    # Add connections (arrows)
    for i in range(len(agents) - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i] + 0.15, x_positions[i + 1] - 0.15],
            y=[0, 0],
            mode='lines',
            line=dict(color='#94a3b8', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        # Arrow head
        fig.add_annotation(
            x=x_positions[i + 1] - 0.15,
            y=0,
            ax=x_positions[i + 1] - 0.25,
            ay=0,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor='#94a3b8'
        )
    
    # Add agent nodes
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='markers+text',
        marker=dict(
            size=60,
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=['1', '2', '3', '4'],
        textposition='middle center',
        textfont=dict(color='white', size=20, family='Arial Black'),
        hovertext=agents,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add agent labels below
    for i, agent in enumerate(agents):
        fig.add_annotation(
            x=x_positions[i],
            y=-0.3,
            text=agent,
            showarrow=False,
            font=dict(size=11, color='#374151'),
            xanchor='center'
        )
        
        # Add timing if available
        if i < len(agent_results) and "processing_time_ms" in agent_results[i]:
            time_ms = agent_results[i]["processing_time_ms"]
            fig.add_annotation(
                x=x_positions[i],
                y=-0.5,
                text=f"{time_ms:.0f}ms",
                showarrow=False,
                font=dict(size=9, color='#6b7280'),
                xanchor='center'
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="RadioFlow Agent Pipeline",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 3.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.8, 0.5]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=200,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_findings_overlay(
    image: Image.Image,
    findings: List[Dict],
    opacity: float = 0.4
) -> Image.Image:
    """
    Create an overlay on the X-ray image highlighting findings.
    
    Args:
        image: Original chest X-ray image
        findings: List of findings with regions
        opacity: Overlay opacity
    
    Returns:
        Image with findings highlighted
    """
    # Convert to RGBA if needed
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create overlay
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Color mapping for finding severity
    severity_colors = {
        'critical': (239, 68, 68, int(255 * opacity)),    # Red
        'high': (249, 115, 22, int(255 * opacity)),       # Orange
        'moderate': (234, 179, 8, int(255 * opacity)),    # Yellow
        'low': (34, 197, 94, int(255 * opacity)),         # Green
        'normal': (59, 130, 246, int(255 * opacity))      # Blue
    }
    
    for finding in findings:
        region = finding.get('region', {})
        severity = finding.get('severity', 'moderate')
        color = severity_colors.get(severity, severity_colors['moderate'])
        
        if 'bbox' in region:
            # Draw bounding box
            x1, y1, x2, y2 = region['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=3)
            
            # Add label
            label = finding.get('label', 'Finding')
            draw.text((x1, y1 - 15), label, fill=color[:3])
        
        elif 'center' in region and 'radius' in region:
            # Draw circle for point findings
            cx, cy = region['center']
            r = region['radius']
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color[:3], width=2)
    
    # Composite
    result = Image.alpha_composite(image, overlay)
    return result.convert('RGB')


def create_radar_chart(scores: Dict[str, float], title: str = "Analysis Scores") -> go.Figure:
    """
    Create a radar chart showing multi-dimensional analysis scores.
    
    Args:
        scores: Dictionary of category -> score (0-1)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    # Add the score trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=2),
        name='Current Analysis'
    ))
    
    # Add reference (normal) trace
    normal_values = [0.85] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(34, 197, 94, 0.1)',
        line=dict(color='#22c55e', width=1, dash='dash'),
        name='Normal Reference'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0]
            )
        ),
        showlegend=True,
        title=dict(text=title, x=0.5),
        height=350,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    return fig


def create_timeline_chart(
    agent_timings: List[Dict],
    total_time_ms: float
) -> go.Figure:
    """
    Create a timeline/Gantt chart showing agent processing times.
    
    Args:
        agent_timings: List of {name, start_ms, duration_ms}
        total_time_ms: Total workflow time
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
    
    for i, timing in enumerate(agent_timings):
        fig.add_trace(go.Bar(
            x=[timing['duration_ms']],
            y=[timing['name']],
            orientation='h',
            marker=dict(color=colors[i % len(colors)]),
            text=f"{timing['duration_ms']:.0f}ms",
            textposition='inside',
            name=timing['name'],
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text=f"Processing Timeline (Total: {total_time_ms:.0f}ms)",
            x=0.5
        ),
        xaxis=dict(title="Time (ms)"),
        yaxis=dict(title=""),
        height=200,
        margin=dict(l=120, r=20, t=50, b=40),
        barmode='stack'
    )
    
    return fig


def create_priority_gauge(priority_score: float, priority_level: str) -> go.Figure:
    """
    Create a gauge chart showing priority/urgency level.
    
    Args:
        priority_score: Score from 0 to 1
        priority_level: Text label for priority
    
    Returns:
        Plotly figure
    """
    # Determine color based on score
    if priority_score >= 0.7:
        color = "#ef4444"  # Red - urgent
    elif priority_score >= 0.4:
        color = "#f59e0b"  # Yellow - moderate
    else:
        color = "#22c55e"  # Green - routine
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=priority_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Priority: {priority_level}", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#dcfce7'},
                {'range': [33, 66], 'color': '#fef3c7'},
                {'range': [66, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for display."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()
