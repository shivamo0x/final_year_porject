# report_plot.py - Optimized Report Generation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def make_report(csv_path="session_log.csv", out_png="focus_report.png", figsize=(15, 8)):
    """
    Generate a comprehensive focus report with multiple visualizations
    """
    try:
        # Validate input file
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"Log file not found at {csv_path}")
            return False
        
        # Read data
        df = pd.read_csv(csv_path)
        
        if df.empty:
            logger.error("Log file is empty. Cannot generate report.")
            return False
        
        logger.info(f"Processing {len(df)} data points from session log")
        
        # Data validation and cleaning
        df = clean_data(df)
        
        # Create the report
        create_comprehensive_report(df, out_png, figsize)
        
        # Generate summary statistics
        generate_summary_stats(df, csv_path.replace('.csv', '_summary.txt'))
        
        logger.info(f"Report successfully generated: {out_png}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False

def clean_data(df):
    """Clean and validate the data"""
    # Define expected columns
    expected_columns = ['t_sec', 'status', 'ear', 'mar', 'yaw_ratio', 'pitch_ratio']
    
    # Check for missing columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    # Fill missing values
    numeric_columns = ['t_sec', 'ear', 'mar', 'yaw_ratio', 'pitch_ratio']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Remove duplicate timestamps
    df = df.drop_duplicates(subset=['t_sec'], keep='first')
    
    # Sort by time
    df = df.sort_values('t_sec').reset_index(drop=True)
    
    return df

def create_comprehensive_report(df, output_path, figsize):
    """Create a comprehensive multi-panel report"""
    
    # Define status order and colors
    status_order = ['Sleepy', 'Yawning', 'No Face', 'Distracted', 'Blinking', 'Focused']
    status_colors = {
        'Sleepy': '#d62728',      # Red
        'Yawning': '#9467bd',     # Purple
        'No Face': '#7f7f7f',     # Gray
        'Distracted': '#ff7f0e',  # Orange
        'Blinking': '#17becf',    # Cyan
        'Focused': '#2ca02c'      # Green
    }
    
    # Create categorical status for plotting
    df['status_code'] = pd.Categorical(df['status'], categories=status_order, ordered=True).codes
    
    # Set up the plot
    plt.style.use('default')  # Use default style instead of deprecated seaborn
    fig = plt.figure(figsize=figsize)
    
    # Create subplots using gridspec for better layout control
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Main timeline plot
    ax1 = fig.add_subplot(gs[0, :])
    plot_timeline(ax1, df, status_order, status_colors)
    
    # Metrics plots
    ax2 = fig.add_subplot(gs[1, 0])
    plot_ear_mar_metrics(ax2, df)
    
    ax3 = fig.add_subplot(gs[1, 1])
    plot_head_pose_metrics(ax3, df)
    
    # Summary statistics
    ax4 = fig.add_subplot(gs[2, :])
    plot_status_distribution(ax4, df, status_colors)
    
    # Add overall title
    fig.suptitle('AI Focus Monitor - Session Report', fontsize=16, fontweight='bold')
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_timeline(ax, df, status_order, status_colors):
    """Plot the main focus timeline"""
    # Plot the step function
    ax.step(df['t_sec'], df['status_code'], where='post', color='black', linewidth=2, alpha=0.8)
    
    # Fill areas with colors
    for i, status in enumerate(status_order):
        mask = df['status_code'] == i
        if mask.any():
            ax.fill_between(df['t_sec'], df['status_code'], i, 
                          where=mask, facecolor=status_colors[status], 
                          alpha=0.7, step='post', label=status)
    
    # Customize the plot
    ax.set_yticks(range(len(status_order)))
    ax.set_yticklabels(status_order)
    ax.set_title('Focus State Timeline', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Focus State', fontsize=12)
    ax.set_ylim(-0.5, len(status_order) - 0.5)
    ax.set_xlim(0, df['t_sec'].max())
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_ear_mar_metrics(ax, df):
    """Plot EAR and MAR metrics over time"""
    ax2 = ax.twinx()  # Create secondary y-axis
    
    # Plot EAR
    line1 = ax.plot(df['t_sec'], df['ear'], color='blue', linewidth=2, alpha=0.7, label='EAR (Eye Aspect Ratio)')
    ax.set_ylabel('EAR', color='blue', fontsize=12)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(0, df['ear'].max() * 1.1)
    
    # Plot MAR
    line2 = ax2.plot(df['t_sec'], df['mar'], color='red', linewidth=2, alpha=0.7, label='MAR (Mouth Aspect Ratio)')
    ax2.set_ylabel('MAR', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, df['mar'].max() * 1.1)
    
    ax.set_title('Eye & Mouth Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')

def plot_head_pose_metrics(ax, df):
    """Plot head pose metrics (yaw and pitch)"""
    ax2 = ax.twinx()  # Create secondary y-axis
    
    # Plot Yaw
    line1 = ax.plot(df['t_sec'], df['yaw_ratio'], color='green', linewidth=2, alpha=0.7, label='Yaw Ratio')
    ax.set_ylabel('Yaw Ratio', color='green', fontsize=12)
    ax.tick_params(axis='y', labelcolor='green')
    
    # Plot Pitch
    line2 = ax2.plot(df['t_sec'], df['pitch_ratio'], color='orange', linewidth=2, alpha=0.7, label='Pitch Ratio')
    ax2.set_ylabel('Pitch Ratio', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    
    ax.set_title('Head Pose Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')

def plot_status_distribution(ax, df, status_colors):
    """Plot status distribution as horizontal bar chart"""
    status_counts = df['status'].value_counts()
    total_frames = len(df)
    status_percentages = (status_counts / total_frames * 100).round(1)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(status_counts)), status_percentages.values)
    
    # Color bars according to status
    for i, (status, bar) in enumerate(zip(status_counts.index, bars)):
        bar.set_color(status_colors.get(status, '#cccccc'))
        bar.set_alpha(0.8)
    
    ax.set_yticks(range(len(status_counts)))
    ax.set_yticklabels(status_counts.index)
    ax.set_xlabel('Percentage of Time (%)', fontsize=12)
    ax.set_title('Status Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels on bars
    for i, (status, percentage) in enumerate(zip(status_counts.index, status_percentages.values)):
        ax.text(percentage + 0.5, i, f'{percentage}%', 
               va='center', ha='left', fontweight='bold')

def generate_summary_stats(df, output_path):
    """Generate text summary statistics"""
    try:
        total_time = df['t_sec'].max()
        total_frames = len(df)
        
        # Calculate status percentages
        status_counts = df['status'].value_counts()
        status_percentages = (status_counts / total_frames * 100).round(1)
        
        # Calculate focus score
        focus_score = status_percentages.get('Focused', 0)
        
        # Calculate average metrics
        avg_ear = df['ear'].mean()
        avg_mar = df['mar'].mean()
        avg_yaw = df['yaw_ratio'].mean()
        avg_pitch = df['pitch_ratio'].mean()
        
        # Generate summary text
        summary = f"""
AI FOCUS MONITOR - SESSION SUMMARY
=================================

Session Duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)
Total Frames: {total_frames}
Overall Focus Score: {focus_score:.1f}%

STATUS DISTRIBUTION:
"""
        
        for status, percentage in status_percentages.items():
            time_spent = (percentage / 100) * total_time
            summary += f"  {status:<12}: {percentage:>5.1f}% ({time_spent:>6.1f}s)\n"
        
        summary += f"""
AVERAGE METRICS:
  Eye Aspect Ratio (EAR): {avg_ear:.3f}
  Mouth Aspect Ratio (MAR): {avg_mar:.3f}
  Head Yaw Ratio: {avg_yaw:.2f}
  Head Pitch Ratio: {avg_pitch:.2f}

ANALYSIS:
"""
        
        # Add basic analysis
        if focus_score >= 80:
            summary += "  Excellent focus throughout the session!\n"
        elif focus_score >= 60:
            summary += "  Good focus with some minor distractions.\n"
        elif focus_score >= 40:
            summary += "  Moderate focus. Consider reducing distractions.\n"
        else:
            summary += "  Low focus detected. Review your environment and rest.\n"
        
        # Specific recommendations
        if status_percentages.get('Sleepy', 0) > 10:
            summary += "  Consider taking breaks or ensuring adequate sleep.\n"
        
        if status_percentages.get('Distracted', 0) > 20:
            summary += "  Try to minimize head movement and maintain forward gaze.\n"
        
        if status_percentages.get('No Face', 0) > 5:
            summary += "  Ensure camera positioning allows consistent face detection.\n"
        
        # Write summary to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Summary statistics saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating summary statistics: {e}")

def create_simple_report(csv_path="session_log.csv", out_png="focus_report_simple.png"):
    """Create a simplified single-panel report for quick viewing"""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.error("Log file is empty")
            return False
        
        # Define status order and colors
        status_order = ['Sleepy', 'Yawning', 'No Face', 'Distracted', 'Blinking', 'Focused']
        status_colors = {
            'Sleepy': '#d62728', 'Yawning': '#9467bd', 'No Face': '#7f7f7f',
            'Distracted': '#ff7f0e', 'Blinking': '#17becf', 'Focused': '#2ca02c'
        }
        
        df['status_code'] = pd.Categorical(df['status'], categories=status_order, ordered=True).codes
        
        # Create simple plot
        plt.figure(figsize=(12, 6))
        plt.step(df['t_sec'], df['status_code'], where='post', color='black', linewidth=2)
        
        # Fill areas
        for i, status in enumerate(status_order):
            mask = df['status_code'] == i
            if mask.any():
                plt.fill_between(df['t_sec'], df['status_code'], i,
                               where=mask, facecolor=status_colors[status],
                               alpha=0.7, step='post')
        
        plt.yticks(range(len(status_order)), status_order)
        plt.title('Focus State Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Focus State', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simple report saved to: {out_png}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating simple report: {e}")
        return False