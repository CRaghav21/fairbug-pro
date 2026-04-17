"""
Time-Aware Bug Analysis - NOVEL FEATURE
Analyzes when performance bugs appear and their trends
Author: Your Name
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class TimeAwareAnalyzer:
    """
    Novel: Analyzes temporal patterns in bug reports
    """
    
    def __init__(self):
        self.temporal_data = {}
    
    def analyze_bug_trends(self, reports, labels, dates):
        """
        Find patterns in bug occurrence over time
        
        Args:
            reports: List of bug reports
            labels: List of labels (1=performance, 0=non-performance)
            dates: List of dates (strings like '2023-01-15')
        
        Returns:
            Dictionary with trend analysis
        """
        # Converting dates to datetime
        parsed_dates = []
        for d in dates:
            try:
                if pd.isna(d) or d is None:
                    parsed_dates.append(pd.Timestamp.now())
                else:
                    parsed_dates.append(pd.to_datetime(d))
            except:
                parsed_dates.append(pd.Timestamp.now())
        
        # Creating DataFrame for analysis
        df = pd.DataFrame({
            'date': parsed_dates,
            'report': reports,
            'is_performance': labels
        })
        
        # Extracting time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month_name'] = df['date'].dt.month_name()
        
        # Groupping by month
        monthly_grouped = df.groupby(['year', 'month']).agg({
            'is_performance': ['count', 'sum', 'mean']
        }).reset_index()
        
        # Renaming columns for easier access
        monthly_grouped.columns = ['year', 'month', 'total_bugs', 'perf_bugs', 'perf_rate']
        
        # Finding peak months
        if len(monthly_grouped) > 0:
            # Find month with highest number of performance bugs
            peak_month_idx = monthly_grouped['perf_bugs'].idxmax()
            peak_year = int(monthly_grouped.iloc[peak_month_idx]['year'])
            peak_month_num = int(monthly_grouped.iloc[peak_month_idx]['month'])
            peak_month = f"{peak_year}-{peak_month_num:02d}"
        else:
            peak_month = "No data"
        
        # Detecting trend
        perf_rates = monthly_grouped['perf_rate'].values
        if len(perf_rates) >= 3:
            # Compare first 3 months vs last 3 months
            first_avg = np.mean(perf_rates[:3])
            last_avg = np.mean(perf_rates[-3:])
            
            if last_avg > first_avg * 1.1:
                trend = "INCREASING ⬆️ (More performance bugs lately)"
            elif last_avg < first_avg * 0.9:
                trend = "DECREASING ⬇️ (Fewer performance bugs lately)"
            else:
                trend = "STABLE ➡️ (No significant change)"
        else:
            trend = "INSUFFICIENT DATA"
        
        # Seasonal patterns (by month name)
        seasonal = df.groupby('month_name')['is_performance'].mean().to_dict()
        
        # Finding busiest month for performance bugs
        if seasonal:
            busiest_month = max(seasonal, key=seasonal.get)
        else:
            busiest_month = None
        
        # Calculating performance rate by quarter
        quarterly = df.groupby('quarter')['is_performance'].mean().to_dict()
        
        return {
            'peak_performance_month': peak_month,
            'trend': trend,
            'seasonal_pattern': seasonal,
            'quarterly_pattern': quarterly,
            'busiest_month': busiest_month,
            'total_performance_bugs': int(sum(labels)),
            'total_bugs': int(len(labels)),
            'performance_rate': float(sum(labels)/len(labels)) if len(labels) > 0 else 0,
            'monthly_stats': monthly_grouped,
            'df': df  # Keep for plotting
        }
    
    def plot_temporal_analysis(self, trend_data, project_name):
        """
        Create visualization of temporal patterns
        """
        monthly_stats = trend_data['monthly_stats']
        
        if len(monthly_stats) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Performance bugs count over time
        ax1 = axes[0, 0]
        months = [f"{int(row['year'])}-{int(row['month']):02d}" 
                  for _, row in monthly_stats.iterrows()]
        perf_counts = monthly_stats['perf_bugs'].values
        
        ax1.bar(months, perf_counts, color='red', alpha=0.7, edgecolor='darkred')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Performance Bugs')
        ax1.set_title(f'{project_name}: Performance Bugs Over Time')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance rate over time
        ax2 = axes[0, 1]
        perf_rates = monthly_stats['perf_rate'].values
        
        ax2.plot(months, perf_rates, marker='o', color='blue', linewidth=2, markersize=6)
        ax2.axhline(y=trend_data['performance_rate'], color='red', 
                    linestyle='--', linewidth=2, label=f"Overall Rate: {trend_data['performance_rate']:.1%}")
        ax2.fill_between(months, perf_rates, trend_data['performance_rate'], 
                         where=(perf_rates > trend_data['performance_rate']), 
                         color='green', alpha=0.3, label='Above Average')
        ax2.fill_between(months, perf_rates, trend_data['performance_rate'], 
                         where=(perf_rates < trend_data['performance_rate']), 
                         color='red', alpha=0.3, label='Below Average')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Performance Bug Rate')
        ax2.set_title(f'{project_name}: Performance Bug Rate Trend')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Seasonal pattern (by month)
        ax3 = axes[1, 0]
        if trend_data['seasonal_pattern']:
            months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            seasonal_rates = []
            for month in months_order:
                if month in trend_data['seasonal_pattern']:
                    seasonal_rates.append(trend_data['seasonal_pattern'][month])
                else:
                    seasonal_rates.append(0)
            
            ax3.bar(months_order, seasonal_rates, color='purple', alpha=0.7)
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Performance Bug Rate')
            ax3.set_title(f'{project_name}: Seasonal Pattern')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trend summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        📊 TEMPORAL ANALYSIS SUMMARY - {project_name.upper()}
        
        📈 Overall Statistics:
        • Total bugs: {trend_data['total_bugs']:,}
        • Performance bugs: {trend_data['total_performance_bugs']:,}
        • Overall performance rate: {trend_data['performance_rate']:.1%}
        
        🎯 Key Findings:
        • Peak performance month: {trend_data['peak_performance_month']}
        • Busiest month for bugs: {trend_data['busiest_month']}
        • Trend: {trend_data['trend']}
        
        📅 Seasonal Analysis:
        """
        
        if trend_data['seasonal_pattern']:
            # Add top 3 months
            sorted_seasons = sorted(trend_data['seasonal_pattern'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            for month, rate in sorted_seasons:
                summary_text += f"\n  • {month}: {rate:.1%} performance bugs"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'data/processed/temporal_analysis_{project_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Temporal analysis plot saved for {project_name}")
    
    def get_prediction_for_month(self, month, year, trend_data):
        """
        Predict performance bug rate for a specific month
        """
        if trend_data['seasonal_pattern']:
            month_name = datetime(year, month, 1).strftime('%B')
            seasonal_rate = trend_data['seasonal_pattern'].get(month_name, 
                                                              trend_data['performance_rate'])
            
            # Adjust based on trend
            if "INCREASING" in trend_data['trend']:
                adjustment = 0.05
            elif "DECREASING" in trend_data['trend']:
                adjustment = -0.05
            else:
                adjustment = 0
            
            predicted_rate = seasonal_rate + adjustment
            predicted_rate = max(0, min(1, predicted_rate))
            
            return {
                'predicted_rate': predicted_rate,
                'expected_bugs': int(predicted_rate * trend_data['total_bugs'] / 12),
                'confidence': 'HIGH' if len(trend_data['monthly_stats']) > 6 else 'MEDIUM'
            }
        
        return None