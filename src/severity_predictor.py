"""
Bug Severity Predictor - NOVEL FEATURE
Predicts how severe and urgent a bug is
Author: Your Name
"""

import numpy as np
import pandas as pd

class SeverityPredictor:
    """
    Novel: Predicts bug severity and priority
    """
    
    def __init__(self):
        self.severity_keywords = {
            'critical': ['crash', 'data loss', 'corruption', 'security', 'exploit'],
            'high': ['memory leak', 'out of memory', 'hang', 'freeze', 'deadlock'],
            'medium': ['slow', 'performance', 'latency', 'timeout', 'memory'],
            'low': ['documentation', 'typo', 'formatting', 'style']
        }
        
        self.urgency_keywords = {
            'urgent': ['asap', 'urgent', 'critical', 'blocker', 'emergency'],
            'high': ['important', 'breaking', 'major', 'severe'],
            'medium': ['should', 'needs', 'please fix'],
            'low': ['maybe', 'consider', 'optional']
        }
    
    def predict_severity(self, bug_report, comments_count=0, has_code_snippet=False):
        """
        Predict bug severity on scale 1-5
        
        Args:
            bug_report: Text of bug report
            comments_count: Number of comments on issue
            has_code_snippet: Whether report contains code
        
        Returns:
            Dictionary with severity and priority
        """
        severity_score = 1  # Start with lowest severity
        report_lower = bug_report.lower()
        
        # Check severity keywords
        for level, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in report_lower:
                    if level == 'critical':
                        severity_score += 4
                    elif level == 'high':
                        severity_score += 3
                    elif level == 'medium':
                        severity_score += 2
                    elif level == 'low':
                        severity_score += 1
                    break  # Only count once per keyword type
        
        # Additional severity indicators
        if '!!!' in bug_report or '!!!!' in bug_report:
            severity_score += 1
        if bug_report.isupper() and len(bug_report) > 20:
            severity_score += 1  # ALL CAPS = urgent
        if comments_count > 10:
            severity_score += 1  # Many comments = severe
        if has_code_snippet:
            severity_score += 1  # Code snippets help fix bugs
        
        # Normalize to 1-5 scale
        severity_score = min(5, severity_score)
        
        # Determine priority based on severity and urgency
        urgency_score = 1
        for level, keywords in self.urgency_keywords.items():
            for keyword in keywords:
                if keyword in report_lower:
                    if level == 'urgent':
                        urgency_score += 3
                    elif level == 'high':
                        urgency_score += 2
                    elif level == 'medium':
                        urgency_score += 1
                    break
        
        # Map to priority levels
        priority_map = {
            5: 'CRITICAL - Fix immediately (today)',
            4: 'HIGH - Fix soon (this week)',
            3: 'MEDIUM - Fix next sprint',
            2: 'LOW - Fix when possible',
            1: 'TRIVIAL - Can be ignored'
        }
        
        # Combined severity (70% severity, 30% urgency)
        combined_score = int(severity_score * 0.7 + urgency_score * 0.3)
        combined_score = min(5, max(1, combined_score))
        
        # Estimate fix time
        fix_time_map = {
            5: '1-3 days (critical)',
            4: '3-5 days',
            3: '1-2 weeks',
            2: '2-4 weeks',
            1: '1-2 months'
        }
        
        return {
            'severity_score': severity_score,
            'severity_level': priority_map[severity_score],
            'urgency_score': urgency_score,
            'combined_priority': priority_map[combined_score],
            'estimated_fix_time': fix_time_map[severity_score],
            'recommended_action': self.get_recommended_action(severity_score, bug_report)
        }
    
    def get_recommended_action(self, severity, bug_report):
        """Get actionable recommendation"""
        if severity >= 4:
            return "🚨 IMMEDIATE: Assign to senior developer, notify manager"
        elif severity == 3:
            return "⚡ PRIORITY: Add to next sprint, assign to experienced developer"
        elif severity == 2:
            return "📋 PLAN: Add to backlog, schedule for next quarter"
        else:
            return "💡 NOTE: Can be fixed when time permits or closed as won't fix"
    
    def batch_analyze_severity(self, reports, comments_list=None):
        """Analyze severity for multiple reports"""
        results = []
        
        for i, report in enumerate(reports):
            comments = comments_list[i] if comments_list else 0
            severity = self.predict_severity(report, comments)
            
            results.append({
                'report_index': i,
                'text_preview': report[:100],
                'severity_score': severity['severity_score'],
                'severity_level': severity['severity_level'],
                'estimated_fix_time': severity['estimated_fix_time'],
                'recommended_action': severity['recommended_action']
            })
        
        return pd.DataFrame(results)