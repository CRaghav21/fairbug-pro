"""
Cross-Language Performance Bug Detection - NOVEL FEATURE
Detects if performance bugs in one language can be found in another
Author: Your Name
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CrossLanguageDetector:
    """
    Novel: Detects performance bug patterns across programming languages
    """
    
    def __init__(self):
        # Language-specific performance keywords
        self.language_patterns = {
            'python': ['slow', 'memory', 'gpu', 'cpu', 'tensor', 'numpy', 'pandas', 'loop'],
            'c_cpp': ['malloc', 'free', 'pointer', 'heap', 'stack', 'segfault', 'leak'],
            'java': ['gc', 'jvm', 'heap', 'thread', 'synchronized', 'outofmemory'],
            'javascript': ['callback', 'async', 'promise', 'memory', 'leak', 'event loop']
        }
        
        # Translation dictionary
        self.translation = {
            'python_slow': 'cpp_performance_degradation',
            'python_memory': 'cpp_memory_allocation',
            'python_gpu': 'cpp_cuda_kernel',
            'slow': 'performance_degradation',
            'memory leak': 'memory_allocation_issue',
            'crash': 'segmentation_fault'
        }
    
    def detect_cross_language_pattern(self, bug_report, source_lang='python', target_lang='c_cpp'):
        """
        Detect if bug pattern exists in another language
        
        Args:
            bug_report: Text of bug report
            source_lang: Language of the bug report
            target_lang: Language to translate patterns to
        
        Returns:
            Dictionary with cross-language analysis
        """
        report_lower = bug_report.lower()
        
        # Get patterns for source and target
        source_patterns = self.language_patterns.get(source_lang, [])
        target_patterns = self.language_patterns.get(target_lang, [])
        
        # Find which source patterns appear
        matched_source = []
        for pattern in source_patterns:
            if pattern in report_lower:
                matched_source.append(pattern)
        
        # Translate to target language equivalents
        translated_patterns = []
        for pattern in matched_source:
            # Simple translation logic
            if pattern in ['slow', 'performance']:
                translated_patterns.append('performance_degradation')
            elif pattern in ['memory', 'leak']:
                translated_patterns.append('memory_allocation')
            elif pattern in ['gpu', 'cuda']:
                translated_patterns.append('gpu_kernel')
            elif pattern in ['crash', 'fail']:
                translated_patterns.append('segmentation_fault')
            else:
                translated_patterns.append(pattern)
        
        # Check if these patterns would be detectable in target language
        target_detectable = []
        for trans_pattern in translated_patterns:
            for target_pattern in target_patterns:
                if trans_pattern in target_pattern or target_pattern in trans_pattern:
                    target_detectable.append(target_pattern)
        
        return {
            'source_language': source_lang,
            'target_language': target_lang,
            'matched_patterns_in_source': matched_source,
            'translated_patterns': translated_patterns,
            'detectable_in_target': target_detectable,
            'cross_language_score': len(target_detectable) / max(1, len(matched_source)),
            'recommendation': self.get_recommendation(len(target_detectable), len(matched_source))
        }
    
    def get_recommendation(self, detected, total):
        """Get recommendation based on cross-language detection"""
        ratio = detected / total if total > 0 else 0
        
        if ratio >= 0.8:
            return "✅ HIGH transferability - Performance patterns in this language generalize well"
        elif ratio >= 0.5:
            return "⚠️ MEDIUM transferability - Some patterns transfer, others are language-specific"
        else:
            return "❌ LOW transferability - Performance issues are language-specific"
    
    def find_language_agnostic_features(self, bug_report):
        """
        Find features that work across all languages
        """
        report_lower = bug_report.lower()
        
        # Language-agnostic performance indicators
        agnostic_features = []
        
        if any(word in report_lower for word in ['slow', 'fast', 'speed']):
            agnostic_features.append('speed_related')
        if any(word in report_lower for word in ['memory', 'ram', 'storage']):
            agnostic_features.append('memory_related')
        if any(word in report_lower for word in ['crash', 'fail', 'error']):
            agnostic_features.append('failure_related')
        if any(word in report_lower for word in ['timeout', 'wait', 'delay']):
            agnostic_features.append('timing_related')
        
        return {
            'agnostic_features': agnostic_features,
            'universal_performance_indicators': len(agnostic_features) > 0,
            'confidence': len(agnostic_features) / 4
        }