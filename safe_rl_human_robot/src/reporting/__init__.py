"""
Automated reporting package for Safe RL analysis.

This package provides comprehensive automated reporting capabilities including
performance reports, safety analysis reports, comparison reports, and executive summaries.
"""

from .automated_reports import (
    ReportGenerator,
    PerformanceReportGenerator, 
    SafetyReportGenerator,
    ComparisonReportGenerator,
    ExecutiveReportGenerator,
    generate_all_reports
)

__all__ = [
    # Base Report Generator
    "ReportGenerator",
    
    # Specialized Report Generators
    "PerformanceReportGenerator",
    "SafetyReportGenerator", 
    "ComparisonReportGenerator",
    "ExecutiveReportGenerator",
    
    # Utility Functions
    "generate_all_reports"
]