"""
Safety analysis for Safe RL training results.

This module provides comprehensive safety analysis including constraint violation
analysis, risk assessment, failure mode analysis, and safety margin distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings


@dataclass
class SafetyMetrics:
    """Container for safety analysis results."""
    
    # Violation metrics
    total_violations: int = 0
    violation_rate: float = 0.0
    mean_violation_severity: float = 0.0
    max_violation_severity: float = 0.0
    
    # Safety margins
    mean_safety_margin: float = 0.0
    min_safety_margin: float = 0.0
    safety_margin_std: float = 0.0
    
    # Risk metrics
    value_at_risk_95: float = 0.0  # 95% VaR
    conditional_value_at_risk: float = 0.0  # CVaR (Expected shortfall)
    time_to_violation: float = 0.0
    recovery_time: float = 0.0
    
    # Constraint-specific metrics
    constraint_costs: Dict[str, float] = None
    constraint_activation_rates: Dict[str, float] = None
    
    def __post_init__(self):
        if self.constraint_costs is None:
            self.constraint_costs = {}
        if self.constraint_activation_rates is None:
            self.constraint_activation_rates = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_violations': self.total_violations,
            'violation_rate': self.violation_rate,
            'mean_violation_severity': self.mean_violation_severity,
            'max_violation_severity': self.max_violation_severity,
            'mean_safety_margin': self.mean_safety_margin,
            'min_safety_margin': self.min_safety_margin,
            'safety_margin_std': self.safety_margin_std,
            'value_at_risk_95': self.value_at_risk_95,
            'conditional_value_at_risk': self.conditional_value_at_risk,
            'time_to_violation': self.time_to_violation,
            'recovery_time': self.recovery_time,
            'constraint_costs': self.constraint_costs,
            'constraint_activation_rates': self.constraint_activation_rates
        }


class ConstraintAnalyzer:
    """Analyzes individual constraints and their violations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_constraint_violations(self, data: pd.DataFrame,
                                    constraint_columns: List[str] = None) -> Dict[str, Any]:
        """Analyze constraint violations in detail."""
        if constraint_columns is None:
            # Auto-detect constraint columns
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost', 'margin'])]
        
        if not constraint_columns:
            self.logger.warning("No constraint columns found in data")
            return {'no_constraint_data': True}
        
        analysis = {
            'constraint_columns': constraint_columns,
            'individual_constraints': {},
            'constraint_correlations': {},
            'violation_patterns': {},
            'temporal_analysis': {}
        }
        
        # Analyze each constraint individually
        for col in constraint_columns:
            if col in data.columns:
                analysis['individual_constraints'][col] = self._analyze_single_constraint(
                    data[col].dropna().values, col
                )
        
        # Constraint correlations
        constraint_data = data[constraint_columns].dropna()
        if len(constraint_data) > 1 and len(constraint_columns) > 1:
            correlation_matrix = constraint_data.corr()
            analysis['constraint_correlations'] = correlation_matrix.to_dict()
        
        # Violation patterns
        analysis['violation_patterns'] = self._analyze_violation_patterns(
            data, constraint_columns
        )
        
        # Temporal analysis
        if 'iteration' in data.columns or 'step' in data.columns:
            time_col = 'iteration' if 'iteration' in data.columns else 'step'
            analysis['temporal_analysis'] = self._analyze_temporal_patterns(
                data, constraint_columns, time_col
            )
        
        return analysis
    
    def _analyze_single_constraint(self, constraint_values: np.ndarray, 
                                  constraint_name: str) -> Dict[str, Any]:
        """Analyze a single constraint in detail."""
        if len(constraint_values) == 0:
            return {'no_data': True}
        
        analysis = {
            'name': constraint_name,
            'statistics': {},
            'violations': {},
            'distribution': {},
            'risk_metrics': {}
        }
        
        # Basic statistics
        analysis['statistics'] = {
            'count': len(constraint_values),
            'mean': float(np.mean(constraint_values)),
            'std': float(np.std(constraint_values)),
            'min': float(np.min(constraint_values)),
            'max': float(np.max(constraint_values)),
            'median': float(np.median(constraint_values)),
            'q25': float(np.percentile(constraint_values, 25)),
            'q75': float(np.percentile(constraint_values, 75))
        }
        
        # Violation analysis (assuming positive values are violations)
        violations = constraint_values[constraint_values > 0]
        analysis['violations'] = {
            'count': len(violations),
            'rate': float(len(violations) / len(constraint_values)),
            'mean_severity': float(np.mean(violations)) if len(violations) > 0 else 0.0,
            'max_severity': float(np.max(violations)) if len(violations) > 0 else 0.0,
            'total_cost': float(np.sum(violations)) if len(violations) > 0 else 0.0
        }
        
        # Distribution analysis
        if len(constraint_values) > 5:
            # Test for normality
            shapiro_stat, shapiro_p = stats.shapiro(constraint_values[:5000])  # Limit for performance
            
            # Fit distributions
            distribution_fits = self._fit_distributions(constraint_values)
            
            analysis['distribution'] = {
                'normality_test': {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': float(shapiro_p) > 0.05
                },
                'best_fit_distribution': distribution_fits
            }
        
        # Risk metrics
        if len(constraint_values) > 10:
            analysis['risk_metrics'] = self._compute_constraint_risk_metrics(constraint_values)
        
        return analysis
    
    def _fit_distributions(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit common distributions to constraint data."""
        distributions = ['norm', 'gamma', 'beta', 'lognorm', 'weibull_min']
        fits = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))
                
                fits[dist_name] = {
                    'parameters': [float(p) for p in params],
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'log_likelihood': float(np.sum(dist.logpdf(data, *params)))
                }
            except Exception as e:
                self.logger.debug(f"Failed to fit {dist_name} distribution: {e}")
                continue
        
        # Find best fit based on log-likelihood
        if fits:
            best_fit = max(fits.items(), key=lambda x: x[1]['log_likelihood'])
            return {'best_distribution': best_fit[0], 'all_fits': fits}
        else:
            return {'no_successful_fits': True}
    
    def _compute_constraint_risk_metrics(self, constraint_values: np.ndarray) -> Dict[str, Any]:
        """Compute risk metrics for constraint values."""
        # Value at Risk (VaR)
        var_95 = float(np.percentile(constraint_values, 95))
        var_99 = float(np.percentile(constraint_values, 99))
        
        # Conditional Value at Risk (CVaR / Expected Shortfall)
        cvar_95 = float(np.mean(constraint_values[constraint_values >= var_95])) if np.any(constraint_values >= var_95) else 0.0
        cvar_99 = float(np.mean(constraint_values[constraint_values >= var_99])) if np.any(constraint_values >= var_99) else 0.0
        
        # Maximum Drawdown (for time series)
        cumulative = np.cumsum(constraint_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'conditional_value_at_risk_95': cvar_95,
            'conditional_value_at_risk_99': cvar_99,
            'maximum_drawdown': max_drawdown,
            'tail_risk_ratio': cvar_95 / var_95 if var_95 > 0 else 0.0
        }
    
    def _analyze_violation_patterns(self, data: pd.DataFrame, 
                                   constraint_columns: List[str]) -> Dict[str, Any]:
        """Analyze patterns in constraint violations."""
        patterns = {
            'co_occurrence': {},
            'sequential_patterns': {},
            'clustering': {}
        }
        
        # Co-occurrence analysis
        violation_data = data[constraint_columns] > 0  # Boolean violation matrix
        
        if len(violation_data) > 1:
            # Pairwise co-occurrence
            co_occurrence_matrix = violation_data.T.dot(violation_data)
            patterns['co_occurrence'] = co_occurrence_matrix.to_dict()
            
            # Simultaneous violations
            simultaneous_violations = (violation_data.sum(axis=1) > 1).sum()
            patterns['simultaneous_violation_rate'] = float(simultaneous_violations / len(violation_data))
        
        # Sequential patterns (if time information available)
        if 'iteration' in data.columns or 'step' in data.columns:
            patterns['sequential_patterns'] = self._analyze_sequential_violations(
                data, constraint_columns
            )
        
        # Clustering violation episodes
        violation_episodes = violation_data.values
        if violation_episodes.shape[0] > 10 and violation_episodes.shape[1] > 1:
            patterns['clustering'] = self._cluster_violation_patterns(violation_episodes)
        
        return patterns
    
    def _analyze_sequential_violations(self, data: pd.DataFrame,
                                     constraint_columns: List[str]) -> Dict[str, Any]:
        """Analyze sequential patterns in violations."""
        time_col = 'iteration' if 'iteration' in data.columns else 'step'
        sequential_analysis = {}
        
        for col in constraint_columns:
            if col not in data.columns:
                continue
            
            violations = (data[col] > 0).values
            times = data[time_col].values
            
            # Find violation episodes
            episodes = self._find_violation_episodes(violations, times)
            
            if episodes:
                episode_lengths = [ep['length'] for ep in episodes]
                inter_episode_times = [ep['recovery_time'] for ep in episodes if ep['recovery_time'] is not None]
                
                sequential_analysis[col] = {
                    'num_episodes': len(episodes),
                    'mean_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                    'max_episode_length': float(np.max(episode_lengths)) if episode_lengths else 0.0,
                    'mean_recovery_time': float(np.mean(inter_episode_times)) if inter_episode_times else 0.0,
                    'episodes': episodes[:10]  # Store first 10 episodes for detailed analysis
                }
        
        return sequential_analysis
    
    def _find_violation_episodes(self, violations: np.ndarray, 
                                times: np.ndarray) -> List[Dict[str, Any]]:
        """Find continuous violation episodes."""
        episodes = []
        in_episode = False
        episode_start = None
        
        for i, (violation, time) in enumerate(zip(violations, times)):
            if violation and not in_episode:
                # Start of new episode
                in_episode = True
                episode_start = i
            elif not violation and in_episode:
                # End of episode
                episode_length = i - episode_start
                
                # Calculate recovery time (time to next violation or end)
                recovery_time = None
                for j in range(i + 1, len(violations)):
                    if violations[j]:
                        recovery_time = times[j] - times[i]
                        break
                
                episodes.append({
                    'start_time': float(times[episode_start]),
                    'end_time': float(times[i-1]),
                    'length': episode_length,
                    'recovery_time': float(recovery_time) if recovery_time is not None else None
                })
                
                in_episode = False
        
        # Handle episode that continues to end of data
        if in_episode:
            episodes.append({
                'start_time': float(times[episode_start]),
                'end_time': float(times[-1]),
                'length': len(times) - episode_start,
                'recovery_time': None
            })
        
        return episodes
    
    def _cluster_violation_patterns(self, violation_data: np.ndarray) -> Dict[str, Any]:
        """Cluster violation patterns to identify common failure modes."""
        # Only cluster time steps with violations
        violation_indices = np.any(violation_data, axis=1)
        violation_patterns = violation_data[violation_indices]
        
        if len(violation_patterns) < 5:
            return {'insufficient_data': True}
        
        clustering_results = {}
        
        try:
            # K-means clustering
            n_clusters = min(5, len(violation_patterns) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(violation_patterns)
            
            clustering_results['kmeans'] = {
                'n_clusters': n_clusters,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'labels': kmeans_labels.tolist(),
                'inertia': float(kmeans.inertia_)
            }
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            dbscan_labels = dbscan.fit_predict(violation_patterns)
            
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            clustering_results['dbscan'] = {
                'n_clusters': n_clusters_dbscan,
                'labels': dbscan_labels.tolist(),
                'n_noise_points': int(np.sum(dbscan_labels == -1))
            }
            
        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            clustering_results['error'] = str(e)
        
        return clustering_results
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame,
                                  constraint_columns: List[str],
                                  time_col: str) -> Dict[str, Any]:
        """Analyze temporal patterns in constraints."""
        temporal_analysis = {}
        
        for col in constraint_columns:
            if col not in data.columns:
                continue
            
            constraint_values = data[col].values
            times = data[time_col].values
            
            # Trend analysis
            if len(constraint_values) > 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, constraint_values)
                
                temporal_analysis[col] = {
                    'trend': {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'significant_trend': float(p_value) < 0.05,
                        'improving': float(slope) < -0.001 and p_value < 0.05
                    }
                }
                
                # Seasonality analysis (if enough data)
                if len(constraint_values) > 50:
                    temporal_analysis[col]['seasonality'] = self._analyze_seasonality(
                        constraint_values, times
                    )
        
        return temporal_analysis
    
    def _analyze_seasonality(self, values: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonal patterns in constraint values."""
        try:
            # Simple autocorrelation analysis
            autocorr_lags = min(20, len(values) // 4)
            autocorrelations = [np.corrcoef(values[:-lag], values[lag:])[0, 1] 
                              for lag in range(1, autocorr_lags + 1)]
            
            # Find significant autocorrelations
            significant_lags = [i + 1 for i, corr in enumerate(autocorrelations) 
                              if abs(corr) > 0.3]  # Threshold for significance
            
            return {
                'autocorrelations': autocorrelations,
                'significant_lags': significant_lags,
                'max_autocorr': float(max(autocorrelations)) if autocorrelations else 0.0,
                'seasonal_periods': significant_lags[:3]  # Top 3 potential periods
            }
            
        except Exception as e:
            self.logger.warning(f"Seasonality analysis failed: {e}")
            return {'error': str(e)}


class RiskAnalyzer:
    """Analyzes risk and safety margins."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_safety_margins(self, data: pd.DataFrame,
                              margin_columns: List[str] = None) -> Dict[str, Any]:
        """Analyze safety margins and risk metrics."""
        if margin_columns is None:
            margin_columns = [col for col in data.columns 
                            if 'margin' in col.lower() or 'distance' in col.lower()]
        
        if not margin_columns:
            # Try to infer safety margins from constraint columns
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost'])]
            
            if constraint_columns:
                # Safety margin is negative constraint value (distance to violation)
                margin_columns = constraint_columns
            else:
                return {'no_margin_data': True}
        
        analysis = {
            'margin_columns': margin_columns,
            'individual_margins': {},
            'combined_risk_analysis': {},
            'state_space_risk': {}
        }
        
        # Analyze each margin individually
        for col in margin_columns:
            if col in data.columns:
                margin_values = data[col].dropna().values
                
                # Convert constraint costs to safety margins (negative values)
                if 'constraint' in col.lower() or 'cost' in col.lower():
                    margin_values = -margin_values
                
                analysis['individual_margins'][col] = self._analyze_single_margin(
                    margin_values, col
                )
        
        # Combined risk analysis
        if len(margin_columns) > 1:
            analysis['combined_risk_analysis'] = self._analyze_combined_risk(
                data, margin_columns
            )
        
        # State space risk analysis (if state information available)
        state_columns = [col for col in data.columns 
                        if any(keyword in col.lower() for keyword in ['state', 'position', 'velocity'])]
        
        if state_columns and margin_columns:
            analysis['state_space_risk'] = self._analyze_state_space_risk(
                data, state_columns, margin_columns
            )
        
        return analysis
    
    def _analyze_single_margin(self, margin_values: np.ndarray, 
                              margin_name: str) -> Dict[str, Any]:
        """Analyze a single safety margin."""
        if len(margin_values) == 0:
            return {'no_data': True}
        
        analysis = {
            'name': margin_name,
            'statistics': {},
            'risk_metrics': {},
            'distribution_analysis': {}
        }
        
        # Basic statistics
        analysis['statistics'] = {
            'count': len(margin_values),
            'mean': float(np.mean(margin_values)),
            'std': float(np.std(margin_values)),
            'min': float(np.min(margin_values)),
            'max': float(np.max(margin_values)),
            'median': float(np.median(margin_values)),
            'q5': float(np.percentile(margin_values, 5)),
            'q95': float(np.percentile(margin_values, 95))
        }
        
        # Risk metrics
        analysis['risk_metrics'] = {
            'probability_of_violation': float(np.mean(margin_values <= 0)),
            'expected_margin': float(np.mean(margin_values)),
            'margin_volatility': float(np.std(margin_values)),
            'worst_case_margin': float(np.min(margin_values)),
            'value_at_risk_95': float(np.percentile(margin_values, 5)),  # 5th percentile for margins
            'value_at_risk_99': float(np.percentile(margin_values, 1))   # 1st percentile for margins
        }
        
        # Conditional Value at Risk (Expected Shortfall)
        var_95 = analysis['risk_metrics']['value_at_risk_95']
        worst_5_percent = margin_values[margin_values <= var_95]
        analysis['risk_metrics']['conditional_value_at_risk_95'] = float(np.mean(worst_5_percent)) if len(worst_5_percent) > 0 else var_95
        
        # Distribution analysis
        if len(margin_values) > 10:
            analysis['distribution_analysis'] = self._analyze_margin_distribution(margin_values)
        
        return analysis
    
    def _analyze_margin_distribution(self, margin_values: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of safety margins."""
        distribution_analysis = {}
        
        # Test for normality
        if len(margin_values) <= 5000:  # Limit for performance
            shapiro_stat, shapiro_p = stats.shapiro(margin_values)
            distribution_analysis['normality_test'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': float(shapiro_p) > 0.05
            }
        
        # Estimate probability density
        try:
            # Kernel density estimation
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(margin_values)
            
            # Evaluate at risk points
            risk_points = np.linspace(np.min(margin_values), 0, 100)
            density_at_risk = kde(risk_points)
            
            distribution_analysis['density_analysis'] = {
                'risk_points': risk_points.tolist(),
                'density_values': density_at_risk.tolist(),
                'density_at_zero': float(kde(0)[0])
            }
            
        except Exception as e:
            self.logger.debug(f"KDE analysis failed: {e}")
        
        # Extreme value analysis
        if len(margin_values) > 50:
            distribution_analysis['extreme_value_analysis'] = self._analyze_extreme_values(margin_values)
        
        return distribution_analysis
    
    def _analyze_extreme_values(self, margin_values: np.ndarray) -> Dict[str, Any]:
        """Analyze extreme values (tail behavior)."""
        # Block maxima approach
        block_size = max(10, len(margin_values) // 20)
        n_blocks = len(margin_values) // block_size
        
        block_minima = []  # Use minima for safety margins (worst cases)
        for i in range(n_blocks):
            block = margin_values[i*block_size:(i+1)*block_size]
            if len(block) > 0:
                block_minima.append(np.min(block))
        
        if len(block_minima) < 3:
            return {'insufficient_data': True}
        
        try:
            # Fit Generalized Extreme Value (GEV) distribution
            gev_params = stats.genextreme.fit(block_minima)
            
            # Estimate return levels (values expected to be exceeded once in N blocks)
            return_periods = [10, 50, 100]
            return_levels = {}
            
            for period in return_periods:
                prob = 1 - 1/period
                return_level = stats.genextreme.ppf(prob, *gev_params)
                return_levels[f'{period}_block_return_level'] = float(return_level)
            
            return {
                'gev_parameters': [float(p) for p in gev_params],
                'return_levels': return_levels,
                'block_size': block_size
            }
            
        except Exception as e:
            self.logger.debug(f"Extreme value analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_combined_risk(self, data: pd.DataFrame,
                              margin_columns: List[str]) -> Dict[str, Any]:
        """Analyze combined risk across multiple safety margins."""
        margin_data = data[margin_columns].dropna()
        
        if len(margin_data) < 10:
            return {'insufficient_data': True}
        
        # Convert constraint costs to margins if needed
        for col in margin_columns:
            if 'constraint' in col.lower() or 'cost' in col.lower():
                margin_data[col] = -margin_data[col]
        
        analysis = {
            'correlation_analysis': {},
            'joint_risk_metrics': {},
            'principal_component_analysis': {}
        }
        
        # Correlation analysis
        correlation_matrix = margin_data.corr()
        analysis['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'max_correlation': float(correlation_matrix.abs().max().max()),
            'mean_correlation': float(correlation_matrix.abs().mean().mean())
        }
        
        # Joint risk metrics
        # Probability that ANY margin is violated
        any_violation = (margin_data <= 0).any(axis=1)
        prob_any_violation = float(any_violation.mean())
        
        # Probability that ALL margins are violated simultaneously
        all_violation = (margin_data <= 0).all(axis=1)
        prob_all_violation = float(all_violation.mean())
        
        # Minimum margin across all constraints
        min_margins = margin_data.min(axis=1)
        
        analysis['joint_risk_metrics'] = {
            'probability_any_violation': prob_any_violation,
            'probability_all_violations': prob_all_violation,
            'expected_minimum_margin': float(min_margins.mean()),
            'worst_case_minimum_margin': float(min_margins.min()),
            'minimum_margin_std': float(min_margins.std())
        }
        
        # Principal Component Analysis
        if len(margin_columns) > 1:
            try:
                pca = PCA()
                pca_result = pca.fit_transform(margin_data.values)
                
                analysis['principal_component_analysis'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': pca.components_.tolist(),
                    'n_components_95_variance': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1)
                }
                
            except Exception as e:
                self.logger.debug(f"PCA analysis failed: {e}")
                analysis['principal_component_analysis'] = {'error': str(e)}
        
        return analysis
    
    def _analyze_state_space_risk(self, data: pd.DataFrame,
                                 state_columns: List[str],
                                 margin_columns: List[str]) -> Dict[str, Any]:
        """Analyze risk in state space."""
        # Prepare state and margin data
        relevant_columns = state_columns + margin_columns
        clean_data = data[relevant_columns].dropna()
        
        if len(clean_data) < 20:
            return {'insufficient_data': True}
        
        states = clean_data[state_columns].values
        margins = clean_data[margin_columns].values
        
        # Convert constraint costs to margins
        for i, col in enumerate(margin_columns):
            if 'constraint' in col.lower() or 'cost' in col.lower():
                margins[:, i] = -margins[:, i]
        
        analysis = {
            'risk_heat_map': {},
            'high_risk_regions': {},
            'safe_regions': {}
        }
        
        try:
            # Create risk heat map (2D projection if high-dimensional)
            if states.shape[1] > 2:
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=2)
                states_2d = pca.fit_transform(states)
                state_labels = [f'PC{i+1}' for i in range(2)]
            else:
                states_2d = states[:, :2]
                state_labels = state_columns[:2]
            
            # Compute risk level for each point (minimum margin)
            risk_levels = np.min(margins, axis=1)
            
            analysis['risk_heat_map'] = {
                'state_points': states_2d.tolist(),
                'risk_levels': risk_levels.tolist(),
                'state_labels': state_labels,
                'dimensionality_reduction': states.shape[1] > 2
            }
            
            # Identify high-risk regions (bottom 10% of margins)
            high_risk_threshold = np.percentile(risk_levels, 10)
            high_risk_indices = risk_levels <= high_risk_threshold
            
            if np.any(high_risk_indices):
                high_risk_states = states_2d[high_risk_indices]
                analysis['high_risk_regions'] = {
                    'threshold': float(high_risk_threshold),
                    'proportion': float(np.mean(high_risk_indices)),
                    'representative_states': high_risk_states[:10].tolist()  # First 10 high-risk states
                }
            
            # Identify safe regions (top 10% of margins)
            safe_threshold = np.percentile(risk_levels, 90)
            safe_indices = risk_levels >= safe_threshold
            
            if np.any(safe_indices):
                safe_states = states_2d[safe_indices]
                analysis['safe_regions'] = {
                    'threshold': float(safe_threshold),
                    'proportion': float(np.mean(safe_indices)),
                    'representative_states': safe_states[:10].tolist()  # First 10 safe states
                }
            
        except Exception as e:
            self.logger.warning(f"State space risk analysis failed: {e}")
            analysis = {'error': str(e)}
        
        return analysis


class SafetyAnalyzer:
    """Main safety analysis class combining constraint and risk analysis."""
    
    def __init__(self):
        self.constraint_analyzer = ConstraintAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze_safety_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive safety analysis of training data."""
        safety_analysis = {
            'overall_safety_metrics': {},
            'constraint_analysis': {},
            'risk_analysis': {},
            'temporal_safety_trends': {},
            'safety_violations_summary': {}
        }
        
        try:
            # Overall safety metrics
            safety_analysis['overall_safety_metrics'] = self._compute_overall_safety_metrics(data)
            
            # Detailed constraint analysis
            safety_analysis['constraint_analysis'] = self.constraint_analyzer.analyze_constraint_violations(data)
            
            # Risk analysis
            safety_analysis['risk_analysis'] = self.risk_analyzer.analyze_safety_margins(data)
            
            # Temporal trends
            if 'iteration' in data.columns or 'step' in data.columns:
                safety_analysis['temporal_safety_trends'] = self._analyze_safety_trends(data)
            
            # Violations summary
            safety_analysis['safety_violations_summary'] = self._summarize_violations(data)
            
        except Exception as e:
            self.logger.error(f"Error in safety analysis: {e}")
            safety_analysis['error'] = str(e)
        
        return safety_analysis
    
    def _compute_overall_safety_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute high-level safety metrics."""
        metrics = {}
        
        # Find constraint-related columns
        constraint_columns = [col for col in data.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['constraint', 'violation', 'cost', 'margin'])]
        
        if not constraint_columns:
            return {'no_constraint_data': True}
        
        # Total violations across all constraints
        total_steps = len(data)
        violation_steps = 0
        total_violation_cost = 0.0
        
        for col in constraint_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if 'margin' in col.lower():
                    # For margins, violations are negative values
                    violations = col_data < 0
                    violation_costs = np.abs(col_data[violations])
                else:
                    # For constraint costs, violations are positive values
                    violations = col_data > 0
                    violation_costs = col_data[violations]
                
                violation_steps += violations.sum()
                total_violation_cost += violation_costs.sum()
        
        metrics['total_violation_steps'] = int(violation_steps)
        metrics['violation_rate'] = float(violation_steps / total_steps) if total_steps > 0 else 0.0
        metrics['total_violation_cost'] = float(total_violation_cost)
        metrics['mean_violation_cost_per_step'] = float(total_violation_cost / total_steps) if total_steps > 0 else 0.0
        
        # Safety budget utilization (assuming a budget of 5% violation rate)
        safety_budget = 0.05
        metrics['safety_budget_utilization'] = float(metrics['violation_rate'] / safety_budget)
        
        # Consecutive violation analysis
        if len(constraint_columns) > 0:
            # Use first constraint column for consecutive violation analysis
            first_constraint = data[constraint_columns[0]].dropna().values
            if 'margin' in constraint_columns[0].lower():
                violations = first_constraint < 0
            else:
                violations = first_constraint > 0
            
            consecutive_violations = self._find_consecutive_violations(violations)
            metrics['max_consecutive_violations'] = consecutive_violations['max_length']
            metrics['mean_consecutive_violations'] = consecutive_violations['mean_length']
            metrics['num_violation_episodes'] = consecutive_violations['num_episodes']
        
        return metrics
    
    def _find_consecutive_violations(self, violations: np.ndarray) -> Dict[str, Any]:
        """Find consecutive violation episodes."""
        episodes = []
        current_episode_length = 0
        
        for violation in violations:
            if violation:
                current_episode_length += 1
            else:
                if current_episode_length > 0:
                    episodes.append(current_episode_length)
                    current_episode_length = 0
        
        # Handle case where violations continue to end of data
        if current_episode_length > 0:
            episodes.append(current_episode_length)
        
        if episodes:
            return {
                'max_length': max(episodes),
                'mean_length': np.mean(episodes),
                'num_episodes': len(episodes),
                'episode_lengths': episodes
            }
        else:
            return {
                'max_length': 0,
                'mean_length': 0.0,
                'num_episodes': 0,
                'episode_lengths': []
            }
    
    def _analyze_safety_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal trends in safety metrics."""
        time_col = 'iteration' if 'iteration' in data.columns else 'step'
        trends = {}
        
        # Find constraint columns
        constraint_columns = [col for col in data.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['constraint', 'violation', 'cost', 'margin'])]
        
        for col in constraint_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                time_data = data[time_col].iloc[:len(col_data)].values
                
                if len(col_data) > 3:
                    # Linear trend analysis
                    slope, intercept, r_value, p_value, std_err = stats.linregress(time_data, col_data)
                    
                    # Determine if safety is improving
                    if 'margin' in col.lower():
                        # For margins, positive slope means improving safety
                        improving = slope > 0.001 and p_value < 0.05
                    else:
                        # For constraint costs, negative slope means improving safety
                        improving = slope < -0.001 and p_value < 0.05
                    
                    trends[col] = {
                        'slope': float(slope),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'improving': improving,
                        'significant_trend': float(p_value) < 0.05
                    }
        
        return trends
    
    def _summarize_violations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create a summary of all safety violations."""
        summary = {
            'by_constraint': {},
            'severity_distribution': {},
            'time_to_first_violation': None,
            'recovery_analysis': {}
        }
        
        # Find constraint columns
        constraint_columns = [col for col in data.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['constraint', 'violation', 'cost'])]
        
        # Analyze each constraint
        for col in constraint_columns:
            if col in data.columns:
                col_data = data[col].dropna().values
                
                if 'margin' in col.lower():
                    violations = col_data < 0
                    violation_costs = np.abs(col_data[violations]) if len(col_data[violations]) > 0 else np.array([])
                else:
                    violations = col_data > 0
                    violation_costs = col_data[violations] if len(col_data[violations]) > 0 else np.array([])
                
                summary['by_constraint'][col] = {
                    'total_violations': int(violations.sum()),
                    'violation_rate': float(violations.mean()),
                    'mean_severity': float(np.mean(violation_costs)) if len(violation_costs) > 0 else 0.0,
                    'max_severity': float(np.max(violation_costs)) if len(violation_costs) > 0 else 0.0
                }
                
                # Time to first violation
                if summary['time_to_first_violation'] is None and violations.any():
                    first_violation_idx = np.argmax(violations)
                    summary['time_to_first_violation'] = int(first_violation_idx)
        
        return summary
    
    def compare_safety_performance(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare safety performance across multiple algorithms."""
        comparison = {
            'algorithms': list(algorithms_data.keys()),
            'individual_safety_analyses': {},
            'comparative_metrics': {},
            'safety_rankings': {}
        }
        
        # Analyze each algorithm
        for alg_name, data in algorithms_data.items():
            comparison['individual_safety_analyses'][alg_name] = self.analyze_safety_performance(data)
        
        # Comparative metrics
        comparison['comparative_metrics'] = self._compare_safety_metrics(
            comparison['individual_safety_analyses']
        )
        
        # Safety rankings
        comparison['safety_rankings'] = self._rank_algorithms_by_safety(
            comparison['individual_safety_analyses']
        )
        
        return comparison
    
    def _compare_safety_metrics(self, safety_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare safety metrics across algorithms."""
        comparisons = {}
        
        # Extract violation rates
        violation_rates = {}
        total_costs = {}
        
        for alg_name, analysis in safety_analyses.items():
            overall_metrics = analysis.get('overall_safety_metrics', {})
            if 'violation_rate' in overall_metrics:
                violation_rates[alg_name] = overall_metrics['violation_rate']
            if 'total_violation_cost' in overall_metrics:
                total_costs[alg_name] = overall_metrics['total_violation_cost']
        
        if violation_rates:
            comparisons['violation_rate_comparison'] = {
                'values': violation_rates,
                'best_algorithm': min(violation_rates, key=violation_rates.get),
                'worst_algorithm': max(violation_rates, key=violation_rates.get),
                'range': max(violation_rates.values()) - min(violation_rates.values())
            }
        
        if total_costs:
            comparisons['violation_cost_comparison'] = {
                'values': total_costs,
                'best_algorithm': min(total_costs, key=total_costs.get),
                'worst_algorithm': max(total_costs, key=total_costs.get),
                'range': max(total_costs.values()) - min(total_costs.values())
            }
        
        return comparisons
    
    def _rank_algorithms_by_safety(self, safety_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Rank algorithms by safety performance."""
        rankings = {}
        
        # Rank by violation rate (lower is better)
        violation_rates = {}
        for alg_name, analysis in safety_analyses.items():
            overall_metrics = analysis.get('overall_safety_metrics', {})
            if 'violation_rate' in overall_metrics:
                violation_rates[alg_name] = overall_metrics['violation_rate']
        
        if violation_rates:
            sorted_by_violations = sorted(violation_rates.items(), key=lambda x: x[1])
            rankings['by_violation_rate'] = [alg for alg, _ in sorted_by_violations]
        
        # Rank by total violation cost (lower is better)
        violation_costs = {}
        for alg_name, analysis in safety_analyses.items():
            overall_metrics = analysis.get('overall_safety_metrics', {})
            if 'total_violation_cost' in overall_metrics:
                violation_costs[alg_name] = overall_metrics['total_violation_cost']
        
        if violation_costs:
            sorted_by_costs = sorted(violation_costs.items(), key=lambda x: x[1])
            rankings['by_violation_cost'] = [alg for alg, _ in sorted_by_costs]
        
        return rankings