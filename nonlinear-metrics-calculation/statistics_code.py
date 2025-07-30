# Author:      Natasha Kovacheva <natasha.kovacheva@stud.hslu.ch>
# Created:     SS 2025
# Description: This script is built for calculating the statistical 
#               evaluation related to NSI due to its specific table structure.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, shapiro, pearsonr, spearmanr
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import friedmanchisquare
from statsmodels.stats.anova import AnovaRM


# class defined due to specific structure of NSI tables
class NSIStats:
    """
    Generalized NSI (Non-Stationarity Index) analyzer for different cohorts
    with consistent table structure.
    """
    
    def __init__(self, cohort_name: str = "Cohort"):
        self.cohort_name = cohort_name
        self.nsi_metrics = {
            'temporal_nsi': 'Temporal NSI (gait pacing stability)',
            'spatial_nsi_x': 'Spatial NSI X (medio-lateral drift)', 
            'spatial_nsi_y': 'Spatial NSI Y (anterior-posterior drift)',
            'spatial_nsi_z': 'Spatial NSI Z (vertical drift)'
        }
        
        self.hypotheses = {
            'temporal_nsi': {
                'H0': 'μ_before = μ_after (no difference in temporal NSI)',
                'H1': 'μ_before ≠ μ_after (temporal NSI differs between conditions)',
                'interpretation': 'Lower values indicate better temporal stationarity (more stable gait pacing)'
            },
            'spatial_nsi_y': {
                'H0': 'μ_before = μ_after (no difference in anterior-posterior drift)',
                'H1': 'μ_before ≠ μ_after (AP drift differs between conditions)',
                'interpretation': 'Lower values indicate less forward/backward CoM drift'
            },
            'spatial_nsi_x': {
                'H0': 'μ_before = μ_after (no difference in medio-lateral drift)',
                'H1': 'μ_before ≠ μ_after (ML drift differs between conditions)',
                'interpretation': 'Lower values indicate less side-to-side CoM drift'
            },
            'spatial_nsi_z': {
                'H0': 'μ_before = μ_after (no difference in vertical drift)',
                'H1': 'μ_before ≠ μ_after (vertical drift differs between conditions)',
                'interpretation': 'Lower values indicate less vertical CoM drift'
            }
        }
    
    def sensitivity_checks(self, before_vals: np.ndarray, after_vals: np.ndarray, 
                          label: str = "") -> Dict[str, Any]:
        """
        Perform sensitivity analyses including bootstrap CI and leave-one-out tests
        """
        diffs = before_vals - after_vals
        n = len(diffs)
        results = {}
        
        if n < 2:
            print(f"   [{label}] Not enough data for sensitivity.")
            return results
        
        # Bootstrap 95% confidence interval
        boots = np.random.choice(diffs, (10000, n), replace=True)
        lo, hi = np.percentile(boots.mean(axis=1), [2.5, 97.5])
        print(f"   [{label}] Bootstrap 95% CI on Δmean: [{lo:.3f}, {hi:.3f}]")
        results['bootstrap_ci'] = (lo, hi)
        
        # Leave-one-out sensitivity analysis
        p_t, p_w = [], []
        for i in range(n):
            lb = np.delete(before_vals, i)
            la = np.delete(after_vals, i)
            if len(lb) > 1:
                p_t.append(ttest_rel(lb, la).pvalue)
                p_w.append(wilcoxon(lb, la).pvalue)
        
        if p_t and p_w:
            print(f"   [{label}] LOO t-test p-range:    [{np.min(p_t):.3f}, {np.max(p_t):.3f}]")
            print(f"   [{label}] LOO Wilcoxon p-range: [{np.min(p_w):.3f}, {np.max(p_w):.3f}]")
            results['loo_ttest_range'] = (np.min(p_t), np.max(p_t))
            results['loo_wilcoxon_range'] = (np.min(p_w), np.max(p_w))
        
        return results
    
    def axis_coupling_analysis(self, before_vals_dict: Dict[str, np.ndarray], 
                              after_vals_dict: Dict[str, np.ndarray], 
                              diffs_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze ML-VT axis coupling: do changes in one axis relate to changes in the other?
        """
        print("\n" + "="*60)
        print("ML-VT AXIS COUPLING ANALYSIS")
        print("="*60)
        print("Research Question: Are ML and VT axis changes coupled?")
        print("Do subjects who worsen in one axis also worsen in the other?")
        print()
        
        results = {}
        
        # Calculate change magnitudes for available axes
        ml_changes = diffs_dict.get('spatial_nsi_x', np.array([]))
        vt_changes = diffs_dict.get('spatial_nsi_z', np.array([]))
        
        if len(ml_changes) == 0 or len(vt_changes) == 0:
            print("Insufficient data for axis coupling analysis")
            return results
        
        # Ensure same length
        min_len = min(len(ml_changes), len(vt_changes))
        ml_changes = ml_changes[:min_len]
        vt_changes = vt_changes[:min_len]
        
        print(f"Analyzing axis coupling (n={min_len} subjects)")
        print()
        
        # 1. ML ↔ VT coupling
        print("1. MEDIO-LATERAL ↔ VERTICAL COUPLING:")
        if len(ml_changes) >= 3:
            r_coupling, p_coupling = pearsonr(ml_changes, vt_changes)
            rho_coupling, p_rho_coupling = spearmanr(ml_changes, vt_changes)
            print(f"   Pearson r = {r_coupling:.3f}, p = {p_coupling:.3f}")
            print(f"   Spearman ρ = {rho_coupling:.3f}, p = {p_rho_coupling:.3f}")
            
            results['ml_vt_correlation'] = r_coupling
            results['ml_vt_p_value'] = p_coupling
            results['ml_vt_spearman'] = rho_coupling
            
            if p_coupling < 0.05:
                direction = "positive" if r_coupling > 0 else "negative"
                print(f"   → SIGNIFICANT {direction} coupling between ML-VT axes")
                if r_coupling > 0:
                    print("     (Subjects who worsen in ML also worsen in VT)")
                else:
                    print("     (Subjects who worsen in ML improve in VT, or vice versa)")
            else:
                print(f"   → NO significant coupling between axes")
                print("     (ML and VT changes are independent)")
        else:
            print("   Insufficient data for correlation analysis")
        
        print()
        
        # 2. Baseline correlation analysis
        print("2. BASELINE ML-VT RELATIONSHIP:")
        ml_before = before_vals_dict.get('spatial_nsi_x', np.array([]))[:min_len]
        vt_before = before_vals_dict.get('spatial_nsi_z', np.array([]))[:min_len]
        
        if len(ml_before) >= 3:
            r_baseline, p_baseline = pearsonr(ml_before, vt_before)
            print(f"   Baseline correlation: r = {r_baseline:.3f}, p = {p_baseline:.3f}")
            results['baseline_correlation'] = r_baseline
            results['baseline_p_value'] = p_baseline
            
            if p_baseline < 0.05:
                print("   → Baseline ML-VT coupling exists")
            else:
                print("   → No baseline ML-VT coupling")
        
        print()
        
        # 3. Change magnitude comparison
        print("3. CHANGE MAGNITUDE COMPARISON:")
        ml_abs_change = np.abs(ml_changes)
        vt_abs_change = np.abs(vt_changes)
        
        print(f"   ML change magnitude: {ml_abs_change.mean():.4f} ± {ml_abs_change.std():.4f}")
        print(f"   VT change magnitude: {vt_abs_change.mean():.4f} ± {vt_abs_change.std():.4f}")
        
        # Test which axis shows larger changes
        if len(ml_abs_change) >= 3:
            t_mag, p_mag = ttest_rel(ml_abs_change, vt_abs_change)
            print(f"   Magnitude comparison: t = {t_mag:.3f}, p = {p_mag:.3f}")
            results['magnitude_t'] = t_mag
            results['magnitude_p'] = p_mag
            
            if p_mag < 0.05:
                larger_axis = "VT" if vt_abs_change.mean() > ml_abs_change.mean() else "ML"
                print(f"   → {larger_axis} axis shows significantly larger changes")
            else:
                print("   → No significant difference in change magnitudes")
        
        return results
    
    def analyze_nsi_with_sensitivity(self, results_nsi: pd.DataFrame, 
                                   plot: bool = True) -> Dict[str, Any]:
        """
        Analyze NSI data with sensitivity checks
        """
        # Get unique subjects that have both before and after data
        before_subjects = set(results_nsi[results_nsi['condition'] == 'Before']['subject'])
        after_subjects = set(results_nsi[results_nsi['condition'] == 'After']['subject'])
        common_subjects = before_subjects.intersection(after_subjects)
        
        print(f"\n=== NSI SENSITIVITY ANALYSIS FOR {self.cohort_name.upper()} ({len(common_subjects)} subjects) ===")
        
        all_results = {}
        before_vals_dict = {}
        after_vals_dict = {}
        diffs_dict = {}
        
        if plot:
            # Create figure for all metrics
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            plt.suptitle(f'NSI Normality & Sensitivity Analysis - {self.cohort_name}', fontsize=16)
        
        for idx, (metric_col, metric_label) in enumerate(self.nsi_metrics.items()):
            print(f"\n=== {metric_label.upper()} ===")
            
            # Get before/after data for common subjects
            before_df = results_nsi[results_nsi['condition'] == 'Before'].set_index('subject')
            after_df = results_nsi[results_nsi['condition'] == 'After'].set_index('subject')
            
            # Extract values for common subjects only
            common = list(common_subjects)
            bvals = before_df.loc[common][metric_col].dropna().values
            avals = after_df.loc[common][metric_col].dropna().values
            
            # Ensure same length (in case of missing data)
            min_len = min(len(bvals), len(avals))
            bvals = bvals[:min_len]
            avals = avals[:min_len]
            diffs = bvals - avals
            
            # Store for axis coupling analysis
            before_vals_dict[metric_col] = bvals
            after_vals_dict[metric_col] = avals
            diffs_dict[metric_col] = diffs
            
            metric_results = {
                'n': len(diffs),
                'before_mean': bvals.mean() if len(bvals) > 0 else np.nan,
                'before_std': bvals.std(ddof=1) if len(bvals) > 0 else np.nan,
                'after_mean': avals.mean() if len(avals) > 0 else np.nan,
                'after_std': avals.std(ddof=1) if len(avals) > 0 else np.nan,
                'diff_mean': diffs.mean() if len(diffs) > 0 else np.nan,
                'diff_std': diffs.std(ddof=1) if len(diffs) > 0 else np.nan
            }
            
            print(f"\n--- {metric_label} (n={len(diffs)}) ---")
            
            # Descriptive statistics by condition
            if len(bvals) > 0:
                print(f" Before: mean={bvals.mean():.4f}, SD={bvals.std(ddof=1):.4f}")
            if len(avals) > 0:
                print(f" After:  mean={avals.mean():.4f}, SD={avals.std(ddof=1):.4f}")
            
            # Normality test for differences
            if len(diffs) >= 3:
                W, p_sw = shapiro(diffs)
                normal = (p_sw > 0.05)
                test_name = "Paired t-test" if normal else "Wilcoxon signed-rank"
                print(f" Shapiro–Wilk p={p_sw:.4f} → {test_name}")
                
                metric_results['shapiro_p'] = p_sw
                metric_results['is_normal'] = normal
                
                # Paired test
                if normal:
                    t, pval = ttest_rel(bvals, avals)
                    print(f" Paired t-test: t={t:.4f}, p={pval:.4f}")
                    metric_results['test_stat'] = t
                    metric_results['test_name'] = 't-test'
                else:
                    w, pval = wilcoxon(bvals, avals)
                    print(f" Wilcoxon: W={w:.4f}, p={pval:.4f}")
                    metric_results['test_stat'] = w
                    metric_results['test_name'] = 'wilcoxon'
                
                metric_results['p_value'] = pval
                
                print(f" Mean Δ={diffs.mean():.4f}, SD={diffs.std(ddof=1):.4f}")
                
                # Effect size (Cohen's d)
                cohen_d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0
                print(f" Cohen's d={cohen_d:.4f}")
                metric_results['cohen_d'] = cohen_d
                
                # Clinical interpretation
                print(f" Hypothesis: {self.hypotheses[metric_col]['H0']}")
                print(f" Clinical meaning: {self.hypotheses[metric_col]['interpretation']}")
                
                if pval < 0.05:
                    direction = "improved (decreased)" if diffs.mean() > 0 else "worsened (increased)"
                    print(f" Result: SIGNIFICANT - NSI {direction}")
                else:
                    print(f" Result: NOT SIGNIFICANT - no change detected")
                
                # Sensitivity analyses
                print(" Sensitivity analyses:")
                sensitivity_results = self.sensitivity_checks(bvals, avals, label=f"{metric_col}")
                metric_results.update(sensitivity_results)
                
                if plot:
                    # Q-Q plot (top row)
                    ax_qq = axes[0, idx]
                    stats.probplot(diffs, dist="norm", plot=ax_qq)
                    ax_qq.set_title(f"{metric_label}\nQ-Q Plot")
                    ax_qq.set_xlabel('Theoretical Quantiles')
                    ax_qq.set_ylabel('Sample Quantiles')
                    ax_qq.grid(True, alpha=0.3)
                    
                    # Histogram of differences (bottom row)
                    ax_hist = axes[1, idx]
                    ax_hist.hist(diffs, bins=min(15, max(5, len(diffs)//3)), density=True,
                                 color='skyblue', edgecolor='black', alpha=0.6)
                    mu, sigma = diffs.mean(), diffs.std(ddof=1)
                    x = np.linspace(diffs.min()-sigma, diffs.max()+sigma, 100)
                    ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
                    ax_hist.legend()
                    ax_hist.axvline(0, color='black', linestyle='--', alpha=0.7, label='No change')
                    ax_hist.set_title(f"{metric_label}\nHistogram of Differences")
                    ax_hist.set_xlabel('Before - After')
                    ax_hist.set_ylabel('Density')
                    ax_hist.grid(True, alpha=0.3)
            else:
                print(f" Insufficient data for normality testing (n={len(diffs)})")
            
            all_results[metric_col] = metric_results
        
        if plot:
            plt.tight_layout()
            plt.show()
        
        # Perform axis coupling analysis
        coupling_results = self.axis_coupling_analysis(before_vals_dict, after_vals_dict, diffs_dict)
        all_results['axis_coupling'] = coupling_results
        
        # Summary table of results
        self._print_summary()
        
        return all_results
    
    def plot_paired_comparisons(self, results_nsi: pd.DataFrame):
        """
        Create paired comparison plots with box plots and individual changes
        """
        nsi_types = list(self.nsi_metrics.keys())
        titles = ['Temporal NSI', 'Spatial NSI X (ML)', 'Spatial NSI Y (AP)', 'Spatial NSI Z (VT)']
        
        # Calculate common scale for spatial NSI
        spatial_data = []
        for nsi_type in ['spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']:
            spatial_data.extend(results_nsi[nsi_type].values)
        spatial_min = min(spatial_data)
        spatial_max = max(spatial_data)
        spatial_range = spatial_max - spatial_min
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # First row: Clean box plots with significance annotations
        for idx, (nsi_type, title) in enumerate(zip(nsi_types, titles)):
            ax = axes[0, idx]
            
            before_data = results_nsi[results_nsi['condition'] == 'Before'][nsi_type].values
            after_data = results_nsi[results_nsi['condition'] == 'After'][nsi_type].values
            
            # Create clean boxplot
            bp = ax.boxplot([before_data, after_data], labels=['Before', 'After'], 
                            patch_artist=True, widths=0.5)
            
            # Customize boxplot colors
            bp['boxes'][0].set_facecolor("#3734db")
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor("#3cb7e7")
            bp['boxes'][1].set_alpha(0.6)
            
            # Make box edges more prominent
            bp['boxes'][0].set_edgecolor('#2a2a2a')
            bp['boxes'][0].set_linewidth(2)
            bp['boxes'][1].set_edgecolor('#2a2a2a')
            bp['boxes'][1].set_linewidth(2)
            
            # Customize other elements
            for element in ['whiskers', 'fliers', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1.5)
            
            # Make medians more prominent
            plt.setp(bp['medians'], color='black', linewidth=2.5)
            
            # Add individual points
            np.random.seed(42)
            jitter_amount = 0.05
            x_before = np.ones(len(before_data)) + np.random.normal(0, jitter_amount, len(before_data))
            x_after = 2 * np.ones(len(after_data)) + np.random.normal(0, jitter_amount, len(after_data))
            
            ax.scatter(x_before, before_data, alpha=0.4, s=40, color='#CC0000', 
                       edgecolors='black', linewidth=0.5)
            ax.scatter(x_after, after_data, alpha=0.4, s=40, color='#2E8B8B', 
                       edgecolors='black', linewidth=0.5)
            
            # Add significance annotation
            t_stat, p_value = stats.ttest_rel(before_data, after_data)
            
            # Set y-axis limits
            if nsi_type == 'temporal_nsi':
                y_max = max(np.max(before_data), np.max(after_data))
                y_min = min(np.min(before_data), np.min(after_data))
                y_range = y_max - y_min
                ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.25)
                bracket_height = y_max + y_range * 0.1
            else:  # spatial NSI - use common scale
                ax.set_ylim(spatial_min - spatial_range*0.1, spatial_max + spatial_range*0.25)
                bracket_height = spatial_max + spatial_range * 0.1
                y_range = spatial_range
            
            # Draw significance bracket
            ax.plot([1, 2], [bracket_height, bracket_height], 'k-', linewidth=1.5)
            ax.plot([1, 1], [bracket_height - y_range*0.02, bracket_height], 'k-', linewidth=1.5)
            ax.plot([2, 2], [bracket_height - y_range*0.02, bracket_height], 'k-', linewidth=1.5)
            
            if p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            ax.text(1.5, bracket_height + y_range*0.02, sig_text, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
            ax.text(1.5, bracket_height + y_range*0.08, f'p={p_value:.3f}', ha='center', 
                    va='bottom', fontsize=10)
            
            # Set background color
            ax.set_facecolor('#F5F5F5')
            
            # Improve grid visibility
            ax.grid(axis='y', alpha=0.4, color='gray', linestyle='-', linewidth=0.8)
            
            # Make axis labels more prominent
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('NSI Value', fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Second row: Individual changes
        for idx, (nsi_type, title) in enumerate(zip(nsi_types, titles)):
            ax = axes[1, idx]
            
            before_df = results_nsi[results_nsi['condition'] == 'Before'].sort_values('subject')
            after_df = results_nsi[results_nsi['condition'] == 'After'].sort_values('subject')
            
            before_data = before_df[nsi_type].values
            after_data = after_df[nsi_type].values
            subjects = before_df['subject'].values
            
            # Plot lines connecting before and after for each subject
            for i in range(len(subjects)):
                color = 'green' if after_data[i] < before_data[i] else 'red'
                ax.plot([0, 1], [before_data[i], after_data[i]], 'o-', 
                        color=color, alpha=0.6, linewidth=2, markersize=8)
            
            # Add mean lines
            ax.axhline(np.mean(before_data), xmin=0, xmax=0.3, color='black', 
                    linestyle='--', linewidth=2, label='Mean Before')
            ax.axhline(np.mean(after_data), xmin=0.7, xmax=1, color='black', 
                    linestyle='--', linewidth=2, label='Mean After')
            
            ax.set_xlim(-0.2, 1.2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Before', 'After'])
            ax.set_ylabel('NSI Value', fontsize=11)
            ax.set_title(f'{title} - Individual Changes', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add improvement percentage
            improved = sum(after_data < before_data)
            ax.text(0.5, ax.get_ylim()[1] * 0.95, f'{improved}/{len(subjects)} improved', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.suptitle(f'NSI Statistical Analysis: Paired Comparisons - {self.cohort_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _print_summary(self):
        """Print summary of analysis results"""
        print("\n" + "="*80)
        print("SUMMARY OF SENSITIVITY ANALYSIS RESULTS")
        print("="*80)
        print("Effect size interpretation (Cohen's d):")
        print("  Small effect: |d| = 0.2")
        print("  Medium effect: |d| = 0.5") 
        print("  Large effect: |d| = 0.8")
        print("\nPositive d = NSI decreased (improvement)")
        print("Negative d = NSI increased (worsening)")
    
    def generate_summary_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all results
        """
        summary_data = []
        
        for metric, metric_results in results.items():
            if metric == 'axis_coupling':
                continue
                
            row = {
                'metric': metric,
                'n': metric_results.get('n', np.nan),
                'before_mean': metric_results.get('before_mean', np.nan),
                'after_mean': metric_results.get('after_mean', np.nan),
                'diff_mean': metric_results.get('diff_mean', np.nan),
                'cohen_d': metric_results.get('cohen_d', np.nan),
                'p_value': metric_results.get('p_value', np.nan),
                'test_used': metric_results.get('test_name', ''),
                'is_significant': metric_results.get('p_value', 1.0) < 0.05
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


# Example usage function
def analyze_cohort(data: pd.DataFrame, cohort_name: str = "Cohort"):
    """
    Function to analyze a cohort - full.
    """
    analyzer = NSIStats(cohort_name=cohort_name)
    
    # Run sensitivity analysis
    results = analyzer.analyze_nsi_with_sensitivity(data, plot=True)
    
    # Create paired comparison plots
    analyzer.plot_paired_comparisons(data)
    
    # Generate summary report
    summary = analyzer.generate_summary_report(results)
    
    print(f"\n=== SUMMARY TABLE FOR {cohort_name.upper()} ===")
    print(summary.to_string(index=False))
    
    return results, summary


# Example of how to use with multiple cohorts
def compare_cohorts(cohort_dict: Dict[str, pd.DataFrame]):
    """
    Compare multiple cohorts
    
    Parameters:
    -----------
    cohort_dict : dict
        Dictionary mapping cohort names to their DataFrames
    """
    all_results = {}
    all_summaries = {}
    
    for cohort_name, cohort_data in cohort_dict.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING {cohort_name.upper()}")
        print(f"{'='*80}")
        
        results, summary = analyze_cohort(cohort_data, cohort_name)
        all_results[cohort_name] = results
        all_summaries[cohort_name] = summary
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    metrics = ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']
    metric_labels = ['Temporal', 'ML', 'AP', 'VT']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        cohort_names = []
        effect_sizes = []
        p_values = []
        
        for cohort_name, summary in all_summaries.items():
            metric_data = summary[summary['metric'] == metric]
            if not metric_data.empty:
                cohort_names.append(cohort_name)
                effect_sizes.append(metric_data['cohen_d'].values[0])
                p_values.append(metric_data['p_value'].values[0])
        
        
    
    return all_results, all_summaries




def analyze_speed_sensitivity(results_nsi, cohort_name="Neurological Patients"):
    """
    Analyze NSI sensitivity across different walking speeds with normality testing
    
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    
    if 'patient_id' not in results_nsi.columns:
        results_nsi['patient_id'] = results_nsi['subject'].str[1:]
    
    nsi_metrics = ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']
    speed_conditions = results_nsi['speed'].unique()
    comparisons = []
    
    for i, speed1 in enumerate(speed_conditions):
        for j, speed2 in enumerate(speed_conditions):
            if i < j:
                comparisons.append((speed1, speed2))
    
    print(f"\n{'='*80}")
    print(f"SPEED SENSITIVITY ANALYSIS - {cohort_name.upper()}")
    print(f"{'='*80}")
    print(f"Speed conditions: {list(speed_conditions)}")
    print(f"Comparisons: {comparisons}")
    print("="*70)
    print("NORMALITY TESTING FOR WALKING SPEED COMPARISONS")
    print("="*70)
    print("Testing differences between speed pairs for normality")
    print("If p > 0.05: Use paired t-test (normal)")
    print("If p < 0.05: Use Wilcoxon signed-rank test (non-normal)")
    print("-"*70)
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'Normality Assessment for Walking Speed Paired Differences - {cohort_name}', 
                 fontsize=18, fontweight='bold')
    
    plot_idx = 0
    all_recommendations = {}
    speed_results = {}
    
    for nsi_idx, nsi_type in enumerate(nsi_metrics):
        print(f"\n{'='*50}")
        print(f"{nsi_type.upper()}")
        print(f"{'='*50}")
        
        recommendations = {}
        metric_comparisons = {}
        
        for comp_idx, (speed1, speed2) in enumerate(comparisons):
            # Find patients that have both speeds
            patients_with_pair = []
            all_patients = results_nsi['patient_id'].unique()
            
            for patient in all_patients:
                patient_data = results_nsi[results_nsi['patient_id'] == patient]
                speeds_present = patient_data['speed'].unique()
                if speed1 in speeds_present and speed2 in speeds_present:
                    patients_with_pair.append(patient)
            
            # Get paired data
            vals1 = []
            vals2 = []
            
            for patient in patients_with_pair:
                patient_data = results_nsi[results_nsi['patient_id'] == patient]
                val1 = patient_data[patient_data['speed'] == speed1][nsi_type].values
                val2 = patient_data[patient_data['speed'] == speed2][nsi_type].values
                
                if len(val1) > 0 and len(val2) > 0:
                    vals1.append(val1[0])
                    vals2.append(val2[0])
            
            vals1 = np.array(vals1)
            vals2 = np.array(vals2)
            
            if len(vals1) >= 3:  
                # Calculate differences
                differences = vals1 - vals2
                n_pairs = len(differences)
                
                # Shapiro-Wilk test
                statistic, p_value = stats.shapiro(differences)
                
                if p_value > 0.05:
                    test_recommendation = "Paired t-test"
                    normality_status = "Normal"
                    use_parametric = True
                else:
                    test_recommendation = "Wilcoxon signed-rank"
                    normality_status = "Non-normal"
                    use_parametric = False
                
                recommendations[f"{speed1}_vs_{speed2}"] = test_recommendation
                
                # Perform the appropriate test
                if use_parametric:
                    t_stat, test_p = ttest_rel(vals1, vals2)
                    test_statistic = t_stat
                    test_name = "Paired t-test"
                else:
                    w_stat, test_p = wilcoxon(vals1, vals2)
                    test_statistic = w_stat
                    test_name = "Wilcoxon"
                
                # Calculate effect size
                cohen_d = (vals1.mean() - vals2.mean()) / np.sqrt((vals1.var() + vals2.var()) / 2)
                
                # Store comparison results
                metric_comparisons[f"{speed1}_vs_{speed2}"] = {
                    'n_pairs': n_pairs,
                    'mean_diff': differences.mean(),
                    'test_statistic': test_statistic,
                    'p_value': test_p,
                    'cohen_d': cohen_d,
                    'test_type': test_name,
                    'significant': test_p < 0.05,
                    'normality_p': p_value
                }
                
                # Print results
                print(f"\n{speed1} vs {speed2}:")
                print(f"  Sample size: n = {n_pairs} pairs")
                print(f"  Mean {speed1}: {np.mean(vals1):.4f}")
                print(f"  Mean {speed2}: {np.mean(vals2):.4f}")
                print(f"  Mean difference: {np.mean(differences):.4f} ± {np.std(differences):.4f}")
                print(f"  Shapiro-Wilk p-value: {p_value:.4f} ({normality_status})")
                print(f"  {test_name}: stat={test_statistic:.3f}, p={test_p:.3f}")
                print(f"  Cohen's d: {cohen_d:.3f}")
                print(f"  Result: {'SIGNIFICANT' if test_p < 0.05 else 'NOT SIGNIFICANT'}")
                
                # Plotting
                if plot_idx < 24:  
                    # Calculate subplot position
                    row = plot_idx // 3
                    col = plot_idx % 3
                    
                    # Q-Q plot
                    ax_qq = plt.subplot(8, 3, plot_idx + 1)
                    stats.probplot(differences, dist="norm", plot=ax_qq)
                    ax_qq.set_title(f'{nsi_type.replace("_", " ").title()}\n{speed1} vs {speed2}', fontsize=10)
                    ax_qq.grid(True, alpha=0.3)
                    
                    # Add p-value to plot
                    ax_qq.text(0.05, 0.95, f'SW p={p_value:.3f}', 
                              transform=ax_qq.transAxes, 
                              bbox=dict(boxstyle='round', 
                                       facecolor='yellow' if p_value < 0.05 else 'lightgreen', 
                                       alpha=0.7),
                              verticalalignment='top')
                    
                    # Histogram with normal overlay
                    ax_hist = plt.subplot(8, 3, plot_idx + 13)  # Second half of figure
                    
                    # Plot histogram
                    n_bins = min(15, max(5, n_pairs // 3))
                    ax_hist.hist(differences, bins=n_bins, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black')
                    
                    # Overlay normal distribution
                    mu, sigma = np.mean(differences), np.std(differences, ddof=1)
                    x = np.linspace(differences.min() - sigma, differences.max() + sigma, 100)
                    ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
                    
                    ax_hist.set_xlabel(f'Difference ({speed1} - {speed2})')
                    ax_hist.set_ylabel('Density')
                    ax_hist.set_title(f'{nsi_type.replace("_", " ").title()}\nHistogram', fontsize=10)
                    ax_hist.legend()
                    ax_hist.grid(True, alpha=0.3)
                    
                    plot_idx += 1
            else:
                print(f"\n{speed1} vs {speed2}: Insufficient data (n={len(vals1)})")
                recommendations[f"{speed1}_vs_{speed2}"] = "Insufficient data"
        
        all_recommendations[nsi_type] = recommendations
        speed_results[nsi_type] = metric_comparisons
    
    plt.tight_layout()
    plt.show()
    
    # Summary table for all comparisons
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY OF SPEED SENSITIVITY RESULTS")
    print("="*80)
    print(f"{'NSI Type':<20} {'Comparison':<15} {'n':<5} {'Test':<20} {'p-value':<10} {'Effect':<10}")
    print("-"*80)
    
    for nsi_type in nsi_metrics:
        if nsi_type in speed_results:
            for comp_name, results in speed_results[nsi_type].items():
                sig_marker = "*" if results['significant'] else "ns"
                effect_size = f"d={results['cohen_d']:.3f}"
                print(f"{nsi_type:<20} {comp_name:<15} {results['n_pairs']:<5} "
                      f"{results['test_type']:<20} {results['p_value']:<10.3f} {effect_size:<10} {sig_marker}")
    
    # Speed sensitivity summary
    print(f"\n{'='*80}")
    print("SPEED SENSITIVITY SUMMARY")
    print(f"{'='*80}")
    
    sensitive_metrics = []
    for nsi_type in nsi_metrics:
        if nsi_type in speed_results:
            significant_comparisons = sum(1 for comp in speed_results[nsi_type].values() 
                                        if comp.get('significant', False))
            total_comparisons = len(speed_results[nsi_type])
            
            if significant_comparisons > 0:
                sensitive_metrics.append(nsi_type)
                print(f"{nsi_type:15s}: {significant_comparisons}/{total_comparisons} significant comparisons")
            else:
                print(f"{nsi_type:15s}: {significant_comparisons}/{total_comparisons} significant comparisons")
    
    if sensitive_metrics:
        print(f"\nMost speed-sensitive metrics: {', '.join(sensitive_metrics)}")
    else:
        print(f"\nNo metrics showed significant speed sensitivity")
    
    print("\n" + "="*70)
    print("QUICK DECISION GUIDE")
    print("="*70)
    print("\nBased on Shapiro-Wilk test results, these tests were used:")
    print("\nFor each NSI type and speed comparison:")
    
    for nsi_type in nsi_metrics:
        print(f"\n{nsi_type.upper()}:")
        if nsi_type in all_recommendations:
            for comp_name, test in all_recommendations[nsi_type].items():
                comp_display = comp_name.replace('_vs_', ' vs ')
                print(f"  {comp_display}: {test}")
    
    print("\nNote: With small sample sizes (n<30), normality tests may lack power.")
    print("Visual inspection (Q-Q plots) is equally important for decision-making.")
    print("\nFor comparisons with n<3, no statistical test is recommended.")
    
    return {
        'speed_results': speed_results,
        'recommendations': all_recommendations,
        'sensitive_metrics': sensitive_metrics,
        'cohort_name': cohort_name
    }


def analyze_by_patient(results_nsi, cohort_name="Neurological Patients"):
    """
    Enhanced speed sensitivity analysis including Repeated Measures ANOVA
    
    Parameters:
    -----------
    results_nsi : pandas.DataFrame
        DataFrame with columns: subject, speed, temporal_nsi, spatial_nsi_x, spatial_nsi_y, spatial_nsi_z
        speed should contain values like: 'PWS', 'SWS', 'FWS'
    """
    
    # Extract patient ID from subject names if needed
    if 'patient_id' not in results_nsi.columns:
        results_nsi['patient_id'] = results_nsi['subject'].str[1:]
    
    nsi_metrics = ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']
    speed_conditions = results_nsi['speed'].unique()
    
    # Define hypotheses for each NSI type
    hypotheses = {
        'temporal_nsi': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in temporal NSI across speeds)',
            'H1': 'At least one speed differs in temporal NSI',
            'interpretation': 'Lower values indicate better temporal stationarity (more stable gait pacing)'
        },
        'spatial_nsi_y': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in anterior-posterior drift)',
            'H1': 'At least one speed differs in AP drift',
            'interpretation': 'Lower values indicate less forward/backward CoM drift'
        },
        'spatial_nsi_x': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in medio-lateral drift)',
            'H1': 'At least one speed differs in ML drift',
            'interpretation': 'Lower values indicate less side-to-side CoM drift'
        },
        'spatial_nsi_z': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in vertical drift)',
            'H1': 'At least one speed differs in vertical drift',
            'interpretation': 'Lower values indicate less vertical CoM drift'
        }
    }
    
    print(f"\n{'='*80}")
    print(f"ENHANCED SPEED SENSITIVITY ANALYSIS WITH ANOVA - {cohort_name.upper()}")
    print(f"{'='*80}")
    print(f"Speed conditions: {list(speed_conditions)}")
    
    # Store results for summary
    anova_results = {}
    pairwise_results = {}
    
    # REPEATED MEASURES ANOVA
    print(f"\n{'='*70}")
    print("REPEATED MEASURES ANOVA ANALYSIS")
    print(f"{'='*70}")
    
    for nsi_type in nsi_metrics:
        print(f"\n{nsi_type.upper()}:")
        print(f"  H₀: {hypotheses[nsi_type]['H0']}")
        print(f"  H₁: {hypotheses[nsi_type]['H1']}")
        print(f"  Note: {hypotheses[nsi_type]['interpretation']}")
        
        # Find patients with all three speeds (complete cases)
        all_patients = results_nsi['patient_id'].unique()
        complete_patients = []
        
        for patient in all_patients:
            patient_data = results_nsi[results_nsi['patient_id'] == patient]
            speeds_present = patient_data['speed'].unique()
            if set(['PWS', 'SWS', 'FWS']).issubset(set(speeds_present)):
                complete_patients.append(patient)
        
        print(f"  Complete cases: {len(complete_patients)} / {len(all_patients)}")
        
        if len(complete_patients) < 3:
            print("  → Insufficient data for ANOVA (n<3)")
            anova_results[nsi_type] = {'status': 'insufficient_data'}
            continue
        
        # Extract values for complete cases
        pws_vals = []
        sws_vals = []
        fws_vals = []
        
        for patient in complete_patients:
            patient_data = results_nsi[results_nsi['patient_id'] == patient]
            pws_vals.append(patient_data[patient_data['speed'] == 'PWS'][nsi_type].values[0])
            sws_vals.append(patient_data[patient_data['speed'] == 'SWS'][nsi_type].values[0])
            fws_vals.append(patient_data[patient_data['speed'] == 'FWS'][nsi_type].values[0])
        
        pws_vals = np.array(pws_vals)
        sws_vals = np.array(sws_vals)
        fws_vals = np.array(fws_vals)
        
        # Descriptive statistics
        print("  Means ± SD:")
        print(f"    PWS: {pws_vals.mean():.4f} ± {pws_vals.std(ddof=1):.4f}")
        print(f"    SWS: {sws_vals.mean():.4f} ± {sws_vals.std(ddof=1):.4f}")
        print(f"    FWS: {fws_vals.mean():.4f} ± {fws_vals.std(ddof=1):.4f}")
        
        # Friedman test (non-parametric)
        friedman_stat, friedman_p = friedmanchisquare(pws_vals, sws_vals, fws_vals)
        
        # Prepare data for RM-ANOVA
        df_long = pd.DataFrame({
            'patient_id': np.repeat(complete_patients, 3),
            'speed': ['PWS', 'SWS', 'FWS'] * len(complete_patients),
            'value': np.concatenate([pws_vals, sws_vals, fws_vals])
        })
        
        # Repeated Measures ANOVA
        try:
            aov = AnovaRM(df_long, depvar='value', subject='patient_id', within=['speed']).fit()
            anova_f = aov.anova_table['F Value'][0]
            anova_p = aov.anova_table['Pr > F'][0]
            anova_success = True
        except Exception as e:
            print(f"  → RM-ANOVA failed: {str(e)}")
            anova_success = False
            anova_f = np.nan
            anova_p = np.nan
        
        # Test ANOVA assumptions
        if anova_success:
            # Calculate residuals manually
            grand_mean = df_long['value'].mean()
            subj_mean = df_long.groupby('patient_id')['value'].transform('mean')
            speed_mean = df_long.groupby('speed')['value'].transform('mean')
            df_long['resid'] = df_long['value'] - (subj_mean + speed_mean - grand_mean)
            
            # Test residual normality
            residuals = df_long['resid'].values
            shapiro_stat, shapiro_p = shapiro(residuals)
        else:
            shapiro_stat = np.nan
            shapiro_p = np.nan
        
        print("\n  Inferential Statistics:")
        print(f"    Friedman χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}")
        if anova_success:
            print(f"    RM-ANOVA F = {anova_f:.3f}, p = {anova_p:.4f}")
            print(f"    ANOVA residuals Shapiro–Wilk: W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")
            
            # Show ANOVA table
            print("\n    Detailed ANOVA Results:")
            print(aov.summary())
        
        # Store results
        anova_results[nsi_type] = {
            'status': 'complete',
            'n_complete': len(complete_patients),
            'friedman_stat': friedman_stat,
            'friedman_p': friedman_p,
            'anova_f': anova_f,
            'anova_p': anova_p,
            'anova_success': anova_success,
            'shapiro_p': shapiro_p,
            'means': {
                'PWS': pws_vals.mean(),
                'SWS': sws_vals.mean(),
                'FWS': fws_vals.mean()
            },
            'sds': {
                'PWS': pws_vals.std(ddof=1),
                'SWS': sws_vals.std(ddof=1),
                'FWS': fws_vals.std(ddof=1)
            }
        }
        
        # Interpretation
        if friedman_p < 0.05:
            print("    → Friedman: SIGNIFICANT omnibus (reject H₀)")
        else:
            print("    → Friedman: NOT SIGNIFICANT omnibus (fail to reject H₀)")
            
        if anova_success and anova_p < 0.05:
            print("    → RM-ANOVA: SIGNIFICANT omnibus (reject H₀)")
        elif anova_success:
            print("    → RM-ANOVA: NOT SIGNIFICANT omnibus (fail to reject H₀)")
    
    # PART 2: POST-HOC PAIRWISE COMPARISONS
    print(f"\n{'='*70}")
    print("POST-HOC PAIRWISE COMPARISONS")
    print(f"{'='*70}")
    print("Note: Using all available patients for each pairwise comparison")
    print("Bonferroni correction: α = 0.05/3 = 0.0167 for 3 comparisons")
    
    comparisons = [('PWS', 'SWS'), ('PWS', 'FWS'), ('SWS', 'FWS')]
    
    for nsi_type in nsi_metrics:
        print(f"\n{nsi_type.upper()} - Pairwise Comparisons:")
        
        metric_comparisons = {}
        
        for speed1, speed2 in comparisons:
            # Find patients that have both speeds
            all_patients = results_nsi['patient_id'].unique()
            patients_with_pair = []
            
            for patient in all_patients:
                patient_data = results_nsi[results_nsi['patient_id'] == patient]
                speeds_present = patient_data['speed'].unique()
                if speed1 in speeds_present and speed2 in speeds_present:
                    patients_with_pair.append(patient)
            
            # Get paired data
            vals1 = []
            vals2 = []
            
            for patient in patients_with_pair:
                patient_data = results_nsi[results_nsi['patient_id'] == patient]
                val1 = patient_data[patient_data['speed'] == speed1][nsi_type].values
                val2 = patient_data[patient_data['speed'] == speed2][nsi_type].values
                
                if len(val1) > 0 and len(val2) > 0:
                    vals1.append(val1[0])
                    vals2.append(val2[0])
            
            vals1 = np.array(vals1)
            vals2 = np.array(vals2)
            
            if len(vals1) >= 3:
                # Check normality and decide test
                diff = vals1 - vals2
                _, shapiro_p = shapiro(diff)
                
                if shapiro_p > 0.05:
                    # Use paired t-test
                    t_stat, p_value = ttest_rel(vals1, vals2)
                    test_used = "Paired t-test"
                    test_stat = t_stat
                else:
                    # Use Wilcoxon signed-rank test
                    w_stat, p_value = wilcoxon(vals1, vals2)
                    test_used = "Wilcoxon signed-rank"
                    test_stat = w_stat
                
                # Effect size
                cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                
                print(f"\n  {speed1} vs {speed2} (n={len(vals1)} pairs):")
                print(f"    {speed1} Mean: {np.mean(vals1):.4f}, {speed2} Mean: {np.mean(vals2):.4f}")
                print(f"    Mean difference: {np.mean(diff):.4f}")
                print(f"    Test used: {test_used} (Shapiro-Wilk p={shapiro_p:.4f})")
                print(f"    Test statistic: {test_stat:.4f}")
                print(f"    p-value: {p_value:.4f}")
                print(f"    Cohen's d: {cohen_d:.4f}")
                
                # Bonferroni corrected significance
                if p_value < 0.0167:
                    print(f"    Result: SIGNIFICANT (Bonferroni corrected α=0.0167)")
                    significant = True
                else:
                    print(f"    Result: NOT SIGNIFICANT after Bonferroni correction")
                    significant = False
                
                # Store results
                metric_comparisons[f"{speed1}_vs_{speed2}"] = {
                    'n_pairs': len(vals1),
                    'mean_diff': np.mean(diff),
                    'test_statistic': test_stat,
                    'p_value': p_value,
                    'cohen_d': cohen_d,
                    'test_type': test_used,
                    'significant': significant,
                    'bonferroni_significant': significant,
                    'normality_p': shapiro_p
                }
            else:
                print(f"\n  {speed1} vs {speed2}: Insufficient paired data (n={len(vals1)})")
                metric_comparisons[f"{speed1}_vs_{speed2}"] = {
                    'n_pairs': len(vals1),
                    'status': 'insufficient_data'
                }
        
        pairwise_results[nsi_type] = metric_comparisons
    
    # PART 3: VISUALIZATION
    create_speed_sensitivity_plots(results_nsi, cohort_name)
    
    # PART 4: SUMMARY
    print_speed_sensitivity_summary(anova_results, pairwise_results, cohort_name)
    
    return {
        'anova_results': anova_results,
        'pairwise_results': pairwise_results,
        'cohort_name': cohort_name
    }


def analyze_by_diagnosis(results_nsi, cohort_name="Neurological Patients"):
    """
    Complete diagnosis-based analysis with repeated measures ANOVA for each diagnosis
    
    Parameters:
    -----------
    results_nsi : pandas.DataFrame
        DataFrame with columns: subject, condition, speed, temporal_nsi, spatial_nsi_x, spatial_nsi_y, spatial_nsi_z
        condition should contain diagnosis names
    """
    
    # Extract patient ID from subject names if needed
    if 'patient_id' not in results_nsi.columns:
        results_nsi['patient_id'] = results_nsi['subject'].str[1:]
    
    # Get unique conditions (diagnoses)
    conditions = results_nsi['condition'].unique()
    
    print("="*80)
    print(f"NSI ANALYSIS BY DIAGNOSIS WITH WALKING SPEED COMPARISONS - {cohort_name.upper()}")
    print("="*80)
    print(f"Diagnoses found: {', '.join(conditions)}")
    
    # Define the hypotheses for each NSI type
    hypotheses = {
        'temporal_nsi': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in temporal NSI across speeds)',
            'H1': 'At least one speed differs in temporal NSI',
            'interpretation': 'Lower values indicate better temporal stationarity (more stable gait pacing)'
        },
        'spatial_nsi_y': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in anterior-posterior drift)',
            'H1': 'At least one speed differs in AP drift',
            'interpretation': 'Lower values indicate less forward/backward CoM drift'
        },
        'spatial_nsi_x': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in medio-lateral drift)',
            'H1': 'At least one speed differs in ML drift',
            'interpretation': 'Lower values indicate less side-to-side CoM drift'
        },
        'spatial_nsi_z': {
            'H0': 'μ_PWS = μ_SWS = μ_FWS (no difference in vertical drift)',
            'H1': 'At least one speed differs in vertical drift',
            'interpretation': 'Lower values indicate less vertical CoM drift'
        }
    }
    
    all_diagnosis_results = {}
    
    # Process each diagnosis separately
    for condition in conditions:
        print(f"\n{'='*80}")
        print(f"DIAGNOSIS: {condition}")
        print(f"{'='*80}")
        
        # Filter data for this diagnosis
        condition_data = results_nsi[results_nsi['condition'] == condition]
        patients = condition_data['patient_id'].unique()
        
        print(f"\nREPEATED MEASURES ANOVA RESULTS FOR {condition}")
        print("-"*80)
        
        diagnosis_results = {
            'anova_results': {},
            'pairwise_results': {},
            'summary_stats': {}
        }
        
        # ANOVA analysis for each NSI metric
        for nsi_type in ['temporal_nsi','spatial_nsi_x','spatial_nsi_y','spatial_nsi_z']:
            # Select only subjects with all three speeds
            complete_patients = []
            for pid in patients:
                patient_speeds = condition_data.loc[condition_data['patient_id']==pid, 'speed'].unique()
                if set(['PWS','SWS','FWS']).issubset(set(patient_speeds)):
                    complete_patients.append(pid)
            
            # Gather values for complete cases
            pws = condition_data.loc[
                (condition_data['patient_id'].isin(complete_patients)) & (condition_data['speed']=='PWS'),
                nsi_type
            ].values
            sws = condition_data.loc[
                (condition_data['patient_id'].isin(complete_patients)) & (condition_data['speed']=='SWS'),
                nsi_type
            ].values
            fws = condition_data.loc[
                (condition_data['patient_id'].isin(complete_patients)) & (condition_data['speed']=='FWS'),
                nsi_type
            ].values
            
            print(f"\n{nsi_type.upper()}:")
            print(f"  H₀: {hypotheses[nsi_type]['H0']}")
            print(f"  H₁: {hypotheses[nsi_type]['H1']}")
            print(f"  Note: {hypotheses[nsi_type]['interpretation']}")
            print(f"  Complete cases: {len(complete_patients)} / {len(patients)} patients")
            
            if len(complete_patients) < 3:
                print("  → Insufficient data for ANOVA (n<3)")
                diagnosis_results['anova_results'][nsi_type] = {'status': 'insufficient_data'}
                continue
            
            # Descriptive statistics
            print("  Means ± SD:")
            print(f"    PWS: {pws.mean():.4f} ± {pws.std(ddof=1):.4f}")
            print(f"    SWS: {sws.mean():.4f} ± {sws.std(ddof=1):.4f}")
            print(f"    FWS: {fws.mean():.4f} ± {fws.std(ddof=1):.4f}")
            
            # Store summary statistics
            diagnosis_results['summary_stats'][nsi_type] = {
                'PWS': {'mean': pws.mean(), 'std': pws.std(ddof=1)},
                'SWS': {'mean': sws.mean(), 'std': sws.std(ddof=1)},
                'FWS': {'mean': fws.mean(), 'std': fws.std(ddof=1)},
                'n_complete': len(complete_patients)
            }
            
            # Friedman non-parametric omnibus test
            fr_stat, fr_p = friedmanchisquare(pws, sws, fws)
            
            # Parametric repeated-measures ANOVA
            df_long = pd.DataFrame({
                'patient_id': np.repeat(complete_patients, 3),
                'speed': ['PWS', 'SWS', 'FWS'] * len(complete_patients),
                'value': np.concatenate([pws, sws, fws])
            })
            
            try:
                aov = AnovaRM(df_long, depvar='value',
                            subject='patient_id', within=['speed']).fit()
                anova_success = True
                anova_f = aov.anova_table['F Value'][0]
                anova_p = aov.anova_table['Pr > F'][0]
            except Exception as e:
                print(f"  → RM-ANOVA failed: {str(e)}")
                anova_success = False
                anova_f = np.nan
                anova_p = np.nan
                aov = None
            
            # Compute ANOVA residuals and test normality
            if anova_success:
                grand_mean = df_long['value'].mean()
                subj_mean = df_long.groupby('patient_id')['value'].transform('mean')
                speed_mean = df_long.groupby('speed')['value'].transform('mean')
                df_long['resid'] = df_long['value'] - (subj_mean + speed_mean - grand_mean)
                
                # Test residual normality
                resid = df_long['resid'].values
                W, p_resid = shapiro(resid)
            else:
                W = np.nan
                p_resid = np.nan
            
            print("\n  Inferential Statistics:")
            print(f"    Friedman χ² = {fr_stat:.3f}, p = {fr_p:.4f}")
            if anova_success:
                print("    Repeated-Measures ANOVA:")
                print(aov.summary())
                print(f"    ANOVA residuals Shapiro–Wilk: W = {W:.3f}, p = {p_resid:.4f}")
            
            # Store ANOVA results
            diagnosis_results['anova_results'][nsi_type] = {
                'n_complete': len(complete_patients),
                'friedman_stat': fr_stat,
                'friedman_p': fr_p,
                'anova_success': anova_success,
                'anova_f': anova_f,
                'anova_p': anova_p,
                'residual_normality_p': p_resid,
                'anova_summary': aov.summary() if anova_success else None
            }
            
            # Interpretation
            if fr_p < 0.05:
                print("    → Friedman: SIGNIFICANT omnibus (reject H₀)")
            else:
                print("    → Friedman: NOT SIGNIFICANT omnibus (fail to reject H₀)")
                
            if anova_success and anova_p < 0.05:
                print("    → RM-ANOVA: SIGNIFICANT omnibus (reject H₀)")
            elif anova_success:
                print("    → RM-ANOVA: NOT SIGNIFICANT omnibus (fail to reject H₀)")
        
        # Post-hoc pairwise comparisons for this diagnosis
        print(f"\n{'='*80}")
        print(f"POST-HOC PAIRWISE COMPARISONS FOR {condition}")
        print(f"{'='*80}")
        print("Note: Using all available patients for each pairwise comparison")
        print("Bonferroni correction: α = 0.05/3 = 0.0167 for 3 comparisons")
        
        condition_patients = condition_data['patient_id'].unique()
        
        for nsi_type in ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']:
            print(f"\n{nsi_type.upper()} - Pairwise Comparisons:")
            
            # Perform all pairwise comparisons
            comparisons = [('PWS', 'SWS'), ('PWS', 'FWS'), ('SWS', 'FWS')]
            pairwise_comparisons = {}
            
            for speed1, speed2 in comparisons:
                # Find patients that have both speeds in this diagnosis
                patients_with_pair = []
                
                for patient in condition_patients:
                    patient_data = condition_data[condition_data['patient_id'] == patient]
                    speeds_present = patient_data['speed'].unique()
                    if speed1 in speeds_present and speed2 in speeds_present:
                        patients_with_pair.append(patient)
                
                # Get paired data
                vals1 = []
                vals2 = []
                
                for patient in patients_with_pair:
                    patient_data = condition_data[condition_data['patient_id'] == patient]
                    val1 = patient_data[patient_data['speed'] == speed1][nsi_type].values
                    val2 = patient_data[patient_data['speed'] == speed2][nsi_type].values
                    
                    if len(val1) > 0 and len(val2) > 0:
                        vals1.append(val1[0])
                        vals2.append(val2[0])
                
                vals1 = np.array(vals1)
                vals2 = np.array(vals2)
                
                if len(vals1) >= 3:  # Need at least 3 pairs for meaningful test
                    # Check normality and decide test
                    diff = vals1 - vals2
                    _, shapiro_p = shapiro(diff)
                    
                    if shapiro_p > 0.05:
                        # Use paired t-test
                        t_stat, p_value = ttest_rel(vals1, vals2)
                        test_used = "Paired t-test"
                        test_stat = t_stat
                    else:
                        # Use Wilcoxon signed-rank test
                        wilcoxon_stat, p_value = wilcoxon(vals1, vals2)
                        test_used = "Wilcoxon signed-rank"
                        test_stat = wilcoxon_stat
                    
                    # Effect size
                    cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                    
                    print(f"\n  {speed1} vs {speed2} (n={len(vals1)} pairs):")
                    print(f"    {speed1} Mean: {np.mean(vals1):.4f}, {speed2} Mean: {np.mean(vals2):.4f}")
                    print(f"    Mean difference: {np.mean(diff):.4f}")
                    print(f"    Test used: {test_used} (Shapiro-Wilk p={shapiro_p:.4f})")
                    print(f"    Test statistic: {test_stat:.4f}")
                    print(f"    p-value: {p_value:.4f}")
                    print(f"    Cohen's d: {cohen_d:.4f}")
                    
                    # Bonferroni corrected significance
                    bonferroni_significant = p_value < 0.0167
                    if bonferroni_significant:
                        print(f"    Result: SIGNIFICANT (Bonferroni corrected α=0.0167)")
                    else:
                        print(f"    Result: NOT SIGNIFICANT after Bonferroni correction")
                    
                    # Store pairwise results
                    pairwise_comparisons[f"{speed1}_vs_{speed2}"] = {
                        'n_pairs': len(vals1),
                        'mean_diff': np.mean(diff),
                        'test_statistic': test_stat,
                        'p_value': p_value,
                        'cohen_d': cohen_d,
                        'test_type': test_used,
                        'significant': bonferroni_significant,
                        'normality_p': shapiro_p
                    }
                else:
                    print(f"\n  {speed1} vs {speed2}: Insufficient paired data (n={len(vals1)})")
                    pairwise_comparisons[f"{speed1}_vs_{speed2}"] = {
                        'n_pairs': len(vals1),
                        'status': 'insufficient_data'
                    }
            
            diagnosis_results['pairwise_results'][nsi_type] = pairwise_comparisons
        
        all_diagnosis_results[condition] = diagnosis_results
    
    # Create visualization for each diagnosis
    create_diagnosis_visualization(results_nsi, conditions, cohort_name)
    
    # Print summary tables
    print_diagnosis_summary(all_diagnosis_results, conditions, cohort_name)
    
    return all_diagnosis_results


def create_speed_sensitivity_plots(results_nsi, cohort_name):
    """Create visualization plots for speed sensitivity analysis"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    nsi_types = ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']
    titles = ['Temporal NSI', 'Spatial NSI X (ML)', 'Spatial NSI Y (AP)', 'Spatial NSI Z (VT)']
    speeds = ['PWS', 'SWS', 'FWS']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # First row: Box plots
    for idx, (nsi_type, title) in enumerate(zip(nsi_types, titles)):
        ax = axes[0, idx]
        
        data_by_speed = []
        sample_sizes = []
        
        for speed in speeds:
            speed_data = results_nsi[results_nsi['speed'] == speed][nsi_type].dropna().values
            data_by_speed.append(speed_data)
            sample_sizes.append(len(speed_data))
        
        if all(len(d) > 0 for d in data_by_speed):
            bp = ax.boxplot(data_by_speed, labels=speeds, patch_artist=True, widths=0.6)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add individual points with jitter
            np.random.seed(42)
            for i, (data, color) in enumerate(zip(data_by_speed, colors)):
                jitter = np.random.normal(0, 0.04, size=len(data))
                ax.scatter(np.ones(len(data)) * (i+1) + jitter, data, 
                          alpha=0.5, s=40, color=color, edgecolors='black', linewidth=0.5)
            
            # Add sample sizes
            for i, (speed, n) in enumerate(zip(speeds, sample_sizes)):
                ax.text(i+1, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                        f'n={n}', ha='center', va='top', fontsize=9)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('NSI Value')
        ax.grid(axis='y', alpha=0.3)
    
    # Second row: Individual trajectory plots
    for idx, (nsi_type, title) in enumerate(zip(nsi_types, titles)):
        ax = axes[1, idx]
        
        patients = results_nsi['patient_id'].unique()
        
        for patient in patients:
            patient_data = results_nsi[results_nsi['patient_id'] == patient]
            
            x_vals = []
            y_vals = []
            
            for i, speed in enumerate(speeds):
                val = patient_data[patient_data['speed'] == speed][nsi_type].values
                if len(val) > 0:
                    x_vals.append(i)
                    y_vals.append(val[0])
            
            if len(x_vals) >= 2:
                ax.plot(x_vals, y_vals, 'o-', alpha=0.6, linewidth=1.5, markersize=6)
        
        # Add mean trajectory
        mean_values = []
        for i, speed in enumerate(speeds):
            speed_vals = results_nsi[results_nsi['speed'] == speed][nsi_type].dropna().values
            if len(speed_vals) > 0:
                mean_values.append(np.mean(speed_vals))
        
        if len(mean_values) == 3:
            ax.plot([0, 1, 2], mean_values, 'o-', color='black', linewidth=3, 
                    markersize=10, label='Mean', zorder=10)
        
        ax.set_xlim(-0.2, 2.2)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(speeds)
        ax.set_ylabel('NSI Value')
        ax.set_title(f'{title} - Individual Trajectories', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if idx == 0:
            ax.legend()
    
    plt.suptitle(f'Speed Sensitivity Analysis - {cohort_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_diagnosis_visualization(results_nsi, conditions, cohort_name):
    """Create visualization for each diagnosis"""
    
    for condition in conditions:
        condition_data = results_nsi[results_nsi['condition'] == condition]
        
        if len(condition_data) == 0:
            continue
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        nsi_types = ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']
        titles = ['Temporal NSI', 'Spatial NSI X (ML)', 'Spatial NSI Y (AP)', 'Spatial NSI Z (VT)']
        speeds = ['PWS', 'SWS', 'FWS']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # First row: Box plots with all available data
        for idx, (nsi_type, title) in enumerate(zip(nsi_types, titles)):
            ax = axes[0, idx]
            
            data_by_speed = []
            sample_sizes = []
            
            for speed in speeds:
                speed_data = condition_data[condition_data['speed'] == speed][nsi_type].dropna().values
                data_by_speed.append(speed_data)
                sample_sizes.append(len(speed_data))
            
            if all(len(d) > 0 for d in data_by_speed):
                bp = ax.boxplot(data_by_speed, labels=speeds, patch_artist=True, widths=0.6)
                
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add individual points
                np.random.seed(42)
                for i, (data, color) in enumerate(zip(data_by_speed, colors)):
                    jitter = np.random.normal(0, 0.04, size=len(data))
                    ax.scatter(np.ones(len(data)) * (i+1) + jitter, data, 
                              alpha=0.5, s=40, color=color, edgecolors='black', linewidth=0.5)
                
                # Add sample sizes
                for i, (speed, n) in enumerate(zip(speeds, sample_sizes)):
                    ax.text(i+1, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                            f'n={n}', ha='center', va='top', fontsize=9)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('NSI Value')
            ax.grid(axis='y', alpha=0.3)
        
        # Second row: Individual trajectory plots
        for idx, (nsi_type, title) in enumerate(zip(nsi_types, titles)):
            ax = axes[1, idx]
            
            condition_patients = condition_data['patient_id'].unique()
            
            for patient in condition_patients:
                patient_data = condition_data[condition_data['patient_id'] == patient]
                
                x_vals = []
                y_vals = []
                
                for i, speed in enumerate(speeds):
                    val = patient_data[patient_data['speed'] == speed][nsi_type].values
                    if len(val) > 0:
                        x_vals.append(i)
                        y_vals.append(val[0])
                
                if len(x_vals) >= 2:
                    ax.plot(x_vals, y_vals, 'o-', alpha=0.6, linewidth=1.5, markersize=6)
            
            # Add mean trajectory for complete cases
            mean_values = []
            for i, speed in enumerate(speeds):
                speed_vals = condition_data[condition_data['speed'] == speed][nsi_type].dropna().values
                if len(speed_vals) > 0:
                    mean_values.append(np.mean(speed_vals))
            
            if len(mean_values) == 3:
                ax.plot([0, 1, 2], mean_values, 'o-', color='black', linewidth=3, 
                        markersize=10, label='Mean', zorder=10)
            
            ax.set_xlim(-0.2, 2.2)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(speeds)
            ax.set_ylabel('NSI Value')
            ax.set_title(f'{title} - Individual Trajectories', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            if idx == 0:
                ax.legend()
        
        plt.suptitle(f'NSI Analysis for {condition}: Walking Speed Comparisons', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def print_speed_sensitivity_summary(anova_results, pairwise_results, cohort_name):
    """Print comprehensive summary of speed sensitivity analysis"""
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SUMMARY - {cohort_name.upper()}")
    print(f"{'='*80}")
    
    # ANOVA Summary
    print("\nREPEATED MEASURES ANOVA SUMMARY:")
    print("-" * 50)
    print(f"{'NSI Type':<20} {'n':<5} {'Friedman p':<12} {'RM-ANOVA p':<12} {'Significant'}")
    print("-" * 50)
    
    for nsi_type, results in anova_results.items():
        if results['status'] == 'complete':
            friedman_sig = "*" if results['friedman_p'] < 0.05 else "ns"
            anova_sig = "*" if results['anova_success'] and results['anova_p'] < 0.05 else "ns"
            
            print(f"{nsi_type:<20} {results['n_complete']:<5} "
                  f"{results['friedman_p']:<12.4f} "
                  f"{results['anova_p']:<12.4f} {friedman_sig}/{anova_sig}")
        else:
            print(f"{nsi_type:<20} {'N/A':<5} {'N/A':<12} {'N/A':<12} {'N/A'}")
    
    # Pairwise Summary
    print(f"\nPAIRWISE COMPARISONS SUMMARY (Bonferroni corrected α=0.0167):")
    print("-" * 80)
    print(f"{'NSI Type':<20} {'Comparison':<15} {'n':<5} {'Test':<20} {'p-value':<10} {'Effect':<10} {'Sig'}")
    print("-" * 80)
    
    for nsi_type in ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']:
        if nsi_type in pairwise_results:
            for comp_name, results in pairwise_results[nsi_type].items():
                if 'status' not in results:  # Valid comparison
                    sig_marker = "*" if results['bonferroni_significant'] else "ns"
                    effect_size = f"d={results['cohen_d']:.3f}"
                    print(f"{nsi_type:<20} {comp_name.replace('_vs_', ' vs '):<15} {results['n_pairs']:<5} "
                          f"{results['test_type']:<20} {results['p_value']:<10.4f} {effect_size:<10} {sig_marker}")
                else:
                    print(f"{nsi_type:<20} {comp_name.replace('_vs_', ' vs '):<15} {results['n_pairs']:<5} "
                          f"{'Insufficient data':<20} {'N/A':<10} {'N/A':<10} {'N/A'}")


def print_diagnosis_summary(all_diagnosis_results, conditions, cohort_name):
    """Print summary tables for diagnosis-based analysis"""
    
    # Summary table across all diagnoses
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE: COMPLETE CASES BY DIAGNOSIS - {cohort_name.upper()}")
    print(f"{'='*80}")
    
    summary_data = []
    for condition in conditions:
        if condition in all_diagnosis_results:
            total_patients = 0
            complete_patients = 0
            
            # Count from first available NSI metric
            for nsi_type in ['temporal_nsi', 'spatial_nsi_x', 'spatial_nsi_y', 'spatial_nsi_z']:
                if nsi_type in all_diagnosis_results[condition]['summary_stats']:
                    complete_patients = all_diagnosis_results[condition]['summary_stats'][nsi_type]['n_complete']
                    break
            
            # Calculate total patients from original data
            condition_data = [d for d in [all_diagnosis_results[condition]] if d]
            if condition_data:
                # This is approximate - in practice you'd pass the original data
                total_patients = complete_patients  # Simplified for this example
            
            completion_rate = f"{(complete_patients/total_patients*100):.1f}%" if total_patients > 0 else "N/A"
            
            summary_data.append({
                'Diagnosis': condition,
                'Complete_Cases': complete_patients,
                'Completion_Rate': completion_rate
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Significant findings summary
    print(f"\n{'='*80}")
    print("SUMMARY OF SIGNIFICANT FINDINGS BY DIAGNOSIS")
    print(f"{'='*80}")
    
    for condition in conditions:
        if condition in all_diagnosis_results:
            print(f"\n{condition.upper()}:")
            
            # Check ANOVA results
            significant_anovas = []
            for nsi_type, anova_result in all_diagnosis_results[condition]['anova_results'].items():
                if 'friedman_p' in anova_result and anova_result['friedman_p'] < 0.05:
                    significant_anovas.append(f"{nsi_type} (Friedman p={anova_result['friedman_p']:.4f})")
                elif 'anova_p' in anova_result and anova_result['anova_success'] and anova_result['anova_p'] < 0.05:
                    significant_anovas.append(f"{nsi_type} (RM-ANOVA p={anova_result['anova_p']:.4f})")
            
            if significant_anovas:
                print(f"  Significant omnibus tests: {', '.join(significant_anovas)}")
            else:
                print("  No significant omnibus tests")
            
            # Check pairwise results
            significant_pairwise = []
            for nsi_type, pairwise_dict in all_diagnosis_results[condition]['pairwise_results'].items():
                for comp_name, comp_result in pairwise_dict.items():
                    if 'significant' in comp_result and comp_result['significant']:
                        significant_pairwise.append(f"{nsi_type}:{comp_name.replace('_vs_', ' vs ')}")
            
            if significant_pairwise:
                print(f"  Significant pairwise comparisons: {', '.join(significant_pairwise)}")
            else:
                print("  No significant pairwise comparisons")
    
    

def run_patient_speed_analysis(data: pd.DataFrame, cohort_name: str = "Neurological Patients"):
    """
    Run complete speed sensitivity analysis for patients
    """
    print("Running enhanced speed sensitivity analysis with ANOVA...")
    results = analyze_by_patient(data, cohort_name)
    return results


def run_diagnosis_analysis(data: pd.DataFrame, cohort_name: str = "Neurological Patients"):
    """
    Run complete diagnosis-based analysis
    """
    print("Running diagnosis-based NSI analysis...")
    results = analyze_by_diagnosis(data, cohort_name)
    return results


def rain_plot_all_axes(df):
    """
    Creates a single rain plot showing Floquet multipliers for all axes,
    split by Before vs After conditions.
    """
    
    axes = [ax for ax in sorted(df['axis'].unique())]
    # Define conditions and colors
    conditions = sorted(df['condition'].unique())
    colors = {'before': '#3734db', 'after': '#3cb7e7'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.6
    offsets = {'before': -0.15, 'after': +0.15}
    
    # Plot for each axis
    col = 'lyapunov_exponent'  
    
    for i, axis in enumerate(axes):
        df_axis = df[df['axis'] == axis]
        
        for cond in conditions:
            vals = df_axis[df_axis['condition'] == cond][col].dropna().values
            if len(vals) == 0:
                continue
            
            pos = i + offsets[cond.lower()]
            
            # Half-violin plot
            parts = ax.violinplot([vals], positions=[pos], widths=width,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                verts = pc.get_paths()[0].vertices
                mx = verts[:, 0].mean()
                if cond.lower() == 'before':
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, mx)
                else:
                    verts[:, 0] = np.clip(verts[:, 0], mx, np.inf)
                pc.set_facecolor(colors[cond.lower()])
                pc.set_alpha(0.4)
            
            # Boxplot
            bp = ax.boxplot([vals], positions=[pos], widths=width*0.3, patch_artist=True,
                            showfliers=False)
            bp['boxes'][0].set_facecolor('white')
            bp['boxes'][0].set_edgecolor('black')
            
            # Rain points (jittered scatter)
            xpts = np.random.normal(pos, 0.02, size=len(vals))
            ax.scatter(xpts, vals, color=colors[cond.lower()], edgecolor='k',
                       alpha=0.7, s=30)
            
            # Mean line
            ax.hlines(vals.mean(), pos-width*0.15, pos+width*0.15,
                      colors='k', linestyles='--', linewidth=2)
    
    # Set labels and title
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([f'Axis {axis.upper()}' for axis in axes])
    ax.set_ylabel('Floquet Multiplier Value')
    ax.set_title('Floquet Multiplier Across All Axes (Before vs After)')
    
    # Create legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[c.lower()], edgecolor='k', label=c.title())
              for c in conditions]
    legend.append(plt.Line2D([0], [0], color='k', linestyle='--', label='Mean'))
    ax.legend(handles=legend, loc='upper right')
    
    plt.tight_layout()
    plt.show()