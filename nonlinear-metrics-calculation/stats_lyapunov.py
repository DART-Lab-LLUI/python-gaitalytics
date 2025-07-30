# Author:      Natasha Kovacheva <natasha.kovacheva@stud.hslu.ch>
# Created:     SS 2025
# Description: This script contains a miscellaneous of functions used for 
#              the lyapunov assessment.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, ttest_rel, wilcoxon
import warnings
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

def analyze_normality_with_plots(df, subject_col='subject', 
                                        condition_col='condition', 
                                        value_col='lyapunov_exponent',
                                        axis_col='axis',
                                        sheet_name=0,
                                        cohort_name='Cohort'):
    """
    Comprehensive normality testing for Lyapunov exponent paired differences with visualization.
    """
    
    print(f"=== {cohort_name.upper()} LYAPUNOV NORMALITY ANALYSIS ===")
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Subjects: {df[subject_col].nunique()}")
    print("="*70)
    
    # Get unique axes
    axes = sorted(df[axis_col].unique()) if axis_col in df.columns else ['all']
    
    # Create figure for Q-Q plots and histograms
    n_axes = len(axes)
    fig, axes_plot = plt.subplots(2, n_axes, figsize=(4*n_axes, 8))
    if n_axes == 1:
        axes_plot = axes_plot.reshape(2, 1)
    
    fig.suptitle(f'Normality Assessment for Before/After Differences - {cohort_name} Lyapunov Exponents', 
                 fontsize=16, fontweight='bold')
    
    print("SHAPIRO-WILK TEST RESULTS (α = 0.05)")
    print("If p > 0.05: Use paired t-test (normal)")
    print("If p < 0.05: Use Wilcoxon signed-rank test (non-normal)")
    print("-"*70)
    
    recommendations = {}
    analysis_results = {}
    
    for idx, axis in enumerate(axes):
        # Get paired data for this axis
        if axis_col in df.columns and axis != 'all':
            axis_data = df[df[axis_col] == axis]
            axis_label = f"{axis.upper()}"
        else:
            axis_data = df
            axis_label = "All Axes"
        
        # Separate before and after data (case-insensitive)
        before_df = axis_data[axis_data[condition_col].str.lower() == 'before']
        after_df = axis_data[axis_data[condition_col].str.lower() == 'after']
        
        # Create proper pairing by subject
        if axis_col in df.columns and axis != 'all':
            before_pivot = before_df.set_index(subject_col)[value_col]
            after_pivot = after_df.set_index(subject_col)[value_col]
        else:
            before_pivot = before_df.set_index([subject_col, axis_col])[value_col]
            after_pivot = after_df.set_index([subject_col, axis_col])[value_col]
        
        # Get common indices (paired data)
        common_indices = before_pivot.index.intersection(after_pivot.index)
        
        # Extract paired values
        before_values = before_pivot.loc[common_indices].values
        after_values = after_pivot.loc[common_indices].values
        differences = before_values - after_values
        n_pairs = len(differences)
        
        # Check if we have paired data
        if n_pairs == 0:
            print(f"\n{axis_label}: NO PAIRED DATA FOUND")
            print("  Skipping this axis - no matching subjects between conditions")
            continue
        
        # Shapiro-Wilk test for normality of differences
        statistic, p_value = shapiro(differences)
        
        # Determine test recommendation
        if p_value > 0.05:
            test_recommendation = "Paired t-test"
            normality_status = "Normal"
        else:
            test_recommendation = "Wilcoxon signed-rank"
            normality_status = "Non-normal"
        
        recommendations[axis_label] = test_recommendation
        
        # Store analysis results
        analysis_results[axis_label] = {
            'n_pairs': n_pairs,
            'before_values': before_values,
            'after_values': after_values,
            'differences': differences,
            'shapiro_stat': statistic,
            'shapiro_p': p_value,
            'test_recommendation': test_recommendation,
            'normality_status': normality_status
        }
        
        # Print detailed results
        print(f"\n{axis_label}:")
        print(f"  Sample size: n = {n_pairs} pairs")
        print(f"  Shapiro-Wilk statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Normality assessment: {normality_status}")
        print(f"  Recommended test: {test_recommendation}")
        print(f"  Mean before: {np.mean(before_values):.4f} ± {np.std(before_values, ddof=1):.4f}")
        print(f"  Mean after: {np.mean(after_values):.4f} ± {np.std(after_values, ddof=1):.4f}")
        print(f"  Mean difference: {np.mean(differences):.4f} ± {np.std(differences, ddof=1):.4f}")
        print(f"  Median difference: {np.median(differences):.4f}")
        
        # Q-Q plot for visual normality check
        ax_qq = axes_plot[0, idx]
        stats.probplot(differences, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'{axis_label}\nQ-Q Plot', fontsize=10, fontweight='bold')
        ax_qq.grid(True, alpha=0.3)
        
        # Add p-value annotation to Q-Q plot
        ax_qq.text(0.05, 0.95, f'SW p={p_value:.3f}', 
                   transform=ax_qq.transAxes, 
                   bbox=dict(boxstyle='round', 
                           facecolor='yellow' if p_value < 0.05 else 'lightgreen', 
                           alpha=0.7),
                   verticalalignment='top', fontsize=9)
        
        # Histogram with normal overlay
        ax_hist = axes_plot[1, idx]
        
        if len(differences) > 0:
            # Adaptive bin size
            n_bins = min(15, max(5, n_pairs // 3))
            ax_hist.hist(differences, bins=n_bins, density=True, alpha=0.7, 
                        color='skyblue', edgecolor='black')
            
            # Overlay theoretical normal distribution
            mu, sigma = np.mean(differences), np.std(differences, ddof=1)
            if sigma > 0:
                x = np.linspace(differences.min() - sigma, differences.max() + sigma, 100)
                ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
            
            # Add vertical line at zero (no difference)
            ax_hist.axvline(0, color='green', linestyle='--', alpha=0.7, label='No change')
        else:
            ax_hist.text(0.5, 0.5, 'No paired data', 
                        transform=ax_hist.transAxes, 
                        ha='center', va='center', fontsize=12)
        
        ax_hist.set_xlabel('Difference (Before - After)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'{axis_label}\nHistogram of Differences', fontsize=10, fontweight='bold')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Additional normality tests for robustness
        if len(differences) >= 3:
            # Anderson-Darling test
            ad_result = anderson(differences, dist='norm')
            ad_normal = ad_result.statistic <= ad_result.critical_values[2]  # 5% level
            
            # D'Agostino and Pearson's test (requires n>=8)
            if len(differences) >= 8:
                k2_stat, k2_p = normaltest(differences)
                k2_normal = k2_p > 0.05
            else:
                k2_stat, k2_p = np.nan, np.nan
                k2_normal = None
            
            print(f"  Additional tests:")
            print(f"    Anderson-Darling: {ad_result.statistic:.4f} ({'Normal' if ad_normal else 'Non-normal'})")
            if not np.isnan(k2_p):
                print(f"    D'Agostino-Pearson: p={k2_p:.4f} ({'Normal' if k2_normal else 'Non-normal'})")
        
        # Sensitivity analyses
        print(f"  Sensitivity analyses:")
        _perform_sensitivity_checks(before_values, after_values, f"{cohort_name} {axis_label}")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary tables
    print("\n" + "="*70)
    print("SUMMARY OF NORMALITY TEST RECOMMENDATIONS")
    print("="*70)
    print(f"{'Axis':<20} {'Normality':<15} {'Recommended Test':<25}")
    print("-"*70)
    for axis_label, test in recommendations.items():
        normality = "Normal" if test == "Paired t-test" else "Non-normal"
        print(f"{axis_label:<20} {normality:<15} {test:<25}")
    
    print("\n" + "="*70)
    print("DECISION GUIDE")
    print("="*70)
    print("Based on Shapiro-Wilk test results:")
    print("\nRecommended statistical tests for each axis:")
    for axis_label, test in recommendations.items():
        print(f"- {axis_label}: {test}")
    
    print("\nNote: With small sample sizes (n<30), normality tests may lack power.")
    print("Visual inspection (Q-Q plots) is equally important for decision-making.")
    
    return recommendations, analysis_results


def statistical_tests(df, subject_col='subject', 
                                     condition_col='condition', 
                                     value_col='lyapunov_exponent',
                                     axis_col='axis',
                                     sheet_name=0,
                                     cohort_name='Cohort',
                                     recommendations=None):
    """
    Perform statistical tests based on normality results and provides comprehensive analysis.
    """
    
    # Read data
    
    axes = sorted(df[axis_col].unique()) if axis_col in df.columns else ['all']
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL TEST RESULTS - {cohort_name.upper()}")
    print(f"{'='*70}")
    
    results_summary = []
    
    for axis in axes:
        # Filter data for this axis
        if axis_col in df.columns and axis != 'all':
            axis_data = df[df[axis_col] == axis]
            axis_label = f"{axis.upper()}"
        else:
            axis_data = df
            axis_label = "All Axes"
        
        # Get paired data (case-insensitive condition matching)
        before_df = axis_data[axis_data[condition_col].str.lower() == 'before']
        after_df = axis_data[axis_data[condition_col].str.lower() == 'after']
        
        # Create proper pairing
        if axis_col in df.columns and axis != 'all':
            before_pivot = before_df.set_index(subject_col)[value_col]
            after_pivot = after_df.set_index(subject_col)[value_col]
        else:
            before_pivot = before_df.set_index([subject_col, axis_col])[value_col]
            after_pivot = after_df.set_index([subject_col, axis_col])[value_col]
        
        common_indices = before_pivot.index.intersection(after_pivot.index)
        
        # Skip if no paired data
        if len(common_indices) == 0:
            print(f"\n{axis_label}: NO PAIRED DATA - SKIPPING")
            continue
        
        before_values = before_pivot.loc[common_indices].values
        after_values = after_pivot.loc[common_indices].values
        differences = before_values - after_values
        
        print(f"\n{axis_label} (n={len(common_indices)} pairs):")
        print(f"  Before: Mean = {np.mean(before_values):.4f}, SD = {np.std(before_values, ddof=1):.4f}")
        print(f"  After:  Mean = {np.mean(after_values):.4f}, SD = {np.std(after_values, ddof=1):.4f}")
        print(f"  Mean difference (Before - After): {np.mean(differences):.4f}")
        print(f"  SD of differences: {np.std(differences, ddof=1):.4f}")
        
        # Determine which test to use
        if recommendations and axis_label in recommendations:
            test_to_use = recommendations[axis_label]
        else:
            # Default decision based on sample size and quick normality check
            if len(differences) >= 3:
                _, p_shapiro = shapiro(differences)
                test_to_use = "Paired t-test" if p_shapiro > 0.05 else "Wilcoxon signed-rank"
            else:
                test_to_use = "Wilcoxon signed-rank"  # Conservative for small samples
        
        # Perform the appropriate statistical test
        if test_to_use == "Paired t-test":
            t_stat, p_value = ttest_rel(before_values, after_values)
            print(f"  Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")
            test_used = "Paired t-test"
            test_statistic = t_stat
        else:
            try:
                w_stat, p_value = wilcoxon(before_values, after_values)
                print(f"  Wilcoxon signed-rank test: W = {w_stat:.4f}, p = {p_value:.4f}")
                test_used = "Wilcoxon"
                test_statistic = w_stat
            except ValueError as e:
                print(f"  Wilcoxon test failed: {e}")
                # Fall back to t-test if Wilcoxon fails
                t_stat, p_value = ttest_rel(before_values, after_values)
                print(f"  Fallback paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")
                test_used = "Paired t-test (fallback)"
                test_statistic = t_stat
        
        # Calculate effect size (Cohen's d)
        if np.std(differences, ddof=1) > 0:
            cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        else:
            cohen_d = 0
        
        print(f"  Effect size (Cohen's d): {cohen_d:.4f}")
        
        # Interpret effect size
        if abs(cohen_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohen_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohen_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"  Effect size interpretation: {effect_interpretation}")
        
        # Statistical significance interpretation
        alpha = 0.05
        if p_value < alpha:
            significance = "Statistically significant"
            print(f"  Result: {significance} difference (p < {alpha})")
            if np.mean(differences) > 0:
                direction = "Lyapunov exponent decreased (improved stability)"
            else:
                direction = "Lyapunov exponent increased (reduced stability)"
            print(f"  Clinical interpretation: {direction}")
        else:
            significance = "Not statistically significant"
            print(f"  Result: {significance} difference (p ≥ {alpha})")
            print(f"  Clinical interpretation: No significant change in stability")
        
        # Confidence interval for the mean difference
        if len(differences) >= 3:
            se_diff = stats.sem(differences)
            ci_lower, ci_upper = stats.t.interval(0.95, len(differences)-1, 
                                                 loc=np.mean(differences), 
                                                 scale=se_diff)
            print(f"  95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Add to results summary
        results_summary.append({
            'Axis': axis_label,
            'n_pairs': len(common_indices),
            'Before_Mean': np.mean(before_values),
            'After_Mean': np.mean(after_values),
            'Mean_Difference': np.mean(differences),
            'Test_Used': test_used,
            'Test_Statistic': test_statistic,
            'p_value': p_value,
            'Cohen_d': cohen_d,
            'Effect_Size': effect_interpretation,
            'Significant': 'Yes' if p_value < alpha else 'No',
            'Clinical_Interpretation': direction if p_value < alpha else "No significant change"
        })
        
        # Perform sensitivity analyses
        print(f"  Sensitivity analyses:")
        _perform_sensitivity_checks(before_values, after_values, f"{cohort_name} {axis_label}")
    
    # Create and display summary table
    summary_df = pd.DataFrame(results_summary)
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE RESULTS SUMMARY - {cohort_name.upper()}")
    print(f"{'='*70}")
    
    if not summary_df.empty:
        # Display key results in a formatted table
        print(f"{'Axis':<8} {'n':<4} {'Test':<15} {'p-value':<9} {'Cohen_d':<8} {'Significant':<11}")
        print("-" * 70)
        for _, row in summary_df.iterrows():
            print(f"{row['Axis']:<8} {row['n_pairs']:<4} {row['Test_Used']:<15} "
                  f"{row['p_value']:<9.4f} {row['Cohen_d']:<8.3f} {row['Significant']:<11}")
        
        # Summary of significant results
        significant_axes = summary_df[summary_df['Significant'] == 'Yes']['Axis'].tolist()
        if significant_axes:
            print(f"\nSignificant changes found in: {', '.join(significant_axes)}")
        else:
            print(f"\nNo statistically significant changes detected in any axis")
        
        # Effect size summary
        large_effects = summary_df[abs(summary_df['Cohen_d']) >= 0.8]['Axis'].tolist()
        medium_effects = summary_df[(abs(summary_df['Cohen_d']) >= 0.5) & 
                                   (abs(summary_df['Cohen_d']) < 0.8)]['Axis'].tolist()
        if large_effects:
            print(f"Large effect sizes (|d| ≥ 0.8): {', '.join(large_effects)}")
        if medium_effects:
            print(f"Medium effect sizes (0.5 ≤ |d| < 0.8): {', '.join(medium_effects)}")
    else:
        print("No valid paired data found for analysis")
    
    return summary_df


def _perform_sensitivity_checks(before_vals, after_vals, label=""):
    """
    Internal function to perform sensitivity analyses:
    1) Bootstrap 95% CI on the mean difference
    2) Leave-one-out t-test & Wilcoxon signed-rank p-value ranges
    """
    diffs = before_vals - after_vals
    n = len(diffs)
    
    if n < 2:
        print(f"   [{label}] Not enough data for sensitivity analysis.")
        return
    
    # Bootstrap 95% confidence interval for mean difference
    np.random.seed(42)  # For reproducibility
    n_bootstrap = 10000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(diffs, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    ci_lo, ci_hi = np.percentile(bootstrap_means, [2.5, 97.5])
    print(f"   [{label}] Bootstrap 95% CI on mean difference: [{ci_lo:.4f}, {ci_hi:.4f}]")
    
    # Leave-one-out sensitivity analysis
    if n >= 3:
        p_t_values = []
        p_w_values = []
        
        for i in range(n):
            loo_before = np.delete(before_vals, i)
            loo_after = np.delete(after_vals, i)
            
            if len(loo_before) > 1:
                try:
                    # Leave-one-out t-test
                    _, p_t = ttest_rel(loo_before, loo_after)
                    p_t_values.append(p_t)
                    
                    # Leave-one-out Wilcoxon test
                    _, p_w = wilcoxon(loo_before, loo_after)
                    p_w_values.append(p_w)
                except:
                    continue
        
        if p_t_values and p_w_values:
            print(f"   [{label}] LOO t-test p-value range: [{np.min(p_t_values):.4f}, {np.max(p_t_values):.4f}]")
            print(f"   [{label}] LOO Wilcoxon p-value range: [{np.min(p_w_values):.4f}, {np.max(p_w_values):.4f}]")
            
            # Check stability of significance
            significant_t = sum(1 for p in p_t_values if p < 0.05)
            significant_w = sum(1 for p in p_w_values if p < 0.05)
            print(f"   [{label}] LOO significance stability: t-test {significant_t}/{len(p_t_values)}, "
                  f"Wilcoxon {significant_w}/{len(p_w_values)}")




def analyze_single_cohort_by_axis(df, cohort_name='Young', sheet_name=0):
    """
    Analyze how lag, embedding dimension, and average stride duration differ
    between conditions and axes for a single cohort.
    """
   
    fig = plt.figure(figsize=(24, 20))
    
    # Define params
    params = ['lag', 'embedding_dimension', 'average stride duration', 'lyapunov_exponent']
    param_labels = ['Lag', 'Embedding Dimension', 'Avg Stride Duration (s)', 'Lyapunov Exponent']
    
    # Color scheme
    condition_colors = {'Before': '#3498db', 'After': '#da1515'}
    axis_colors = {'x': '#e74c3c', 'z': '#2ecc71'}  # Red for X, Green for Z
    
    axes_list = sorted(df['axis'].unique())
    
    # Box plots comparing parameters across conditions and axes
    plot_idx = 1
    for param, label in zip(params, param_labels):
        ax = plt.subplot(5, 4, plot_idx)
        plot_idx += 1
        
        # Prepare data for plotting
        plot_data = []
        plot_labels = []
        plot_colors = []
        positions = []
        pos = 0
        
        for axis in axes_list:
            for condition in ['Before', 'After']:
                data = df[(df['axis'] == axis) & (df['condition'] == condition)][param].dropna()
                
                if len(data) > 0:
                    plot_data.append(data.values)
                    plot_labels.append(f'{axis}\n{condition}')
                    # Use a blend of axis and condition colors
                    plot_colors.append(axis_colors[axis])
                    positions.append(pos)
                pos += 1
            pos += 0.5  # Space between axes
        
        # Create box plot
        bp = ax.boxplot(plot_data, positions=positions, widths=0.6, 
                       patch_artist=True, showfliers=True)
        
        # Color the boxes
        for i, (patch, color) in enumerate(zip(bp['boxes'], plot_colors)):
            patch.set_facecolor(color)
            
            if i % 2 == 1:
                patch.set_alpha(0.9)
            else:  
                patch.set_alpha(0.5)
        
        # Add mean values as text
        for i, (data, pos) in enumerate(zip(plot_data, positions)):
            mean_val = np.mean(data)
            ax.text(pos, ax.get_ylim()[1] * 0.95, f'{mean_val:.2f}', 
                   ha='center', va='top', fontsize=8, fontweight='bold')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.set_title(f'{label} by Axis and Condition')
        ax.grid(axis='y', alpha=0.3)
    
    # Correlation between parameters by axis
    plot_idx = 5
    for param in ['lag', 'embedding_dimension']:
        ax = plt.subplot(5, 4, plot_idx)
        plot_idx += 1
        
        for axis in axes_list:
            for condition in ['Before', 'After']:
                data = df[(df['axis'] == axis) & (df['condition'] == condition)]
                
                if len(data) > 0:
                    x = data['average stride duration'].values
                    y = data[param].values
                    
                    # Remove NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[mask], y[mask]
                    
                    if len(x) > 2:
                        # Scatter plot
                        marker = 'o' if condition == 'Before' else '^'
                        ax.scatter(x, y, label=f'{axis}-{condition}', 
                                 color=axis_colors[axis], alpha=0.6 if condition == 'Before' else 0.9,
                                 s=50, marker=marker)
        
        ax.set_xlabel('Average Stride Duration (s)')
        ax.set_ylabel(param.replace('_', ' ').title())
        ax.set_title(f'{param.replace("_", " ").title()} vs Stride Duration by Axis')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Parameter changes (Before vs After) by axis
    plot_idx = 9
    for param, label in zip(params, param_labels):
        ax = plt.subplot(5, 4, plot_idx)
        plot_idx += 1
        
        changes_by_axis = {axis: [] for axis in axes_list}
        
        # Calculate changes for each subject and axis
        subjects = df['subject'].unique()
        
        for subject in subjects:
            for axis in axes_list:
                before = df[(df['condition'] == 'Before') & 
                           (df['subject'] == subject) & 
                           (df['axis'] == axis)][param].values
                after = df[(df['condition'] == 'After') & 
                          (df['subject'] == subject) & 
                          (df['axis'] == axis)][param].values
                
                if len(before) > 0 and len(after) > 0:
                    change = after[0] - before[0]
                    changes_by_axis[axis].append(change)
        
        # Plot violin plots for each axis
        positions = []
        for i, axis in enumerate(axes_list):
            if len(changes_by_axis[axis]) > 0:
                parts = ax.violinplot([changes_by_axis[axis]], positions=[i], 
                                     widths=0.7, showmeans=True)
                
                # Color the violin
                for pc in parts['bodies']:
                    pc.set_facecolor(axis_colors[axis])
                    pc.set_alpha(0.3)
                
                # Add individual points
                y_jitter = np.random.normal(0, 0.02, size=len(changes_by_axis[axis]))
                ax.scatter(np.full_like(changes_by_axis[axis], i) + y_jitter, 
                          changes_by_axis[axis], color=axis_colors[axis], alpha=0.6, s=40)
                
                # Add mean change
                mean_change = np.mean(changes_by_axis[axis])
                ax.text(i, ax.get_ylim()[1] * 0.95, f'μ={mean_change:.3f}', 
                       ha='center', va='top', fontsize=9, fontweight='bold')
                
                positions.append(i)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'Axis {axis}' for axis in axes_list])
        ax.set_ylabel(f'Change in {label}')
        ax.set_title(f'{label} Change (After - Before) by Axis')
        ax.grid(axis='y', alpha=0.3)
    
    
    ax = plt.subplot(5, 4, 13)
    
    lyap_data = []
    labels = []
    colors = []
    positions = []
    pos = 0
    
    for axis in axes_list:
        for condition in ['Before', 'After']:
            data = df[(df['axis'] == axis) & (df['condition'] == condition)]['lyapunov_exponent'].dropna()
            if len(data) > 0:
                lyap_data.append(data.values)
                labels.append(f'{axis}-{condition}')
                colors.append(axis_colors[axis])
                positions.append(pos)
            pos += 1
        pos += 0.5
    
    bp = ax.boxplot(lyap_data, positions=positions, widths=0.6, patch_artist=True)
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.5 if i % 2 == 0 else 0.9)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Lyapunov Exponent')
    ax.set_title('Lyapunov Exponent by Axis and Condition')
    ax.grid(axis='y', alpha=0.3)
    
    # Heatmap showing all parameters by axis and condition
    ax_heatmap = plt.subplot(5, 4, (14, 16))
    
    # Create pivot table for heatmap
    pivot_data = []
    row_labels = []
    
    for axis in axes_list:
        for condition in ['Before', 'After']:
            row_data = []
            row_labels.append(f'{axis}-{condition}')
            for param in params:
                val = df[(df['axis'] == axis) & (df['condition'] == condition)][param].mean()
                row_data.append(val)
            pivot_data.append(row_data)
    
    # Create heatmap
    heatmap_df = pd.DataFrame(pivot_data, columns=param_labels, index=row_labels)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='coolwarm', 
               center=heatmap_df.mean().mean(), ax=ax_heatmap,
               cbar_kws={'label': 'Parameter Value'})
    ax_heatmap.set_title('Average Parameter Values by Axis and Condition')
   
    # Print detailed statistics by axis
    print("\n" + "="*60)
    print("DETAILED PARAMETER STATISTICS BY AXIS")
    print("="*60)
    
    for param, label in zip(params, param_labels):
        print(f"\n{label.upper()}:")
        
        for axis in axes_list:
            print(f"\n  Axis {axis}:")
            for condition in ['Before', 'After']:
                data = df[(df['axis'] == axis) & (df['condition'] == condition)][param].dropna()
                
                if len(data) > 0:
                    print(f"    {condition}:")
                    print(f"      Mean: {data.mean():.4f}")
                    print(f"      Median: {data.median():.4f}")
                    print(f"      Std: {data.std():.4f}")
                    print(f"      Range: [{data.min():.4f}, {data.max():.4f}]")
            
            # Statistical comparison between conditions for this axis
            before_data = df[(df['axis'] == axis) & (df['condition'] == 'Before')][param].dropna()
            after_data = df[(df['axis'] == axis) & (df['condition'] == 'After')][param].dropna()
            
            if len(before_data) > 0 and len(after_data) > 0:
                t_stat, p_val = stats.ttest_rel(before_data, after_data)
                print(f"\n    Paired t-test (Before vs After): t = {t_stat:.4f}, p = {p_val:.4f}")
    
    # Create axis-specific correlation plots
    fig_corr, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig_corr.suptitle(f'{cohort_name} - Correlations with Lyapunov Exponent by Axis', fontsize=16)
    
    plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, x_param in enumerate(['lag', 'embedding_dimension']):
        for axis_idx, axis in enumerate(axes_list):
            ax = axes[plot_positions[idx*2 + axis_idx][0], plot_positions[idx*2 + axis_idx][1]]
            
            for condition in ['Before', 'After']:
                sub = df[(df['axis'] == axis) & (df['condition'] == condition)]
                x = sub[x_param].values
                y = sub['lyapunov_exponent'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
                
                if len(x) >= 3:
                    # Scatter
                    marker = 'o' if condition == 'Before' else '^'
                    ax.scatter(x, y, label=condition, color=axis_colors[axis], 
                             alpha=0.6 if condition == 'Before' else 0.9, 
                             marker=marker, s=60)
                    
                    # Fit & line
                    # m, b = np.polyfit(x, y, 1)
                    # line_x = np.linspace(x.min(), x.max(), 100)
                    # ax.plot(line_x, m*line_x + b, color=axis_colors[axis], 
                    #        linestyle='--' if condition == 'Before' else '-', 
                    #        alpha=0.8)
                    # Prepare the design matrix (add intercept)
                    X = sm.add_constant(x)      # shape (n,2): [1, x]

                    # Fit OLS
                    model = sm.OLS(y, X).fit()

                    # Print the summary
                    print(model.summary())

                    # Plot the fitted line
                    line_x = np.linspace(x.min(), x.max(), 100)
                    line_X = sm.add_constant(line_x)
                    line_y = model.predict(line_X)
                    ax.plot(line_x, line_y, color=axis_colors[axis], linestyle='--' if condition=='Before' else '-', alpha=0.8)
                                        
                    # Pearson correlation
                    r, p = stats.pearsonr(x, y)
                    ax.text(0.05, 0.95 - 0.1 * ['Before','After'].index(condition),
                            f'{condition}: r={r:.2f}, p={p:.3f}',
                            transform=ax.transAxes, fontsize=9)
            
            ax.set_xlabel(x_param.replace('_', ' ').title())
            ax.set_ylabel('Lyapunov Exponent')
            ax.set_title(f'Axis {axis.upper()} - {x_param.replace("_", " ").title()} vs Lyapunov')
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_axis_comparison_dashboard(df, cohort_name='Young', sheet_name=0):
    """
    Create a dashboard comparing axes for key parameters.
    """
  
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "AXIS COMPARISON SUMMARY\n" + "="*30 + "\n\n"
    
    for param in ['lyapunov_exponent', 'lag', 'embedding_dimension']:
        x_before = df[(df['axis'] == 'x') & (df['condition'] == 'Before')][param].mean()
        x_after = df[(df['axis'] == 'x') & (df['condition'] == 'After')][param].mean()
        z_before = df[(df['axis'] == 'z') & (df['condition'] == 'Before')][param].mean()
        z_after = df[(df['axis'] == 'z') & (df['condition'] == 'After')][param].mean()
        
        summary_text += f"{param.upper()}:\n"
        summary_text += f"  X-axis: {x_before:.3f} → {x_after:.3f}\n"
        summary_text += f"  Z-axis: {z_before:.3f} → {z_after:.3f}\n"
        summary_text += f"  X change: {((x_after-x_before)/x_before*100):.1f}%\n"
        summary_text += f"  Z change: {((z_after-z_before)/z_before*100):.1f}%\n\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{cohort_name} - Axis Comparison Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()




def check_normality_lyapunov(df, subject_col='subject', 
                             condition_col='condition', 
                             value_col='lyapunov_exponent',
                             axis_col='axis',
                             sheet_name=0):
    """
    Comprehensive normality testing for Lyapunov exponent paired differences.
    Tests normality of (before - after) differences to determine appropriate statistical test.
    """
    
    # Read Excel file
    
    # Get unique axes
    axes = sorted(df[axis_col].unique()) if axis_col in df.columns else ['all']
    
    # Create figure for Q-Q plots and histograms
    n_axes = len(axes)
    fig, axes_plot = plt.subplots(2, n_axes, figsize=(4*n_axes, 8))
    if n_axes == 1:
        axes_plot = axes_plot.reshape(2, 1)
    
    fig.suptitle('Normality Test for Before/After Differences - Lyapunov Exponents', 
                 fontsize=16, fontweight='bold')
    
    print("Shapiro-Wilk Test Results (α = 0.05)")
    print("If p > 0.05: Use paired t-test (normal)")
    print("If p < 0.05: Use Wilcoxon signed-rank test (non-normal)")
    print("-"*70)
    
    recommendations = {}
    
    for idx, axis in enumerate(axes):
        # Get paired data for this axis
        if axis_col in df.columns and axis != 'all':
            axis_data = df[df[axis_col] == axis]
            axis_label = f"{axis.upper()}"  # Just use the axis letter
        else:
            axis_data = df
            axis_label = "All Axes"
        
        # Separate before and after data
        before_df = axis_data[axis_data[condition_col].str.lower() == 'before']
        after_df = axis_data[axis_data[condition_col].str.lower() == 'after']
        
        # Create a proper pairing by subject (and axis if we're looking at specific axis)
        if axis_col in df.columns and axis != 'all':
            # For specific axis, we already filtered by axis, just need to match subjects
            before_pivot = before_df.set_index(subject_col)[value_col]
            after_pivot = after_df.set_index(subject_col)[value_col]
        else:
            # For all axes combined, need to match subject-axis pairs
            before_pivot = before_df.set_index([subject_col, axis_col])[value_col]
            after_pivot = after_df.set_index([subject_col, axis_col])[value_col]
        
        # Get common indices (paired data)
        common_indices = before_pivot.index.intersection(after_pivot.index)
        
        # Extract paired values
        before_values = before_pivot.loc[common_indices].values
        after_values = after_pivot.loc[common_indices].values
        
        differences = before_values - after_values
        n_pairs = len(differences)
        
        # Check if  any paired data
        if n_pairs == 0:
            print(f"\n{axis_label.upper()}: NO PAIRED DATA FOUND")
            print("  Skipping this axis - no matching subjects between conditions")
            continue
        
        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(differences)
        
        if p_value > 0.05:
            test_recommendation = "Paired t-test"
            normality_status = "Normal"
        else:
            test_recommendation = "Wilcoxon signed-rank"
            normality_status = "Non-normal"
        
        recommendations[axis_label] = test_recommendation
        
        # Print results
        print(f"\n{axis_label.upper()}:")
        print(f"  Sample size: n = {n_pairs} pairs")
        print(f"  Shapiro-Wilk statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Normality assessment: {normality_status}")
        print(f"  Recommended test: {test_recommendation}")
        
        # Descriptive stats of differences
        print(f"  Mean difference: {np.mean(differences):.4f}")
        print(f"  Std of differences: {np.std(differences, ddof=1):.4f}")
        print(f"  Median difference: {np.median(differences):.4f}")
        
        # Q-Q plot for visual check
        ax_qq = axes_plot[0, idx]
        stats.probplot(differences, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'{axis_label}\nQ-Q Plot', fontsize=10)
        ax_qq.grid(True, alpha=0.3)
        
        # Add p-value to plot
        ax_qq.text(0.05, 0.95, f'SW p={p_value:.3f}', 
                   transform=ax_qq.transAxes, 
                   bbox=dict(boxstyle='round', 
                           facecolor='yellow' if p_value < 0.05 else 'lightgreen', 
                           alpha=0.7),
                   verticalalignment='top')
        
        # Histogram with normal overlay (bottom row)
        ax_hist = axes_plot[1, idx]
        
        # Check if we have data to plot
        if len(differences) > 0:
            # Plot histogram
            n_bins = min(15, max(5, n_pairs // 3))  # Adaptive bin size
            ax_hist.hist(differences, bins=n_bins, density=True, alpha=0.7, 
                        color='skyblue', edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = np.mean(differences), np.std(differences, ddof=1)
            if sigma > 0:  # Only plot if there's variation
                x = np.linspace(differences.min() - sigma, differences.max() + sigma, 100)
                ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        else:
            ax_hist.text(0.5, 0.5, 'No paired data', 
                        transform=ax_hist.transAxes, 
                        ha='center', va='center', fontsize=12)
        
        ax_hist.set_xlabel('Difference (Before - After)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'{axis_label}\nHistogram', fontsize=10)
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY OF RECOMMENDATIONS")
    print("="*70)
    print(f"{'Axis':<20} {'Normality':<15} {'Recommended Test':<25}")
    print("-"*60)
    for axis_label, test in recommendations.items():
        normality = "Normal" if test == "Paired t-test" else "Non-normal"
        print(f"{axis_label:<20} {normality:<15} {test:<25}")
    
    # Additional normality tests for robustness
    print("\n" + "="*70)
    print("ADDITIONAL NORMALITY TESTS (for comparison)")
    print("="*70)
    
    for idx, axis in enumerate(axes):
        # Get differences again
        if axis_col in df.columns and axis != 'all':
            axis_data = df[df[axis_col] == axis]
            axis_label = f"{axis.upper()}"  # Just use the axis letter
        else:
            axis_data = df
            axis_label = "All Axes"
        
        before_df = axis_data[axis_data[condition_col] == 'before']
        after_df = axis_data[axis_data[condition_col] == 'after']
        
        # Create proper pairing
        if axis_col in df.columns and axis != 'all':
            before_pivot = before_df.set_index(subject_col)[value_col]
            after_pivot = after_df.set_index(subject_col)[value_col]
        else:
            before_pivot = before_df.set_index([subject_col, axis_col])[value_col]
            after_pivot = after_df.set_index([subject_col, axis_col])[value_col]
        
        common_indices = before_pivot.index.intersection(after_pivot.index)
        differences = before_pivot.loc[common_indices].values - after_pivot.loc[common_indices].values
        
        # Skip if no paired data
        if len(differences) == 0:
            continue
        
        # Anderson-Darling test
        ad_result = stats.anderson(differences, dist='norm')
        
        # D'Agostino and Pearson's test
        if len(differences) >= 8:  # Minimum sample size requirement
            k2_stat, k2_p = stats.normaltest(differences)
        else:
            k2_stat, k2_p = np.nan, np.nan
        
        print(f"\n{axis_label}:")
        print(f"  Anderson-Darling statistic: {ad_result.statistic:.4f}")
        print(f"  Critical value (5%): {ad_result.critical_values[2]:.4f}")
        print(f"  AD Test: {'Reject normality' if ad_result.statistic > ad_result.critical_values[2] else 'Fail to reject normality'}")
        
        if not np.isnan(k2_p):
            print(f"  D'Agostino-Pearson K² test p-value: {k2_p:.4f}")
            print(f"  K² Test: {'Reject normality' if k2_p < 0.05 else 'Fail to reject normality'}")
    
    # Quick decision guide
    print("\n" + "="*70)
    print("DECISION GUIDE")
    print("="*70)
    print("Based on the Shapiro-Wilk test results:")
    print("\nRecommended statistical tests for each axis:")
    for axis_label, test in recommendations.items():
        print(f"- {axis_label}: {test}")
    
    print("\nNote: With small sample sizes (n<30), normality tests may lack power.")
    print("Visual inspection (Q-Q plots) is equally important for decision-making.")
    
    # Return recommendations for use in subsequent analyses
    return recommendations

def perform_statistical_tests(df, recommendations=None, subject_col='subject', 
                            condition_col='condition', value_col='lyapunov_exponent',
                            axis_col='axis', sheet_name=0):
    """
    Perform appropriate statistical tests based on normality results.
    """
    
    axes = sorted(df[axis_col].unique()) if axis_col in df.columns else ['all']
    
    print("\n" + "="*70)
    print("STATISTICAL TEST RESULTS")
    print("="*70)
    
    results_summary = []
    
    for axis in axes:
        if axis_col in df.columns and axis != 'all':
            axis_data = df[df[axis_col] == axis]
            axis_label = f"{axis.upper()}"  # Just use the axis letter
        else:
            axis_data = df
            axis_label = "All Axes"
        
        # Get paired data
        before_df = axis_data[axis_data[condition_col].str.lower() == 'before']
        after_df = axis_data[axis_data[condition_col].str.lower() == 'after']
        
        # Create proper pairing
        if axis_col in df.columns and axis != 'all':
            before_pivot = before_df.set_index(subject_col)[value_col]
            after_pivot = after_df.set_index(subject_col)[value_col]
        else:
            before_pivot = before_df.set_index([subject_col, axis_col])[value_col]
            after_pivot = after_df.set_index([subject_col, axis_col])[value_col]
        
        common_indices = before_pivot.index.intersection(after_pivot.index)
        
        # Skip if no paired data
        if len(common_indices) == 0:
            print(f"\n{axis_label}: NO PAIRED DATA - SKIPPING")
            continue
        
        before_values = before_pivot.loc[common_indices].values
        after_values = after_pivot.loc[common_indices].values
        
        differences = before_values - after_values
        
        print(f"\n{axis_label}:")
        print(f"  Before: Mean = {np.mean(before_values):.4f}, SD = {np.std(before_values, ddof=1):.4f}")
        print(f"  After: Mean = {np.mean(after_values):.4f}, SD = {np.std(after_values, ddof=1):.4f}")
        print(f"  Mean difference (Before - After): {np.mean(differences):.4f}")
        
        # Perform appropriate test
        if recommendations and axis_label in recommendations:
            test_to_use = recommendations[axis_label]
        else:
            # Default to Wilcoxon for small samples
            test_to_use = "Wilcoxon signed-rank" if len(differences) < 30 else "Paired t-test"
        
        if test_to_use == "Paired t-test":
            t_stat, p_value = stats.ttest_rel(before_values, after_values)
            print(f"  Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")
            test_used = "Paired t-test"
        else:
            stat, p_value = stats.wilcoxon(before_values, after_values)
            print(f"  Wilcoxon signed-rank test: W = {stat:.4f}, p = {p_value:.4f}")
            test_used = "Wilcoxon"
        
        # Effect size
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        print(f"  Effect size (Cohen's d): {cohen_d:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            print(f"  Result: Statistically significant difference (p < 0.05)")
        else:
            print(f"  Result: No statistically significant difference (p ≥ 0.05)")
        
        results_summary.append({
            'Axis': axis_label,
            'Test': test_used,
            'p-value': p_value,
            'Effect Size': cohen_d,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    return summary_df


def sensitivity_checks(before_vals, after_vals, label=""):
    """
    1) Bootstrap a 95% CI on the mean difference
    2) Leave-one-out t-test & Wilcoxon signed-rank p-value ranges
    """
    diffs = before_vals - after_vals
    n = len(diffs)
    if n < 2:
        print(f"   [{label}] Not enough data for sensitivity.")
        return

    # Bootstrap 95% CI
    boots = np.random.choice(diffs, size=(10000, n), replace=True)
    ci_lo, ci_hi = np.percentile(boots.mean(axis=1), [2.5, 97.5])
    print(f"   [{label}] Bootstrap 95% CI on Δmean: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Leave-one-out p-value ranges
    p_t, p_w = [], []
    for i in range(n):
        loo_before = np.delete(before_vals, i)
        loo_after  = np.delete(after_vals,  i)
        if len(loo_before) > 1:
            p_t .append(ttest_rel (loo_before, loo_after).pvalue)
            p_w .append(wilcoxon(loo_before, loo_after).pvalue)
    print(f"   [{label}] LOO t-test p-range:    [{np.min(p_t):.3f}, {np.max(p_t):.3f}]")
    print(f"   [{label}] LOO Wilcoxon p-range: [{np.min(p_w):.3f}, {np.max(p_w):.3f}]")


def check_normality_and_test_lyapunov(df,cohort_name='Cohort'):
    """
    Checks normality, runs statistical tests, and performs sensitivity checks
    """
    
    axes = sorted(df['axis'].unique())
    print(f"\n=== {cohort_name} Lyapunov Analysis ===")

    for axis in axes:
        sub = df[df['axis']==axis]
        before = sub[sub['condition']=='Before'].set_index('subject')['lyapunov_exponent']
        after  = sub[sub['condition']=='After' ].set_index('subject')['lyapunov_exponent']
        common = before.index.intersection(after.index)
        bvals = before.loc[common].values
        avals = after .loc[common].values
        diffs = bvals - avals

        print(f"\n-- Axis {axis.upper()} (n={len(common)}) --")
        # Shapiro-Wilk
        stat, p_sw = shapiro(diffs)
        normal = (p_sw>0.05)
        test_name = "Paired t-test" if normal else "Wilcoxon signed-rank"
        print(f" Shapiro-Wilk p={p_sw:.3f} → {test_name}")

        # Run chosen test
        if normal:
            t, pval = ttest_rel(bvals, avals)
            print(f" Paired t-test:     t={t:.3f}, p={pval:.3f}")
        else:
            w, pval = wilcoxon(bvals, avals)
            print(f" Wilcoxon signed-rank: W={w:.3f}, p={pval:.3f}")

        # Descriptives
        print(f" Mean before={bvals.mean():.3f}, after={avals.mean():.3f}, Δmean={diffs.mean():.3f}")
        print(f" Std of diffs={diffs.std(ddof=1):.3f}")

        # Sensitivity checks
        print(" Sensitivity analyses:")
        sensitivity_checks(bvals, avals, label=f"{cohort_name} Axis {axis.upper()}")