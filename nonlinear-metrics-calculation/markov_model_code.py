
# Description: This script contains two classes used for the construction of MSM.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.linalg import eig
from scipy.interpolate import interp1d
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis as BSpline
from typing import Optional, Dict, Tuple, List

sns.set_theme(style="whitegrid", font_scale=1.1)

class EventAlignedFPCA_MSM_RMCE:
    """
    Event-aligned MSM for gait stability using FPCA + K-means:
    each stride (event-to-event) → functional representation → FPCA → K-means → Stable/Marginal/Unstable.
    """
    def __init__(self,
                 patient_id: str,
                 n_states: int = 3,
                 n_grid_points: int = 100,  # Number of points to interpolate each stride to
                 n_components: int = 7,
                 n_basis: int = 15):
        self.patient_id        = patient_id
        self.n_states          = n_states
        self.n_grid_points     = n_grid_points  # For interpolation
        self.n_components      = n_components
        self.n_basis           = n_basis
        self.scaler            = StandardScaler()
        self.metadata          = {}
        self.fpca              = None
        self.kmeans            = None
        self.transition_matrix = None
        self.stationary_dist   = None
        self.state_labels      = {0:"Stable", 1:"Marginally Stable", 2:"Unstable"}
        self.metrics_history   = None
        self.fdata             = None
        self.fpca_scores       = None
        self.stride_info       = None  # Store stride timing information
        self.validation_metrics = {}  # Store validation metrics

    def compute_event_aligned_functional_data(self, signal: np.ndarray, events: np.ndarray, fs: float) -> FDataGrid:
        """
        Convert event-aligned segments into functional data objects.
        Each segment is from one event to the next (e.g., foot strike to foot strike).
        """
        # Collect stride segments
        segments = []
        stride_times = []
        stride_durations = []
        valid_strides = []
        
        for i in range(len(events) - 1):
            start_time = events[i]
            end_time = events[i + 1]
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # Check bounds
            if start_idx >= 0 and end_idx <= len(signal):
                segment = signal[start_idx:end_idx]
                
                # Skip very short or very long strides (outliers)
                stride_duration = end_time - start_time
                if 0.5 < stride_duration < 2.0:  # Reasonable stride duration range
                    segments.append(segment)
                    stride_times.append(start_time)
                    stride_durations.append(stride_duration)
                    valid_strides.append(i)
        
        # Interpolate all segments to common grid
        interpolated_segments = []
        common_grid = np.linspace(0, 1, self.n_grid_points)
        
        for segment in segments:
            # Create interpolation function
            original_grid = np.linspace(0, 1, len(segment))
            f = interp1d(original_grid, segment, kind='cubic', fill_value='extrapolate')
            
            # Interpolate to common grid
            interpolated_segment = f(common_grid)
            interpolated_segments.append(interpolated_segment)
        
        # Convert to FDataGrid
        self.fdata = FDataGrid(
            data_matrix=np.array(interpolated_segments),
            grid_points=common_grid
        )
        
        # Store metadata - later for visuals and analysis
        self.metrics_history = pd.DataFrame({
            'stride_idx': range(len(segments)),
            'original_event_idx': valid_strides,
            'time': stride_times,
            'duration': stride_durations
        })
        
        # Store stride info for later use
        self.stride_info = {
            'events': events,
            'valid_strides': valid_strides,
            'fs': fs
        }
        
        print(f"Created functional data from {len(segments)} valid strides")
        print(f"Average stride duration: {np.mean(stride_durations):.3f}s ± {np.std(stride_durations):.3f}s")
        
        return self.fdata

    def apply_fpca(self) -> np.ndarray:
        """Apply Functional PCA to extract principal components."""
        try:
            # smooth the functional data using B-spline basis
            basis = BSpline(n_basis=self.n_basis, domain_range=(0, 1))
            
            smooth_fdata = self.fdata.to_basis(basis)
            
            # apply FPCA to smoothed data
            self.fpca = FPCA(n_components=self.n_components)
            self.fpca_scores = self.fpca.fit_transform(smooth_fdata)
        except:
            # If smoothing fails, apply FPCA directly to the discrete data
            print("Note: B-spline smoothing failed, using direct FPCA on discrete data")
            self.fpca = FPCA(n_components=self.n_components)
            self.fpca_scores = self.fpca.fit_transform(self.fdata)
        
        # add FPCA scores to metrics history
        for i in range(self.n_components):
            self.metrics_history[f'fpca_{i}'] = self.fpca_scores[:, i]
        
        # compute FPCA reconstruction error
        self._compute_fpca_reconstruction_error()
        
        return self.fpca_scores


    def _compute_fpca_reconstruction_error(self):
        """Compute reconstruction error from FPCA, plus NRMSE wrt range & std."""
        try:
            recon = self.fpca.inverse_transform(self.fpca_scores)
            if not isinstance(recon, FDataGrid):
                recon = recon.to_grid(self.fdata.grid_points)

            orig = self.fdata.data_matrix.squeeze()  # (n_strides, n_grid)
            rec  = recon.data_matrix.squeeze()

            # 1) MSE & RMSE
            mse  = np.mean((orig - rec)**2)
            rmse = np.sqrt(mse)
            self.validation_metrics['fpca_reconstruction_mse'] = mse
            self.validation_metrics['fpca_reconstruction_rmse'] = rmse

            # 2) Signal range & std
            sig_min   = orig.min()
            sig_max   = orig.max()
            sig_range = sig_max - sig_min
            sig_std   = orig.std()
            self.validation_metrics.update({
                'signal_min': sig_min,
                'signal_max': sig_max,
                'signal_range': sig_range,
                'signal_std': sig_std,
            })

            # 3) Normalized errors
            nrmse_range = rmse / sig_range if sig_range>0 else np.nan
            nrmse_std   = rmse / sig_std   if sig_std>0   else np.nan
            self.validation_metrics['nrmse_range'] = nrmse_range
            self.validation_metrics['nrmse_std']   = nrmse_std

            print(f"FPCA reconstruction MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            print(f"Signal range: {sig_range:.4f}, σ: {sig_std:.4f}")
            print(f"NRMSE (range): {nrmse_range:.2%},  NRMSE (σ): {nrmse_std:.2%}")

        except Exception as e:
            print(f"Errror: {e}")
            self.validation_metrics['fpca_reconstruction_mse'] = np.nan
    
    
    def _compute_kmeans_variance_explained(self):
        """ Compute variance explained by K-means clustering on FPCA scores."""
        # Pull out the FPCA scores 
        X = self.fpca_scores[:, :self.n_components]
        labels = self.metrics_history['state'].values
        
        # 1) global mean & TSS
        global_mean = X.mean(axis=0)
        TSS = np.sum((X - global_mean) ** 2)
        
        # 2) within‑cluster sum of squares (WSS)
        WSS = 0.0
        for k in range(self.n_states):
            mask = labels == k
            if mask.sum() == 0:
                continue
            cluster_data = X[mask]
            cluster_mean = cluster_data.mean(axis=0)
            WSS += np.sum((cluster_data - cluster_mean) ** 2)
        
        # 3) fraction explained
        R2 = 1 - WSS / TSS if TSS > 0 else 0.0
        self.validation_metrics['fpca_variance_explained_by_clustering'] = R2
        print(f"FPCA‑KMeans variance explained: {R2:.1%}")

    def identify_stability_states(self) -> np.ndarray:
        """Apply K-means clustering to FPCA scores to identify states."""
        
        # Standardize FPCA scores
        X = self.scaler.fit_transform(self.fpca_scores[:,:self.n_components])  
        
        # Apply K-means
        self.kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        raw_states = self.kmeans.fit_predict(X)
        
        # Order states by distance from origin (stability proxy)
        cluster_centers = self.kmeans.cluster_centers_
        norms = np.linalg.norm(cluster_centers, axis=1)
        order = np.argsort(norms)
        
        # Remap states to ensure 0=stable, 2=unstable
        state_mapping = {old: new for new, old in enumerate(order)}
        states = np.array([state_mapping[s] for s in raw_states])
        
        self.metrics_history['state'] = states
        
        # Compute clustering validation metric
        self._compute_clustering_silhouette_score(X, states)
        self.metrics_history['state'] = states
        
        
        self._compute_clustering_silhouette_score(X, states)
       
        self._compute_kmeans_variance_explained()
    
        return states

    def _compute_clustering_silhouette_score(self, X: np.ndarray, labels: np.ndarray):
        """Compute silhouette score for clustering validation."""
        
        try:
            # Compute silhouette score
            sil_score = silhouette_score(X, labels)
            self.validation_metrics['clustering_silhouette_score'] = sil_score
            
            print(f"Clustering silhouette score: {sil_score:.3f}")
            
            # Interpretation
            if sil_score > 0.7:
                interpretation = "Strong clustering"
            elif sil_score > 0.5:
                interpretation = "Reasonable clustering"
            elif sil_score > 0.25:
                interpretation = "Weak clustering"
            else:
                interpretation = "Poor clustering"
            
            print(f"Clustering quality: {interpretation}")
            
        except Exception as e:
            print(f"Warning: Could not compute silhouette score: {e}")
            self.validation_metrics['clustering_silhouette_score'] = np.nan

    def compute_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Compute state transition probabilities."""

        T = np.zeros((self.n_states, self.n_states))
        for a, b in zip(states[:-1], states[1:]):
            T[a, b] += 1
        
        # Normalize rows
        row_sums = T.sum(axis=1)
        row_sums[row_sums == 0] = 1
        T = T / row_sums[:, None]
        
        self.transition_matrix = T
        return T

    def compute_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution of the Markov chain."""

        vals, vecs = eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(vals - 1))
        pi = np.real(vecs[:, idx])
        pi = pi / pi.sum()
        self.stationary_dist = np.abs(pi)
        return self.stationary_dist

    def compute_mean_dwell_times(self) -> np.ndarray:
        """Compute mean dwell time in each state."""

        τ = np.zeros(self.n_states)
        for i in range(self.n_states):
            p = self.transition_matrix[i, i]
            τ[i] = np.inf if p >= 1 else 1 / (1 - p)
        return τ

    def compute_stability_metrics(self) -> dict:
        """Compute overall stability metrics."""

        df = self.metrics_history
        out = {'patient_id': self.patient_id}
        
        # State fractions
        for i in range(self.n_states):
            out[f'frac_state_{i}'] = (df.state == i).mean()
        
        # Mean FPCA scores per state
        for i in range(self.n_states):
            state_data = df[df.state == i]
            for j in range(self.n_components):
                out[f'mean_fpca_{j}_state_{i}'] = state_data[f'fpca_{j}'].mean()
        
        # Mean stride duration per state
        for i in range(self.n_states):
            state_data = df[df.state == i]
            out[f'mean_duration_state_{i}'] = state_data['duration'].mean()
        
        # Add validation metrics
        out.update(self.validation_metrics)
        
        return out

    def visualize_patient_analysis(self):
        """2×2 summary figure with FPCA results and transitions."""
        df = self.metrics_history.copy()
        names = list(self.state_labels.values())
        colors = sns.color_palette(n_colors=self.n_states)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1) FPCA explained variance
        if self.fpca is not None and hasattr(self.fpca, 'explained_variance_ratio_'):
            explained_var = self.fpca.explained_variance_ratio_
            pc_labels = [f'PC{i+1}' for i in range(len(explained_var))]
            axes[0, 0].bar(pc_labels, explained_var, color='skyblue')
            axes[0, 0].set_title('FPCA Explained Variance Ratio')
            axes[0, 0].set_ylabel('Variance Explained')
            for i, v in enumerate(explained_var):
                axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center')
                
        # 2) Transition matrix heatmap
        sns.heatmap(self.transition_matrix,
                    annot=True, fmt=".2f",
                    xticklabels=names, yticklabels=names,
                    cmap="Blues", ax=axes[0, 1])
        axes[0, 1].set_title("Transition Probabilities")
        
        # Add silhouette score
        if 'clustering_silhouette_score' in self.validation_metrics:
            sil_score = self.validation_metrics['clustering_silhouette_score']
            axes[0, 1].text(0.02, 0.98, f'Silhouette Score: {sil_score:.3f}', 
                           transform=axes[0, 1].transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3) FPCA scores scatter (first 2 PCs)
        if self.n_components >= 2 and 'fpca_0' in df.columns and 'fpca_1' in df.columns:
            for i, (state, name) in enumerate(self.state_labels.items()):
                mask = df.state == state
                axes[1, 0].scatter(df.loc[mask, 'fpca_0'], 
                                 df.loc[mask, 'fpca_1'],
                                 label=name, color=colors[i], alpha=0.6)
            axes[1, 0].set_xlabel('FPCA Component 1')
            axes[1, 0].set_ylabel('FPCA Component 2')
            axes[1, 0].set_title('State Distribution in FPCA Space')
            axes[1, 0].legend()
        
        # 4) Stride duration by state
        # More informative than stationary distribution for event-aligned data
        duration_data = []
        for state, name in self.state_labels.items():
            durations = df[df.state == state]['duration'].values
            duration_data.extend([{'State': name, 'Duration': d} for d in durations])
        
        duration_df = pd.DataFrame(duration_data)
        # sns.boxplot(data=duration_df, x='State', y='Duration', palette=colors, ax=axes[1, 1])
        sns.boxplot(
            data=duration_df,
            x='State',
            y='Duration',
            hue='State',       
            palette=colors,
            dodge=False,        
            ax=axes[1, 1]
                )   
        axes[1, 1].set_title('Stride Duration by State')
        axes[1, 1].set_ylabel('Duration (s)')
        
        plt.tight_layout()
        plt.show()

    def visualize_functional_modes(self):
        """Visualize the functional principal components."""
        if self.fpca is None:
            return
        
        if not hasattr(self.fpca, 'mean_') or not hasattr(self.fpca, 'components_'):
            print("FPCA components not available for visualization")
            return
        
        fig, axes = plt.subplots(1, min(3, self.n_components), 
                                figsize=(5*min(3, self.n_components), 4))
        
        if self.n_components == 1:
            axes = [axes]
        
        # Create evaluation grid
        grid = np.linspace(0, 1, 100)
        
        try:
            # Get mean function and components
            mean_func = self.fpca.mean_
            
            # Evaluate mean function on grid
            mean_values = mean_func(grid).flatten()
            
            for i in range(min(3, self.n_components)):
                # Get principal component
                pc = self.fpca.components_[i]
                pc_values = pc(grid).flatten()
                
                # Plot mean plus PC
                axes[i].plot(grid, mean_values, 'k-', 
                            label='Mean', linewidth=2)
                
                # Scale by explained variance if available
                if hasattr(self.fpca, 'explained_variance_'):
                    scale = np.sqrt(self.fpca.explained_variance_[i])
                else:
                    scale = 1.0
                    
                axes[i].plot(grid, mean_values + 2*scale*pc_values, 
                            'b--', label='+2 SD', alpha=0.7)
                axes[i].plot(grid, mean_values - 2*scale*pc_values, 
                            'r--', label='-2 SD', alpha=0.7)
                
                if hasattr(self.fpca, 'explained_variance_ratio_'):
                    axes[i].set_title(f'PC{i+1} ({self.fpca.explained_variance_ratio_[i]:.1%} var)')
                else:
                    axes[i].set_title(f'PC{i+1}')
                    
                axes[i].set_xlabel('Normalized Stride Time')
                axes[i].legend()
        except Exception as e:
            print(f"Error visualizing functional modes: {e}")
            for ax in axes:
                ax.text(0.5, 0.5, 'Functional modes\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()

    def visualize_stride_alignment(self, signal: np.ndarray, max_strides: int = 20):
        """Visualize how strides are aligned and colored by state."""
        if self.stride_info is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        colors = sns.color_palette(n_colors=self.n_states)
        events = self.stride_info['events']
        fs = self.stride_info['fs']
        
        # Original signal with event markers and state colors
        time_axis = np.arange(len(signal)) / fs
        ax1.plot(time_axis, signal, 'k-', alpha=0.5, linewidth=0.5)
        
        # Add stride coloring
        for idx, stride_idx in enumerate(self.stride_info['valid_strides'][:max_strides]):
            if stride_idx < len(events) - 1:
                start_time = events[stride_idx]
                end_time = events[stride_idx + 1]
                state = int(self.metrics_history.iloc[idx]['state'])  # Convert to int
                ax1.axvspan(start_time, end_time, color=colors[state], alpha=0.3)
        
        # Add event markers
        ax1.scatter(events, np.interp(events, time_axis, signal), 
                   color='red', s=50, zorder=5, label='Foot Strikes')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal Value')
        ax1.set_title('Original Signal with Event-Aligned State Classification')
        ax1.legend()
        
        # Overlaid normalized strides colored by state
        grid = np.linspace(0, 1, self.n_grid_points)
        for i in range(min(max_strides, len(self.fdata))):
            state = int(self.metrics_history.iloc[i]['state'])  # Convert to int
            ax2.plot(grid, self.fdata[i].data_matrix[0], 
                    color=colors[state], alpha=0.5, linewidth=1)
        
        # Add mean curves per state
        for state, name in self.state_labels.items():
            state_mask = self.metrics_history['state'] == state
            if state_mask.sum() > 0:
                state_curves = self.fdata[state_mask].data_matrix
                mean_curve = np.mean(state_curves, axis=0)
                ax2.plot(grid, mean_curve, color=colors[state], 
                        linewidth=3, label=name)
        
        ax2.set_xlabel('Normalized Stride Time (0=strike, 1=next strike)')
        ax2.set_ylabel('Signal Value')
        ax2.set_title('Event-Aligned Strides by State')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def fit(self, signal: np.ndarray, events: np.ndarray, fs: float) -> 'EventAlignedFPCA_MSM_RMCE':
        """
        Complete pipeline: signal + events → functional data → FPCA → clustering → MSM.
        """
        print(f"Processing patient {self.patient_id} with event-aligned strides...")
        
        # 1. Create functional data from event-aligned segments
        print("Creating event-aligned functional data...")
        self.compute_event_aligned_functional_data(signal, events, fs)
        
        # 2. Apply FPCA
        print(f"Applying FPCA with {self.n_components} components...")
        self.apply_fpca()
        
        # 3. Identify states via K-means
        print("Identifying stability states via K-means...")
        states = self.identify_stability_states()
        
        # 4. Build Markov model
        print("Computing transition matrix...")
        self.compute_transition_matrix(states)
        
        # 5. Compute stationary distribution
        print("Computing stationary distribution...")
        self.compute_stationary_distribution()
             
        metrics = self.compute_stability_metrics()
        print(f"\nStability metrics for {self.patient_id}:")
        for key, value in metrics.items():
            if key == 'patient_id':
                continue
            # if it's an array, summarize it rather than try to format directly
            if isinstance(value, np.ndarray):
                print(f"  {key}: array, n={value.size}, mean={value.mean():.3f}, std={value.std():.3f}")
            else:
                print(f"  {key}: {value:.3f}")

        # Print validation summary
        print(f"\nValidation Summary:")
       
        if 'fpca_reconstruction_rmse' in self.validation_metrics:
            rmse = self.validation_metrics['fpca_reconstruction_rmse']
            nrmse_r = self.validation_metrics.get('nrmse_range', np.nan)
            nrmse_s = self.validation_metrics.get('nrmse_std',   np.nan)
            print(f"  FPCA reconstruction → RMSE = {rmse:.4f}  "
            f"(NRMSE range = {nrmse_r:.2%}, NRMSE σ = {nrmse_s:.2%})")
        sil = self.validation_metrics.get('clustering_silhouette_score', None)
        if sil is not None and np.isscalar(sil):
            print(f"  Clustering quality: Silhouette = {sil:.3f}")

        return self
    def plot_fpca_reconstruction_error_distribution(self, bins: int = 20):
        """
        Histogram of per‑stride FPCA reconstruction MSE to spot outliers.
        """
        errors = self.validation_metrics.get('fpca_reconstruction_mse_per_stride')
        if errors is None:
            print("No per‑stride MSE data available.")
            return

        plt.figure(figsize=(8, 4))
        sns.histplot(errors, bins=bins, kde=True)
        plt.title('Per‑Stride FPCA Reconstruction MSE')
        plt.xlabel('Reconstruction MSE')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


    def plot_silhouette_samples(self):
        """
        Scatter of silhouette coefficient per stride.
        """
        from sklearn.metrics import silhouette_samples
        from sklearn.preprocessing import StandardScaler

        # Rescale all PCs afresh
        X = StandardScaler().fit_transform(self.fpca_scores)
        labels = self.metrics_history['state'].values
        sil_vals = silhouette_samples(X, labels)

        self.validation_metrics['silhouette_samples'] = sil_vals

        plt.figure(figsize=(10, 5))
        palette = sns.color_palette(n_colors=self.n_states)
        sns.scatterplot(
            x=np.arange(len(sil_vals)),
            y=sil_vals,
            hue=labels,
            palette=palette,
            legend='brief',
            alpha=0.7
        )
        plt.axhline(sil_vals.mean(), color='k', linestyle='--',
                    label=f'Mean = {sil_vals.mean():.3f}')
        plt.title('Silhouette Coefficient per Stride')
        plt.xlabel('Stride Index')
        plt.ylabel('Silhouette Coefficient')
        plt.legend(title='State')
        plt.tight_layout()
        plt.show()


    def plot_silhouette_analysis(self):
        """
        Classic silhouette‑analysis bar plot by cluster.
        """
        from sklearn.metrics import silhouette_samples, silhouette_score
        from sklearn.preprocessing import StandardScaler
        import matplotlib.cm as cm

        X = StandardScaler().fit_transform(self.fpca_scores)
        y = self.metrics_history['state'].values
        sil_vals = silhouette_samples(X, y)
        sil_avg  = silhouette_score(X, y)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        y_lower = 10
        for i in range(self.n_states):
            ith_sil = np.sort(sil_vals[y == i])
            size_i  = ith_sil.shape[0]
            if size_i == 0:
                continue
            y_upper = y_lower + size_i
            color   = cm.nipy_spectral(float(i) / self.n_states)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_sil,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_i, f"State {i}")
            y_lower = y_upper + 10

        ax1.axvline(sil_avg, color="red", linestyle="--",
                    label=f'Average silhouette = {sil_avg:.3f}')
        ax1.set_title("Silhouette Analysis per State")
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster")
        ax1.set_yticks([])
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        
    def plot_all_feature_correlations(self):
        # 1) gather metadata columns
        meta_cols = ['stride_idx', 'time', 'duration']
        
        # 2) FPCA score columns
        fpca_cols = [f"fpca_{i}" for i in range(self.n_components)]
        
        # 3) validation metric columns (filter numeric entries)
        val_cols = [k for k, v in self.validation_metrics.items() 
                    if isinstance(v, (int, float, np.floating, np.integer))]
        
        # 4) combine and drop any that aren't in metrics_history
        all_cols = [c for c in (meta_cols + fpca_cols + val_cols) 
                    if c in self.metrics_history.columns]
        
        # build the DataFrame
        df = self.metrics_history[all_cols].copy()
        
        # if some validation metrics are arrays, skip them
        # (we only plot scalar metrics)
        
        # 5) compute correlation matrix
        corr = df.corr()
        
        # 6) plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            cbar_kws={"shrink": .8},
            linewidths=0.5
        )
        plt.title(f"Patient {self.patient_id} — All Feature Correlations")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        
    def plot_features_vs_clusters(self):
        # 1) pick numeric features (e.g., duration, FPCA scores, RMSE, silhouette…)
        meta_cols  = ['duration']
        fpca_cols  = [f"fpca_{i}" for i in range(self.n_components)]
        val_cols   = [k for k, v in self.validation_metrics.items() 
                    if isinstance(v, (int, float, np.integer, np.floating))]
        feat_cols  = [c for c in (meta_cols + fpca_cols + val_cols) 
                    if c in self.metrics_history.columns]
        
        df_feats   = self.metrics_history[feat_cols].copy()
        
        # 2) one‑hot encode the cluster labels
        df_states  = pd.get_dummies(self.metrics_history['state'], 
                                    prefix='state')
        
        # 3) combine
        df_all     = pd.concat([df_feats, df_states], axis=1)
        
        # 4) compute correlation
        corr = df_all.corr()
        
        # 5) plot
        plt.figure(figsize=(10,8))
        sns.heatmap(
            corr.loc[feat_cols, df_states.columns],   # show only feature vs state correlations
            annot=True, fmt=".2f", cmap="vlag", center=0,
            cbar_kws={"shrink":.8}
        )
        plt.title(f"Patient {self.patient_id} — Feature vs. Cluster Correlations")
        plt.xlabel("Cluster (one‑hot)")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()




sns.set_theme(style="whitegrid", font_scale=1.1)
class DistributionMSM_Strides:
    """
    MSM for gait stability from single-stride distributions
    """
    def __init__(self,
                 patient_id: str,
                 n_states: int = 3,
                 n_bins: int = 64, n_grid_points: int = 100):
        self.patient_id        = patient_id
        self.n_states          = n_states
        self.n_bins            = n_bins
        self.n_grid_points     = n_grid_points
        self.scaler            = StandardScaler()
        self.gmm               = None
        self.transition_matrix = None
        self.stationary_dist   = None
        self.state_labels      = {0:"Stable",1:"Marginally Stable",2:"Unstable"}
        self.metrics_history   = None
        self.validation_metrics = {}  # Store validation metrics
        self.histogram_data    = None  # Store original histogram data

    def compute_stride_metrics(self,
                               signal: np.ndarray,
                               events: np.ndarray,
                               fs: float) -> pd.DataFrame:
        """
        Build one histogram per stride, from one event to the next.
        """
        # edges = np.linspace(signal.min(), signal.max(), self.n_bins+1)
        edges = np.linspace(signal.min(), signal.max(), self.n_bins + 1)
        common_grid = np.linspace(0, 1, self.n_grid_points)
        recs = []
        histograms = []
        
        for i in range(len(events)-1):
            start_idx = int(events[i] * fs)
            end_idx   = int(events[i+1] * fs)
            # skip if out of bounds
            if start_idx<0 or end_idx>len(signal):
                continue
            stride = signal[start_idx:end_idx]
            # skip too-short/long strides
            dur = (events[i+1] - events[i])
            if not (0.3 < dur < 2.0):
                continue
            
            # —— time‐normalize to [0,1] on a fixed grid:
            orig_grid = np.linspace(0, 1, stride.shape[0])
            f_interp  = interp1d(orig_grid, stride,
                                kind='cubic',
                                 fill_value='extrapolate')
            stride = f_interp(common_grid)

            h, _ = np.histogram(stride, bins=edges, density=True)
            h    = h / h.sum()  # Normalize to probability
            histograms.append(h)
            
            rec = {
                'stride_idx': i,
                'time': events[i],        # start time of stride
                'duration': dur
            }
            # add bin probabilities
            for b in range(self.n_bins):
                rec[f'bin_{b}'] = h[b]
            recs.append(rec)

        self.metrics_history = pd.DataFrame(recs)
        self.histogram_data = np.array(histograms)  # Store for validation
        
        print(f"Created histograms from {len(recs)} valid strides")
        print(f"Average stride duration: {self.metrics_history['duration'].mean():.3f}s ± {self.metrics_history['duration'].std():.3f}s")
        
        return self.metrics_history

    def _compute_gmm_reconstruction_error(self):
        """
        Compute reconstruction error from GMM clustering.
        For each stride, reconstruct its histogram from the assigned cluster's mean.
        """
        try:
            if self.gmm is None or self.histogram_data is None:
                print("Warning: GMM or histogram data not available")
                self.validation_metrics['gmm_reconstruction_rmse'] = np.nan
                return
            
            # Get the standardized data that was used for clustering
            cols = [c for c in self.metrics_history.columns if c.startswith('bin_')]
            X_scaled = self.scaler.transform(self.metrics_history[cols].values)
            
            # Get cluster assignments
            cluster_labels = self.gmm.predict(X_scaled)
            
            # Reconstruct each histogram using its cluster mean (in scaled space)
            cluster_means = self.gmm.means_
            reconstructed_scaled = cluster_means[cluster_labels]
            
            # Transform back to original space
            reconstructed_original = self.scaler.inverse_transform(reconstructed_scaled)
            
            # Ensure reconstructed histograms are valid probabilities (non-negative, sum to 1)
            reconstructed_original = np.maximum(0, reconstructed_original)
            row_sums = reconstructed_original.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            reconstructed_original = reconstructed_original / row_sums
            
            # Compare with original histograms
            original_histograms = self.metrics_history[cols].values
            
            # Compute RMSE
            mse = np.mean((original_histograms - reconstructed_original)**2)
            rmse = np.sqrt(mse)
            
            # Compute per-stride RMSE
            per_stride_mse = np.mean((original_histograms - reconstructed_original)**2, axis=1)
            per_stride_rmse = np.sqrt(per_stride_mse)
            
            self.validation_metrics['gmm_reconstruction_rmse'] = rmse
            self.validation_metrics['gmm_reconstruction_rmse_per_stride'] = per_stride_rmse
            
            print(f"GMM reconstruction RMSE: {rmse:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not compute GMM reconstruction error: {e}")
            self.validation_metrics['gmm_reconstruction_rmse'] = np.nan

    def _compute_clustering_silhouette_score(self, X: np.ndarray, labels: np.ndarray):
        """Compute silhouette score for clustering validation."""
        try:
            # Compute silhouette score
            sil_score = silhouette_score(X, labels)
            self.validation_metrics['clustering_silhouette_score'] = sil_score
            
            print(f"Clustering silhouette score: {sil_score:.3f}")
            
            # Interpretation
            if sil_score > 0.7:
                interpretation = "Strong clustering"
            elif sil_score > 0.5:
                interpretation = "Reasonable clustering"
            elif sil_score > 0.25:
                interpretation = "Weak clustering"
            else:
                interpretation = "Poor clustering"
            
            print(f"Clustering quality: {interpretation}")
            
        except Exception as e:
            print(f"Warning: Could not compute silhouette score: {e}")
            self.validation_metrics['clustering_silhouette_score'] = np.nan


    def _compute_histogram_variance_captured(self):
        """ Compute variance explained by clustering."""


        cols = [c for c in self.metrics_history.columns if c.startswith('bin_')]
        data = self.metrics_history[cols].values  # (N, B)
        labels = self.metrics_history['state'].values

        # 1) Global mean
        global_mean = data.mean(axis=0)

        # 2) TSS
        TSS = np.sum((data - global_mean)**2)

        # 3) WSS
        WSS = 0.0
        for k in range(self.n_states):
            mask = labels == k
            if mask.sum() == 0:
                continue
            cluster_data = data[mask]
            cluster_mean = cluster_data.mean(axis=0)
            WSS += np.sum((cluster_data - cluster_mean)**2)

        # 4) R^2
        R2 = 1 - WSS / TSS if TSS > 0 else 0.0
        self.validation_metrics['variance_explained_by_clustering'] = R2
        print(f"Variance explained by clustering: {R2:.1%}")


    def identify_stability_states(self) -> np.ndarray:
        """ Identify stability states using GMM clustering on stride histograms.
        """
        cols = [c for c in self.metrics_history.columns if c.startswith('bin_')]
        X    = self.scaler.fit_transform(self.metrics_history[cols].values)
        self.gmm = GaussianMixture(n_components=self.n_states,
                                   covariance_type='full',
                                   random_state=42)
        raw = self.gmm.fit_predict(X)

        # order clusters by ascending distance from origin
        norms = np.linalg.norm(self.gmm.means_, axis=1)
        order = np.argsort(norms)
        mapping = {old:new for new,old in enumerate(order)}
        states = np.array([mapping[r] for r in raw])

        self.metrics_history['state'] = states
        
        # Compute validation metrics
        self._compute_gmm_reconstruction_error()
        self._compute_clustering_silhouette_score(X, states)
        self._compute_histogram_variance_captured()
        
        return states

    def compute_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """ Compute transition matrix from state sequence.
        """
        T = np.zeros((self.n_states, self.n_states))
        for a,b in zip(states[:-1], states[1:]):
            T[a,b] += 1
        row_sums = T.sum(axis=1); row_sums[row_sums==0]=1
        T /= row_sums[:,None]
        self.transition_matrix = T
        return T

    def compute_stationary_distribution(self) -> np.ndarray:
        """ Compute stationary distribution from transition matrix."""
        vals, vecs = eig(self.transition_matrix.T)
        idx        = np.argmin(np.abs(vals-1))
        pi         = np.real(vecs[:,idx])
        pi /= pi.sum()
        self.stationary_dist = np.abs(pi)
        return self.stationary_dist

    def compute_mean_dwell_times(self) -> np.ndarray:
        """ Compute mean dwell times in each state."""
        τ = np.zeros(self.n_states)
        for i in range(self.n_states):
            p = self.transition_matrix[i,i]
            τ[i] = np.inf if p>=1 else 1/(1-p)
        return τ

    def compute_stability_metrics(self) -> dict:
        """ Compute stability metrics."""
        df  = self.metrics_history
        out = {'patient_id':self.patient_id}
        for i in range(self.n_states):
            out[f'frac_state_{i}'] = (df.state==i).mean()
        
        # Add validation metrics
        out.update(self.validation_metrics)
        
        return out

    def visualize_patient_analysis(self):
        """summary: average histograms & transitions with validation metrics."""
        df    = self.metrics_history.copy()
        names = list(self.state_labels.values())
        colors= sns.color_palette(n_colors=self.n_states)

        fig, axes = plt.subplots(2,2, figsize=(12,10))
        
        # 1) Average histogram
        for st in range(self.n_states):
            sub = df[df.state==st]
            cols= [c for c in sub.columns if c.startswith('bin_')]
            mean_hist = sub[cols].mean().values
            centers   = np.linspace(0,1,self.n_bins)
            axes[0,0].plot(centers, mean_hist, label=names[st], color=colors[st])
        axes[0,0].set_title("Average Distribution by State")
        axes[0,0].legend()
        
        # Add validation metrics as text
        if 'gmm_reconstruction_rmse' in self.validation_metrics:
            rmse = self.validation_metrics['gmm_reconstruction_rmse']
            if not np.isnan(rmse):
                axes[0,0].text(0.02, 0.98, f'Reconstruction RMSE: {rmse:.4f}', 
                              transform=axes[0,0].transAxes, va='top', 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2) Transition heatmap
        sns.heatmap(self.transition_matrix, annot=True, fmt=".2f",
                    xticklabels=names, yticklabels=names,
                    cmap="Blues", ax=axes[0,1])
        axes[0,1].set_title("Transition Probabilities")
        
        # Add silhouette score
        if 'clustering_silhouette_score' in self.validation_metrics:
            sil_score = self.validation_metrics['clustering_silhouette_score']
            if not np.isnan(sil_score):
                axes[0,1].text(0.02, 0.98, f'Silhouette Score: {sil_score:.3f}', 
                              transform=axes[0,1].transAxes, va='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3) Stationary dist
        axes[1,0].bar(names, self.stationary_dist, color=colors)
        axes[1,0].set_ylim(0,1)
        axes[1,0].set_title("Stationary Distribution")
        for i,v in enumerate(self.stationary_dist):
            axes[1,0].text(i, v+0.02, f"{v:.1%}", ha='center')
        
        # Add variance explained
        if 'variance_explained_by_clustering' in self.validation_metrics:
            var_exp = self.validation_metrics['variance_explained_by_clustering']
            if not np.isnan(var_exp):
                axes[1,0].text(0.02, 0.98, f'Variance Explained: {var_exp:.1%}', 
                              transform=axes[1,0].transAxes, va='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4) Transition arrows
        from_, to_ = [], []
        for a,b in zip(df.state[:-1], df.state[1:]):
            from_.append(a); to_.append(b)
        counts = pd.DataFrame({'from':from_,'to':to_}).value_counts().reset_index()
        for _, row in counts.iterrows():
            a,b,c = row['from'], row['to'], row[0]
            axes[1,1].arrow(a, b, 0.2,0.2,
                           width=c/len(from_)*0.5,
                           color='navy', alpha=0.7)
        axes[1,1].set_xlim(-0.5,self.n_states-0.5)
        axes[1,1].set_ylim(-0.5,self.n_states-0.5)
        axes[1,1].set_xticks(range(self.n_states))
        axes[1,1].set_yticks(range(self.n_states))
        axes[1,1].set_xticklabels(names)
        axes[1,1].set_yticklabels(names)
        axes[1,1].set_title("State Transition Patterns")

        plt.tight_layout()
        plt.show()

    def fit(self, signal: np.ndarray, events: np.ndarray, fs: float):
        """
        Runs the pipeline:
          1) Histogram per stride
          2) GMM clustering
          3) MSM construction
        """
        print(f"Processing patient {self.patient_id} using stride-based histograms…")
        self.compute_stride_metrics(signal, events, fs)
        print("Identifying stability states via GMM…")
        states = self.identify_stability_states()
        print("Computing transition matrix…")
        self.compute_transition_matrix(states)
        print("Computing stationary distribution…")
        self.compute_stationary_distribution()
        metrics = self.compute_stability_metrics()
        
        print(f"\nStability metrics for {self.patient_id}:")
        for key, value in metrics.items():
            if key != 'patient_id':
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"  {key}: {value:.3f}")
        
        # Print validation summary
        print(f"\nValidation Summary:")
        if 'gmm_reconstruction_rmse' in self.validation_metrics:
            rmse = self.validation_metrics['gmm_reconstruction_rmse']
            if not np.isnan(rmse):
                print(f"  GMM reconstruction quality: RMSE = {rmse:.4f}")
        if 'clustering_silhouette_score' in self.validation_metrics:
            sil = self.validation_metrics['clustering_silhouette_score']
            if not np.isnan(sil):
                print(f"  Clustering quality: Silhouette = {sil:.3f}")
        if 'variance_explained_by_clustering' in self.validation_metrics:
            var_exp = self.validation_metrics['variance_explained_by_clustering']
            if not np.isnan(var_exp):
                print(f"  Variance explained by clustering: {var_exp:.1%}")
        
        return self
    

    def plot_gmm_reconstruction_error_distribution(self, bins=20):
        """ Plot distribution of GMM reconstruction RMSE per stride."""
        errs = self.validation_metrics.get('gmm_reconstruction_rmse_per_stride')
        if errs is None:
            print("No per‑stride GMM RMSE data.")
            return
        plt.figure(figsize=(8,4))
        sns.histplot(errs, bins=bins, kde=True)
        plt.title("Per‑Stride GMM Reconstruction RMSE")
        plt.xlabel("RMSE")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
            
