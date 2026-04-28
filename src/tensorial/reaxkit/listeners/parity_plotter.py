import pathlib
from typing import Any, Final

import jraph
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import reax
from typing_extensions import override

from ... import base, gcnn
from ...gcnn import _tree
from ..utils import pylogger

__all__ = ("ParityPlotter", "GraphParityPlotter", "TensorsGraphParityPlotter")

_LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)


class ParityPlotter(reax.TrainerListener):
    """
    A TrainerListener that collects true and predicted values to create a
    parity plot at the end of each stage (Train, Validate, Test, Predict).
    """

    def __init__(
        self,
        save_dir: str | pathlib.Path = "plots/",
        fit_plot_every: int = 10,
        x_label: str = "True Values (y)",
        y_label: str = "Predicted Values (y')",
        plot_final_combined: bool = False,
    ):
        # Params
        self._save_dir: Final[pathlib.Path] = pathlib.Path(save_dir)
        self._plot_every: Final[int] = fit_plot_every
        self._x_label: Final[str] = x_label
        self._y_label: Final[str] = y_label
        self.plot_final_combined = plot_final_combined


        # State
        self._last_plotted_epoch: dict[str, int] = {}

        # Stores data for each stage: {'stage_name': (list of true y's, list of predicted y's)}
        self.data_store: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
            "train": ([], []),
            "validation": ([], []),
            "test": ([], []),
            "predict": ([], []),
        }

    def _should_collect(self, stage_name: str, epoch: int) -> bool:
        if stage_name in ("test", "predict"):
            return True
        if epoch == 0 or (epoch + 1) % self._plot_every == 0:
            return True
        last = self._last_plotted_epoch.get(stage_name, 0)
        return (epoch - last) >= self._plot_every

    def _collect_batch_data(self, stage_name: str, outputs: Any | None, batch: Any) -> bool:
        """Helper to collect true (y) and predicted (y') values."""
        # Assuming batch is a tuple (x, y) and outputs is y' (the prediction).
        # We take the second element of the batch tuple for the true value (y).
        if outputs is None:
            return False

        y_true, y_pred = self.get_target_predicted(batch, outputs)
        # _LOGGER.info(f"Collected {len(y_true)} points for {stage_name}") # Log temporaneo
        # Flatten y_true and y_pred if they are multi-dimensional (e.g., shape (batch_size, 1))
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        self.data_store[stage_name][0].append(y_true)
        self.data_store[stage_name][1].append(y_pred)

        return True

    def _plot_parity(self, stage_name: str, save_dir: pathlib.Path, epoch: int | None = None, data_to_plot: tuple[list, list] | None = None):
        """Helper to create and display the parity plot for the current data."""
        # true_y_list, pred_y_list = self.data_store[stage_name]
        true_y_list, pred_y_list = data_to_plot or self.data_store[stage_name]

        if not true_y_list:
            _LOGGER.debug("Skipping parity plot for %s: No data collected.", stage_name)
            return

        if epoch is not None:
            self._last_plotted_epoch[stage_name] = epoch

        # 1. Combine all batches into one large numpy array
        # The user requested one large output array which is passed to matplotlib.
        y_true_all = np.concatenate(true_y_list)
        y_pred_all = np.concatenate(pred_y_list)

        # 2. Clear the stage data for the next run (e.g., next 'fit' call)
        # self.data_store[stage_name] = ([], [])

        # 3. Create the Parity Plot
        _LOGGER.debug("Generating Parity Plot for %s stage...", stage_name)

        fig = plt.figure(figsize=(8, 8))

        # Determine the range for the ideal line (y=x)
        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())

        # Add a buffer for visualization
        range_buffer = (max_val - min_val) * 0.1
        plot_range = (min_val - range_buffer, max_val + range_buffer)

        # Scatter plot of True vs. Predicted
        plt.scatter(
            y_true_all, y_pred_all, alpha=0.6, label=f"{stage_name} points (N={len(y_true_all)})"
        )

        # Ideal Line (y = x)
        plt.plot(
            plot_range, plot_range, color="red", linestyle="--", label="Ideal Parity Line (y=x)"
        )

        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.title(f"Parity Plot: True vs. Predicted ({stage_name.capitalize()} Stage)")
        plt.legend()
        plt.grid(True)

        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{stage_name}_epoch_{epoch}.pdf" if epoch is not None else f"{stage_name}.pdf"
        plt.savefig(str(save_dir / filename), bbox_inches="tight")
        # full_path = save_dir / filename
        # print(f"DEBUG: Saving plot to {full_path.absolute()}")

        plt.close(fig)

        # Triggering an image for context/illustration
        _LOGGER.debug("Parity Plot for %s generated.", stage_name)

    def reset(self):
        self.data_store: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
            "train": ([], []),
            "validation": ([], []),
            "test": ([], []),
            "predict": ([], []),
        }
        self._last_plotted_epoch.clear()

    def _plot_combined_all_stages(self, trainer: "reax.Trainer", key_name: str | None = None):
        """Method to plot Train, Val e Test stages results in a single parity plot."""
        save_dir = self._get_save_dir(trainer) / "combined_final"
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        
        stage_cfg = [
            ("train", "#b2df8a", "Train"),
            ("validation", "#1f78b4", "Validation"),
            ("test", "#f19428", "Test")
        ]

        all_true, all_pred = [], []

        for stage, color, label in stage_cfg:
            true_list, pred_list = self.data_store[stage]
            if not true_list:
                continue

            if key_name is not None:
                y_true = np.concatenate([np.atleast_1d(d[key_name]).flatten() for d in true_list])
                y_pred = np.concatenate([np.atleast_1d(d[key_name]).flatten() for d in pred_list])
            else:
                y_true = np.concatenate([np.atleast_1d(x).flatten() for x in true_list])
                y_pred = np.concatenate([np.atleast_1d(x).flatten() for x in pred_list])

            ax.scatter(y_true, y_pred, c=color, alpha=0.5, s=15, label=label, edgecolors='none')
            all_true.append(y_true)
            all_pred.append(y_pred)

        if not all_true:
            plt.close(fig)
            return

        # Parity line based on all data
        combined_true = np.concatenate(all_true)
        combined_pred = np.concatenate(all_pred)
        lims = [
            min(combined_true.min(), combined_pred.min()),
            max(combined_true.max(), combined_pred.max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0, label="Ideal")

        ax.set_xlabel(f"True {key_name}")
        ax.set_ylabel(f"Predicted {key_name}")
        ax.set_title(f"Global Parity: {key_name}")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

        filename = f"final_combined_{key_name}.pdf"
        plt.savefig(save_dir / filename, bbox_inches="tight")
        plt.close(fig)


    # --- Implement Batch End Hooks to Collect Data ---

    @override
    def on_train_batch_end(
        self,
        _trainer: reax.Trainer,
        stage: reax.stages.Train,
        outputs: Any,
        batch: Any,
        _batch_idx: int,
        /,
    ) -> None:
        if self._should_collect("train", stage.epoch):
            self._collect_batch_data("train", outputs, batch)

    @override
    def on_validation_batch_end(
        self,
        _trainer: reax.Trainer,
        stage: reax.stages.Validate,
        outputs: Any,
        batch: Any,
        _batch_idx: int,
        /,
    ) -> None:
        if self._should_collect("validation", stage.epoch):
            self._collect_batch_data("validation", outputs, batch)

    @override
    def on_test_batch_end(
        self,
        _trainer: reax.Trainer,
        stage: reax.stages.Test,
        outputs: Any,
        batch: Any,
        _batch_idx: int,
        /,
    ) -> None:
        if self._should_collect("test", stage.epoch):
            self._collect_batch_data("test", outputs, batch)

    @override
    def on_predict_batch_end(
        self,
        _trainer: reax.Trainer,
        stage: reax.stages.Predict,
        outputs: Any,
        batch: Any,
        _batch_idx: int,
        /,
    ) -> None:
        if self._should_collect("predict", stage.epoch):
            self._collect_batch_data("predict", outputs, batch)

    
    # --- Implement Stage End Hooks to Trigger Plotting ---

    @override
    def on_train_end(self, trainer: reax.Trainer, stage: reax.stages.Train, /):
        """Training is ending, plot the collected training data."""
        self._plot_parity("train", self._get_save_dir(trainer), stage.epoch - 1)

    @override
    def on_validation_end(self, trainer: reax.Trainer, stage: reax.stages.Validate, /) -> None:
        """Validation has ended, plot the collected validation data."""
        self._plot_parity("validation", self._get_save_dir(trainer), stage.epoch)

    @override
    def on_fit_end(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """Fit has ended, plot the collected training data."""
        self._plot_parity("train", self._get_save_dir(trainer), stage.epoch - 1)
        self._plot_parity("validation", self._get_save_dir(trainer), stage.epoch - 1)

    @override
    def on_test_end(self, trainer, stage, /):
        """Test has ended, plot the collected test data and the combined plot if requested."""
        self._plot_parity("test", self._get_save_dir(trainer), stage.epoch)
        
        if self.plot_final_combined:
            self._plot_combined_all_stages(trainer)

    @override
    def on_predict_end(self, trainer: reax.Trainer, stage: reax.stages.Predict, /) -> None:
        """Predict is ending, plot the collected prediction data."""
        self._plot_parity("predict", self._get_save_dir(trainer), stage.epoch)

    
    # --- Implement Stage End Hooks to clean stored stage data ---
    
    @override
    def on_train_start(self, trainer, stage, /):
        # Clean train data
        self.data_store["train"] = ([], [])

    @override
    def on_validation_start(self, trainer, stage, /):
        # Clean validation data
        self.data_store["validation"] = ([], [])

    @override
    def on_test_start(self, trainer, stage, /):
        # Clean test data
        self.data_store["test"] = ([], [])

    def get_target_predicted(self, batch, outputs) -> tuple[np.ndarray, np.ndarray]:
        targets, predictions = self._get_target_predicted(batch, outputs)
        return np.array(base.as_array(targets)), np.array(base.as_array(predictions))

    def _get_target_predicted(self, batch, outputs) -> tuple[Any, Any]:
        if isinstance(outputs, dict):
            if isinstance(batch, tuple):
                targets = self._get_batch_targets(batch)
            else:
                targets = outputs["targets"]
            predictions = outputs["predictions"]

            return targets, predictions

        targets = batch[1] if len(batch) > 1 and batch[1] is not None else batch[0]
        predictions = outputs
        return targets, predictions

    def _get_batch_targets(self, batch: tuple) -> Any:
        if len(batch) == 1:
            return batch[0]

        if batch[1] is None:
            return batch[0]

        return batch[1]

    def _get_save_dir(self, trainer: reax.Trainer) -> pathlib.Path:
        if self._save_dir.is_absolute():
            return self._save_dir

        return trainer.log_dir / self._save_dir


class GraphParityPlotter(ParityPlotter):
    def __init__(
        self,
        targets: gcnn.typing.TreePathLike,
        predictions: gcnn.typing.TreePathLike | None = None,
        save_dir: str | pathlib.Path = "plots/",
        fit_plot_every: int = 100,
        x_label: str | None = None,
        y_label: str | None = None,
        **kwargs,
    ):
        target_path = gcnn.utils.path_from_str(targets)
        prediction_path = self._init_prediction_path(predictions, target_path)

        x_label = x_label or f"True Values ({target_path[-1]})"
        y_label = y_label or f"Predicted Values ({prediction_path[-1]})"

        super().__init__(
            save_dir=save_dir,
            fit_plot_every=fit_plot_every,
            x_label=x_label,
            y_label=y_label,
            **kwargs, 
        )
        self._target_path = target_path
        self._prediction_path = prediction_path

    @staticmethod
    def _init_prediction_path(
        prediction_path, target_path: gcnn.typing.TreePath
    ) -> gcnn.typing.TreePath:
        if prediction_path is not None:
            return prediction_path

        path = list(target_path)
        path[-1] = gcnn.keys.predicted(path[-1])
        return tuple(path)

    @override
    def _get_target_predicted(
        self, batch: tuple[jraph.GraphsTuple, jraph.GraphsTuple | None], outputs: jraph.GraphsTuple
    ) -> tuple[Any, Any]:
        targets_graph, predictions_graph = super()._get_target_predicted(batch, outputs)

        targets = _tree.get(targets_graph, self._target_path)
        predictions = _tree.get(predictions_graph, self._prediction_path)

        mask_path = self._target_path[:-1] + ("mask",)
        try:
            mask = np.array(_tree.get(targets_graph, mask_path))
        except KeyError:
            pass
        else:
            targets = np.array(base.as_array(targets))
            predictions = np.array(base.as_array(predictions))
            targets = targets[mask]
            predictions = predictions[mask]

        return targets, predictions

class TensorsGraphParityPlotter(GraphParityPlotter):
    """
    Computes selected scalars from true/predicted tensors and generates a parity plot for each.
    """
    def __init__(
        self,
        targets: str = "nodes.nmr_tensors",
        predictions: str = "nodes.predicted_nmr_tensors",
        scalar_keys: list[str] | None = None,
        save_dir: str | pathlib.Path = "parity_plots",
        fit_plot_every: int = 100,
        **kwargs
    ):
        super().__init__(
            targets=targets, 
            predictions=predictions, 
            save_dir=save_dir, 
            fit_plot_every=fit_plot_every, 
            **kwargs
        )

        self.scalar_keys = scalar_keys or [
            "sigma xx",
            "sigma yy",
            "sigma zz",
            "sigma iso",
            "delta sigma",
            "eta",
            "frobenius norm",
            "symmetric part",
            "antisymmetric part",
            "eigenvalues",
        ]       

    def _compute_all_scalars(self, tensors: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Compute all requested scalars."""
        results = {}
        
        if "symmetric part" in self.scalar_keys:
            # (T + T.T) / 2
            results["symmetric part"] = (tensors + tensors.swapaxes(-1, -2)) / 2.0
            
        if "antisymmetric part" in self.scalar_keys:
            # (T - T.T) / 2
            results["antisymmetric part"] = (tensors - tensors.swapaxes(-1, -2)) / 2.0

        sym_tensors = (tensors + tensors.swapaxes(-1, -2)) / 2.0
        
        # Check if autoval are needed
        needs_eig = any(k in self.scalar_keys for k in [
            "sigma xx", "sigma yy", "sigma zz", "sigma iso", 
            "delta sigma", "eta", "eigenvalues"
        ])

        if needs_eig:
            eigvals = jnp.linalg.eigvalsh(sym_tensors)
            #  zz >= yy >= xx
            eigvals_sorted = jnp.sort(eigvals, axis=-1)[:, ::-1] 
            
            s_zz = eigvals_sorted[..., 0]
            s_yy = eigvals_sorted[..., 1]
            s_xx = eigvals_sorted[..., 2]
            s_iso = (s_zz + s_yy + s_xx) / 3.0

            if "eigenvalues" in self.scalar_keys:
                results["eigenvalues"] = eigvals_sorted
            if "sigma zz" in self.scalar_keys:
                results["sigma zz"] = s_zz
            if "sigma yy" in self.scalar_keys:
                results["sigma yy"] = s_yy
            if "sigma xx" in self.scalar_keys:
                results["sigma xx"] = s_xx
            if "sigma iso" in self.scalar_keys:
                results["sigma iso"] = s_iso
            if "delta sigma" in self.scalar_keys:
                # Δσ = σzz - (σxx + σyy)/2
                results["delta sigma"] = s_zz - (s_xx + s_yy) / 2.0
            if "eta" in self.scalar_keys:
                # η = (σxx - σyy) / (σzz - σiso)
                denominator = s_zz - s_iso
                results["eta"] = jnp.where(jnp.abs(denominator) > 1e-6, (s_xx - s_yy) / denominator, 0.0)

        if "frobenius norm" in self.scalar_keys:
            results["frobenius norm"] = jnp.linalg.norm(tensors, axis=(-2, -1))
            
        return results

    @override
    def _collect_batch_data(self, stage_name: str, outputs: Any | None, batch: Any) -> bool:
        """
        Modified function to collect scalars dictionary.
        """
        if outputs is None: return False

        tensors_gt, tensors_pred = super()._get_target_predicted(batch, outputs)
        
        gt_dict = self._compute_all_scalars(jnp.array(tensors_gt))
        pred_dict = self._compute_all_scalars(jnp.array(tensors_pred))

        self.data_store[stage_name][0].append(gt_dict)
        self.data_store[stage_name][1].append(pred_dict)
        
        _LOGGER.info(f"Collected tensors scalars for {stage_name}")
        return True

    @override
    def _plot_parity(self, stage_name: str, save_dir: pathlib.Path, epoch: int | None = None):
        dicts_gt, dicts_pred = self.data_store[stage_name]
        if not dicts_gt:
            return

        for key in self.scalar_keys:
            y_true = [np.array(d[key]).flatten() for d in dicts_gt]
            y_pred = [np.array(d[key]).flatten() for d in dicts_pred]

            self._x_label = f"True {key}"
            self._y_label = f"Predicted {key}"

            super()._plot_parity(
                stage_name=stage_name, 
                save_dir=save_dir / key, 
                epoch=epoch,
                data_to_plot=(y_true, y_pred)
            )

    @override
    def _plot_combined_all_stages(self, trainer):
        """
        Override for tensors: modifies the single request in a loop over the requested tensor properties.
        """
        # Iteration over proeprties keys
        for key in self.scalar_keys:
            # Single call for each key
            super()._plot_combined_all_stages(trainer, key_name=key)
