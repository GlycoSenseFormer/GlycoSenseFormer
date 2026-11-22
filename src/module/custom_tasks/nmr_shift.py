from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from module.custom_data import load_train_test_data
from module.custom_models import build_nmr_model


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_interval: int = 0,
        gpu_id: int = 0,
    ) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.checkpoint_interval = checkpoint_interval
        self.derive_mean_and_std()

    def _target_mask(self, data: Data) -> torch.Tensor:
        carbon_mask = data.x[:, 0] == 2.0
        nmr_mask = data.x[:, -1] > -0.5
        return nmr_mask.logical_and(carbon_mask)

    def _forward_model(self, data: Data) -> torch.Tensor:
        return self.model(
            x=data.x[:, 0:2].long(),
            pos=data.pos.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.long(),
            batch=data.batch,
        )

    def _run_batch(self, data: Data, criterion: torch.nn.Module) -> float:
        self.optimizer.zero_grad()
        mask = self._target_mask(data)
        nmr_true = data.x[:, -1]
        nmr_masked = (nmr_true[mask] - self.mean) / self.std
        out = self._forward_model(data)
        loss = criterion(out[mask].flatten(), nmr_masked.clone())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int, criterion: torch.nn.Module) -> set[float]:
        losses = []
        for i, data in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            data = data.to(self.gpu_id)
            losses.append(self._run_batch(data, criterion))
            if (
                self.gpu_id == 0
                and self.checkpoint_interval > 0
                and i > 0
                and i % self.checkpoint_interval == 0
            ):
                self._save_checkpoint(epoch)
        return {sum(losses) / len(losses)}

    def _save_checkpoint(self, epoch: int) -> None:
        path = "_checkpoint_epoch_" + str(epoch) + ".pkl"
        torch.save(self.model.state_dict(), path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

    def derive_mean_and_std(self) -> None:
        nmrs = []
        for data in self.train_loader:
            mask = self._target_mask(data)
            nmrs.append(data.x[:, -1][mask])
        nmrs = torch.cat(nmrs)
        self.mean = nmrs.mean()
        self.std = nmrs.std()

    def train(self, max_epochs: int, criterion: torch.nn.Module, lr_reduce: float) -> None:
        for epoch in range(max_epochs):
            loss = self._run_epoch(epoch, criterion)
            print(f"Train loss {loss}")
            self._save_checkpoint(epoch)
            self.optimizer.param_groups[0]["lr"] *= lr_reduce

    def test(self, test_data: Data, message: str) -> dict[str, float | str]:
        torch.cuda.empty_cache()
        mae_criterion = torch.nn.L1Loss()
        mse_criterion = torch.nn.MSELoss()

        nmr_trues = []
        nmr_preds = []
        self.model.eval()
        with torch.no_grad():
            for loader_dict in test_data:
                n_confs = loader_dict["n_mols"]
                n_nmr = loader_dict["n_nmr"]
                for data in loader_dict["loader"]:
                    data = data.to(self.gpu_id)
                    mask = self._target_mask(data)
                    nmr_masked = data.x[:, -1][mask]
                    out = self._forward_model(data)
                    out_masked = out[mask] * self.std + self.mean
                    out = (out_masked.reshape(n_confs, n_nmr).T).mean(dim=1)
                    trues = nmr_masked[:n_nmr].detach().flatten()
                    nmr_trues.append(trues)
                    nmr_preds.append(out.detach().flatten())

            if nmr_trues:
                nmr_trues_ = torch.cat(nmr_trues)
                nmr_preds_ = torch.cat(nmr_preds)
                mae = mae_criterion(nmr_trues_.flatten(), nmr_preds_.flatten()).item()
                rmse = torch.sqrt(mse_criterion(nmr_trues_.flatten(), nmr_preds_.flatten())).item()
                residual_sum_squares = torch.sum((nmr_trues_.flatten() - nmr_preds_.flatten()) ** 2)
                total_sum_squares = torch.sum((nmr_trues_.flatten() - nmr_trues_.flatten().mean()) ** 2)
                if total_sum_squares.item() == 0:
                    r2 = float("nan")
                else:
                    r2 = (1.0 - residual_sum_squares / total_sum_squares).item()
                print(f"Test MAE is {mae}, RMSE is {rmse}, and R2 is {r2} for {message}.")
                self.model.train()
                return {"split": message, "mae": mae, "rmse": rmse, "r2": r2}

        self.model.train()
        return {"split": message, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()


def build_model_kwargs(args) -> dict:
    if args.architecture in {
        "gcn",
        "gin",
        "gat",
        "mpnn",
        "painn",
        "egnn",
        "rgcn",
        "compgcn",
        "glycanaa_atom",
        "glycanaa_hetero_atom",
    }:
        kwargs = {
            "architecture": args.architecture,
            "hidden_channels": args.baseline_hidden_channels,
            "num_layers": args.baseline_num_layers,
            "dropout": args.baseline_dropout,
            "head_hidden_channels": args.baseline_head_hidden_channels,
        }
        if args.architecture in {"mpnn", "painn", "egnn", "glycanaa_atom", "glycanaa_hetero_atom"}:
            kwargs["edge_hidden_channels"] = args.baseline_edge_hidden_channels
        if args.architecture == "gat":
            kwargs["heads"] = args.baseline_attention_heads
        if args.architecture in {"rgcn", "compgcn", "glycanaa_atom", "glycanaa_hetero_atom"}:
            kwargs["num_relations"] = args.baseline_num_relations
        return kwargs
    if args.architecture == "geqshift":
        return {
            "architecture": args.architecture,
            "n_input": args.geqshift_n_input,
            "n_node_attr": args.geqshift_n_node_attr,
            "n_output": args.geqshift_n_output,
            "n_layers": args.geqshift_n_layers,
        }
    if args.architecture == "glycangt_atom":
        return {
            "architecture": args.architecture,
            "glycangt_root": args.glycangt_root,
            "config_name": args.glycangt_config_name,
            "distance_bins": args.glycangt_distance_bins,
            "distance_cutoff": args.glycangt_distance_cutoff,
            "dropout": args.baseline_dropout,
            "head_hidden_channels": args.baseline_head_hidden_channels,
            "freeze_encoder": args.glycangt_freeze_encoder,
        }
    if args.architecture in {
        "eqv3_glycanaa_hetero_fusion",
        "eqv3_glycanaa_cross_attention_fusion",
        "eqv3_glycanaa_cross_attention_residual_fusion",
        "eqv3_glycanaa_cross_attention_identity_residual_fusion",
        "eqv3_glycanaa_cross_attention_zero_mono_token_residual_fusion",
        "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_fusion",
        "eqv3_glycanaa_cross_attention_bidirectional_residual_fusion",
        "eqv3_glycanaa_cross_attention_identity_residual_bidirectional_fusion",
        "eqv3_glycanaa_cross_attention_zero_mono_token_bidirectional_residual_fusion",
        "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_bidirectional_fusion",
        "eqv3_glycanaa_cross_attention_mono_context_residual_fusion",
        "eqv3_glycanaa_cross_attention_identity_residual_mono_context_fusion",
        "eqv3_glycanaa_gated_residual_fusion",
        "eqv3_glycanaa_atom_uncertainty_gated_fusion",
        "eqv3_glycanaa_linkage_aware_gated_fusion",
        "eqv3_glycan_chain_atom_context",
        "eqv3_glycan_chain_residual_bias",
        "eqv3_glycan_chain_feature_adapter",
        "eqv3_glycan_chain_feature_adapter_residual_bias",
        "eqv3_glycan_chain_tiny_feature_residual_bias",
        "eqv3_glycan_chain_anchored_feature_residual_bias",
        "eqv3_glycan_chain_stable_residual_bias",
        "eqv3_glycan_chain_stable_chain_only_residual_bias",
        "eqv3_glycan_chain_stable_hard_carbon_residual_bias",
        "eqv3_glycan_chain_stable_film_residual_bias",
        "eqv3_glycan_chain_stable_scalar_concat_fusion",
        "eqv3_glycan_chain_stable_scalar_residual_concat_fusion",
    }:
        return {
            "architecture": args.architecture,
            "equiformer_v3_root": args.equiformer_v3_root,
            "eqv3_num_layers": args.eqv3_num_layers,
            "eqv3_num_channels": args.eqv3_num_channels,
            "eqv3_attn_hidden_channels": args.eqv3_attn_hidden_channels,
            "eqv3_ffn_hidden_channels": args.eqv3_ffn_hidden_channels,
            "eqv3_num_radial_basis": args.eqv3_num_radial_basis,
            "eqv3_lmax": args.eqv3_lmax,
            "eqv3_mmax": args.eqv3_mmax,
            "eqv3_max_radius": args.eqv3_max_radius,
            "glycanaa_hidden_channels": args.baseline_hidden_channels,
            "glycanaa_num_layers": args.baseline_num_layers,
            "glycanaa_edge_hidden_channels": args.baseline_edge_hidden_channels,
            "glycanaa_dropout": args.baseline_dropout,
            "glycanaa_head_hidden_channels": args.baseline_head_hidden_channels,
            "glycanaa_num_relations": args.baseline_num_relations,
            "fusion_hidden_channels": args.fusion_hidden_channels,
            "fusion_dropout": args.fusion_dropout,
            "fusion_zero_init": not args.no_fusion_zero_init,
        }
    if args.architecture == "eqv3_glycanaa_mono_bias":
        return {
            "architecture": args.architecture,
            "equiformer_v3_root": args.equiformer_v3_root,
            "eqv3_num_layers": args.eqv3_num_layers,
            "eqv3_num_channels": args.eqv3_num_channels,
            "eqv3_attn_hidden_channels": args.eqv3_attn_hidden_channels,
            "eqv3_ffn_hidden_channels": args.eqv3_ffn_hidden_channels,
            "eqv3_num_radial_basis": args.eqv3_num_radial_basis,
            "eqv3_lmax": args.eqv3_lmax,
            "eqv3_mmax": args.eqv3_mmax,
            "eqv3_max_radius": args.eqv3_max_radius,
            "glycanaa_hidden_channels": args.baseline_hidden_channels,
            "glycanaa_num_layers": args.baseline_num_layers,
            "glycanaa_edge_hidden_channels": args.baseline_edge_hidden_channels,
            "glycanaa_dropout": args.baseline_dropout,
            "glycanaa_head_hidden_channels": args.baseline_head_hidden_channels,
            "glycanaa_num_relations": args.baseline_num_relations,
            "bias_hidden_channels": args.bias_hidden_channels,
            "bias_dropout": args.bias_dropout,
            "bias_zero_init": not args.no_bias_zero_init,
        }
    return {
        "architecture": args.architecture,
        "equiformer_v3_root": args.equiformer_v3_root,
        "num_layers": args.eqv3_num_layers,
        "num_channels": args.eqv3_num_channels,
        "attn_hidden_channels": args.eqv3_attn_hidden_channels,
        "ffn_hidden_channels": args.eqv3_ffn_hidden_channels,
        "num_radial_basis": args.eqv3_num_radial_basis,
        "lmax": args.eqv3_lmax,
        "mmax": args.eqv3_mmax,
        "max_radius": args.eqv3_max_radius,
    }


def run_training(
    epochs: int,
    batch_size: int,
    train_path: str,
    test_path_mo: str,
    test_path_di: str,
    test_path_tri: str,
    lr: float,
    seed: int,
    checkpoint_interval: int,
    model_kwargs: dict,
    lr_reduce: float = 0.1,
) -> None:
    set_seed(seed)
    train_set, test_data_mo, test_data_di, test_data_tri = load_train_test_data(
        train_path,
        test_path_mo,
        test_path_di,
        test_path_tri,
    )
    model_kwargs = dict(model_kwargs)
    architecture = model_kwargs.pop("architecture", "equiformer_v3")
    model = build_nmr_model(architecture=architecture, **model_kwargs)
    train_loader = prepare_dataloader(train_set, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, optimizer, checkpoint_interval)

    trainer.train(epochs, torch.nn.L1Loss(), lr_reduce)
    trainer.test(test_data_mo, "mono")
    trainer.test(test_data_di, "di")
    trainer.test(test_data_tri, "tri")
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"CUDA peak allocated GB: {peak_allocated:.3f}; peak reserved GB: {peak_reserved:.3f}.")
