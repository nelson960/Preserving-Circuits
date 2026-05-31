"""Shared PyTorch model architectures for the real book continual learning benchmark."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTraceAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_slots: int,
        rank: int,
        top_k: int,
        init_scale: float,
        state_update_rate: float = 0.05,
        state_decay: float = 0.99,
        initial_strength_logit: float = -4.0,
    ) -> None:
        super().__init__()
        if n_slots <= 0:
            raise ValueError("SparseTraceAdapter requires n_slots > 0.")
        if rank <= 0:
            raise ValueError("SparseTraceAdapter requires rank > 0.")
        if top_k <= 0 or top_k > n_slots:
            raise ValueError(f"SparseTraceAdapter top_k must be in [1, n_slots], got top_k={top_k}, n_slots={n_slots}.")
        if init_scale <= 0.0:
            raise ValueError("SparseTraceAdapter init_scale must be positive.")
        if not 0.0 <= state_update_rate <= 1.0:
            raise ValueError("SparseTraceAdapter state_update_rate must be in [0, 1].")
        if not 0.0 <= state_decay <= 1.0:
            raise ValueError("SparseTraceAdapter state_decay must be in [0, 1].")
        if not math.isfinite(initial_strength_logit):
            raise ValueError("SparseTraceAdapter initial_strength_logit must be finite.")
        self.n_slots = n_slots
        self.rank = rank
        self.top_k = top_k
        self.state_update_rate = state_update_rate
        self.state_decay = state_decay
        self.keys = nn.Parameter(torch.empty(n_slots, d_model))
        self.down = nn.Parameter(torch.empty(n_slots, d_model, rank))
        self.up = nn.Parameter(torch.empty(n_slots, rank, d_model))
        self.strength_logits = nn.Parameter(torch.full((n_slots,), initial_strength_logit))
        read_feature_dim = 4 * d_model
        self.read_reasoner = nn.Sequential(
            nn.LayerNorm(read_feature_dim),
            nn.Linear(read_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.source_mix_head = nn.Linear(read_feature_dim, 1)
        self.reason_gate_head = nn.Linear(read_feature_dim, 1)
        self.reason_gain_logit = nn.Parameter(torch.tensor(-2.0))
        self.reason_norm = nn.LayerNorm(d_model)
        self.state_input = nn.Linear(d_model, rank)
        self.state_cell = nn.GRUCell(rank, rank)
        state_context_dim = rank + 2
        self.pressure_head = nn.Linear(state_context_dim, 1)
        self.write_head = nn.Linear(state_context_dim, 1)
        self.residual_head = nn.Linear(state_context_dim, 1)
        self.consolidation_head = nn.Linear(state_context_dim, 1)
        self.capacity_head = nn.Linear(state_context_dim, 1)
        self.compression_head = nn.Linear(state_context_dim, 1)
        self.forget_head = nn.Linear(state_context_dim, 1)
        self.fast_read_gain = nn.Parameter(torch.tensor(1.0))
        self.routing_homeostasis_gain_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("slot_state", torch.zeros(n_slots, rank))
        self.register_buffer("slot_frequency", torch.zeros(n_slots))
        self.register_buffer("slot_write_mass", torch.zeros(n_slots))
        self.register_buffer("slot_usage_ema", torch.full((n_slots,), 1.0 / float(n_slots)))
        self.register_buffer("fast_key_delta", torch.zeros(n_slots, d_model))
        self.register_buffer("fast_memory", torch.zeros(n_slots, rank))
        self.register_buffer("fast_value_memory", torch.zeros(n_slots, d_model))
        self.register_buffer("fast_memory_strength", torch.zeros(n_slots))
        self.temperature = math.sqrt(float(d_model))
        self.last_scores: torch.Tensor | None = None
        self.last_base_scores: torch.Tensor | None = None
        self.last_top_indices: torch.Tensor | None = None
        self.last_top_gates: torch.Tensor | None = None
        self.last_pressure: torch.Tensor | None = None
        self.last_write_gate: torch.Tensor | None = None
        self.last_residual_gate: torch.Tensor | None = None
        self.last_consolidation_gate: torch.Tensor | None = None
        self.last_frequency: torch.Tensor | None = None
        self.last_state_norm: torch.Tensor | None = None
        self.last_state_delta: torch.Tensor | None = None
        self.last_token_pressure: torch.Tensor | None = None
        self.last_update_energy: torch.Tensor | None = None
        self.last_state_delta_energy: torch.Tensor | None = None
        self.last_fast_update_energy: torch.Tensor | None = None
        self.last_fast_key_delta_norm: torch.Tensor | None = None
        self.last_fast_memory_norm: torch.Tensor | None = None
        self.last_fast_value_norm: torch.Tensor | None = None
        self.last_fast_memory_strength: torch.Tensor | None = None
        self.last_write_rate: torch.Tensor | None = None
        self.last_error_pressure: torch.Tensor | None = None
        self.last_error_write_rate: torch.Tensor | None = None
        self.last_source_mix: torch.Tensor | None = None
        self.last_reason_gate: torch.Tensor | None = None
        self.last_reason_gain: torch.Tensor | None = None
        self.last_reason_update_energy: torch.Tensor | None = None
        self.last_capacity_pressure: torch.Tensor | None = None
        self.last_compression_gate: torch.Tensor | None = None
        self.last_forget_gate: torch.Tensor | None = None
        self.last_forget_rate: torch.Tensor | None = None
        self.last_fast_read_gain: torch.Tensor | None = None
        self.last_slot_usage_ema: torch.Tensor | None = None
        self.last_routing_homeostasis: torch.Tensor | None = None
        self.last_routing_homeostasis_gain: torch.Tensor | None = None
        self.last_slot_summary: torch.Tensor | None = None
        self.last_candidate_state: torch.Tensor | None = None
        nn.init.normal_(self.keys, std=1.0 / math.sqrt(float(d_model)))
        nn.init.normal_(self.down, std=init_scale)
        nn.init.normal_(self.up, std=init_scale)

    def forward(self, x: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SparseTraceAdapter input must be [batch, seq, d_model], got {x.shape}.")
        if source is None:
            source = x
        if source.shape != x.shape:
            raise ValueError(f"SparseTraceAdapter source must match x shape, got source={source.shape}, x={x.shape}.")
        read_features = torch.cat([x, source, x - source, x * source], dim=-1)
        source_mix = torch.sigmoid(self.source_mix_head(read_features))
        reason_gate = torch.sigmoid(self.reason_gate_head(read_features))
        reason_gain = F.softplus(self.reason_gain_logit)
        reason_delta = torch.tanh(self.read_reasoner(read_features))
        reason_update = source_mix * reason_gate * reason_gain * reason_delta
        reasoned = self.reason_norm(x + reason_update)
        self.last_source_mix = source_mix.detach()
        self.last_reason_gate = reason_gate.detach()
        self.last_reason_gain = reason_gain.detach()
        self.last_reason_update_energy = (reason_update ** 2).mean(dim=-1).detach()

        x_norm = F.normalize(reasoned, dim=-1)
        fast_key_snapshot = self.fast_key_delta.detach().clone()
        fast_memory_snapshot = self.fast_memory.detach().clone()
        fast_value_snapshot = self.fast_value_memory.detach().clone()
        fast_strength_snapshot = self.fast_memory_strength.detach().clone()
        effective_keys = self.keys + fast_key_snapshot
        key_norm = F.normalize(effective_keys, dim=-1)
        base_scores = torch.matmul(x_norm, key_norm.t()) * self.temperature
        self.last_base_scores = base_scores

        state_snapshot = self.slot_state.detach().clone()
        frequency_snapshot = self.slot_frequency.detach().clone()
        write_mass_snapshot = self.slot_write_mass.detach().clone()
        preliminary_probabilities = F.softmax(base_scores, dim=-1)
        slot_mass = preliminary_probabilities.sum(dim=(0, 1)).clamp_min(torch.finfo(x.dtype).eps)
        slot_summary = torch.einsum("bts,btd->sd", preliminary_probabilities, reasoned) / slot_mass.unsqueeze(-1)
        state_input = self.state_input(slot_summary)
        candidate_state = self.state_cell(state_input, state_snapshot)
        state_context = torch.cat(
            [
                candidate_state,
                frequency_snapshot.unsqueeze(-1),
                write_mass_snapshot.unsqueeze(-1),
            ],
            dim=-1,
        )
        pressure = torch.sigmoid(self.pressure_head(state_context)).squeeze(-1)
        write_gate = torch.sigmoid(self.write_head(state_context)).squeeze(-1)
        residual_gate = torch.sigmoid(self.residual_head(state_context)).squeeze(-1)
        consolidation_gate = torch.sigmoid(self.consolidation_head(state_context)).squeeze(-1)
        capacity_pressure = torch.sigmoid(self.capacity_head(state_context)).squeeze(-1)
        compression_gate = torch.sigmoid(self.compression_head(state_context)).squeeze(-1)
        forget_gate = torch.sigmoid(self.forget_head(state_context)).squeeze(-1)
        usage_snapshot = self.slot_usage_ema.detach().clone().to(dtype=x.dtype, device=x.device)
        usage_snapshot = usage_snapshot / usage_snapshot.sum().clamp_min(torch.finfo(x.dtype).eps)
        uniform_usage = torch.full_like(usage_snapshot, 1.0 / float(self.n_slots))
        routing_homeostasis = torch.log(
            (uniform_usage + torch.finfo(x.dtype).eps) / (usage_snapshot + torch.finfo(x.dtype).eps)
        )
        routing_homeostasis = routing_homeostasis - routing_homeostasis.mean()
        routing_homeostasis_gain = F.softplus(self.routing_homeostasis_gain_logit)
        routing_homeostasis = routing_homeostasis_gain * routing_homeostasis
        scores = (
            base_scores
            + torch.log(write_gate.clamp_min(torch.finfo(x.dtype).eps)).view(1, 1, -1)
            + routing_homeostasis.view(1, 1, -1)
        )
        self.last_slot_summary = slot_summary.detach()
        self.last_candidate_state = candidate_state.detach()
        self.last_scores = scores
        top_values, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        top_gates = F.softmax(top_values, dim=-1)
        flat_top_indices = top_indices.reshape(-1)
        gate_scale = torch.sigmoid(self.strength_logits).index_select(0, flat_top_indices)
        residual_scale = residual_gate.index_select(0, flat_top_indices)
        gate_scale = gate_scale.reshape_as(top_gates)
        residual_scale = residual_scale.reshape_as(top_gates)
        top_gates = top_gates * gate_scale * residual_scale
        top_pressure = pressure.index_select(0, flat_top_indices).reshape_as(top_gates)
        self.last_top_indices = top_indices.detach().cpu()
        self.last_top_gates = top_gates.detach().cpu()
        self.last_pressure = pressure
        self.last_write_gate = write_gate
        self.last_residual_gate = residual_gate
        self.last_consolidation_gate = consolidation_gate
        self.last_capacity_pressure = capacity_pressure
        self.last_compression_gate = compression_gate
        self.last_forget_gate = forget_gate
        self.last_frequency = self.slot_frequency.detach().clone()
        self.last_state_norm = self.slot_state.norm(dim=-1).detach().clone()
        state_delta = candidate_state - state_snapshot
        self.last_state_delta = state_delta.norm(dim=-1)
        self.last_state_delta_energy = (state_delta ** 2).mean(dim=-1)
        self.last_token_pressure = (top_gates * top_pressure).sum(dim=-1)
        self.last_fast_key_delta_norm = fast_key_snapshot.norm(dim=-1)
        self.last_fast_memory_norm = fast_memory_snapshot.norm(dim=-1)
        self.last_fast_value_norm = fast_value_snapshot.norm(dim=-1)
        self.last_fast_memory_strength = fast_strength_snapshot.detach().clone()
        self.last_write_rate = torch.zeros_like(pressure.detach())
        self.last_error_pressure = torch.zeros_like(pressure.detach())
        self.last_error_write_rate = torch.zeros_like(pressure.detach())
        self.last_forget_rate = torch.zeros_like(pressure.detach())
        self.last_fast_read_gain = F.softplus(self.fast_read_gain.detach())
        self.last_slot_usage_ema = usage_snapshot.detach().clone()
        self.last_routing_homeostasis = routing_homeostasis.detach().clone()
        self.last_routing_homeostasis_gain = routing_homeostasis_gain.detach().clone()

        flat_x = reasoned.reshape(-1, reasoned.shape[-1])
        selected_down = self.down.index_select(0, flat_top_indices)
        selected_up = self.up.index_select(0, flat_top_indices)
        selected_fast_memory = fast_memory_snapshot.index_select(0, flat_top_indices)
        selected_fast_value = fast_value_snapshot.index_select(0, flat_top_indices)
        repeated_x = flat_x.repeat_interleave(self.top_k, dim=0)
        hidden = torch.bmm(repeated_x.unsqueeze(1), selected_down).squeeze(1)
        hidden = F.gelu(hidden)
        param_updates = torch.bmm(hidden.unsqueeze(1), selected_up).squeeze(1)
        fast_updates = F.softplus(self.fast_read_gain) * (
            torch.bmm(selected_fast_memory.unsqueeze(1), selected_up).squeeze(1) + selected_fast_value
        )
        updates = (param_updates + fast_updates).reshape(*x.shape[:2], self.top_k, x.shape[-1])
        fast_updates = fast_updates.reshape(*x.shape[:2], self.top_k, x.shape[-1])
        update = (updates * top_gates.unsqueeze(-1)).sum(dim=2)
        fast_update = (fast_updates * top_gates.unsqueeze(-1)).sum(dim=2)
        self.last_update_energy = (update ** 2).mean(dim=-1)
        self.last_fast_update_energy = (fast_update ** 2).mean(dim=-1)
        if self.training and self.state_update_rate > 0.0:
            with torch.no_grad():
                routed_probabilities = F.softmax(scores.detach(), dim=-1)
                usage = routed_probabilities.mean(dim=(0, 1))
                usage_for_homeostasis = usage.to(dtype=self.slot_usage_ema.dtype, device=self.slot_usage_ema.device)
                self.slot_usage_ema.mul_(self.state_decay).add_(usage_for_homeostasis * (1.0 - self.state_decay))
                self.slot_usage_ema.div_(
                    self.slot_usage_ema.sum().clamp_min(torch.finfo(self.slot_usage_ema.dtype).eps)
                )
                plasticity = write_gate.detach() * (1.0 - pressure.detach() * consolidation_gate.detach())
                state_rate = (self.state_update_rate * usage * plasticity).clamp(0.0, 1.0)
                self._apply_slot_write(
                    slot_summary.detach(),
                    candidate_state.detach(),
                    usage,
                    state_rate,
                    write_gate.detach(),
                    compression_gate.detach(),
                    forget_gate.detach(),
                    capacity_pressure.detach(),
                )
                self.last_write_rate = state_rate.detach().clone()
        return x + update

    def _apply_slot_write(
        self,
        slot_summary: torch.Tensor,
        candidate_state: torch.Tensor,
        usage: torch.Tensor,
        write_rate: torch.Tensor,
        write_gate: torch.Tensor,
        compression_gate: torch.Tensor,
        forget_gate: torch.Tensor,
        capacity_pressure: torch.Tensor,
    ) -> None:
        if slot_summary.shape != self.fast_key_delta.shape:
            raise ValueError(
                f"slot_summary must match fast_key_delta shape, got {slot_summary.shape} and {self.fast_key_delta.shape}."
            )
        if candidate_state.shape != self.fast_memory.shape:
            raise ValueError(
                f"candidate_state must match fast_memory shape, got {candidate_state.shape} and {self.fast_memory.shape}."
            )
        if usage.shape != self.slot_frequency.shape:
            raise ValueError(f"usage must match slot_frequency shape, got {usage.shape} and {self.slot_frequency.shape}.")
        if write_rate.shape != self.slot_frequency.shape:
            raise ValueError(
                f"write_rate must match slot_frequency shape, got {write_rate.shape} and {self.slot_frequency.shape}."
            )
        if write_gate.shape != self.slot_frequency.shape:
            raise ValueError(
                f"write_gate must match slot_frequency shape, got {write_gate.shape} and {self.slot_frequency.shape}."
            )
        if compression_gate.shape != self.slot_frequency.shape:
            raise ValueError(
                "compression_gate must match slot_frequency shape, "
                f"got {compression_gate.shape} and {self.slot_frequency.shape}."
            )
        if forget_gate.shape != self.slot_frequency.shape:
            raise ValueError(
                f"forget_gate must match slot_frequency shape, got {forget_gate.shape} and {self.slot_frequency.shape}."
            )
        if capacity_pressure.shape != self.slot_frequency.shape:
            raise ValueError(
                "capacity_pressure must match slot_frequency shape, "
                f"got {capacity_pressure.shape} and {self.slot_frequency.shape}."
            )
        for name, tensor in {
            "slot_summary": slot_summary,
            "candidate_state": candidate_state,
            "usage": usage,
            "write_rate": write_rate,
            "write_gate": write_gate,
            "compression_gate": compression_gate,
            "forget_gate": forget_gate,
            "capacity_pressure": capacity_pressure,
        }.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Non-finite tensor in SparseTraceAdapter write path: {name}.")

        rate = write_rate.clamp(0.0, 1.0)
        rate_column = rate.unsqueeze(-1)
        forget_rate = ((1.0 - self.state_decay) * (1.0 + capacity_pressure * forget_gate * (1.0 - usage))).clamp(0.0, 1.0)
        forget_column = forget_rate.unsqueeze(-1)
        target_key_delta = F.normalize(slot_summary, dim=-1) - self.keys.detach()
        compressed_state = (1.0 - compression_gate.unsqueeze(-1)) * candidate_state + compression_gate.unsqueeze(-1) * (
            0.5 * (candidate_state + self.fast_memory.detach())
        )
        compressed_value = (1.0 - compression_gate.unsqueeze(-1)) * slot_summary + compression_gate.unsqueeze(-1) * (
            0.5 * (slot_summary + self.fast_value_memory.detach())
        )
        self.slot_state.mul_(1.0 - rate_column).add_(compressed_state * rate_column)
        self.fast_key_delta.mul_(1.0 - rate_column).add_(target_key_delta * rate_column)
        self.fast_memory.mul_(1.0 - rate_column).add_(compressed_state * rate_column)
        self.fast_value_memory.mul_(1.0 - rate_column).add_(compressed_value * rate_column)
        self.fast_key_delta.mul_(1.0 - forget_column)
        self.fast_memory.mul_(1.0 - forget_column)
        self.fast_value_memory.mul_(1.0 - forget_column)
        strength_target = usage * write_gate
        self.fast_memory_strength.mul_(1.0 - rate).add_(strength_target * rate)
        self.fast_memory_strength.mul_(1.0 - forget_rate)
        self.slot_frequency.mul_(self.state_decay).add_(usage * (1.0 - self.state_decay))
        self.slot_write_mass.mul_(self.state_decay).add_(strength_target * (1.0 - self.state_decay))
        self.last_frequency = self.slot_frequency.detach().clone()
        self.last_state_norm = self.slot_state.norm(dim=-1).detach().clone()
        self.last_fast_key_delta_norm = self.fast_key_delta.norm(dim=-1).detach().clone()
        self.last_fast_memory_norm = self.fast_memory.norm(dim=-1).detach().clone()
        self.last_fast_value_norm = self.fast_value_memory.norm(dim=-1).detach().clone()
        self.last_fast_memory_strength = self.fast_memory_strength.detach().clone()
        self.last_forget_rate = forget_rate.detach().clone()

    def apply_loss_feedback(self, token_losses: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if token_losses.ndim != 2:
            raise ValueError(f"token_losses must be [batch, seq], got {token_losses.shape}.")
        required = {
            "scores": self.last_scores,
            "slot_summary": self.last_slot_summary,
            "candidate_state": self.last_candidate_state,
            "pressure": self.last_pressure,
            "write_gate": self.last_write_gate,
            "consolidation_gate": self.last_consolidation_gate,
            "compression_gate": self.last_compression_gate,
            "forget_gate": self.last_forget_gate,
            "capacity_pressure": self.last_capacity_pressure,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"SparseTraceAdapter missing feedback diagnostics: {missing}")
        scores = self.last_scores
        slot_summary = self.last_slot_summary
        candidate_state = self.last_candidate_state
        pressure = self.last_pressure
        write_gate = self.last_write_gate
        consolidation_gate = self.last_consolidation_gate
        compression_gate = self.last_compression_gate
        forget_gate = self.last_forget_gate
        capacity_pressure = self.last_capacity_pressure
        assert scores is not None
        assert slot_summary is not None
        assert candidate_state is not None
        assert pressure is not None
        assert write_gate is not None
        assert consolidation_gate is not None
        assert compression_gate is not None
        assert forget_gate is not None
        assert capacity_pressure is not None
        if token_losses.shape != scores.shape[:2]:
            raise ValueError(f"token_losses shape {token_losses.shape} does not match trace scores {scores.shape[:2]}.")
        if mask is None:
            weights = torch.ones_like(token_losses)
        else:
            if mask.shape != token_losses.shape:
                raise ValueError(f"mask shape {mask.shape} does not match token_losses {token_losses.shape}.")
            weights = mask.detach().to(dtype=token_losses.dtype, device=token_losses.device)
        weight_sum = weights.sum()
        if weight_sum.item() <= 0.0:
            raise ValueError("SparseTraceAdapter.apply_loss_feedback received an empty mask.")
        detached_losses = token_losses.detach()
        if not torch.isfinite(detached_losses).all():
            raise FloatingPointError("Non-finite token loss in SparseTraceAdapter feedback write path.")
        probabilities = F.softmax(scores.detach(), dim=-1)
        weighted_probabilities = probabilities * weights.unsqueeze(-1)
        slot_mass = weighted_probabilities.sum(dim=(0, 1)).clamp_min(torch.finfo(probabilities.dtype).eps)
        mean_loss = (detached_losses * weights).sum() / weight_sum
        relative_loss = detached_losses / mean_loss.clamp_min(torch.finfo(detached_losses.dtype).eps)
        slot_relative_loss = torch.einsum("bts,bt->s", weighted_probabilities, relative_loss) / slot_mass
        error_pressure = slot_relative_loss / (1.0 + slot_relative_loss)
        usage = slot_mass / weight_sum
        plasticity = write_gate.detach() * error_pressure * (1.0 - pressure.detach() * consolidation_gate.detach())
        write_rate = (self.state_update_rate * usage * plasticity).clamp(0.0, 1.0)
        with torch.no_grad():
            self._apply_slot_write(
                slot_summary.detach(),
                candidate_state.detach(),
                usage.detach(),
                write_rate.detach(),
                write_gate.detach(),
                compression_gate.detach(),
                forget_gate.detach(),
                capacity_pressure.detach(),
            )
            self.last_error_pressure = error_pressure.detach().clone()
            self.last_error_write_rate = write_rate.detach().clone()

    def native_cl_terms(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        scores = self.last_scores
        if scores is None:
            raise RuntimeError("SparseTraceAdapter.native_cl_terms called before forward recorded routing scores.")
        if scores.ndim != 3:
            raise ValueError(f"Native trace scores must be [batch, seq, slots], got {tuple(scores.shape)}.")
        slot_count = int(scores.shape[-1])
        if slot_count <= 1:
            raise ValueError("Native trace terms require at least two slots.")
        probabilities = F.softmax(scores, dim=-1)
        token_entropy = -(probabilities * torch.log(probabilities.clamp_min(torch.finfo(scores.dtype).eps))).sum(dim=-1)
        normalized_entropy = token_entropy / math.log(float(slot_count))
        usage = probabilities.mean(dim=(0, 1))
        uniform = torch.full_like(usage, 1.0 / float(slot_count))

        required = {
            "pressure": self.last_pressure,
            "write_gate": self.last_write_gate,
            "residual_gate": self.last_residual_gate,
            "consolidation_gate": self.last_consolidation_gate,
            "frequency": self.last_frequency,
            "state_norm": self.last_state_norm,
            "state_delta": self.last_state_delta,
            "token_pressure": self.last_token_pressure,
            "update_energy": self.last_update_energy,
            "state_delta_energy": self.last_state_delta_energy,
            "fast_update_energy": self.last_fast_update_energy,
            "fast_key_delta_norm": self.last_fast_key_delta_norm,
            "fast_memory_norm": self.last_fast_memory_norm,
            "fast_value_norm": self.last_fast_value_norm,
            "fast_memory_strength": self.last_fast_memory_strength,
            "write_rate": self.last_write_rate,
            "error_pressure": self.last_error_pressure,
            "error_write_rate": self.last_error_write_rate,
            "source_mix": self.last_source_mix,
            "reason_gate": self.last_reason_gate,
            "reason_gain": self.last_reason_gain,
            "reason_update_energy": self.last_reason_update_energy,
            "capacity_pressure": self.last_capacity_pressure,
            "compression_gate": self.last_compression_gate,
            "forget_gate": self.last_forget_gate,
            "forget_rate": self.last_forget_rate,
            "fast_read_gain": self.last_fast_read_gain,
            "slot_usage_ema": self.last_slot_usage_ema,
            "routing_homeostasis": self.last_routing_homeostasis,
            "routing_homeostasis_gain": self.last_routing_homeostasis_gain,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"SparseTraceAdapter missing native CL diagnostics: {missing}")

        pressure = self.last_pressure
        write_gate = self.last_write_gate
        residual_gate = self.last_residual_gate
        consolidation_gate = self.last_consolidation_gate
        frequency = self.last_frequency
        state_norm = self.last_state_norm
        state_delta = self.last_state_delta
        token_pressure = self.last_token_pressure
        update_energy = self.last_update_energy
        state_delta_energy = self.last_state_delta_energy
        fast_update_energy = self.last_fast_update_energy
        fast_key_delta_norm = self.last_fast_key_delta_norm
        fast_memory_norm = self.last_fast_memory_norm
        fast_value_norm = self.last_fast_value_norm
        fast_memory_strength = self.last_fast_memory_strength
        write_rate = self.last_write_rate
        error_pressure = self.last_error_pressure
        error_write_rate = self.last_error_write_rate
        source_mix = self.last_source_mix
        reason_gate = self.last_reason_gate
        reason_gain = self.last_reason_gain
        reason_update_energy = self.last_reason_update_energy
        capacity_pressure = self.last_capacity_pressure
        compression_gate = self.last_compression_gate
        forget_gate = self.last_forget_gate
        forget_rate = self.last_forget_rate
        fast_read_gain = self.last_fast_read_gain
        slot_usage_ema = self.last_slot_usage_ema
        routing_homeostasis = self.last_routing_homeostasis
        routing_homeostasis_gain = self.last_routing_homeostasis_gain
        assert pressure is not None
        assert write_gate is not None
        assert residual_gate is not None
        assert consolidation_gate is not None
        assert frequency is not None
        assert state_norm is not None
        assert state_delta is not None
        assert token_pressure is not None
        assert update_energy is not None
        assert state_delta_energy is not None
        assert fast_update_energy is not None
        assert fast_key_delta_norm is not None
        assert fast_memory_norm is not None
        assert fast_value_norm is not None
        assert fast_memory_strength is not None
        assert write_rate is not None
        assert error_pressure is not None
        assert error_write_rate is not None
        assert source_mix is not None
        assert reason_gate is not None
        assert reason_gain is not None
        assert reason_update_energy is not None
        assert capacity_pressure is not None
        assert compression_gate is not None
        assert forget_gate is not None
        assert forget_rate is not None
        assert fast_read_gain is not None
        assert slot_usage_ema is not None
        assert routing_homeostasis is not None
        assert routing_homeostasis_gain is not None

        usage_imbalance = (usage - uniform).abs().sum().detach().clamp(0.0, 1.0)
        state_delta_pressure = (state_delta_energy.detach() / (1.0 + state_delta_energy.detach())).clamp(0.0, 1.0)
        capacity_target = usage_imbalance.expand_as(capacity_pressure)
        compression_target = state_delta_pressure
        forget_target = ((1.0 - usage.detach()) * usage_imbalance).clamp(0.0, 1.0)

        return {
            "native_slot_entropy_loss": normalized_entropy.mean(),
            "native_slot_balance_loss": ((usage - uniform) ** 2).mean() * float(slot_count),
            "native_slot_strength_loss": torch.sigmoid(self.strength_logits).mean(),
            "native_pressure_update_loss": (token_pressure * update_energy).mean(),
            "native_state_delta_loss": (pressure * state_delta_energy).mean(),
            "native_pressure_sparsity_loss": pressure.mean(),
            "native_capacity_pressure_loss": ((capacity_pressure - capacity_target) ** 2).mean(),
            "native_compression_pressure_loss": ((compression_gate - compression_target) ** 2).mean(),
            "native_forget_pressure_loss": ((forget_gate - forget_target) ** 2).mean(),
            "native_slot_entropy": normalized_entropy.detach().mean(),
            "native_slot_active_fraction": (usage.detach() > (1.0 / float(slot_count * slot_count))).float().mean(),
            "native_slot_max_share": usage.detach().max(),
            "native_slot_pressure": pressure.detach().mean(),
            "native_slot_write_gate": write_gate.detach().mean(),
            "native_slot_residual_gate": residual_gate.detach().mean(),
            "native_slot_consolidation_gate": consolidation_gate.detach().mean(),
            "native_slot_frequency": frequency.detach().mean(),
            "native_slot_state_norm": state_norm.detach().mean(),
            "native_slot_state_delta": state_delta.detach().mean(),
            "native_fast_update_energy": fast_update_energy.detach().mean(),
            "native_fast_key_delta_norm": fast_key_delta_norm.detach().mean(),
            "native_fast_memory_norm": fast_memory_norm.detach().mean(),
            "native_fast_value_norm": fast_value_norm.detach().mean(),
            "native_fast_memory_strength": fast_memory_strength.detach().mean(),
            "native_write_rate": write_rate.detach().mean(),
            "native_error_pressure": error_pressure.detach().mean(),
            "native_error_write_rate": error_write_rate.detach().mean(),
            "native_source_mix": source_mix.detach().mean(),
            "native_reason_gate": reason_gate.detach().mean(),
            "native_reason_gain": reason_gain.detach().mean(),
            "native_reason_update_energy": reason_update_energy.detach().mean(),
            "native_usage_imbalance": usage_imbalance.detach(),
            "native_capacity_pressure": capacity_pressure.detach().mean(),
            "native_compression_gate": compression_gate.detach().mean(),
            "native_forget_gate": forget_gate.detach().mean(),
            "native_forget_rate": forget_rate.detach().mean(),
            "native_fast_read_gain": fast_read_gain.detach().mean(),
            "native_slot_usage_ema_max": slot_usage_ema.detach().max(),
            "native_slot_usage_ema_min": slot_usage_ema.detach().min(),
            "native_routing_homeostasis": routing_homeostasis.detach().abs().mean(),
            "native_routing_homeostasis_gain": routing_homeostasis_gain.detach().mean(),
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.d_head))
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out(y)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.add_module("0", nn.Linear(d_model, d_ff))
        self.add_module("1", nn.GELU())
        self.add_module("2", nn.Linear(d_ff, d_model))
        self.last_fc1_pathway: torch.Tensor | None = None
        self.last_fc2_pathway: torch.Tensor | None = None

    @property
    def fc1(self) -> nn.Linear:
        layer = self._modules["0"]
        assert isinstance(layer, nn.Linear)
        return layer

    @property
    def activation(self) -> nn.GELU:
        layer = self._modules["1"]
        assert isinstance(layer, nn.GELU)
        return layer

    @property
    def fc2(self) -> nn.Linear:
        layer = self._modules["2"]
        assert isinstance(layer, nn.Linear)
        return layer

    @staticmethod
    def activation_pathway_matrix(post_activation: torch.Tensor, layer_input: torch.Tensor) -> torch.Tensor:
        if post_activation.ndim != 3 or layer_input.ndim != 3:
            raise ValueError(
                "FeedForwardBlock activation pathway tensors must both be [batch, seq, channels], "
                f"got post_activation={post_activation.shape}, layer_input={layer_input.shape}."
            )
        if post_activation.shape[:2] != layer_input.shape[:2]:
            raise ValueError(
                "FeedForwardBlock activation pathway tensors must share batch/sequence axes, "
                f"got post_activation={post_activation.shape}, layer_input={layer_input.shape}."
            )
        flat_post = post_activation.detach().reshape(-1, post_activation.shape[-1]).abs()
        flat_input = layer_input.detach().reshape(-1, layer_input.shape[-1]).abs()
        if flat_post.shape[0] <= 0:
            raise ValueError("FeedForwardBlock cannot build an activation pathway from an empty tensor.")
        pathway = torch.matmul(flat_post.t(), flat_input) / float(flat_post.shape[0])
        if not torch.isfinite(pathway).all():
            raise FloatingPointError("Non-finite activation pathway matrix in FeedForwardBlock.")
        scale = torch.quantile(pathway.reshape(-1), 0.99).clamp_min(torch.finfo(pathway.dtype).eps)
        return (pathway / scale).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_pre = self.fc1(x)
        hidden = self.activation(hidden_pre)
        out = self.fc2(hidden)
        self.last_fc1_pathway = self.activation_pathway_matrix(hidden, x)
        self.last_fc2_pathway = self.activation_pathway_matrix(out, hidden)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        trace_slots: int = 0,
        trace_rank: int = 8,
        trace_top_k: int = 2,
        trace_init_scale: float = 1e-3,
        trace_state_update_rate: float = 0.05,
        trace_state_decay: float = 0.99,
        trace_initial_strength_logit: float = -4.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, d_ff)
        self.trace_adapter = (
            SparseTraceAdapter(
                d_model,
                trace_slots,
                trace_rank,
                trace_top_k,
                trace_init_scale,
                trace_state_update_rate,
                trace_state_decay,
                trace_initial_strength_logit,
            )
            if trace_slots > 0
            else None
        )

    def forward(self, x: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        if self.trace_adapter is not None:
            x = self.trace_adapter(x, source=source)
        return x

class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        trace_slots: int = 0,
        trace_rank: int = 8,
        trace_top_k: int = 2,
        trace_init_scale: float = 1e-3,
        trace_state_update_rate: float = 0.05,
        trace_state_decay: float = 0.99,
        trace_initial_strength_logit: float = -4.0,
        trace_entropy_loss_weight: float = 0.0,
        trace_balance_loss_weight: float = 0.0,
        trace_strength_loss_weight: float = 0.0,
        trace_pressure_update_loss_weight: float = 0.0,
        trace_state_delta_loss_weight: float = 0.0,
        trace_pressure_sparsity_loss_weight: float = 0.0,
        trace_capacity_pressure_loss_weight: float = 0.0,
        trace_compression_pressure_loss_weight: float = 0.0,
        trace_forget_pressure_loss_weight: float = 0.0,
        native_trace_learning_only: bool = False,
    ) -> None:
        super().__init__()
        for name, value in {
            "trace_entropy_loss_weight": trace_entropy_loss_weight,
            "trace_balance_loss_weight": trace_balance_loss_weight,
            "trace_strength_loss_weight": trace_strength_loss_weight,
            "trace_pressure_update_loss_weight": trace_pressure_update_loss_weight,
            "trace_state_delta_loss_weight": trace_state_delta_loss_weight,
            "trace_pressure_sparsity_loss_weight": trace_pressure_sparsity_loss_weight,
            "trace_capacity_pressure_loss_weight": trace_capacity_pressure_loss_weight,
            "trace_compression_pressure_loss_weight": trace_compression_pressure_loss_weight,
            "trace_forget_pressure_loss_weight": trace_forget_pressure_loss_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if not math.isfinite(trace_initial_strength_logit):
            raise ValueError("trace_initial_strength_logit must be finite.")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.trace_entropy_loss_weight = trace_entropy_loss_weight
        self.trace_balance_loss_weight = trace_balance_loss_weight
        self.trace_strength_loss_weight = trace_strength_loss_weight
        self.trace_pressure_update_loss_weight = trace_pressure_update_loss_weight
        self.trace_state_delta_loss_weight = trace_state_delta_loss_weight
        self.trace_pressure_sparsity_loss_weight = trace_pressure_sparsity_loss_weight
        self.trace_capacity_pressure_loss_weight = trace_capacity_pressure_loss_weight
        self.trace_compression_pressure_loss_weight = trace_compression_pressure_loss_weight
        self.trace_forget_pressure_loss_weight = trace_forget_pressure_loss_weight
        self.native_trace_learning_only = native_trace_learning_only
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    trace_slots=trace_slots,
                    trace_rank=trace_rank,
                    trace_top_k=trace_top_k,
                    trace_init_scale=trace_init_scale,
                    trace_state_update_rate=trace_state_update_rate,
                    trace_state_decay=trace_state_decay,
                    trace_initial_strength_logit=trace_initial_strength_logit,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight # weight tieing
        self.configure_trainability(train_embeddings=False)

    def configure_trainability(self, *, train_embeddings: bool) -> list[nn.Parameter]:
        trace_adapters = self.trace_adapters()
        if self.native_trace_learning_only and not trace_adapters:
            raise RuntimeError("native_trace_learning_only requires at least one SparseTraceAdapter.")
        for name, parameter in self.named_parameters():
            if self.native_trace_learning_only:
                parameter.requires_grad_("trace_adapter" in name)
            elif not train_embeddings and (
                "token_embedding" in name or "position_embedding" in name or "lm_head" in name
            ):
                parameter.requires_grad_(False)
            else:
                parameter.requires_grad_(True)
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters remain after applying model trainability configuration.")
        return params

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be shape [batch, seq], got {tokens.shape}")
        batch, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
            
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
        h = self.token_embedding(tokens) + self.position_embedding(positions)
        input_stream = h
        for block in self.blocks:
            h = block(h, source=input_stream)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h

    def trace_adapters(self) -> list[SparseTraceAdapter]:
        adapters: list[SparseTraceAdapter] = []
        for block in self.blocks:
            adapter = getattr(block, "trace_adapter", None)
            if adapter is not None:
                adapters.append(adapter)
        return adapters

    def gco_mlp_pathways(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        layer_count = len(self.blocks)
        for layer_index, block in enumerate(self.blocks):
            ffn = block.ffn
            if ffn.last_fc1_pathway is None or ffn.last_fc2_pathway is None:
                raise RuntimeError("DecoderTransformer.gco_mlp_pathways called before every MLP produced pathways.")
            records.append(
                {
                    "name": f"blocks.{layer_index}.ffn.0.weight",
                    "parameter": ffn.fc1.weight,
                    "pathway": ffn.last_fc1_pathway,
                    "layer_index": layer_index,
                    "layer_count": layer_count,
                }
            )
            records.append(
                {
                    "name": f"blocks.{layer_index}.ffn.2.weight",
                    "parameter": ffn.fc2.weight,
                    "pathway": ffn.last_fc2_pathway,
                    "layer_index": layer_index,
                    "layer_count": layer_count,
                }
            )
        return records

    def native_cl_terms(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        adapters = self.trace_adapters()
        zero = reference.new_zeros(())
        keys = (
            "native_slot_entropy_loss",
            "native_slot_balance_loss",
            "native_slot_strength_loss",
            "native_pressure_update_loss",
            "native_state_delta_loss",
            "native_pressure_sparsity_loss",
            "native_capacity_pressure_loss",
            "native_compression_pressure_loss",
            "native_forget_pressure_loss",
            "native_slot_entropy",
            "native_slot_active_fraction",
            "native_slot_max_share",
            "native_slot_pressure",
            "native_slot_write_gate",
            "native_slot_residual_gate",
            "native_slot_consolidation_gate",
            "native_slot_frequency",
            "native_slot_state_norm",
            "native_slot_state_delta",
            "native_fast_update_energy",
            "native_fast_key_delta_norm",
            "native_fast_memory_norm",
            "native_fast_value_norm",
            "native_fast_memory_strength",
            "native_write_rate",
            "native_error_pressure",
            "native_error_write_rate",
            "native_source_mix",
            "native_reason_gate",
            "native_reason_gain",
            "native_reason_update_energy",
            "native_usage_imbalance",
            "native_capacity_pressure",
            "native_compression_gate",
            "native_forget_gate",
            "native_forget_rate",
            "native_fast_read_gain",
            "native_slot_usage_ema_max",
            "native_slot_usage_ema_min",
            "native_routing_homeostasis",
            "native_routing_homeostasis_gain",
        )
        if not adapters:
            weighted = (
                self.trace_entropy_loss_weight
                + self.trace_balance_loss_weight
                + self.trace_strength_loss_weight
                + self.trace_pressure_update_loss_weight
                + self.trace_state_delta_loss_weight
                + self.trace_pressure_sparsity_loss_weight
                + self.trace_capacity_pressure_loss_weight
                + self.trace_compression_pressure_loss_weight
                + self.trace_forget_pressure_loss_weight
            )
            if weighted > 0.0:
                raise RuntimeError("Native CL loss weights require at least one SparseTraceAdapter.")
            return {key: zero for key in keys}

        collected = {key: [] for key in keys}
        for adapter in adapters:
            adapter_terms = adapter.native_cl_terms(reference)
            for key in keys:
                collected[key].append(adapter_terms[key])
        return {key: torch.stack(values).mean() for key, values in collected.items()}

    def native_internal_loss(self, reference: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms = self.native_cl_terms(reference)
        loss = (
            self.trace_entropy_loss_weight * terms["native_slot_entropy_loss"]
            + self.trace_balance_loss_weight * terms["native_slot_balance_loss"]
            + self.trace_strength_loss_weight * terms["native_slot_strength_loss"]
            + self.trace_pressure_update_loss_weight * terms["native_pressure_update_loss"]
            + self.trace_state_delta_loss_weight * terms["native_state_delta_loss"]
            + self.trace_pressure_sparsity_loss_weight * terms["native_pressure_sparsity_loss"]
            + self.trace_capacity_pressure_loss_weight * terms["native_capacity_pressure_loss"]
            + self.trace_compression_pressure_loss_weight * terms["native_compression_pressure_loss"]
            + self.trace_forget_pressure_loss_weight * terms["native_forget_pressure_loss"]
        )
        return loss, terms

    def native_cl_loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        task_weight: float = 1.0,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if task_weight < 0.0:
            raise ValueError("task_weight must be non-negative.")
        if targets.shape != tokens.shape:
            raise ValueError(f"targets must match tokens shape, got targets={targets.shape}, tokens={tokens.shape}.")
        logits, hidden = self(tokens)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        if mask is None:
            task_loss = token_losses.mean()
        else:
            if mask.shape != targets.shape:
                raise ValueError(f"mask must match targets shape, got mask={mask.shape}, targets={targets.shape}.")
            denom = mask.sum()
            if denom.item() <= 0.0:
                raise ValueError("native_cl_loss received an empty mask.")
            task_loss = (token_losses * mask).sum() / denom
        if self.training and self.trace_adapters():
            feedback_mask = None if mask is None else mask.detach()
            for adapter in self.trace_adapters():
                adapter.apply_loss_feedback(token_losses.detach(), feedback_mask)
        internal_loss, native_terms = self.native_internal_loss(logits)
        total_loss = task_weight * task_loss + internal_loss
        return {
            "loss": total_loss,
            "task_loss": task_loss,
            "internal_loss": internal_loss,
            "logits": logits,
            "hidden": hidden,
            "native_terms": native_terms,
        }

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

class ClosedOperator(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        # Weight initialization
        nn.init.normal_(self.net[0].weight, std=1.0 / math.sqrt(float(d_model)))
        nn.init.constant_(self.net[0].bias, 0.01)
        nn.init.normal_(self.net[2].weight, std=1.0 / math.sqrt(float(hidden_dim)))
        nn.init.zeros_(self.net[2].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h) # Residual flow

class LearnedActionPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
