"""Book-style continual learning benchmark.

This benchmark generates a ~N-word prose "book" from a structured semantic
world, then tests two learning loops on the same chapter relation stream:

1. blind AdamW shared-operator fine-tuning
2. latent-geometry candidate reasoning with a learned action policy

The model still receives extracted fact supervision from the book metadata.
That is deliberate for this stage: it isolates continual-learning dynamics
from the separate problem of open-ended information extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import tiny_lm_geometry_reasoner as geom


def load_world(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "groups",
        "parent_categories",
        "owners",
        "colors",
        "habitats",
        "locations",
        "access_targets",
        "train_streams",
        "test_streams",
        "templates",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Book world spec missing key(s): {missing}.")
    return raw


def validate_nonempty_unique(name: str, values: list[str]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates.")


def build_book_spec(raw: dict[str, Any]) -> geom.SemanticSpec:
    groups = {str(group): [str(item) for item in items] for group, items in raw["groups"].items()}
    for group, items in groups.items():
        validate_nonempty_unique(f"groups.{group}", items)
    group_names = list(groups)
    parent_categories = {str(key): str(value) for key, value in raw["parent_categories"].items()}
    owners = [str(item) for item in raw["owners"]]
    colors = [str(item) for item in raw["colors"]]
    habitats = [str(item) for item in raw["habitats"]]
    locations = [str(item) for item in raw["locations"]]
    access_targets = [str(item) for item in raw["access_targets"]]
    for name, values in [
        ("owners", owners),
        ("colors", colors),
        ("habitats", habitats),
        ("locations", locations),
        ("access_targets", access_targets),
    ]:
        validate_nonempty_unique(name, values)

    input_domain: list[str] = []
    item_to_group: dict[str, str] = {}
    for group, items in groups.items():
        for item in items:
            input_domain.append(item)
            item_to_group[item] = group
    taxonomy_nodes = sorted(set(group_names) | set(parent_categories) | set(parent_categories.values()))
    for node in taxonomy_nodes:
        if node not in input_domain:
            input_domain.append(node)
    validate_nonempty_unique("input_domain", input_domain)

    parent: dict[str, str] = {}
    for token in input_domain:
        if token in item_to_group:
            parent[token] = item_to_group[token]
        elif token in parent_categories:
            parent[token] = parent_categories[token]
        else:
            raise KeyError(f"No parent mapping for input-domain token {token!r}.")

    def compose(mapping: dict[str, str], token: str, depth: int) -> str:
        value = token
        for _ in range(depth):
            if value not in mapping:
                raise KeyError(f"Cannot compose parent mapping through missing token {value!r}.")
            value = mapping[value]
        return value

    relations: dict[str, dict[str, str]] = {
        "COPY": {token: token for token in input_domain},
        "PARENT": parent,
        "GRANDPARENT": {token: compose(parent, token, 2) for token in input_domain},
        "PARENT3": {token: compose(parent, token, 3) for token in input_domain},
    }
    cyclic_sources = {token: index for index, token in enumerate(input_domain)}
    relations["COLOR"] = {token: colors[index % len(colors)] for token, index in cyclic_sources.items()}
    relations["HABITAT"] = {token: habitats[index % len(habitats)] for token, index in cyclic_sources.items()}
    relations["OWNER"] = {token: owners[index % len(owners)] for token, index in cyclic_sources.items()}
    relations["LOCATION"] = {token: locations[index % len(locations)] for token, index in cyclic_sources.items()}
    relations["ACCESS"] = {token: access_targets[index % len(access_targets)] for token, index in cyclic_sources.items()}

    train_streams = tuple(tuple(str(item).upper() for item in stream) for stream in raw["train_streams"])
    test_streams = tuple(tuple(str(item).upper() for item in stream) for stream in raw["test_streams"])
    geom.validate_streams(train_streams, relations, "train_streams")
    geom.validate_streams(test_streams, relations, "test_streams")
    return geom.SemanticSpec(tuple(input_domain), relations, train_streams, test_streams)


def generate_book_text(
    spec: geom.SemanticSpec,
    raw_world: dict[str, Any],
    word_target: int,
) -> tuple[str, list[dict[str, Any]]]:
    if word_target <= 0:
        raise ValueError(f"word_target must be positive, got {word_target}.")
    templates = {str(key).upper(): [str(item) for item in values] for key, values in raw_world["templates"].items()}
    for relation_name in spec.relations:
        if relation_name not in templates:
            raise KeyError(f"No prose template for relation {relation_name}.")
        if not templates[relation_name]:
            raise ValueError(f"Relation {relation_name} has no prose templates.")
    chapter_stream = list(dict.fromkeys(item.removeprefix("REPAIR_") for stream in spec.test_streams for item in stream))
    chapters: list[str] = []
    chapter_rows: list[dict[str, Any]] = []
    total_words = 0
    chapter_index = 0
    while total_words < word_target:
        relation_name = chapter_stream[chapter_index % len(chapter_stream)]
        relation = spec.relations[relation_name]
        sentences: list[str] = [f"Chapter {chapter_index + 1} records the {relation_name.lower()} relation."]
        for fact_index, source in enumerate(spec.input_domain):
            target = relation[source]
            template = templates[relation_name][fact_index % len(templates[relation_name])]
            sentences.append(template.format(source=source, target=target))
        chapter = " ".join(sentences)
        chapters.append(chapter)
        chapter_words = len(chapter.split())
        total_words += chapter_words
        chapter_rows.append(
            {
                "chapter": chapter_index + 1,
                "relation": relation_name,
                "facts": len(relation),
                "words": chapter_words,
            }
        )
        chapter_index += 1
    return "\n\n".join(chapters), chapter_rows


def train_shared_operator_adamw(
    model: geom.TinyCausalLM,
    record: geom.OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: geom.Config,
    epochs: int,
    lr: float,
) -> None:
    if epochs <= 0:
        raise ValueError(f"baseline epochs must be positive, got {epochs}.")
    if lr <= 0.0:
        raise ValueError(f"baseline lr must be positive, got {lr}.")
    optimizer = torch.optim.AdamW(record.module.parameters(), lr=lr)
    input_code = model.code(inputs).detach()
    target_code = model.code(targets).detach()
    for _ in geom.progress_range(epochs, f"blind AdamW {record.origin_task}", cfg):
        optimizer.zero_grad()
        out_code = record.module(input_code)
        logits = model.decode(out_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * geom.closure_loss(out_code, target_code)
        geom.tensor_is_finite("blind AdamW loss", loss)
        loss.backward()
        optimizer.step()
    record.update_count += 1


def run_blind_adamw_baseline(
    spec: geom.SemanticSpec,
    vocab: geom.Vocab,
    cfg: geom.Config,
    args: argparse.Namespace,
    seed: int,
    stream_index: int,
    stream: tuple[str, ...],
) -> list[dict[str, Any]]:
    model, code_token_ids, distance_scale = geom.make_run_state(spec, vocab, cfg, seed)
    shared = geom.new_operator("SHARED", cfg, "OP_SHARED_0")
    learned_tasks: list[str] = []
    rows: list[dict[str, Any]] = []
    for step_index, event in enumerate(stream):
        task_name = event.removeprefix("REPAIR_")
        prior_tasks = list(learned_tasks)
        if event.startswith("REPAIR_"):
            geom.perturb_operator(shared, cfg.repair_noise_std)
        inputs, targets = geom.make_task_data(spec, vocab, task_name, cfg.device)
        train_shared_operator_adamw(
            model,
            shared,
            inputs,
            targets,
            cfg,
            args.baseline_epochs,
            args.baseline_lr,
        )
        current = geom.evaluate_direct_operator(model, shared, inputs, targets, code_token_ids, distance_scale)
        old_accs: list[float] = []
        old_closures: list[float] = []
        for old_task in prior_tasks:
            old_inputs, old_targets = geom.make_task_data(spec, vocab, old_task, cfg.device)
            metrics = geom.evaluate_direct_operator(model, shared, old_inputs, old_targets, code_token_ids, distance_scale)
            old_accs.append(metrics["accuracy"])
            old_closures.append(metrics["closure_norm"])
        if not event.startswith("REPAIR_") and task_name not in learned_tasks:
            learned_tasks.append(task_name)
        rows.append(
            {
                "method": "blind_adamw_shared_operator",
                "seed": seed,
                "stream_index": stream_index,
                "step_index": step_index,
                "event": event,
                "task": task_name,
                "new_acc": current["accuracy"],
                "old_min_acc": geom.safe_min(old_accs, 1.0),
                "old_mean_acc": geom.safe_mean(old_accs, 1.0),
                "new_closure_norm": current["closure_norm"],
                "old_mean_closure_norm": geom.safe_mean(old_closures, 0.0),
                "operator_count": 1,
                "new_parameters_added": shared.parameter_count,
            }
        )
    return rows


def train_geometry_policy(
    spec: geom.SemanticSpec,
    vocab: geom.Vocab,
    cfg: geom.Config,
    args: argparse.Namespace,
) -> tuple[geom.LearnedActionPolicy, list[dict[str, Any]], dict[str, int]]:
    train_features: list[list[float]] = []
    train_labels: list[int] = []
    train_rows: list[dict[str, Any]] = []
    train_seed_count = args.policy_train_seed_count if args.policy_train_seed_count is not None else args.seed_count
    for seed in tqdm(range(train_seed_count), desc="book collect policy", disable=not cfg.progress, dynamic_ncols=True):
        model, code_token_ids, distance_scale = geom.make_run_state(spec, vocab, cfg, seed)
        for stream_index, stream in enumerate(spec.train_streams):
            train_rows.extend(
                geom.run_stream(
                    spec,
                    vocab,
                    model,
                    code_token_ids,
                    distance_scale,
                    stream,
                    cfg,
                    policy=None,
                    phase="book_teacher_train",
                    seed=seed,
                    stream_index=stream_index,
                    collect_features=train_features,
                    collect_labels=train_labels,
                )
            )
    policy = geom.train_policy(train_features, train_labels, cfg, args)
    return policy, train_rows, geom.count_values([geom.ACTION_NAMES[label] for label in train_labels])


def run_geometry_policy_eval(
    spec: geom.SemanticSpec,
    vocab: geom.Vocab,
    cfg: geom.Config,
    args: argparse.Namespace,
    policy: geom.LearnedActionPolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_seed_count = args.eval_seed_count if args.eval_seed_count is not None else args.seed_count
    for seed in tqdm(range(eval_seed_count), desc="book evaluate policy", disable=not cfg.progress, dynamic_ncols=True):
        actual_seed = args.test_seed_offset + seed
        model, code_token_ids, distance_scale = geom.make_run_state(spec, vocab, cfg, actual_seed)
        for stream_index, stream in enumerate(spec.test_streams):
            rows.extend(
                geom.run_stream(
                    spec,
                    vocab,
                    model,
                    code_token_ids,
                    distance_scale,
                    stream,
                    cfg,
                    policy=policy,
                    phase="book_geometry_policy",
                    seed=actual_seed,
                    stream_index=stream_index,
                )
            )
    for row in rows:
        row["method"] = "latent_geometry_policy"
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty rows.")
    final_rows: list[dict[str, Any]] = []
    keys = sorted({(row["seed"], row["stream_index"], row.get("method", row.get("phase", ""))) for row in rows})
    for seed, stream_index, method in keys:
        selected = [
            row
            for row in rows
            if row["seed"] == seed and row["stream_index"] == stream_index and row.get("method", row.get("phase", "")) == method
        ]
        final_rows.append(max(selected, key=lambda row: row["step_index"]))
    summary = {
        "new_acc": float(np.mean([row["new_acc"] for row in rows])),
        "old_min_acc": float(np.mean([row["old_min_acc"] for row in rows])),
        "new_closure_norm": float(np.mean([row["new_closure_norm"] for row in rows])),
        "final_operator_count": float(np.mean([row["operator_count"] for row in final_rows])),
        "final_new_parameters": float(np.mean([row["new_parameters_added"] for row in final_rows])),
        "events": {},
    }
    if "action_correct" in rows[0]:
        summary["action_accuracy"] = float(np.mean([float(row["action_correct"]) for row in rows]))
        summary["unsafe_choice_rate"] = float(np.mean([float(not row["safe"]) for row in rows]))
    for event in sorted({row["event"] for row in rows}):
        event_rows = [row for row in rows if row["event"] == event]
        event_summary = {
            "new_acc": float(np.mean([row["new_acc"] for row in event_rows])),
            "old_min_acc": float(np.mean([row["old_min_acc"] for row in event_rows])),
            "new_closure_norm": float(np.mean([row["new_closure_norm"] for row in event_rows])),
            "operator_count": float(np.mean([row["operator_count"] for row in event_rows])),
        }
        if "chosen_action" in event_rows[0]:
            event_summary["chosen_actions"] = geom.count_values([row["chosen_action"] for row in event_rows])
        summary["events"][event] = event_summary
    return summary


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    raw_world = load_world(args.world_json)
    spec = build_book_spec(raw_world)
    vocab = geom.make_vocab(spec)
    cfg = geom.config_from_args(args)
    book_text, chapters = generate_book_text(spec, raw_world, args.book_word_target)
    if args.book_output_txt is not None:
        if args.book_output_txt.exists():
            raise FileExistsError(f"book output already exists: {args.book_output_txt}")
        args.book_output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.book_output_txt.write_text(book_text, encoding="utf-8")

    policy, train_rows, train_action_counts = train_geometry_policy(spec, vocab, cfg, args)
    geometry_rows = run_geometry_policy_eval(spec, vocab, cfg, args, policy)

    baseline_rows: list[dict[str, Any]] = []
    eval_seed_count = args.eval_seed_count if args.eval_seed_count is not None else args.seed_count
    for seed in tqdm(range(eval_seed_count), desc="book blind AdamW baseline", disable=not cfg.progress, dynamic_ncols=True):
        actual_seed = args.test_seed_offset + seed
        for stream_index, stream in enumerate(spec.test_streams):
            baseline_rows.extend(run_blind_adamw_baseline(spec, vocab, cfg, args, actual_seed, stream_index, stream))

    report = {
        "mode": "book_continual_benchmark",
        "world_json": str(args.world_json),
        "book_word_count": len(book_text.split()),
        "chapter_count": len(chapters),
        "chapters": chapters,
        "vocab_size": vocab.size,
        "input_domain_size": len(spec.input_domain),
        "relations": sorted(spec.relations),
        "config": geom.serializable_config(args),
        "train_action_counts": train_action_counts,
        "summary": {
            "latent_geometry_policy": summarize_rows(geometry_rows),
            "blind_adamw_shared_operator": summarize_rows(baseline_rows),
        },
        "policy_train_rows": train_rows,
        "geometry_rows": geometry_rows,
        "baseline_rows": baseline_rows,
    }
    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\nBOOK CONTINUAL LEARNING BENCHMARK")
    print("=" * 132)
    print(
        f"words={report['book_word_count']} chapters={report['chapter_count']} "
        f"input_domain={report['input_domain_size']} vocab={report['vocab_size']}"
    )
    print(f"train_action_counts={report['train_action_counts']}")
    print("-" * 132)
    print(f"{'method':<32} {'new_acc':<10} {'old_min':<10} {'closure':<10} {'final_ops':<10} {'new_params':<12} {'action_acc':<10} {'unsafe':<10}")
    print("-" * 132)
    for method, summary in report["summary"].items():
        action_acc = summary.get("action_accuracy")
        unsafe = summary.get("unsafe_choice_rate")
        print(
            f"{method:<32} "
            f"{summary['new_acc']:<10.3f} {summary['old_min_acc']:<10.3f} "
            f"{summary['new_closure_norm']:<10.4f} {summary['final_operator_count']:<10.2f} "
            f"{summary['final_new_parameters']:<12.2f} "
            f"{'n/a' if action_acc is None else f'{action_acc:.3f}':<10} "
            f"{'n/a' if unsafe is None else f'{unsafe:.3f}':<10}"
        )
    print("=" * 132)


def write_json_report(report: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output JSON already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-json", type=Path, required=True)
    parser.add_argument("--book-word-target", type=int, default=5000)
    parser.add_argument("--book-output-txt", type=Path)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--policy-train-seed-count", type=int)
    parser.add_argument("--eval-seed-count", type=int)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--operator-hidden-dim", type=int, default=128)
    parser.add_argument("--base-epochs", type=int, default=500)
    parser.add_argument("--operator-epochs", type=int, default=700)
    parser.add_argument("--update-epochs", type=int, default=500)
    parser.add_argument("--shadow-operator-epochs", type=int, default=250)
    parser.add_argument("--shadow-update-epochs", type=int, default=120)
    parser.add_argument("--baseline-epochs", type=int, default=500)
    parser.add_argument("--base-lr", type=float, default=0.003)
    parser.add_argument("--operator-lr", type=float, default=0.01)
    parser.add_argument("--update-lr", type=float, default=0.005)
    parser.add_argument("--baseline-lr", type=float, default=0.005)
    parser.add_argument("--lambda-closure", type=float, default=10.0)
    parser.add_argument("--separation-margin", type=float, default=2.0)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--ae-weight", type=float, default=0.5)
    parser.add_argument("--reuse-acc-threshold", type=float, default=0.98)
    parser.add_argument("--reuse-closure-threshold", type=float, default=0.03)
    parser.add_argument("--update-closure-low", type=float, default=0.001)
    parser.add_argument("--update-closure-high", type=float, default=0.75)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--max-programs", type=int, default=100000)
    parser.add_argument("--repair-noise-std", type=float, default=0.03)
    parser.add_argument("--structural-risk-power", type=float, default=1.0)
    parser.add_argument("--structural-need-power", type=float, default=1.0)
    parser.add_argument("--policy-epochs", type=int, default=500)
    parser.add_argument("--policy-lr", type=float, default=0.01)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--policy-seed", type=int, default=12345)
    parser.add_argument("--test-seed-offset", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.world_json.exists():
        raise FileNotFoundError(f"--world-json does not exist: {args.world_json}")
    positive_ints = [
        "book_word_target",
        "seed_count",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_seq_len",
        "operator_hidden_dim",
        "base_epochs",
        "operator_epochs",
        "update_epochs",
        "shadow_operator_epochs",
        "shadow_update_epochs",
        "baseline_epochs",
        "search_depth",
        "max_programs",
        "policy_epochs",
        "policy_hidden_dim",
    ]
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ["policy_train_seed_count", "eval_seed_count"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when set.")
    positive_floats = [
        "base_lr",
        "operator_lr",
        "update_lr",
        "baseline_lr",
        "lambda_closure",
        "separation_margin",
        "separation_weight",
        "lm_weight",
        "ae_weight",
        "reuse_acc_threshold",
        "reuse_closure_threshold",
        "update_closure_low",
        "update_closure_high",
        "repair_noise_std",
        "structural_risk_power",
        "structural_need_power",
        "policy_lr",
    ]
    for name in positive_floats:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model must be divisible by --n-heads.")
    if args.update_closure_high <= args.update_closure_low:
        raise ValueError("--update-closure-high must be greater than --update-closure-low.")


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    if args.output_json is not None:
        write_json_report(report, args.output_json)
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
