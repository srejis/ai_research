"""One EXP003 seed: baseline, exact completion audit, four controlled pairs."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import time

import numpy as np
import torch

from model import TinyMLP, clone_model, optimizer
from core import (ROOT, PROJECT, json_read, json_write, digest, setup, rng_capture, rng_restore,
                  cpu, load, save, finite, BatchStream, data, evaluate, flatten, layout,
                  assign_flat, layer_vectors, cosine, CSV, XorWriter, xor_path,
                  checkpoint, sync, train_step)


def old_vector(vector, mapping):
    return np.concatenate([vector[v["start"]:v["end"]] for n, v in mapping.items() if n.startswith("fc")])


def restore(cp, cfg, device):
    model = clone_model(cp["model_state_dict"], device=device)
    opt = optimizer(model, cfg, cp["optimizer_state_dict"])
    stream = BatchStream(55000, cfg["seed"], cp["batch_stream"])
    rng_restore(cp["rng"])
    return model, opt, stream


def optimizer_equal(source, model, opt):
    original = source["optimizer_state_dict"]
    assert opt.param_groups[0]["lr"] == original["param_groups"][0]["lr"]
    for parameter_id, (name, param) in zip(original["param_groups"][0]["params"], model.original_parameters()):
        assert torch.equal(param.detach().cpu(), source["model_state_dict"][name]), name
        expected = original["state"][parameter_id]
        for key, value in expected.items():
            actual = opt.state[param][key]
            assert torch.equal(actual.detach().cpu(), value), (name, key)
    if model.growth:
        assert all(p not in opt.state for _, p in model.new_parameters())
    return True


def run(seed, run_id, device):
    cfg = json_read(ROOT / "configs/experiment.json")
    cfg["seed"] = seed
    setup(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    directory = ROOT / "results" / run_id / f"seed_{seed}"
    checkpoints = ROOT / "checkpoints" / run_id / f"seed_{seed}"
    if directory.exists() or checkpoints.exists():
        raise FileExistsError(f"Run already exists: {directory}")
    protected = {}
    for name in ("EXP001_baseline_mlp", "EXP002_single_node"):
        for p in (ROOT.parent / name).rglob("*"):
            if p.is_file():
                protected[p.relative_to(PROJECT).as_posix()] = digest(p)
    directory.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    code = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py") if p.name in {"core.py", "model.py", "run_seed.py"}}
    source = directory / "source"
    source.mkdir()
    for name in code:
        shutil.copy2(Path(__file__).parent / name, source / name)
    json_write(directory / "config.json", cfg)
    json_write(directory / "protected_previous_experiments.json", protected)
    json_write(directory / "environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "numpy": np.__version__, "deterministic_algorithms": True, "tf32": False,
        "code_sha256": code, "new_packages_installed": []})
    (directory / "requirements-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    (tx, ty, vx, vy), provenance = data(cfg, device)
    provenance["code_sha256"] = code
    json_write(directory / "data_provenance.json", provenance)
    baseline_dir = directory / "baseline"
    baseline_dir.mkdir()
    baseline_cp = checkpoints / "baseline"
    baseline_cp.mkdir()
    model = TinyMLP().to(device)
    opt = optimizer(model, cfg)
    stream = BatchStream(len(ty), seed)
    initial_val = evaluate(model, vx, vy, cfg["validation_batch_size"])
    checkpoint(baseline_cp / "step_00000.pt", model, opt, stream, 0, cfg, provenance, initial_val)
    mapping = layout(model, True)
    initial = flatten(model)
    writer = XorWriter(baseline_dir / "path", initial, cfg["trajectory_chunk_steps"])
    step_log, coarse_log = CSV(baseline_dir / "steps.csv"), CSV(baseline_dir / "validation_coarse.csv")
    coarse_log.row({"step": 0, **initial_val})
    source_flat = initial.copy()
    coarse_hit = None
    start_clock = time.perf_counter()
    cumulative_train_wall, cumulative_gpu = 0.0, 0.0
    for step in range(1, cfg["max_baseline_steps"] + 1):
        indices = stream.next(cfg["batch_size"])
        result = train_step(model, opt, tx, ty, indices, device)
        current = flatten(model)
        delta = current.astype(np.float64) - source_flat.astype(np.float64)
        update_layers = layer_vectors(delta, mapping)
        gradients = layer_vectors(result["gradient"], mapping)
        cumulative_train_wall += result["train_wall_seconds"]
        cumulative_gpu += result["gpu_event_seconds"] or 0
        row = {"step": step, "train_loss": result["train_loss"], "learning_rate": opt.param_groups[0]["lr"],
               "samples": stream.samples, "train_wall_seconds": result["train_wall_seconds"],
               "cumulative_train_wall_seconds": cumulative_train_wall, "cumulative_gpu_seconds": cumulative_gpu,
               "epoch": stream.epoch, "next_batch_cursor": stream.cursor}
        for layer in update_layers:
            row[f"{layer}_gradient_norm"] = np.linalg.norm(gradients[layer])
            row[f"{layer}_update_norm"] = np.linalg.norm(update_layers[layer])
        step_log.row(row)
        writer.append(step, current)
        source_flat = current
        if step % cfg["validation_interval"] == 0 or step == cfg["max_baseline_steps"]:
            validation = evaluate(model, vx, vy, cfg["validation_batch_size"])
            coarse_log.row({"step": step, **validation})
            checkpoint(baseline_cp / f"step_{step:05d}.pt", model, opt, stream, step, cfg, provenance, validation)
            print(f"seed={seed} baseline step={step} accuracy={validation['accuracy']:.4f}", flush=True)
            if validation["accuracy"] >= cfg["completion_accuracy"]:
                coarse_hit = step
                break
    writer.flush()
    step_log.close()
    coarse_log.close()
    baseline_search_seconds = time.perf_counter() - start_clock
    if coarse_hit is None:
        json_write(directory / "failure.json", {"reason": "95% not reached", "steps": step})
        raise RuntimeError("Baseline target not reached; branches will not run")
    states = {s: a for s, a in xor_path(baseline_dir / "path")}

    def reconstruct(target, output, validation_each_step=False):
        previous = target - target % cfg["validation_interval"]
        cp = load(baseline_cp / f"step_{previous:05d}.pt")
        local_model, local_opt, local_stream = restore(cp, cfg, device)
        reproduced = 0
        replay_rows = []
        for replay_step in range(previous + 1, target + 1):
            batch_indices = local_stream.next(cfg["batch_size"])
            train_step(local_model, local_opt, tx, ty, batch_indices, device)
            assert np.array_equal(flatten(local_model), states[replay_step]), replay_step
            reproduced += 1
            if validation_each_step:
                v = evaluate(local_model, vx, vy, cfg["validation_batch_size"])
                replay_rows.append({"step": replay_step, **v})
        final_v = evaluate(local_model, vx, vy, cfg["validation_batch_size"])
        if output:
            checkpoint(output, local_model, local_opt, local_stream, target, cfg, provenance, final_v,
                       {"reconstructed_from": previous, "bit_exact_replay_steps": reproduced})
        return local_model, local_opt, local_stream, replay_rows, final_v

    # Required replay of the interval immediately preceding first coarse hit.
    previous = max(0, coarse_hit - cfg["validation_interval"])
    cp = load(baseline_cp / f"step_{previous:05d}.pt")
    local_model, local_opt, local_stream = restore(cp, cfg, device)
    refinement = CSV(baseline_dir / "completion_refinement.csv")
    bracket_complete = None
    for replay_step in range(previous + 1, coarse_hit + 1):
        indices = local_stream.next(cfg["batch_size"])
        train_step(local_model, local_opt, tx, ty, indices, device)
        assert np.array_equal(flatten(local_model), states[replay_step])
        v = evaluate(local_model, vx, vy, cfg["validation_batch_size"])
        refinement.row({"step": replay_step, **v})
        if v["accuracy"] >= cfg["completion_accuracy"]:
            bracket_complete = replay_step
            break
    refinement.close()
    assert bracket_complete is not None
    # Check prior transients without further optimizer steps: lossless weight-path replay.
    audit_model = clone_model(load(baseline_cp / "step_00000.pt")["model_state_dict"], device=device)
    global_validation = CSV(baseline_dir / "validation_exact.csv")
    complete = None
    for audit_step in range(bracket_complete + 1):
        assign_flat(audit_model, states[audit_step])
        v = evaluate(audit_model, vx, vy, cfg["validation_batch_size"])
        global_validation.row({"step": audit_step, **v})
        if audit_step and v["accuracy"] >= cfg["completion_accuracy"]:
            complete = audit_step
            break
    global_validation.close()
    assert complete is not None
    completion_file = baseline_cp / "completion.pt"
    reconstruct(complete, completion_file)
    branch_steps = {name: math.floor(complete * fraction) for name, fraction in cfg["branch_fractions"].items()}
    if min(branch_steps.values()) < 1 or len(set(branch_steps.values())) != len(branch_steps):
        json_write(directory / "failure.json", {"reason": "insufficient distinct positive branch steps", "steps": branch_steps})
        raise RuntimeError("Branch observation interval insufficient")
    json_write(baseline_dir / "completion.json", {"S_complete": complete, "first_coarse_hit": coarse_hit,
        "first_hit_in_refined_bracket": bracket_complete, "earlier_transient_found": complete != bracket_complete,
        "global_first_confirmed": True, "branch_steps": branch_steps,
        "baseline_search_wall_seconds": baseline_search_seconds,
        "search_overshoot_steps": coarse_hit - complete,
        "definition": "first validation >=95% across every integer step, verified by lossless forward-only audit"})
    print(f"seed={seed} S_complete={complete}, branches={branch_steps}; exact global audit passed", flush=True)
    pair_summaries = {}
    for condition, source_step in branch_steps.items():
        pair_dir = directory / "branches" / condition
        pair_dir.mkdir(parents=True)
        pair_cp = checkpoints / condition
        pair_cp.mkdir()
        source_path = pair_cp / "source.pt"
        reconstruct(source_step, source_path)
        source = load(source_path)
        control = clone_model(source["model_state_dict"], device=device)
        growth = clone_model(source["model_state_dict"], True, cfg["new_neuron_seed_offset"] + seed, device)
        control_opt = optimizer(control, cfg, source["optimizer_state_dict"])
        growth_opt = optimizer(growth, cfg, source["optimizer_state_dict"])
        assert optimizer_equal(source, control, control_opt)
        assert optimizer_equal(source, growth, growth_opt)
        assert sum(p.numel() for p in growth.parameters()) - sum(p.numel() for p in control.parameters()) == 801
        rng_restore(source["rng"])
        c_stream = BatchStream(len(ty), seed, source["batch_stream"])
        g_stream = BatchStream(len(ty), seed, source["batch_stream"])
        with torch.no_grad():
            lc, lg = control(vx), growth(vx)
            max_logit_diff = (lc-lg).abs().max().item()
            assert torch.equal(lc, lg)
        initial_c, initial_g = flatten(control), flatten(growth)
        map_c, map_g = layout(control), layout(growth)
        old_mapping = layout(control, True)
        start_layers = layer_vectors(initial_c, old_mapping)
        c_path = XorWriter(pair_dir / "control_path", initial_c, cfg["trajectory_chunk_steps"])
        g_path = XorWriter(pair_dir / "growth_path", initial_g, cfg["trajectory_chunk_steps"])
        performance, directions, new_log = CSV(pair_dir / "performance.csv"), CSV(pair_dir / "directions.csv"), CSV(pair_dir / "new_neuron.csv")
        timing = {"control": {"wall": 0.0, "gpu": 0.0}, "growth": {"wall": 0.0, "gpu": 0.0}}
        batch_hash = hashlib.sha256()
        source_samples = c_stream.samples
        first_new = {"input": None, "output": None}
        for label, m, o, s in (("control", control, control_opt, c_stream), ("growth", growth, growth_opt, g_stream)):
            val = evaluate(m, vx, vy, cfg["validation_batch_size"])
            checkpoint(pair_cp / label / "initial.pt", m, o, s, source_step, cfg, provenance, val,
                       {"additional_step": 0, "source_step": source_step, "optimizer_preserved": True})
            performance.row({"branch": label, "additional_step": 0, "absolute_step": source_step,
                "train_loss": None, "train_batch_accuracy": None, "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                "samples": 0, "cumulative_train_wall_seconds": 0, "cumulative_gpu_seconds": 0,
                "forward_mac_proxy": 0, "validation_seconds": 0})
        observations = []
        pair_clock = time.perf_counter()
        previous_c, previous_g = initial_c, initial_g
        for additional in range(1, complete+1):
            c_indices, g_indices = c_stream.next(cfg["batch_size"]), g_stream.next(cfg["batch_size"])
            assert torch.equal(c_indices, g_indices)
            batch_hash.update(c_indices.numpy().tobytes())
            before_rng = torch.get_rng_state()
            results = {}
            order = ("control", "growth") if additional % 2 else ("growth", "control")
            for label in order:
                m, o = (control, control_opt) if label == "control" else (growth, growth_opt)
                results[label] = train_step(m, o, tx, ty, c_indices, device)
            assert torch.equal(before_rng, torch.get_rng_state())
            current_c, current_g = flatten(control), flatten(growth)
            assert np.isfinite(current_c).all() and np.isfinite(current_g).all()
            c_path.append(additional, current_c)
            g_path.append(additional, current_g)
            old_c, old_g = old_vector(current_c, map_c).astype(np.float64), old_vector(current_g, map_g).astype(np.float64)
            prev_old_c, prev_old_g = old_vector(previous_c, map_c).astype(np.float64), old_vector(previous_g, map_g).astype(np.float64)
            updates_c, updates_g = layer_vectors(old_c-prev_old_c, old_mapping), layer_vectors(old_g-prev_old_g, old_mapping)
            cum_c, cum_g = layer_vectors(old_c-initial_c.astype(np.float64), old_mapping), layer_vectors(old_g-initial_c.astype(np.float64), old_mapping)
            grads_c = layer_vectors(results["control"]["gradient"], old_mapping)
            grads_g = layer_vectors(results["growth"]["gradient"], old_mapping)
            for layer in updates_c:
                distance = float(np.linalg.norm(cum_g[layer]-cum_c[layer]))
                directions.row({"additional_step": additional, "absolute_step": source_step+additional, "layer": layer,
                    "update_cosine": cosine(updates_c[layer], updates_g[layer], cfg["cosine_min_norm"]),
                    "gradient_cosine": cosine(grads_c[layer], grads_g[layer], cfg["cosine_min_norm"]),
                    "control_update_norm": np.linalg.norm(updates_c[layer]), "growth_update_norm": np.linalg.norm(updates_g[layer]),
                    "control_gradient_norm": np.linalg.norm(grads_c[layer]), "growth_gradient_norm": np.linalg.norm(grads_g[layer]),
                    "cumulative_cosine": cosine(cum_c[layer], cum_g[layer], cfg["cosine_min_norm"]),
                    "path_distance": distance, "relative_path_distance": distance/(np.linalg.norm(start_layers[layer])+cfg["distance_epsilon"])})
            new = results["growth"]["new"]
            new_log.row({"additional_step": additional, **new})
            for key, name in (("input", "input_gradient_norm"), ("output", "output_gradient_norm")):
                if first_new[key] is None and new[name] > cfg["cosine_min_norm"]:
                    first_new[key] = additional
            if additional == 1:
                assert new["input_gradient_norm"] == 0 and new["input_bias_gradient"] == 0
                assert all(int(growth_opt.state[p]["step"].item()) == 1 for _, p in growth.new_parameters())
                for label, m, o in (("control", control, control_opt), ("growth", growth, growth_opt)):
                    save(pair_dir / f"{label}_first_task_gradients.pt", {n: p.grad for n, p in m.named_parameters()})
                    assert all(int(o.state[p]["step"].item()) == source_step+1 for _, p in m.original_parameters())
            validation_pair = {}
            for label in order:
                m = control if label == "control" else growth
                sync(device)
                clock = time.perf_counter()
                val = evaluate(m, vx, vy, cfg["validation_batch_size"])
                val_seconds = time.perf_counter()-clock
                validation_pair[label] = val
                timing[label]["wall"] += results[label]["train_wall_seconds"]
                timing[label]["gpu"] += results[label]["gpu_event_seconds"] or 0
                performance.row({"branch": label, "additional_step": additional, "absolute_step": source_step+additional,
                    "train_loss": results[label]["train_loss"], "train_batch_accuracy": results[label]["train_batch_accuracy"],
                    "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                    "samples": c_stream.samples-source_samples, "cumulative_train_wall_seconds": timing[label]["wall"],
                    "cumulative_gpu_seconds": timing[label]["gpu"],
                    "forward_mac_proxy": (c_stream.samples-source_samples)*(25760+(800 if label=="growth" else 0)),
                    "validation_seconds": val_seconds})
                if additional == 1 or additional == complete or additional % cfg["periodic_branch_checkpoint_steps"] == 0:
                    name = "final.pt" if additional == complete else f"additional_{additional:05d}.pt"
                    checkpoint(pair_cp / label / name, m, control_opt if label=="control" else growth_opt,
                        c_stream if label=="control" else g_stream, source_step+additional, cfg, provenance, val,
                        {"additional_step": additional, "source_step": source_step})
            observations.append({"step": additional, **{label: validation_pair[label] for label in validation_pair}})
            previous_c, previous_g = current_c, current_g
            if additional % 250 == 0 or additional == complete:
                print(f"seed={seed} {condition} +{additional}/{complete}: C={validation_pair['control']['accuracy']:.4f} G={validation_pair['growth']['accuracy']:.4f}", flush=True)
        c_path.flush()
        g_path.flush()
        performance.close()
        directions.close()
        new_log.close()
        pair_summary = {"source_step": source_step, "observation_steps": complete,
            "optimizer_state_preserved": True, "old_state_first_step": source_step+1, "new_state_first_step": 1,
            "initial_logits_exactly_equal": True, "initial_max_logit_diff": max_logit_diff,
            "batch_order_sha256": batch_hash.hexdigest(), "identical_batches_each_step": True,
            "training_rng_unchanged_by_deterministic_steps": True,
            "first_nonzero_new_gradient": first_new,
            "timing": timing, "instrumented_pair_seconds": time.perf_counter()-pair_clock,
            "final_control": observations[-1]["control"], "final_growth": observations[-1]["growth"],
            "checkpoint_directory": pair_cp.relative_to(ROOT).as_posix()}
        json_write(pair_dir / "summary.json", pair_summary)
        pair_summaries[condition] = pair_summary
    for relative, expected in protected.items():
        assert digest(PROJECT / relative) == expected, relative
    json_write(directory / "summary.json", {"seed": seed, "S_complete": complete, "branch_steps": branch_steps,
        "pairs": pair_summaries, "previous_experiments_unchanged": True, "test_data_used": False,
        "status": "TRAINING_COMPLETE_AWAITING_ANALYSIS", "baseline_search_seconds": baseline_search_seconds})
    paths = [p for p in directory.rglob("*") if p.is_file()] + [p for p in checkpoints.rglob("*") if p.is_file()]
    json_write(directory / "artifact_sha256.json", {p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
    print(f"SEED COMPLETE: {seed} {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[42,123,2026])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be one directory name")
    run(args.seed, args.run_id, args.device)
