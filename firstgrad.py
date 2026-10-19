#!/usr/bin/env python3
"""
First-gradient tooling for prereg 15 Part 1 (GSM8K, unsloth/Llama-3.2-1B-Instruct).
Standalone: it imports route_sft_grpo (as R) and reuses its merge code; it changes nothing there.

One start per call: --start untrained | gold, --seed S (the SFT adapter / E2 run seed, 42/43/44), --start_mode:
  merged    (primary; route_sft_grpo --merge_sft_start) The base weights, plus the gold SFT LoRA merged into them
            (R.merge_sft_adapter: bf16 merge_and_unload on CUDA, as the run does), plus a fresh LoRA (B = 0) whose A
            is drawn by R.fresh_lora after torch.manual_seed(--lora_init_seed). That gives the same A at every
            start. RL runs come from --run_dir (the method dir holding partition.json and grpo/, with
            grpo/merged_start.json).
  continue  (secondary; prereg-13 E2) The untrained start is the base plus E2's fresh LoRA for seed S
            (transformers.set_seed(S) -> R.load_fresh_model -> get_peft_model, as TRL did). The gold start
            continues the SFT adapter (B != 0). RL runs come from runs/gsm8k/Llama-3.2-1B-Instruct-censored-<loss>/.

Stages:
  sample      --sets A B C. A/B: K samples per item from the start policy; C: from the RL run's final adapter
              -> samples_<tag>_<set>.jsonl (set C: samples_<dtag>_C.jsonl)
  directions  v0_lora, v0_raw, rand_lora, v<T> (run checkpoint-T), sft (gold) -> directions/<dtag>/<name>.safetensors
  phi         phi and the second difference along each direction, on set A
              -> phi_<dtag>_A.jsonl, features_<tag>_A.jsonl, phi_<dtag>_A_eps.json
  all         sample -> directions -> phi
Tags: tag names the start policy: untrained, gold_s<S> (continue), gold_s<S>_merged. The untrained policy is the same
in both modes (B = 0), so its A/B sets are shared. dtag adds the LoRA init: untrained_s<S> / gold_s<S> (continue),
untrained_i<I> / gold_s<S>_merged_i<I> (merged). In the phi file, direction_name = <name>_<dtag>, e.g.
v0_lora_gold_s42_merged_i42.

Conventions (to state at registration):
- Items: the 600 GSM8K test items sorted(random.Random(15).sample(range(1319), 600)). `item` is the pipeline's
  test example_id (R.prepare_data: test shuffled with data_seed 1234), as in E2's predictions*.jsonl (row // k).
- Sampling: T 0.8, top-p 0.95, top-k off, <= 512 new tokens, stop = R.stop_token_ids (EOS + end-of-turn), bf16.
  The model is built as the run and probe.py build it (bf16 weights, LoRA unmerged through PEFT), in eval mode.
  `token_ids` are the sampled ids, cut after the first stop token. `stopped` means a stop token was sampled.
  `correct_*` = R.answer_is_correct(text, gold, marked), with no stop requirement.
- Directions are per-module merged-weight changes dW_l over all 112 LoRA-targeted linears (q,k,v,o,gate,up,down x 16).
  W_l = W_start_l + s B_l A_l with s = alpha/r = 2. W_start is the base or, for merged starts, the bf16-merged SFT base.
  * v0_raw  = -G, G_l = dL/dW_l at the start. L is TRL 1.14's loss_type='grpo' at the first step (ratio 1, beta 0):
    L = -(1/N) sum_i A_i/|y_i| sum_t log pi_T(y_it | x_i, y_i<t). The rollouts are N = 256 pool prompts x 8 samples.
    A_i are TRL's group-normalised advantages: (r - nanmean_g r) / (nanstd_g r + 1e-4), Bessel std, None -> NaN -> 0.
    The reward is R.make_reward(marked, needs_stop, censor '## Step'), as in E2. Completions are cut as TRL cuts them
    (after the first tokenizer.eos_token_id, else the whole row). log pi_T uses T = 0.8, fp32, dropout off.
  * v0_lora = -s^2 (G A^T A + B B^T G): the first-order merged-weight change of one SGD step on (A, B) per unit lr
    (dL/dB = s G A^T, dL/dA = s B^T G). It is exact at B = 0, where it is -s^2 G A_0^T A_0: the same geometry at
    every merged start.
  * rand_lora = the same map applied to R_l ~ iid N(0, 1). The phi stage rescales it so that its RMS phi over set A
    equals v0_lora's (effect-size matched). The factor goes into the eps json.
  * v<T> = s (B_T A_T - B_0 A_0) from <run>/grpo/checkpoint-T (merged starts: s B_T A_T on the merged base).
    sft = s B_S A_S (the SFT LoRA displacement, gold only).
  The low-rank directions are stored as factors (dW = U V, rank <= 2r). v0_raw is stored dense, in bf16.
- phi_v(x, y) = d/de log pi_T,{W + e dW}(y | x) at e = 0 (T = 0.8, full sequence including the stop token): a central
  finite difference on the fp32 start model (TF32 off, eval mode). phi = (f+ - f-) / 2e, and
  second_diff = (f+ - 2 f0 + f-) / e^2 estimates d^2/de^2 log pi (the L2 flag). e = rel * ||W|| / ||dW|| (global
  Frobenius norms). rel is the largest value on a grid for which RMS(second-order term) / RMS(first-order term)
  < 0.1, over 32 calibration samples from set B (set A if B is missing).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.utils.checkpoint
from safetensors.torch import load_file, save_file

import route_sft_grpo as R

MODEL = "unsloth/Llama-3.2-1B-Instruct"
GOLD_SFT = "runs/gsm8k/Llama-3.2-1B-Instruct/seed_{seed}/random_sft_250/sft"
E2_ROOT = "runs/gsm8k/Llama-3.2-1B-Instruct-censored-{loss}/seed_{seed}"
METHOD = {"untrained": "full_grpo", "gold": "random_sft_250"}
TEMPERATURE, TOP_P, MAX_NEW = 0.8, 0.95, 512
N_TEST, N_ITEMS, ITEM_SEED = 1319, 600, 15
DATA_SEED, N_TRAIN = 1234, 1000
CENSOR = "## Step"
LORA = dict(no_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05)
REL_GRID = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)
PHI_FIELDS = ("item", "sample", "start", "direction_name", "phi", "second_diff", "correct_lenient", "correct_marked")


# ---------------------------------------------------------------------------
# Starts, data, models
# ---------------------------------------------------------------------------

def model_args(dtype: str = "bf16", **extra) -> SimpleNamespace:
    return SimpleNamespace(model=MODEL, dataset="gsm8k", bf16=dtype == "bf16", fp16=False, gradient_checkpointing=False,
                           **LORA, **extra)


def e2_run(start: str, seed: int, loss: str = "grpo") -> Path:
    return Path(E2_ROOT.format(loss=loss, seed=seed)) / METHOD[start]


@dataclass
class Start:
    kind: str                 # untrained | gold
    seed: int                 # SFT adapter seed (gold) / E2 run seed
    mode: str                 # merged | continue
    init_seed: int            # LoRA init seed (continue: = seed)
    sft: Optional[str]        # SFT adapter dir (gold) or None
    run_dir: Optional[Path]   # RL method dir: partition.json, grpo/, grpo/checkpoint-T (None: no run yet)

    @property
    def tag(self) -> str:
        if self.kind == "untrained":
            return "untrained"
        return f"gold_s{self.seed}" + ("_merged" if self.mode == "merged" else "")

    @property
    def dtag(self) -> str:
        if self.mode == "continue":
            return f"{self.kind}_s{self.seed}"
        return f"{self.tag}_i{self.init_seed}"


def make_start(kind: str, seed: int, mode: str = "merged", init_seed: Optional[int] = None,
               sft: Optional[str] = None, run_dir: Optional[str] = None, loss: str = "grpo") -> Start:
    sft = (sft or GOLD_SFT.format(seed=seed)) if kind == "gold" else None
    if mode == "continue":
        if init_seed not in (None, seed):
            raise ValueError("continue mode: the LoRA init is E2's, seed = run seed")
        return Start(kind, seed, mode, seed, sft, Path(run_dir) if run_dir else e2_run(kind, seed, loss))
    return Start(kind, seed, mode, seed if init_seed is None else init_seed, sft, Path(run_dir) if run_dir else None)


def test_items() -> List[int]:
    return sorted(random.Random(ITEM_SEED).sample(range(N_TEST), N_ITEMS))


def load_data(tokenizer):
    """(train, test) exactly as the route_sft_grpo runs saw them (prompt style 'train')."""
    return R.prepare_data(tokenizer, "gsm8k", N_TRAIN, -1, DATA_SEED, "train")


def set_precise():
    """fp32 matmuls at full precision: TF32 would round away the finite-difference perturbation."""
    torch.set_float32_matmul_precision("highest")
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def peft_factors(peft_model) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]:
    factors, scaling = {}, None
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            factors[name[len("base_model.model."):]] = (module.lora_A["default"].weight.detach().float().cpu().clone(),
                                                        module.lora_B["default"].weight.detach().float().cpu().clone())
            scaling = module.scaling["default"]
    return factors, scaling


def load_adapter_factors(adapter_dir) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]:
    """{HF module name: (A [r, in], B [out, r])} in fp32 and the LoRA scaling alpha / r, from a saved PEFT adapter."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{adapter_dir}: only plain LoRA is supported")
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    factors = {}
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            name = key[len("base_model.model."):-len(".lora_A.weight")]
            factors[name] = (value.float(), state[key.replace(".lora_A.", ".lora_B.")].float())
    return factors, cfg["lora_alpha"] / cfg["r"]


def init_lora_factors(seed: int):
    """E2's fresh LoRA for seed S (continue mode): run() calls transformers.set_seed(S) at the start of each method,
    then load_fresh_model (CPU, bf16), and GRPOTrainer calls get_peft_model(model, make_lora_config(args)) before
    anything else consumes torch's CPU RNG. A is PEFT's kaiming_uniform(a=sqrt(5)) draw, B = 0. Verified against E2's
    checkpoint-25 in S2_tests.py (test init)."""
    from peft import get_peft_model
    from transformers import set_seed

    args = model_args("bf16")
    set_seed(seed)
    model = R.load_fresh_model(args)
    peft_model = get_peft_model(model, R.make_lora_config(args))
    out = peft_factors(peft_model)
    del peft_model, model
    return out


def fresh_lora_factors(init_seed: int):
    """The fresh LoRA of a --merge_sft_start run, through R.fresh_lora (CPU init after torch.manual_seed(init_seed)).
    A depends only on the seed and the module shapes, not on the weights, so the base model stands in for any start
    (tested against a merged SFT model and a run's checkpoint-1 in S2_tests.py)."""
    args = model_args("bf16", lora_init_seed=init_seed)
    peft_model = R.fresh_lora(R.load_fresh_model(args), args)
    out = peft_factors(peft_model)
    del peft_model
    return out


def start_lora(st: Start):
    """(factors {name: (A, B)}, scaling) of the trainable LoRA at the start of RL."""
    if st.mode == "merged":
        return fresh_lora_factors(st.init_seed)
    if st.kind == "untrained":
        return init_lora_factors(st.seed)
    return load_adapter_factors(st.sft)


def check_run(st: Start) -> dict:
    """For merged starts: the run's grpo/merged_start.json must match this start (SFT adapter, LoRA init seed)."""
    if st.mode != "merged" or st.run_dir is None:
        return {}
    info = R.merged_start_info(st.run_dir / "grpo")
    if info is None:
        raise SystemExit(f"{st.run_dir}/grpo has no {R.MERGED_START}: not a --merge_sft_start run")
    same_sft = (info["sft_adapter"] is None and st.sft is None) or (
        info["sft_adapter"] is not None and st.sft is not None
        and Path(info["sft_adapter_abs"] or info["sft_adapter"]).resolve() == Path(st.sft).resolve())
    if not same_sft or info["lora_init_seed"] != st.init_seed or info["merge_dtype"] != "bfloat16":
        raise SystemExit(f"{st.run_dir}: merged_start.json {info} does not match start {st}")
    return info


def start_base_bf16(st: Start, device="cuda"):
    """bf16 weights the RL LoRA sits on: the base, or (merged gold) the base with the SFT LoRA merged in bf16 on CUDA
    exactly as route_sft_grpo does it (R.merge_sft_adapter)."""
    model = R.load_fresh_model(model_args("bf16"))
    if st.mode == "merged" and st.sft:
        model = R.merge_sft_adapter(model, st.sft)
    return model.to(device)


def load_sampler(st: Start, rl_adapter: Optional[str] = None):
    """The start policy (or, with rl_adapter, an RL adapter on it) in bf16 on GPU, eval mode, LoRA unmerged through
    PEFT, as route_sft_grpo / probe.py build it. A fresh merged-start LoRA has B = 0 and is omitted."""
    from peft import PeftModel

    model = start_base_bf16(st)
    if rl_adapter:
        model = PeftModel.from_pretrained(model, rl_adapter)
    elif st.mode == "continue" and st.sft:
        model = PeftModel.from_pretrained(model, st.sft)
    return model.cuda().eval()


def start_base_fp32(st: Start):
    """fp32 (exact upcast) copy of the weights the start's LoRA sits on, on GPU, eval mode, frozen."""
    set_precise()
    if st.mode == "merged" and st.sft:
        model = start_base_bf16(st).float()
    else:
        model = R.load_fresh_model(model_args("fp32")).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_start_fp32(st: Start, factors, scaling):
    """The start policy in fp32: W = W_start + s B A merged in fp32 (B = 0 everywhere but the continued gold adapter)."""
    model = start_base_fp32(st)
    with torch.no_grad():
        for name, (A, B) in factors.items():
            if bool(B.abs().max() > 0):
                model.get_submodule(name).weight.add_(B.cuda() @ A.cuda(), alpha=scaling)
    return model


def load_merged_fp32(factors, scaling):
    """fp32 base + s B A (continue-mode helper, kept for the tests)."""
    return load_start_fp32(Start("gold", 0, "continue", 0, None, None), factors, scaling)


def grpo_pool(st: Start, train) -> List[int]:
    """The start's GRPO prompt pool: the run's partition.json, else the SFT adapter's (gold: the other 750), else all."""
    for d in ([st.run_dir] if st.run_dir else []) + ([Path(st.sft).parent] if st.sft else []):
        if (d / "partition.json").exists():
            return json.loads((d / "partition.json").read_text())["grpo_example_ids"]
    if st.kind == "gold":
        raise SystemExit("gold start without a partition.json")
    return list(train["example_id"])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def batch_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


@torch.inference_mode()
def sample_ids(model, tokenizer, prompts: Sequence[str], k: int, seed: int, trl_cut: bool = False) -> List[List[int]]:
    """k sampled completions per prompt (prompt-major order) as token id lists.
    trl_cut=False: cut after the first stop token (R.stop_token_ids).
    trl_cut=True:  as TRL 1.14 GRPOTrainer cuts them, after the first tokenizer.eos_token_id; a row without one stays
    whole, including the pads after another stop token, e.g. <|end_of_text|>."""
    torch.manual_seed(seed)
    stop = R.stop_token_ids(tokenizer)
    enc = R.encode_prompts(tokenizer, list(prompts)).to("cuda")
    out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=0, num_return_sequences=k,
                         max_new_tokens=MAX_NEW, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop)
    rows = out[:, enc["input_ids"].shape[1]:].tolist()
    cut_ids = {tokenizer.eos_token_id} if trl_cut else set(stop)
    result = []
    for row in rows:
        cut = next((i for i, t in enumerate(row) if t in cut_ids), None)
        result.append(row[: cut + 1] if cut is not None else row)
    return result


def read_jsonl(path) -> List[dict]:
    rows = []
    if Path(path).exists():
        for line in Path(path).open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by preemption
                pass
    return rows


def stage_sample(args, st: Start, tokenizer, test):
    items = test_items()[: args.n_items] if args.n_items > 0 else test_items()
    by_id = {ex["example_id"]: ex for ex in test.select(items)}
    stop = set(R.stop_token_ids(tokenizer))
    timings = {}
    for set_name in args.sets:
        if set_name == "C":
            if st.run_dir is None:
                print("[sample] set C needs --run_dir (the RL run); skipped", flush=True)
                continue
            check_run(st)
            rl_adapter = str(st.run_dir / "grpo")
            row_start, out = st.dtag, Path(args.out_dir) / f"samples_{st.dtag}_C.jsonl"
        else:
            rl_adapter = None
            row_start, out = st.tag, Path(args.out_dir) / f"samples_{st.tag}_{set_name}.jsonl"
        done = {r["item"] for r in read_jsonl(out)}
        todo = [i for i in items if i not in done]
        if not todo:
            print(f"[sample] {out.name}: complete", flush=True)
            continue
        model = load_sampler(st, rl_adapter)
        n_done, t0 = 0, time.time()
        for b in range(0, len(todo), args.prompts_per_batch):
            chunk = todo[b: b + args.prompts_per_batch]
            seed = batch_seed(args.sample_seed, row_start, set_name, chunk[0], len(chunk))
            groups = sample_ids(model, tokenizer, [by_id[i]["prompt"] for i in chunk], args.k, seed)
            rows = []
            for j, ids in enumerate(groups):
                ex = by_id[chunk[j // args.k]]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                rows.append({
                    "item": ex["example_id"], "sample": j % args.k, "start": row_start, "set": set_name,
                    "completion": text,
                    "correct_marked": R.answer_is_correct(text, ex["gold_answer"], True),
                    "correct_lenient": R.answer_is_correct(text, ex["gold_answer"], False),
                    "stopped": bool(ids) and ids[-1] in stop, "n_tokens": len(ids),
                    "gold": ex["gold_answer"], "start_mode": st.mode, "sft_adapter": st.sft,
                    "rl_adapter": rl_adapter, "token_ids": ids,
                })
            with out.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            n_done += len(rows)
            print(f"[sample] {out.name}: {n_done} completions, {time.time() - t0:.0f}s", flush=True)
        timings[set_name] = {"completions": n_done, "seconds": time.time() - t0,
                             "sec_per_1k": 1000 * (time.time() - t0) / max(n_done, 1)}
        del model
        torch.cuda.empty_cache()
    return timings


# ---------------------------------------------------------------------------
# Teacher-forced log-likelihoods (batched; right padding, causal, no mask needed)
# ---------------------------------------------------------------------------

def make_batches(lengths: Sequence[int], token_budget: int) -> List[List[int]]:
    """Index batches with batch_size * max_len <= token_budget, longest first (fixed, so +e/0/-e passes match)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and new_max * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def seq_logps(model, seqs: Sequence[Tuple[List[int], List[int]]], idx: Sequence[int], temperature: float,
              entropy: bool = False, chunk: int = 4096):
    """Sum over completion tokens of log softmax(logits / T), for seqs[i] = (prompt ids, completion ids), i in idx.
    Returns a float64 tensor [len(idx)] (and the mean per-token entropy at T if asked). Differentiable when grad is
    enabled (the lm_head chunks are then checkpointed, so the [tokens, vocab] logits are not kept for backward).
    Right padding with a causal model: real positions never attend to the pads, so no mask is needed."""
    device = next(model.parameters()).device
    full = [seqs[i][0] + seqs[i][1] for i in idx]
    L = max(map(len, full))
    ids = torch.zeros((len(full), L), dtype=torch.long)
    for b, s in enumerate(full):
        ids[b, : len(s)] = torch.tensor(s)
    hidden = model.model(input_ids=ids.to(device), use_cache=False).last_hidden_state
    rows, cols, tgt, seg = [], [], [], []
    for b, i in enumerate(idx):
        p, c = len(seqs[i][0]), len(seqs[i][1])
        rows += [b] * c
        cols += range(p - 1, p + c - 1)
        tgt += seqs[i][1]
        seg += [b] * c
    rows, cols = torch.tensor(rows, device=device), torch.tensor(cols, device=device)
    tgt, seg = torch.tensor(tgt, device=device), torch.tensor(seg, device=device)
    h = hidden[rows, cols]

    def chunk_lp(hc, tc):
        logits = model.lm_head(hc).float() / temperature
        return logits.gather(1, tc[:, None])[:, 0] - torch.logsumexp(logits, -1)

    lps, ents = [], []
    for s in range(0, len(tgt), chunk):
        hc, tc = h[s: s + chunk], tgt[s: s + chunk]
        if torch.is_grad_enabled():
            lps.append(torch.utils.checkpoint.checkpoint(chunk_lp, hc, tc, use_reentrant=False))
        else:
            lps.append(chunk_lp(hc, tc))
        if entropy:
            with torch.no_grad():
                logits = model.lm_head(hc).float() / temperature
                ents.append(torch.logsumexp(logits, -1) - (logits.softmax(-1) * logits).sum(-1))
    lp = torch.cat(lps).double()
    sums = torch.zeros(len(idx), dtype=torch.float64, device=device).index_add(0, seg, lp)
    if not entropy:
        return sums
    ent = torch.zeros(len(idx), dtype=torch.float64, device=device).index_add_(0, seg, torch.cat(ents).double())
    counts = torch.zeros(len(idx), dtype=torch.float64, device=device).index_add_(
        0, seg, torch.ones_like(seg, dtype=torch.float64))
    return sums, ent / counts.clamp(min=1)


@torch.no_grad()
def all_logps(model, seqs, batches, temperature=TEMPERATURE, entropy=False):
    out = torch.zeros(len(seqs), dtype=torch.float64)
    ent = torch.zeros(len(seqs), dtype=torch.float64)
    for idx in batches:
        res = seq_logps(model, seqs, idx, temperature, entropy)
        if entropy:
            out[idx], ent[idx] = res[0].cpu(), res[1].cpu()
        else:
            out[idx] = res.cpu()
    return (out, ent) if entropy else out


# ---------------------------------------------------------------------------
# GRPO first step: rewards, advantages, gradient w.r.t. merged weights, LoRA geometry
# ---------------------------------------------------------------------------

def grpo_advantages(rewards: Sequence[Optional[float]], group: int) -> torch.Tensor:
    """TRL 1.14 GRPOTrainer._generate_and_score_completions for one reward function (multi_objective_aggregation=
    'sum_then_normalize', scale_rewards='group'): a None reward is unscorable (NaN, out of the baseline), the group
    nan-mean is the baseline, std is the Bessel nan-std, A = (r - mean) / (std + 1e-4), and NaN advantages become 0."""
    from trl.trainer.utils import nanstd

    rpf = torch.tensor([[math.nan if r is None else float(r)] for r in rewards], dtype=torch.float32)
    unscorable = torch.isnan(rpf).all(dim=1)
    rew = (rpf * torch.ones(1)[None]).nansum(dim=1)
    rew[unscorable] = torch.nan
    mean = torch.nanmean(rew.view(-1, group), dim=1).repeat_interleave(group)
    std = nanstd(rew.view(-1, group), dim=1).repeat_interleave(group)
    return torch.nan_to_num((rew - mean) / (std + 1e-4), nan=0.0)


def grpo_loss_coefs(advantages: Sequence[float], lengths: Sequence[int], n_total: int) -> List[float]:
    """loss_type='grpo' at ratio 1: L = sum_i c_i * sum_t log pi(y_it), with c_i = -A_i / (|y_i| * N). TRL averages
    over the 8 sequences of a micro-batch and divides by the 8 accumulation steps, i.e. 1/64 for a 64-rollout step."""
    return [-a / (max(n, 1) * n_total) for a, n in zip(advantages, lengths)]


def target_weights(model, names: Sequence[str]) -> Dict[str, torch.nn.Parameter]:
    return {n: model.get_submodule(n).weight for n in names}


def merged_gradient(model, names, seqs, coefs, temperature=TEMPERATURE, token_budget=4096) -> Dict[str, torch.Tensor]:
    """G_l = dL/dW_l (fp32) for L = sum_i coefs[i] * log pi_T(y_i | x_i), over the merged weights of `names`."""
    weights = target_weights(model, names)
    for w in weights.values():
        w.requires_grad_(True)
        w.grad = None
    active = [i for i, c in enumerate(coefs) if c != 0]
    for idx in make_batches([len(seqs[i][0]) + len(seqs[i][1]) for i in active], token_budget):
        idx = [active[j] for j in idx]
        sums = seq_logps(model, seqs, idx, temperature)
        (sums * torch.tensor([coefs[i] for i in idx], dtype=torch.float64, device=sums.device)).sum().backward()
    grads = {}
    for n, w in weights.items():
        grads[n] = w.grad if w.grad is not None else torch.zeros_like(w)
        w.grad = None
        w.requires_grad_(False)
    return grads


def lora_geometry(G: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float):
    """First-order merged-weight change of one SGD step on (A, B) along -dL/d(A, B), per unit learning rate:
    W = W_start + s B A, dL/dB = s G A^T, dL/dA = s B^T G
    => dW = s (dB A + B dA) = -s^2 (G A^T A + B B^T G) + O(lr)   (exact when B = 0: A does not move then).
    Returns factors (U, V) with dW = U @ V: rank <= 2r, or r when B = 0."""
    U1 = G @ A.T
    if not bool(B.abs().max() > 0):
        return (-s * s) * U1, A.clone()
    V2 = B.T @ G
    return (-s * s) * torch.cat([U1, B], 1), torch.cat([A, V2], 0)


def lowrank_inner(d1, d2) -> float:
    """<dW1, dW2>_F for one module, each {'U','V'} or {'D'}."""
    if "D" in d1 or "D" in d2:
        m1 = d1["D"].double() if "D" in d1 else d1["U"].double() @ d1["V"].double()
        m2 = d2["D"].double() if "D" in d2 else d2["U"].double() @ d2["V"].double()
        return float((m1 * m2).sum())
    return float(((d1["U"].double().T @ d2["U"].double()) * (d1["V"].double() @ d2["V"].double().T)).sum())


def direction_norm(direction) -> float:
    return math.sqrt(sum(lowrank_inner(d, d) for d in direction.values()))


def save_direction(path: Path, direction: Dict[str, dict], meta: dict, dense_dtype=torch.bfloat16):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for n, d in direction.items():
        for k, v in d.items():
            tensors[f"{n}.{k}"] = (v.to(dense_dtype) if k == "D" else v.float()).contiguous().cpu()
    save_file(tensors, str(path))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_direction(path: Path, device="cuda") -> Dict[str, dict]:
    direction = {}
    for key, v in load_file(str(path)).items():
        name, k = key.rsplit(".", 1)
        direction.setdefault(name, {})[k] = v.to(device)
    return direction


def realised_direction(start_f, ckpt_f, s) -> Dict[str, dict]:
    """s (B_T A_T - B_0 A_0) as factors U = s [B_T, -B_0], V = [A_T; A_0] (U = s B_T, V = A_T when B_0 = 0)."""
    out = {}
    for n, (A0, B0) in start_f.items():
        AT, BT = ckpt_f[n]
        if bool(B0.abs().max() > 0):
            out[n] = {"U": s * torch.cat([BT, -B0], 1), "V": torch.cat([AT, A0], 0)}
        else:
            out[n] = {"U": s * BT, "V": AT.clone()}
    return out


def stage_directions(args, st: Start, tokenizer, train):
    ddir = Path(args.out_dir) / "directions" / st.dtag
    ddir.mkdir(parents=True, exist_ok=True)
    factors, s = start_lora(st)
    names = sorted(factors)
    run_info = check_run(st)
    meta_common = {"start": st.tag, "dtag": st.dtag, "mode": st.mode, "seed": st.seed, "init_seed": st.init_seed,
                   "sft_adapter": st.sft, "run_dir": str(st.run_dir) if st.run_dir else None,
                   "merged_start": run_info or None, "scaling": s, "n_modules": len(names)}
    timings = {}

    # Realised updates and the SFT displacement (cheap, factors only).
    for T in args.ckpts:
        path = ddir / f"v{T}.safetensors"
        ck = st.run_dir / "grpo" / f"checkpoint-{T}" if st.run_dir else None
        if path.exists() or ck is None or not ck.exists():
            if not path.exists():
                print(f"[directions] {st.dtag}/v{T}: no checkpoint ({ck}), skipped", flush=True)
            continue
        ckpt_f, s_ck = load_adapter_factors(ck)
        assert abs(s_ck - s) < 1e-12 and sorted(ckpt_f) == names
        d = realised_direction(factors, ckpt_f, s)
        cos = [float(torch.nn.functional.cosine_similarity(ckpt_f[n][0].flatten(), factors[n][0].flatten(), 0))
               for n in names]
        meta = {**meta_common, "name": f"v{T}", "checkpoint": str(ck), "norm": direction_norm(d),
                "cos_A_start_AT": {"min": min(cos), "mean": sum(cos) / len(cos)}}  # init check: A moved little
        save_direction(path, d, meta)
        print(f"[directions] {st.dtag}/v{T}: norm {meta['norm']:.4g}, cos(A_0, A_T) min {min(cos):.5f}", flush=True)
    if st.kind == "gold" and not (ddir / "sft.safetensors").exists():
        sft_f, s_sft = load_adapter_factors(st.sft)
        d = {n: {"U": s_sft * B, "V": A.clone()} for n, (A, B) in sft_f.items()}
        save_direction(ddir / "sft.safetensors", d, {**meta_common, "name": "sft", "norm": direction_norm(d)})

    if all((ddir / f"{n}.safetensors").exists() for n in ("v0_lora", "rand_lora", "v0_raw")):
        print(f"[directions] {st.dtag}: v0 directions exist", flush=True)
        return timings

    # First-step rollouts at the start, from its GRPO pool.
    rollout_path = Path(args.out_dir) / f"rollouts_v0_{st.dtag}.jsonl"
    pool = grpo_pool(st, train)
    chosen = sorted(random.Random(1500 + st.seed).sample(sorted(pool), min(args.n_prompts, len(pool))))
    by_id = {ex["example_id"]: ex for ex in train}
    rows = read_jsonl(rollout_path)
    if len(rows) < len(chosen) * args.group:
        t0 = time.time()
        model = load_sampler(st)
        reward_dir = Path(args.out_dir) / f"reward_log_{st.dtag}"
        reward_dir.mkdir(parents=True, exist_ok=True)
        reward = R.make_reward(reward_dir, R.stop_token_ids(tokenizer), marked=True, dataset="gsm8k",
                               needs_stop=True, censor=CENSOR)
        rows = []
        for b in range(0, len(chosen), args.trl_prompts):  # 8 prompts x 8 = one TRL generation batch of 64
            chunk = chosen[b: b + args.trl_prompts]
            ids = sample_ids(model, tokenizer, [by_id[e]["prompt"] for e in chunk], args.group,
                             batch_seed(args.sample_seed, "v0", st.dtag, b), trl_cut=True)
            texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            golds = [by_id[chunk[j // args.group]]["gold_answer"] for j in range(len(ids))]
            exs = [chunk[j // args.group] for j in range(len(ids))]
            rewards = reward(completions=texts, gold_answer=golds, completion_ids=ids, trainer_state=None, example_id=exs)
            adv = grpo_advantages(rewards, args.group).tolist()
            for j in range(len(ids)):
                rows.append({"example_id": exs[j], "sample": j % args.group, "reward": rewards[j], "advantage": adv[j],
                             "n_tokens": len(ids[j]), "completion": texts[j], "token_ids": ids[j]})
        rollout_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        timings["rollouts"] = {"completions": len(rows), "seconds": time.time() - t0}
        del model
        torch.cuda.empty_cache()

    # G = dL/dW at the start (fp32 weights), then the LoRA-geometry map.
    t0 = time.time()
    model = load_start_fp32(st, factors, s)
    prompt_ids = {e: tokenizer(by_id[e]["prompt"])["input_ids"] for e in chosen}
    seqs = [(prompt_ids[r["example_id"]], r["token_ids"]) for r in rows]
    coefs = grpo_loss_coefs([r["advantage"] for r in rows], [r["n_tokens"] for r in rows], len(rows))
    G = merged_gradient(model, names, seqs, coefs, token_budget=args.grad_token_budget)
    timings["gradient"] = {"sequences": sum(c != 0 for c in coefs), "seconds": time.time() - t0}
    rewards = [r["reward"] for r in rows]
    scored = [r for r in rewards if r is not None]
    adv = torch.tensor([r["advantage"] for r in rows]).view(-1, args.group)
    meta_v0 = {**meta_common, "n_prompts": len(chosen), "group": args.group, "n_rollouts": len(rows),
               "mean_reward": sum(scored) / max(len(scored), 1), "n_censored": sum(r is None for r in rewards),
               "frac_zero_adv_groups": float((adv.abs().sum(1) == 0).float().mean()),
               "n_no_eos_rows": sum(1 for r in rows if not r["token_ids"] or r["token_ids"][-1] != tokenizer.eos_token_id),
               "G_norm": math.sqrt(sum(float((g.double() ** 2).sum()) for g in G.values())),
               "G_layer_norms": {n: float(g.norm()) for n, g in G.items()},
               "pool_prompts": chosen}

    v0 = {}
    for n in names:
        A, B = (t.cuda() for t in factors[n])
        U, V = lora_geometry(G[n], A, B, s)
        v0[n] = {"U": U, "V": V}
    save_direction(ddir / "v0_lora.safetensors", v0, {**meta_v0, "name": "v0_lora", "norm": direction_norm(v0)})

    gen = torch.Generator(device="cuda").manual_seed(batch_seed("rand_lora", st.dtag))
    rand = {}
    for n in names:
        A, B = (t.cuda() for t in factors[n])
        Rn = torch.randn(G[n].shape, generator=gen, device="cuda", dtype=torch.float32)
        U, V = lora_geometry(Rn, A, B, s)
        rand[n] = {"U": U, "V": V}
    save_direction(ddir / "rand_lora.safetensors", rand,
                   {**meta_common, "name": "rand_lora", "norm": direction_norm(rand),
                    "note": "unscaled; the phi stage rescales it to v0_lora's RMS phi on set A (eps json)"})

    raw = {n: {"D": -G[n]} for n in names}
    save_direction(ddir / "v0_raw.safetensors", raw,
                   {**meta_v0, "name": "v0_raw", "norm": meta_v0["G_norm"], "stored_dtype": "bfloat16"})

    # Weight-space cosines between v0_lora and the realised updates (diagnostic).
    cos = {}
    for T in args.ckpts:
        p = ddir / f"v{T}.safetensors"
        if p.exists():
            vT = load_direction(p)
            cos[f"v{T}"] = sum(lowrank_inner(v0[n], vT[n]) for n in names) / (direction_norm(v0) * direction_norm(vT))
    meta_path = ddir / "v0_lora.json"
    meta = json.loads(meta_path.read_text())
    meta["weight_cos"] = cos
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    timings["directions_total"] = time.time() - t0
    print(f"[directions] {st.dtag}: |G| {meta_v0['G_norm']:.4g}, weight cos v0_lora vs {cos}", flush=True)
    del model, G
    torch.cuda.empty_cache()
    return timings


# ---------------------------------------------------------------------------
# phi by central finite differences
# ---------------------------------------------------------------------------

class Perturber:
    """Sets the target weights to W_start + eps * dW (fp32) and back."""

    def __init__(self, model, names):
        self.weights = target_weights(model, names)
        self.base = {n: w.detach().clone() for n, w in self.weights.items()}
        self.base_norm = math.sqrt(sum(float((w.double() ** 2).sum()) for w in self.base.values()))

    @torch.no_grad()
    def set(self, direction: Optional[Dict[str, dict]], eps: float = 0.0):
        for n, w in self.weights.items():
            w.copy_(self.base[n])
            if direction is not None and eps != 0.0 and n in direction:
                d = direction[n]
                if "D" in d:
                    w.add_(d["D"].to(w.dtype), alpha=eps)
                else:
                    w.addmm_(d["U"], d["V"], alpha=eps)


def fd_terms(model, pert, direction, eps, seqs, batches, f0):
    pert.set(direction, eps)
    fp = all_logps(model, seqs, batches)
    pert.set(direction, -eps)
    fm = all_logps(model, seqs, batches)
    pert.set(None)
    return (fp - fm) / (2 * eps), (fp - 2 * f0 + fm) / eps ** 2, fp, fm


def rms(x: torch.Tensor) -> float:
    return float(x.double().pow(2).mean().sqrt())


def calibrate_eps(model, pert, direction, seqs, batches, f0, dnorm, threshold=0.1):
    """Largest eps = rel * |W| / |dW| on REL_GRID with RMS(second-order term) / RMS(first-order term) < threshold,
    where first = (f+ - f-)/2 and second = (f+ - 2 f0 + f-)/2. Also reports phi(eps) vs phi(eps/2)."""
    trials, chosen = [], None
    for rel in REL_GRID:
        eps = rel * pert.base_norm / dnorm
        phi, sec, fp, fm = fd_terms(model, pert, direction, eps, seqs, batches, f0)
        ratio = rms((fp - 2 * f0 + fm) / 2) / max(rms((fp - fm) / 2), 1e-300)
        trials.append({"rel": rel, "eps": eps, "ratio": ratio, "rms_first": rms((fp - fm) / 2), "rms_phi": rms(phi)})
        if ratio < threshold:
            chosen = trials[-1]
            break
    if chosen is None:
        chosen = min(trials, key=lambda t: t["ratio"])
        chosen["warning"] = f"no grid value reached ratio < {threshold}"
    phi1, _, _, _ = fd_terms(model, pert, direction, chosen["eps"], seqs, batches, f0)
    phi2, _, _, _ = fd_terms(model, pert, direction, chosen["eps"] / 2, seqs, batches, f0)
    chosen = {**chosen, "phi_eps_vs_half_rel_rms": rms(phi1 - phi2) / max(rms(phi1), 1e-300)}
    return chosen, trials


def exact_phi(model, names, seqs, idx_list, direction, temperature=TEMPERATURE) -> List[float]:
    """phi by autograd: phi = sum_l <d log pi / dW_l, dW_l>, one sample at a time (the FD check)."""
    weights = target_weights(model, names)
    for w in weights.values():
        w.requires_grad_(True)
    out = []
    for i in idx_list:
        for w in weights.values():
            w.grad = None
        with torch.enable_grad():
            seq_logps(model, seqs, [i], temperature).sum().backward()
        total = 0.0
        for n, w in weights.items():
            d = direction[n]
            if "D" in d:
                total += float((w.grad.double() * d["D"].double()).sum())
            else:
                total += float(((d["U"].T @ w.grad) * d["V"]).double().sum())
        out.append(total)
    for w in weights.values():
        w.grad = None
        w.requires_grad_(False)
    return out


def stage_phi(args, st: Start, tokenizer, test):
    out_dir = Path(args.out_dir)
    samples = read_jsonl(out_dir / f"samples_{st.tag}_A.jsonl")
    if not samples:
        raise SystemExit(f"no set-A samples for {st.tag}; run --stage sample first")
    calib_pool = read_jsonl(out_dir / f"samples_{st.tag}_B.jsonl") or samples
    calib = random.Random(0).sample(calib_pool, min(args.n_calib, len(calib_pool)))
    prompts = {ex["example_id"]: tokenizer(ex["prompt"])["input_ids"] for ex in test}
    seqs = [(prompts[r["item"]], r["token_ids"]) for r in samples]
    cseqs = [(prompts[r["item"]], r["token_ids"]) for r in calib]
    batches = make_batches([len(a) + len(b) for a, b in seqs], args.token_budget)
    cbatches = make_batches([len(a) + len(b) for a, b in cseqs], args.token_budget)

    factors, s = start_lora(st)
    names = sorted(factors)
    model = load_start_fp32(st, factors, s)
    pert = Perturber(model, names)
    timings = {}

    t0 = time.time()
    f0, ent = all_logps(model, seqs, batches, entropy=True)
    timings["f0_pass"] = {"sequences": len(seqs), "seconds": time.time() - t0}
    feat_path = out_dir / f"features_{st.tag}_A.jsonl"
    if not feat_path.exists():
        feat_path.write_text("".join(json.dumps({
            "item": r["item"], "sample": r["sample"], "start": st.tag, "logp": float(f0[i]),
            "mean_entropy": float(ent[i]), "n_tokens": r["n_tokens"], "stopped": r["stopped"],
            "correct_lenient": r["correct_lenient"], "correct_marked": r["correct_marked"]}) + "\n"
            for i, r in enumerate(samples)))
    c0 = all_logps(model, cseqs, cbatches)

    phi_path = out_dir / f"phi_{st.dtag}_A.jsonl"
    eps_path = out_dir / f"phi_{st.dtag}_A_eps.json"
    eps_info = json.loads(eps_path.read_text()) if eps_path.exists() else {}
    existing = read_jsonl(phi_path)
    done = {r["direction_name"] for r in existing}
    for dname in sorted(args.directions, key=lambda d: d.startswith("rand")):  # rand_* after v0_lora
        full_name = f"{dname}_{st.dtag}"
        path = out_dir / "directions" / st.dtag / f"{dname}.safetensors"
        if full_name in done or not path.exists():
            print(f"[phi] {full_name}: {'done' if full_name in done else 'missing direction, skipped'}", flush=True)
            continue
        direction = load_direction(path)
        dnorm = direction_norm(direction)
        t0 = time.time()
        chosen, trials = calibrate_eps(model, pert, direction, cseqs, cbatches, c0, dnorm, args.ratio_threshold)
        t_cal = time.time() - t0
        t0 = time.time()
        phi, sec, _, _ = fd_terms(model, pert, direction, chosen["eps"], seqs, batches, f0)
        t_fd = time.time() - t0
        info = {"eps": chosen, "trials": trials, "dir_norm": dnorm, "W_norm": pert.base_norm, "rms_phi": rms(phi),
                "calib_seconds": t_cal, "fd_seconds": t_fd, "n": len(seqs)}
        if dname.startswith("rand"):
            ref = [r["phi"] for r in existing if r["direction_name"] == f"v0_lora_{st.dtag}"]
            if not ref:
                raise SystemExit("rand_lora needs v0_lora's phi first (same run or file)")
            c = rms(torch.tensor(ref)) / max(rms(phi), 1e-300)
            phi, sec = phi * c, sec * c * c
            info.update({"rms_match_factor": c, "rms_phi_matched": rms(phi), "rms_phi_v0_lora": rms(torch.tensor(ref))})
        if args.verify_exact > 0:
            idx = list(range(min(args.verify_exact, len(cseqs))))
            pert.set(None)
            ex = exact_phi(model, names, cseqs, idx, direction)
            fd_c, _, _, _ = fd_terms(model, pert, direction, chosen["eps"], cseqs, cbatches, c0)
            fdv = [float(fd_c[i]) for i in idx]
            scale = math.sqrt(sum(b * b for b in ex) / len(ex))
            info["exact_check"] = {"fd": fdv, "exact": ex,
                                   "max_err_over_rms": max(abs(a - b) for a, b in zip(fdv, ex)) / max(scale, 1e-300)}
        rows = [dict(zip(PHI_FIELDS, (r["item"], r["sample"], st.tag, full_name, float(phi[i]), float(sec[i]),
                                      r["correct_lenient"], r["correct_marked"]))) for i, r in enumerate(samples)]
        with phi_path.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        existing += rows
        eps_info[full_name] = info
        eps_path.write_text(json.dumps(eps_info, indent=2))
        timings[full_name] = {"calib_seconds": t_cal, "fd_seconds": t_fd, "sequences": len(seqs),
                              "sec_per_1k_phi": 1000 * t_fd / len(seqs)}
        print(f"[phi] {full_name}: rel {chosen['rel']:g} ratio {chosen['ratio']:.3f} rms {info['rms_phi']:.4g} "
              f"({t_fd:.0f}s for {len(seqs)})" + (f" exact-check {info['exact_check']['max_err_over_rms']:.2e}"
                                                  if "exact_check" in info else ""), flush=True)
        del direction
        torch.cuda.empty_cache()
    return timings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["sample", "directions", "phi", "all"], required=True)
    p.add_argument("--start", choices=["untrained", "gold"], required=True)
    p.add_argument("--seed", type=int, required=True, help="SFT adapter seed (gold) / E2 run seed (continue mode).")
    p.add_argument("--start_mode", choices=["merged", "continue"], default="merged")
    p.add_argument("--lora_init_seed", type=int, default=None, help="merged: fresh-LoRA init seed (default --seed).")
    p.add_argument("--run_dir", default=None,
                   help="RL method dir (partition.json, grpo/, grpo/checkpoint-T). continue: defaults to E2's.")
    p.add_argument("--sft_adapter", default=None, help="gold: SFT adapter dir (default: the gold n=250 adapter).")
    p.add_argument("--out_dir", default="runs/firstgrad/part1")
    p.add_argument("--loss", default="grpo", choices=["grpo", "dr_grpo"], help="continue: E2 loss variant.")
    p.add_argument("--sets", nargs="+", default=["A", "B"], choices=["A", "B", "C"])
    p.add_argument("--n_items", type=int, default=-1, help="First n of the 600 items (smoke tests); -1 = all.")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--prompts_per_batch", type=int, default=32)
    p.add_argument("--sample_seed", type=int, default=15)
    p.add_argument("--n_prompts", type=int, default=256, help="v0: pool prompts.")
    p.add_argument("--group", type=int, default=8, help="v0: rollouts per prompt (GRPO group).")
    p.add_argument("--trl_prompts", type=int, default=8, help="v0: prompts per generate call (TRL: 8 x 8 = 64).")
    p.add_argument("--ckpts", type=int, nargs="*", default=[25, 150])
    p.add_argument("--directions", nargs="+", default=["v0_lora", "rand_lora", "v0_raw", "v25", "v150"])
    p.add_argument("--n_calib", type=int, default=32)
    p.add_argument("--ratio_threshold", type=float, default=0.1)
    p.add_argument("--verify_exact", type=int, default=0, help="phi: check FD against autograd on this many samples.")
    p.add_argument("--token_budget", type=int, default=16384, help="phi: tokens per forward batch.")
    p.add_argument("--grad_token_budget", type=int, default=4096, help="directions: tokens per backward batch.")
    p.add_argument("--timings_out", default=None, help="Append this call's timings (json line).")
    return p


def main():
    args = build_parser().parse_args()
    st = make_start(args.start, args.seed, args.start_mode, args.lora_init_seed, args.sft_adapter, args.run_dir,
                    args.loss)
    if st.kind == "untrained" and "sft" in args.directions:
        args.directions.remove("sft")
    tokenizer = R.load_tokenizer(model_args())
    train, test = load_data(tokenizer)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    timings = {"stage": args.stage, "start": st.tag, "dtag": st.dtag, "mode": st.mode, "n_items": args.n_items,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(f"[start] {st}", flush=True)
    if args.stage in ("sample", "all"):
        timings["sample"] = stage_sample(args, st, tokenizer, test)
    if args.stage in ("directions", "all"):
        timings["directions"] = stage_directions(args, st, tokenizer, train)
    if args.stage in ("phi", "all"):
        timings["phi"] = stage_phi(args, st, tokenizer, test)
    print(json.dumps(timings, default=str), flush=True)
    if args.timings_out:
        with open(args.timings_out, "a") as f:
            f.write(json.dumps(timings, default=str) + "\n")


if __name__ == "__main__":
    main()
