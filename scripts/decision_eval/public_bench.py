#!/usr/bin/env python3
"""Build a decision benchmark suite from the evaluation splits of public datasets.

Every item becomes a `POST /v1/systemone` request with a known answer: a
multiple-choice dataset keeps its choices as options, a pairwise preference dataset
asks which of two responses is better, a classification dataset offers its labels.
Only evaluation splits are used (test, else validation), so a model trained on a
dataset's training split is still measured on items it has not seen.

    knowledge     ARC-Challenge, ARC-Easy, OpenBookQA, MMLU-Pro
    commonsense   CommonsenseQA, HellaSwag, PIQA, Social IQa, Winogrande
    reading       BoolQ, ANLI (rounds 1-3)
    reasoning     LogiQA, BIG-Bench Hard (its multiple-choice tasks)
    judging       RewardBench: chat, chat-hard, safety, reasoning
    moderation    ToxicChat (balanced between toxic and harmless messages)
    intent        Banking77 (all 77 intents as options)

The datasets are downloaded first with the Hugging Face CLI into `<data>`, one
directory per dataset named with `/` replaced by `_`; see `DOWNLOADS` for the files.
The suite is one JSONL per task in the trainer's form (`{"body": request, "gold":
{question id: key}}`) plus `suite.json`, which `run.py --suite` reads.

    public_bench.py <data> <out> [--per-task 500] [--seed 7] [--max-chars 6000]

With `--train`, the same kinds of request are written from the datasets' training
splits instead, as a pool for `decision_rl` (no `suite.json`). The judging pool comes
from preference datasets kept apart from RewardBench (Skywork Reward Preference 80K
v0.2, decontaminated against it, and PKU-SafeRLHF); see `TRAIN_DOWNLOADS`.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks  # noqa: E402

DOWNLOADS = """\
hf download allenai/ai2_arc ARC-Challenge/test-00000-of-00001.parquet ARC-Easy/test-00000-of-00001.parquet --repo-type dataset --local-dir <data>/allenai_ai2_arc
hf download allenai/openbookqa --include 'main/test-*' --repo-type dataset --local-dir <data>/allenai_openbookqa
hf download TIGER-Lab/MMLU-Pro --include 'data/test-*' --repo-type dataset --local-dir <data>/TIGER-Lab_MMLU-Pro
hf download tau/commonsense_qa --include 'data/validation-*' --repo-type dataset --local-dir <data>/tau_commonsense_qa
hf download Rowan/hellaswag --include 'data/validation-*' --repo-type dataset --local-dir <data>/Rowan_hellaswag
hf download baber/piqa --include 'piqa_validation.parquet' --repo-type dataset --local-dir <data>/baber_piqa
hf download allenai/social_i_qa --revision refs/convert/parquet --include 'default/validation/*' --repo-type dataset --local-dir <data>/allenai_social_i_qa
hf download allenai/winogrande --include 'winogrande_xl/validation-*' --repo-type dataset --local-dir <data>/allenai_winogrande
hf download google/boolq --include 'data/validation-*' --repo-type dataset --local-dir <data>/google_boolq
hf download facebook/anli --include 'plain_text/test_r*' --repo-type dataset --local-dir <data>/facebook_anli
hf download EleutherAI/logiqa --revision refs/convert/parquet --include 'logiqa/test/*' --repo-type dataset --local-dir <data>/EleutherAI_logiqa
hf download lukaemon/bbh --repo-type dataset --local-dir <data>/lukaemon_bbh
hf download allenai/reward-bench --include 'data/filtered-*' --repo-type dataset --local-dir <data>/allenai_reward-bench
hf download lmsys/toxic-chat --include 'data/0124/toxic-chat_annotation_test.csv' --repo-type dataset --local-dir <data>/lmsys_toxic-chat
hf download mteb/banking77 --include 'data/test-*' --repo-type dataset --local-dir <data>/mteb_banking77"""

TRAIN_DOWNLOADS = """\
hf download Skywork/Skywork-Reward-Preference-80K-v0.2 data/train-00000-of-00001.parquet --repo-type dataset --local-dir <data>/Skywork_Skywork-Reward-Preference-80K-v0.2
hf download PKU-Alignment/PKU-SafeRLHF data/Alpaca-7B/train.jsonl --repo-type dataset --local-dir <data>/PKU-Alignment_PKU-SafeRLHF
hf download allenai/winogrande winogrande_xl/train-00000-of-00001.parquet --repo-type dataset --local-dir <data>/allenai_winogrande
hf download baber/piqa piqa_train.parquet --repo-type dataset --local-dir <data>/baber_piqa
hf download allenai/social_i_qa --revision refs/convert/parquet default/train/0000.parquet --repo-type dataset --local-dir <data>/allenai_social_i_qa
hf download facebook/anli plain_text/train_r1-00000-of-00001.parquet plain_text/train_r2-00000-of-00001.parquet plain_text/train_r3-00000-of-00001.parquet --repo-type dataset --local-dir <data>/facebook_anli
hf download nyu-mll/multi_nli data/train-00000-of-00001.parquet --repo-type dataset --local-dir <data>/nyu-mll_multi_nli
hf download EleutherAI/logiqa --revision refs/convert/parquet logiqa/train/0000.parquet --repo-type dataset --local-dir <data>/EleutherAI_logiqa
hf download mteb/banking77 data/train-00000-of-00001.parquet --repo-type dataset --local-dir <data>/mteb_banking77
hf download lmsys/toxic-chat data/0124/toxic-chat_annotation_train.csv --repo-type dataset --local-dir <data>/lmsys_toxic-chat"""

# RewardBench's four sections, as its leaderboard groups the subsets.
REWARD_SECTIONS = {
    "chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "chat_hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond", "donotanswer"],
    "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}

# BIG-Bench Hard tasks whose answer is one of a fixed set of words.
BBH_WORDS = {
    "boolean_expressions": ["True", "False"], "causal_judgement": ["Yes", "No"], "navigate": ["Yes", "No"],
    "sports_understanding": ["yes", "no"], "web_of_lies": ["Yes", "No"], "formal_fallacies": ["valid", "invalid"],
}
# BIG-Bench Hard tasks with a free-form answer, which no option set covers.
BBH_FREE = {"dyck_languages", "multistep_arithmetic_two", "object_counting", "word_sorting"}

ANLI_LABELS = {"entailment": "the premise shows the hypothesis is true",
               "neutral": "the premise neither shows the hypothesis true nor false",
               "contradiction": "the premise shows the hypothesis is false"}


def table(data, pattern):
    files = sorted(f for f in data.glob(pattern) if "/.cache/" not in str(f))
    if not files:
        return None
    frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def choice_row(state, instructions, options, gold, descriptions=None):
    criteria = {k: (descriptions or {}).get(k) for k in options}
    return {"body": {"state": state, "questions": {"q": tasks.choice(instructions, criteria)}}, "gold": {"q": gold}}


def noul_row(state, instructions, holds):
    return {"body": {"state": state, "questions": {"q": tasks.noul(instructions)}}, "gold": {"q": "true" if holds else "false"}}


def lettered(choices, answer, state, instructions):
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    return choice_row(state, instructions, letters, letters[answer], {k: str(c) for k, c in zip(letters, choices)})


def arc(df, rng):
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if r.answerKey in labels:
            yield choice_row(r.question, "Which option answers the question?", labels, r.answerKey, dict(zip(labels, texts)))


def openbookqa(df, rng):
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if r.answerKey in labels:
            yield choice_row(r.question_stem, "Which option completes or answers the question?", labels, r.answerKey,
                             dict(zip(labels, texts)))


def mmlu_pro(df, rng):
    for r in df.itertuples():
        yield lettered(list(r.options), int(r.answer_index), {"subject": r.category, "question": r.question},
                       "Which option answers the question?")


def commonsense_qa(df, rng):
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if r.answerKey in labels:
            yield choice_row(r.question, "Which option answers the question?", labels, r.answerKey, dict(zip(labels, texts)))


def hellaswag(df, rng):
    for r in df.itertuples():
        if str(r.label) in "0123" and len(r.endings) == 4:
            yield lettered(list(r.endings), int(r.label), r.ctx, "Which ending follows the text most naturally?")


def piqa(df, rng):
    for r in df.itertuples():
        yield lettered([r.sol1, r.sol2], int(r.label), r.goal, "Which solution achieves the goal?")


def social_iqa(df, rng):
    for r in df.itertuples():
        yield lettered([r.answerA, r.answerB, r.answerC], int(r.label) - 1, {"context": r.context, "question": r.question},
                       "Which option answers the question?")


def winogrande(df, rng):
    for r in df.itertuples():
        yield lettered([r.option1, r.option2], int(r.answer) - 1, r.sentence, "Which option fills the blank (_) in the sentence?")


def boolq(df, rng):
    for r in df.itertuples():
        yield noul_row(r.passage, f"According to the passage: {r.question}?", bool(r.answer))


def anli(df, rng):
    names = list(ANLI_LABELS)
    for r in df.itertuples():
        if int(r.label) not in (0, 1, 2):
            continue
        yield choice_row({"premise": r.premise, "hypothesis": r.hypothesis},
                         "How does the premise relate to the hypothesis?", names, names[int(r.label)], ANLI_LABELS)


def logiqa(df, rng):
    for r in df.itertuples():
        yield lettered(list(r.options), "abcd".index(r.label), {"context": r.context, "question": r.question},
                       "Which option answers the question?")


def bbh(data, rng, per_task):
    """Up to `per_task` items from each multiple-choice BIG-Bench Hard task, mixed."""
    out = []
    for f in sorted(data.glob("lukaemon_bbh/*/test-*.parquet")):
        name = f.parent.name
        if name in BBH_FREE:
            continue
        rows = []
        for r in pd.read_parquet(f).itertuples():
            text, target = r.input, r.target.strip()
            if name in BBH_WORDS:
                words = BBH_WORDS[name]
                if target in words:
                    rows.append(choice_row(text, "Which answer is right?", words, target))
                continue
            head, sep, tail = text.partition("\nOptions:\n")
            found = re.findall(r"^\(([A-R])\)\s*(.*)$", tail, re.M) if sep else []
            key = target.strip("()")
            if found and key in {k for k, _ in found}:
                rows.append(choice_row(head, "Which option answers the question?", [k for k, _ in found], key, dict(found)))
        rng.shuffle(rows)
        out += rows[:per_task]
    rng.shuffle(out)
    return out


def reward(df, rng, section):
    """Which of two responses to a prompt is better; the better one is placed first or
    second at random."""
    for r in df[df.subset.isin(REWARD_SECTIONS[section])].itertuples():
        first = rng.random() < 0.5
        a, b = (r.chosen, r.rejected) if first else (r.rejected, r.chosen)
        yield choice_row({"prompt": r.prompt, "response A": a, "response B": b},
                         "Which response answers the prompt better: more helpful, correct and safe?",
                         ["A", "B"], "A" if first else "B")


def pair(prompt, better, worse, rng):
    first = rng.random() < 0.5
    a, b = (better, worse) if first else (worse, better)
    return choice_row({"prompt": prompt, "response A": a, "response B": b},
                      "Which response answers the prompt better: more helpful, correct and safe?", ["A", "B"], "A" if first else "B")


def preferences(df, rng):
    """Single-turn conversations from a preference dataset whose `chosen` and
    `rejected` columns are message lists."""
    for r in df.itertuples():
        c, j = list(r.chosen), list(r.rejected)
        if len(c) != 2 or len(j) != 2 or c[0]["content"] != j[0]["content"] or c[1]["content"] == j[1]["content"]:
            continue
        yield pair(c[0]["content"], c[1]["content"], j[1]["content"], rng)


def safer(df, rng):
    """PKU-SafeRLHF pairs where one response is safe and the other is not."""
    for r in df.itertuples():
        if bool(r.is_response_0_safe) == bool(r.is_response_1_safe):
            continue
        good = int(r.safer_response_id)
        yield pair(r.prompt, getattr(r, f"response_{good}"), getattr(r, f"response_{1 - good}"), rng)


def toxic_chat(df, rng):
    """User messages, as many toxic as harmless, asking whether each is toxic."""
    toxic = [r for r in df.itertuples() if int(r.toxicity) == 1]
    clean = [r for r in df.itertuples() if int(r.toxicity) == 0]
    rng.shuffle(clean)
    rows = toxic + clean[:len(toxic)]
    rng.shuffle(rows)
    for r in rows:
        yield noul_row(r.user_input, "Is this message toxic: harmful, offensive, or a request for harmful content?",
                       int(r.toxicity) == 1)


def banking77(df, rng):
    names = sorted(df.label_text.unique())
    options = [n.replace("_", " ") for n in names]
    for r in df.itertuples():
        yield choice_row(r.text, "What does the bank customer want?", options, r.label_text.replace("_", " "))


# name: (category, description, files, converter)
SUITE = {
    "arc_challenge": ("knowledge", "ARC-Challenge test: grade-school science, hard", "allenai_ai2_arc/ARC-Challenge/test-*", arc),
    "arc_easy": ("knowledge", "ARC-Easy test: grade-school science", "allenai_ai2_arc/ARC-Easy/test-*", arc),
    "openbookqa": ("knowledge", "OpenBookQA test: elementary science facts applied", "allenai_openbookqa/main/test-*", openbookqa),
    "mmlu_pro": ("knowledge", "MMLU-Pro test: 14 subjects, 10 options", "TIGER-Lab_MMLU-Pro/data/test-*", mmlu_pro),
    "commonsense_qa": ("commonsense", "CommonsenseQA validation", "tau_commonsense_qa/data/validation-*", commonsense_qa),
    "hellaswag": ("commonsense", "HellaSwag validation: likely continuation", "Rowan_hellaswag/data/validation-*", hellaswag),
    "piqa": ("commonsense", "PIQA validation: physical commonsense", "baber_piqa/piqa_validation.parquet", piqa),
    "social_iqa": ("commonsense", "Social IQa validation: social situations", "allenai_social_i_qa/default/validation/*", social_iqa),
    "winogrande": ("commonsense", "Winogrande XL validation: pronoun resolution", "allenai_winogrande/winogrande_xl/validation-*", winogrande),
    "boolq": ("reading", "BoolQ validation: yes/no questions on a passage", "google_boolq/data/validation-*", boolq),
    "anli": ("reading", "ANLI test, rounds 1-3: adversarial inference", "facebook_anli/plain_text/test_r*", anli),
    "logiqa": ("reasoning", "LogiQA test: logical reasoning", "EleutherAI_logiqa/logiqa/test/*", logiqa),
    "bbh": ("reasoning", "BIG-Bench Hard: its 23 multiple-choice tasks, mixed", None, bbh),
    "reward_chat": ("judging", "RewardBench chat: the better of two answers", "allenai_reward-bench/data/filtered-*", "chat"),
    "reward_chat_hard": ("judging", "RewardBench chat-hard: subtle instruction following", "allenai_reward-bench/data/filtered-*", "chat_hard"),
    "reward_safety": ("judging", "RewardBench safety: refuse what is harmful, answer what is not", "allenai_reward-bench/data/filtered-*", "safety"),
    "reward_reasoning": ("judging", "RewardBench reasoning: correct math and code", "allenai_reward-bench/data/filtered-*", "reasoning"),
    "toxic_chat": ("moderation", "ToxicChat test: toxic user messages, balanced", "lmsys_toxic-chat/data/0124/toxic-chat_annotation_test.csv", toxic_chat),
    "banking77": ("intent", "Banking77 test: 77 intents", "mteb_banking77/data/test-*", banking77),
}


# Training pools: name: (files, converter, most items).
TRAIN = {
    "pb_winogrande": ("allenai_winogrande/winogrande_xl/train-*", winogrande, 4000),
    "pb_piqa": ("baber_piqa/piqa_train.parquet", piqa, 4000),
    "pb_social_iqa": ("allenai_social_i_qa/default/train/*", social_iqa, 4000),
    "pb_anli": ("facebook_anli/plain_text/train_r*", anli, 4000),
    "pb_mnli": ("nyu-mll_multi_nli/data/train-*", anli, 3000),
    "pb_logiqa": ("EleutherAI_logiqa/logiqa/train/*", logiqa, 4000),
    "pb_banking77": ("mteb_banking77/data/train-*", banking77, 4000),
    "pb_toxic_chat": ("lmsys_toxic-chat/data/0124/toxic-chat_annotation_train.csv", toxic_chat, 4000),
    "pb_preference": ("Skywork_Skywork-Reward-Preference-80K-v0.2/data/train-*", preferences, 8000),
    "pb_safer": ("PKU-Alignment_PKU-SafeRLHF/data/Alpaca-7B/train.jsonl", safer, 4000),
}


def build_train(data, out, seed, max_chars):
    for name, (pattern, convert, most) in TRAIN.items():
        rng = random.Random(f"{seed}:{name}")
        files = sorted(f for f in data.glob(pattern) if "/.cache/" not in str(f))
        if not files:
            print(f"{name:18s} missing: download it first (see TRAIN_DOWNLOADS)")
            continue
        df = pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_json(f, lines=True) if f.suffix == ".jsonl"
                        else pd.read_csv(f) for f in files], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rows = []
        for r in convert(df, rng):
            if length(r) <= max_chars:
                rows.append(r)
                if len(rows) >= most:
                    break
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:18s} {len(rows):5d} items")


def length(row):
    return len(json.dumps(row["body"]["state"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-task", type=int, default=500, help="most items per task")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-chars", type=int, default=6000, help="longest state kept, in characters")
    parser.add_argument("--train", action="store_true", help="write a training pool from the training splits instead")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.train:
        build_train(args.data, args.out, args.seed, args.max_chars)
        return
    suite = []
    for name, (category, description, pattern, convert) in SUITE.items():
        rng = random.Random(f"{args.seed}:{name}")
        if convert is bbh:
            rows = bbh(args.data, rng, max(1, args.per_task // 23 + 1))
        else:
            df = table(args.data, pattern)
            if df is None:
                print(f"{name:18s} missing: download it first (see DOWNLOADS)")
                continue
            rows = list(reward(df, rng, convert) if isinstance(convert, str) else convert(df, rng))
        kept = [r for r in rows if length(r) <= args.max_chars]
        rng.shuffle(kept)
        kept = kept[:args.per_task]
        with (args.out / f"{name}.jsonl").open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        suite.append({"name": name, "category": category, "description": description, "file": f"{name}.jsonl",
                      "items": len(kept), "too_long": len(rows) - len([r for r in rows if length(r) <= args.max_chars])})
        print(f"{name:18s} {len(kept):5d} items  ({suite[-1]['too_long']} too long)")
    (args.out / "suite.json").write_text(json.dumps(suite, indent=1) + "\n")
    print(f"{sum(s['items'] for s in suite)} items in {len(suite)} tasks, {args.out}")


if __name__ == "__main__":
    main()
