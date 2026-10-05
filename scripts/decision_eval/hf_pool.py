#!/usr/bin/env python3
"""Turn public datasets into decision requests for `decision_rl`.

A classification dataset's label set becomes the options of a choice question, a
yes/no dataset a noul, and a multiple-choice dataset keeps its choices, so every row
carries its answer and the teacher need judge nothing. One JSONL per dataset in the
trainer's form (`{"body": request, "gold": {question id: key}}`), sampled to a size
per dataset, long states left out.

The datasets are downloaded first with the Hugging Face CLI into `<data>`, one
directory per dataset named with `/` replaced by `_`:

    hf download cais/mmlu --repo-type dataset --local-dir <data>/cais_mmlu

    hf_pool.py <data> <out> [--per-dataset 3000] [--seed 7]
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks  # noqa: E402

MAX_STATE_CHARS = 2500
LANGUAGES = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fr": "French",
             "hi": "Hindi", "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
             "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"}
CODE_LANGUAGES = ["Python", "C", "C++", "Java", "JavaScript", "Go", "Rust", "Ruby", "Perl", "PHP", "Haskell", "Lua", "Kotlin",
                  "Swift", "Scala", "C sharp", "Julia", "R", "Racket", "Common Lisp", "Fortran", "COBOL", "Pascal", "Ada",
                  "Nim", "D", "Elixir", "Erlang", "OCaml", "F Sharp", "Clojure", "Groovy", "Dart", "TypeScript", "Zig", "Tcl",
                  "AWK", "UNIX Shell", "PowerShell", "SQL", "MATLAB", "Prolog", "Scheme", "Factor", "Forth", "Crystal", "Raku"]


def files_of(data, name, prefer):
    """Data files of dataset `name` (parquet, else CSV or JSONL) whose path holds
    one of `prefer`, else all."""
    root = data / name.replace("/", "_")
    found = []
    for ext in ("parquet", "csv", "jsonl"):
        found = sorted(f for f in glob.glob(str(root / "**" / f"*.{ext}"), recursive=True) if "/.cache/" not in f)
        if found:
            break
    for p in prefer:
        chosen = [f for f in found if p in f]
        if chosen:
            return chosen
    return found


def read_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def class_names(path, column):
    """The label names a parquet's Hugging Face metadata gives `column`, if any."""
    if not path.endswith(".parquet"):
        return None
    meta = pq.read_schema(path).metadata or {}
    info = json.loads(meta.get(b"huggingface", b"{}") or b"{}")
    feature = info.get("info", {}).get("features", {}).get(column, {})
    return feature.get("names")


def load(data, name, prefer=("test", "validation", "train"), limit=None):
    files = files_of(data, name, prefer)
    if not files:
        return None, None
    frames = []
    for f in files:
        frames.append(read_table(f))
        if limit and sum(len(x) for x in frames) >= limit:
            break
    return pd.concat(frames, ignore_index=True), files[0]


def short(text):
    return isinstance(text, str) and 0 < len(text) <= MAX_STATE_CHARS


def choice_row(state, instructions, options, gold, descriptions=None):
    criteria = {k: (descriptions or {}).get(k) for k in options}
    return {"body": {"state": state, "questions": {"q": tasks.choice(instructions, criteria)}}, "gold": {"q": gold}}


def noul_row(state, instructions, holds):
    return {"body": {"state": state, "questions": {"q": tasks.noul(instructions)}}, "gold": {"q": "true" if holds else "false"}}


def lettered(choices, answer_index, state, instructions):
    """A multiple-choice item with lettered keys and the choice texts as descriptions."""
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    return choice_row(state, instructions, letters, letters[answer_index], dict(zip(letters, choices)))


# ---------------------------------------------------------------- converters

def mmlu(df, path, rng):
    for r in df.itertuples():
        choices = list(r.choices)
        if len(choices) != 4 or not short(r.question):
            continue
        subject = str(r.subject).replace("_", " ")
        yield lettered(choices, int(r.answer), {"subject": subject, "question": r.question}, "Which option answers the question?")


def arc(df, path, rng):
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if r.answerKey not in labels or not short(r.question):
            continue
        yield choice_row(r.question, "Which option answers the question?", labels, r.answerKey, dict(zip(labels, texts)))


def openbookqa(df, path, rng):
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if r.answerKey not in labels or not short(r.question_stem):
            continue
        yield choice_row(r.question_stem, "Which option completes or answers the question?", labels, r.answerKey, dict(zip(labels, texts)))


def commonsense_qa(df, path, rng):
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if r.answerKey not in labels or not short(r.question):
            continue
        yield choice_row(r.question, "Which option answers the question?", labels, r.answerKey, dict(zip(labels, texts)))


def sciq(df, path, rng):
    for r in df.itertuples():
        options = [r.correct_answer, r.distractor1, r.distractor2, r.distractor3]
        if any(not isinstance(o, str) or not o for o in options) or len(set(options)) < 4 or not short(r.question):
            continue
        rng.shuffle(options)
        state = {"question": r.question}
        if short(r.support) and r.support.strip():
            state["context"] = r.support
        yield choice_row(state, "Which option answers the question?", options, r.correct_answer)


def boolq(df, path, rng):
    for r in df.itertuples():
        if not short(r.passage) or not short(r.question):
            continue
        yield noul_row(r.passage, f"According to the passage: {r.question}?", bool(r.answer))


def hellaswag(df, path, rng):
    for r in df.itertuples():
        endings = list(r.endings)
        if len(endings) != 4 or not short(r.ctx) or str(r.label) not in "0123":
            continue
        yield lettered(endings, int(r.label), r.ctx, "Which ending follows the text most naturally?")


def truthful_qa(df, path, rng):
    for r in df.itertuples():
        choices, labels = list(r.mc1_targets["choices"]), list(r.mc1_targets["labels"])
        if 1 not in labels or len(choices) > 12:
            continue
        order = list(range(len(choices)))
        rng.shuffle(order)
        yield lettered([choices[i] for i in order], order.index(labels.index(1)), r.question, "Which option is the true answer?")


def labelled(column, text_column, instructions, names=None, state=None):
    """A classification dataset: the label names (from the parquet's metadata unless
    given) are the options."""
    def convert(df, path, rng):
        found = names or class_names(path, column)
        if not found:
            return
        for r in df.itertuples():
            text = getattr(r, text_column)
            label = getattr(r, column)
            if not short(text) or not (0 <= int(label) < len(found)):
                continue
            s = state(r) if state else text
            yield choice_row(s, instructions, list(found), found[int(label)])
    return convert


def labelled_noul(column, text_column, instructions, true_value=1, state=None):
    def convert(df, path, rng):
        for r in df.itertuples():
            text = getattr(r, text_column)
            label = getattr(r, column)
            if not short(text):
                continue
            yield noul_row(state(r) if state else text, instructions, int(label) == true_value)
    return convert


def intents(column, text_column, instructions, shown=8):
    """A many-intent dataset: the right intent among `shown` options drawn at random."""
    def convert(df, path, rng):
        found = class_names(path, column)
        if not found:
            return
        for r in df.itertuples():
            text, label = getattr(r, text_column), int(getattr(r, column))
            if not short(text) or not (0 <= label < len(found)):
                continue
            others = rng.sample([n for i, n in enumerate(found) if i != label], shown - 1)
            options = others + [found[label]]
            rng.shuffle(options)
            yield choice_row(text, instructions, [o.replace("_", " ") for o in options], found[label].replace("_", " "))
    return convert


def language_id(df, path, rng):
    names = list(LANGUAGES.values())
    for r in df.itertuples():
        if r.labels in LANGUAGES and short(r.text):
            yield choice_row(r.text, "Which language is the text written in?", names, LANGUAGES[r.labels])


def snli(df, path, rng):
    names = ["entailment", "neutral", "contradiction"]
    for r in df.itertuples():
        if int(r.label) not in (0, 1, 2) or not short(r.premise) or not short(r.hypothesis):
            continue
        yield choice_row({"premise": r.premise, "hypothesis": r.hypothesis},
                         "Does the premise entail the hypothesis, contradict it, or neither?", names, names[int(r.label)])


def rosetta(df, path, rng):
    wanted = set(CODE_LANGUAGES)
    df = df[df.language_name.isin(wanted)]
    for r in df.itertuples():
        if not short(r.code) or len(r.code) < 40:
            continue
        others = rng.sample([l for l in CODE_LANGUAGES if l != r.language_name], 7)
        options = others + [r.language_name]
        rng.shuffle(options)
        yield choice_row({"code": r.code}, "Which programming language is this code written in?", options, r.language_name)


def defects(df, path, rng):
    for r in df.itertuples():
        if not short(r.func):
            continue
        yield noul_row({"code": r.func}, "Does this C function contain a security defect?", bool(r.target))


# name -> (dataset, preferred split markers, converter)
DATASETS = {
    "hf_mmlu": ("cais/mmlu", ("all/test",), mmlu),
    "hf_arc": ("allenai/ai2_arc", ("train",), arc),
    "hf_openbookqa": ("allenai/openbookqa", ("main/train",), openbookqa),
    "hf_commonsense_qa": ("tau/commonsense_qa", ("train",), commonsense_qa),
    "hf_sciq": ("allenai/sciq", ("train",), sciq),
    "hf_boolq": ("google/boolq", ("train",), boolq),
    "hf_hellaswag": ("Rowan/hellaswag", ("train",), hellaswag),
    "hf_truthful_qa": ("truthfulqa/truthful_qa", ("multiple_choice",), truthful_qa),
    "hf_language_id": ("papluca/language-identification", ("train",), language_id),
    "hf_sst2": ("stanfordnlp/sst2", ("train",), labelled("label", "sentence", "What is the sentiment of the sentence?", ["negative", "positive"])),
    "hf_imdb": ("stanfordnlp/imdb", ("train",), labelled("label", "text", "What is the sentiment of the review?", ["negative", "positive"])),
    "hf_ag_news": ("fancyzhx/ag_news", ("train",), labelled("label", "text", "What is the topic of the news text?", ["world", "sports", "business", "science and technology"])),
    "hf_dbpedia": ("fancyzhx/dbpedia_14", ("train",), labelled("label", "content", "What kind of thing does the text describe?")),
    "hf_emotion": ("SetFit/emotion", ("train",), labelled("label", "text", "Which emotion does the text express?", ["sadness", "joy", "love", "anger", "fear", "surprise"])),
    "hf_sms_spam": ("ucirvine/sms_spam", ("train",), labelled_noul("label", "sms", "Is this message spam?")),
    "hf_clinc": ("clinc/clinc_oos", ("plus/train",), intents("intent", "text", "What is the user's intent?")),
    "hf_tweet_sentiment": ("cardiffnlp/tweet_eval", ("sentiment/train",), labelled("label", "text", "What is the sentiment of the tweet?", ["negative", "neutral", "positive"])),
    "hf_tweet_emotion": ("cardiffnlp/tweet_eval", ("emotion/train",), labelled("label", "text", "Which emotion does the tweet express?", ["anger", "joy", "optimism", "sadness"])),
    "hf_tweet_irony": ("cardiffnlp/tweet_eval", ("irony/train",), labelled_noul("label", "text", "Is the tweet ironic?")),
    "hf_tweet_offensive": ("cardiffnlp/tweet_eval", ("offensive/train",), labelled_noul("label", "text", "Is the tweet offensive?")),
    "hf_snli": ("stanfordnlp/snli", ("train",), snli),
    "hf_rte": ("nyu-mll/glue", ("rte/train",), labelled_noul("label", "sentence1", "Does the first sentence entail the second?", true_value=0,
                                                              state=lambda r: {"first": r.sentence1, "second": r.sentence2})),
    "hf_qnli": ("nyu-mll/glue", ("qnli/train",), labelled_noul("label", "sentence", "Does the sentence answer the question?", true_value=0,
                                                                state=lambda r: {"question": r.question, "sentence": r.sentence})),
    "hf_mrpc": ("nyu-mll/glue", ("mrpc/train",), labelled_noul("label", "sentence1", "Do the two sentences mean the same thing?",
                                                                state=lambda r: {"first": r.sentence1, "second": r.sentence2})),
    "hf_cola": ("nyu-mll/glue", ("cola/train",), labelled_noul("label", "sentence", "Is the sentence grammatically acceptable?")),
    "hf_rosetta": ("christopher/rosetta-code", ("train",), rosetta),
    "hf_defects": ("code_x_glue_cc_defect_detection", ("train",), defects),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-dataset", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    total = 0
    for family, (name, prefer, convert) in DATASETS.items():
        df, path = load(args.data, name, prefer, limit=args.per_dataset * 20)
        if df is None:
            print(f"{family:22s} missing: {name}")
            continue
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        rows = []
        try:
            for row in convert(df, path, rng):
                rows.append(row)
                if len(rows) >= args.per_dataset:
                    break
        except (AttributeError, KeyError, TypeError) as e:
            print(f"{family:22s} schema mismatch in {name}: {e}")
            continue
        if not rows:
            print(f"{family:22s} nothing usable in {name}")
            continue
        with (args.out / f"{family}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{family:22s} {len(rows):6d}")
    print(f"{total} requests in {args.out}")


if __name__ == "__main__":
    main()
