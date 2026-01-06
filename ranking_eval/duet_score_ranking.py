#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from tqdm import tqdm
import pandas as pd

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import requests


from verl.prompt_config import (
    DUET_PROFILE_SYSTEM_PROMPT,
    DUET_PROFILE_GENERATOR_PROMPT,
    DENSE_DUET_RATING_SYSTEM_PROMPT,
    DENSE_DUET_RATING_PREDICTOR_PROMPT
)
def extract_duet_profiles(completion_text: str):
    """
    One-pass extraction of DUET-style profiles.

    Expected structure:
    <user_profile>
        <cue>...</cue>
        <constructed_prompt>...</constructed_prompt>
        <profile>...</profile>
    </user_profile>
    <item_profile>
        <cue>...</cue>
        <constructed_prompt>...</constructed_prompt>
        <profile>...</profile>
    </item_profile>

    Returns:
    {
        "user": {
            "cue": str | None,
            "constructed_prompt": str | None,
            "profile": str | None,
        },
        "item": {
            "cue": str | None,
            "constructed_prompt": str | None,
            "profile": str | None,
        }
    }
    """

    text = completion_text
    lower = completion_text.lower()

    result = {
        "user": {"cue": None, "constructed_prompt": None, "profile": None},
        "item": {"cue": None, "constructed_prompt": None, "profile": None},
    }

    current_scope = None  # None | "user" | "item"
    i = 0
    n = len(lower)

    while i < n:
        # enter scopes
        if lower.startswith("<user_profile>", i):
            current_scope = "user"
            i += len("<user_profile>")
            continue

        if lower.startswith("<item_profile>", i):
            current_scope = "item"
            i += len("<item_profile>")
            continue

        # exit scopes
        if lower.startswith("</user_profile>", i):
            current_scope = None
            i += len("</user_profile>")
            continue

        if lower.startswith("</item_profile>", i):
            current_scope = None
            i += len("</item_profile>")
            continue

        # inside a profile: extract sub-tags
        if current_scope is not None:
            for tag in ("cue", "constructed_prompt", "profile"):
                open_tag = f"<{tag}>"
                close_tag = f"</{tag}>"

                if lower.startswith(open_tag, i):
                    start = i + len(open_tag)
                    end = lower.find(close_tag, start)
                    if end != -1:
                        result[current_scope][tag] = text[start:end].strip()
                        i = end + len(close_tag)
                        break
            else:
                i += 1
        else:
            i += 1

    return result


def profile_content(row, df, train_df):
    uid, pos_iid = row["user_id"], row["item_id"]
    current_time = row["unixReviewTime"]

    user_title = row.get("reviewerName", f"User_{uid}")


    all_df=pd.concat([df,train_df])
    
    user_history = all_df[
        (all_df.user_id == uid) & (all_df.unixReviewTime < current_time)
    ].sort_values("unixReviewTime").tail(10)

    if user_history.empty:
        user_avg_rating = "N/A (no historical data)"
        user_history_text = "[No historical interactions available for this user]"
    else:
        user_avg_rating = f"{user_history['ratings'].mean():.1f}"
        user_history_text = "\n".join([
            f"[History {i+1}] Item: {r['title']}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()}"
            for i, (_, r) in enumerate(user_history.iterrows())
        ])
    item_ids=[pos_iid]+row["negative_item_ids"]

    all_contents=[]

    for iid in item_ids:
        item_title_df=all_df[all_df.item_id==iid]["title"]
        if item_title_df.empty:
            item_title=f"Item_{iid}"
        else:
            item_title=item_title_df.iloc[0]

        item_history = all_df[
            (all_df.item_id == iid) & (all_df.unixReviewTime < current_time)
        ].sort_values("unixReviewTime").tail(10)

        if item_history.empty:
            item_avg_rating = "N/A (no historical data)"
            item_history_text = "[No historical reviews available for this item]"
        else:
            item_avg_rating = f"{item_history['ratings'].mean():.1f}"
            item_history_text = "\n".join([
                f"[Review {i+1}] User: {r.get('reviewerName', 'Anonymous')}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()}"
                for i, (_, r) in enumerate(item_history.iterrows())
            ])

        all_contents.append(
            {
                "user_id": uid,
                "item_id": iid,
                "user_title": user_title,
                "item_title": item_title,
                "user_avg": user_avg_rating,
                "item_avg": item_avg_rating,  
                "user_text": user_history_text,
                "item_text": item_history_text,
            }
        )
    return all_contents

def call_batch(system_prompt, user_prompt,seed=42):
    api = "http://0.0.0.0:8008/chat_batch"

    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": 20,
        "thinking": False,
        "seed":seed,
        "output_logits": False
    }

    res = requests.post(api, json=payload)
    results= res.json()
    res=json.loads(results)
    return res


from utils.api_call import Qwen3_8B



# =========================
# 评估指标
# =========================

def eval_one(scored, k):
    ranked = sorted(scored, key=lambda x: x["confidence"], reverse=True)
    labels = [x["label"] for x in ranked]

    hit = int(1 in labels[:k])
    ndcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(labels[:k])
    )

    rr = 0.0
    for i, rel in enumerate(labels):
        if rel == 1:
            rr = 1 / (i + 1)
            break

    return hit, ndcg, rr

RATING_TAG_PATTERN = re.compile(
    r"<rating>\s*([1-5](?:\.\d+)?)\s*</rating>",
    re.DOTALL
)

def extract_rating(rating_text: str) -> tuple[float, bool]:
    matches = list(RATING_TAG_PATTERN.finditer(rating_text))
    if not matches:
        print("No valid <rating> tag found.")
        return 0.0, False

    tag_content = matches[-1].group(1).strip()
    try:
        return float(tag_content), True
    except Exception:
        return 0.0, False
# =========================
# 主流程
# =========================
def batch_iter(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

import os
from collections import defaultdict
import torch

def main(args):
    print("==============args==============")
    print(args)

    # =====================================================
    # 基础配置
    # =====================================================
    args.tp = torch.cuda.device_count()
    print(f"Using {args.tp} GPUs.")

    BATCH_USERS = 64   # 推荐 8 / 16

    print("📦 Loading data")
    train_df = pd.read_pickle(args.train_pkl)
    test_df = pd.read_pickle(args.test_pkl)

    # test_df=test_df[:20]

    def yelp_adaptive(df):
        df["unixReviewTime"] = (
            pd.to_datetime(df["date"], utc=True)
            .astype("int64") // 10**9
        )
        return df

    if args.dataset_type == "Yelp":
        train_df = yelp_adaptive(train_df)
        test_df = yelp_adaptive(test_df)

    print("🤖 Loading model")
    model = Qwen3_8B(args.model_path, tp=args.tp, lora=args.lora)

    # =====================================================
    # 评测指标
    # =====================================================
    metrics = {f"Hit@{k}": [] for k in args.k_list}
    metrics.update({f"NDCG@{k}": [] for k in args.k_list})
    metrics["MRR"] = []

    os.makedirs(args.dump_dir, exist_ok=True)
    dump_path = os.path.join(args.dump_dir, "eval_dump.jsonl")
    dump_f = open(dump_path, "w", encoding="utf-8")
    print(f"📝 Dumping to {dump_path}")

    progress_bar = tqdm(total=len(test_df), desc="Processing (DUET row-level)")

    for batch_df in batch_iter(test_df, BATCH_USERS):

        # -------- row-level 容器 --------
        row_scored = defaultdict(list)
        records_by_row = defaultdict(list)

        # -------- pair-level 收集 --------
        pair_prompts = []
        pair_meta = []   # (row_idx, row, content)

        for row_idx, row in batch_df.iterrows():
            all_contents = profile_content(row, test_df, train_df)
            for content in all_contents:
                pair_prompts.append(
                    DUET_PROFILE_GENERATOR_PROMPT.format(
                        user_title=content["user_title"],
                        item_title=content["item_title"],
                        user_history_text=content["user_text"],
                        item_history_text=content["item_text"],
                        user_avg_rating=content["user_avg"],
                        item_avg_rating=content["item_avg"],
                    )
                )
                pair_meta.append((row_idx, row, content))

        # =================================================
        # Step 1: DUET joint profile（大 batch）
        # =================================================
        profile_outputs = model.generate_batch(
            [DUET_PROFILE_SYSTEM_PROMPT] * len(pair_prompts),
            pair_prompts,
            max_new_tokens=4096,
        )

        # =================================================
        # Step 2: 构造 rating prompts
        # =================================================
        rank_prompts = []
        rank_meta = []   # (row_idx, row, content, profile)

        for (row_idx, row, content), profile in zip(pair_meta, profile_outputs):
            parsed = extract_duet_profiles(profile)

            rank_prompt = DENSE_DUET_RATING_PREDICTOR_PROMPT.format(
                user_title=content["user_title"],
                item_title=content["item_title"],
                user_profile=parsed["user"]["profile"],
                item_profile=parsed["item"]["profile"],
                user_avg_rating=content["user_avg"],
                item_avg_rating=content["item_avg"],
            )

            rank_prompts.append(rank_prompt)
            rank_meta.append((row_idx, row, content, profile))

        # =================================================
        # Step 3: Rating 推理（大 batch）
        # =================================================
        rank_outputs = call_batch(
            [DENSE_DUET_RATING_SYSTEM_PROMPT] * len(rank_prompts),
            rank_prompts
        )

        # =================================================
        # Step 4: 回收 → row-level
        # =================================================
        for (row_idx, row, content, profile), output in zip(rank_meta, rank_outputs):

            score, ok = extract_rating(output["response"])
            label = int(content["item_id"] == row["item_id"])

            row_scored[row_idx].append({
                "label": label,
                "confidence": float(score)
            })

            records_by_row[row_idx].append({
                "content": content,
                "profile": str(profile),
                "rank_output": output,
                "score": float(score)
            })

       # =================================================
        # Step 5: 评测 + Dump 
        # =================================================
        for row_idx, scored in row_scored.items():
            for k in args.k_list:
                hit, ndcg, rr = eval_one(scored, k)
                metrics[f"Hit@{k}"].append(hit)
                metrics[f"NDCG@{k}"].append(ndcg)
                metrics["MRR"].append(rr)

            dump_f.write(json.dumps({
                "row_idx": int(row_idx),
                "records": records_by_row[row_idx]
            }) + "\n")

        progress_bar.update(len(batch_df))

    progress_bar.close()
    dump_f.close()

    # =====================================================
    # 汇总结果
    # =====================================================
    results = {k: sum(v) / len(v) for k, v in metrics.items()}
    print("✅ Evaluation completed.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pkl", required=True)
    parser.add_argument("--test_pkl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--dump_dir", type=str, default="./duet_eval_dumps")
    parser.add_argument("--dataset_type", type=str, default="amazaon")
    args = parser.parse_args()

    main(args)
