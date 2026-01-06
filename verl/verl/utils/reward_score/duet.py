import sys
from prompt_config import (
    DUET_RATING_SYSTEM_PROMPT,
    DUET_RATING_PREDICTOR_PROMPT,
    DENSE_DUET_RATING_SYSTEM_PROMPT,
    DENSE_DUET_RATING_PREDICTOR_PROMPT
)
import re
import requests, json
def call(system_prompt, user_prompt,seed=42):
    api = "http://0.0.0.0:8008/chat"

    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": 20,
        "thinking": False,
        "seed":seed
    }

    res = requests.post(api, json=payload)
    text = res.json()["response"]
    return text

from typing import Dict, Optional

def extract_duet_profiles(completion_text: str) -> Dict[str, Dict[str, Optional[str]]]:
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


RATING_TAG_PATTERN = re.compile(
    r"<rating>\s*((?:[1-4]\.\d{2}|5\.00))\s*</rating>",
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

def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
            
    current_user_title = ground_truth["user_title"]
    current_item_title = ground_truth["item_title"]
    current_user_avg_rating = ground_truth["user_avg_rating"]
    current_item_avg_rating = ground_truth["item_avg_rating"]
    gt_rating = ground_truth["gt_rating"]
    # print("========message info=========")
    # print(extra_info["messages"])
    # print("===========solution_str============")
    # print(solution_str)


    result=extract_duet_profiles(solution_str)
    """
{
  "user": {
    "cue": "prefers strategic indie content",
    "constructed_prompt": "Describe strategic depth, engagement consistency, and stylistic preferences.",
    "profile": "A user who values strategic challenge and thoughtful gameplay, showing consistent interest in indie titles with depth."
  },
  "item": {
    "cue": "appeals to strategy-focused users",
    "constructed_prompt": "Describe core mechanics, strategic complexity, and target audience.",
    "profile": "This item emphasizes strategic decision-making and appeals to users who enjoy thoughtful and challenging experiences."
  }
}
    """
    user_profile = result["user"]["profile"]
    item_profile = result["item"]["profile"]

    user_cue= result["user"]["cue"]
    item_cue= result["item"]["cue"]
    user_constructed_prompt= result["user"]["constructed_prompt"]
    item_constructed_prompt= result["item"]["constructed_prompt"]


    total_reward=0
    predicted_rating = -1 # means error

            
    # 计算格式奖励
    format_reward = 0.0
    if user_profile is None or item_profile is None or user_cue is None or item_cue is None or user_constructed_prompt is None or item_constructed_prompt is None:
        print("Profile extraction failed or incomplete.....")
        format_reward = -1.0
        # 直接给负格式奖励，准确奖励为0
        total_reward = format_reward
        
    else:
        # 构造评分预测prompt
        rating_prompt = DUET_RATING_PREDICTOR_PROMPT.format(
            user_title=current_user_title,
            item_title=current_item_title,
            user_avg_rating=current_user_avg_rating,
            item_avg_rating=current_item_avg_rating,
            user_profile=user_profile,
            item_profile=item_profile
        )
        
        rating_text= call(
            DUET_RATING_SYSTEM_PROMPT,
            rating_prompt
        )


        predicted_rating, format_valid = extract_rating(rating_text)
          
        # 计算格式奖励
        if format_valid is False:
            format_reward = -1.0
            total_reward = format_reward
        else:
            # 计算准确奖励
            max_possible_reward = 1.0
            error = abs(predicted_rating - gt_rating)/4.0  # 归一化误差到[0,1]
            acc_reward = max_possible_reward - error
            
            # 总奖励 = 格式奖励 + 准确奖励
            total_reward = acc_reward + format_reward
    return {
        "score": total_reward,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "predicted_rating": predicted_rating,
        "gt_rating": gt_rating
    }


def dense_compute_reward(data_source, solution_str, ground_truth, extra_info=None):
            
    current_user_title = ground_truth["user_title"]
    current_item_title = ground_truth["item_title"]
    current_user_avg_rating = ground_truth["user_avg_rating"]
    current_item_avg_rating = ground_truth["item_avg_rating"]
    gt_rating = ground_truth["gt_rating"]
    # print("========message info=========")
    # print(extra_info["messages"])
    # print("===========solution_str============")
    # print(solution_str)


    result=extract_duet_profiles(solution_str)
    """
{
  "user": {
    "cue": "prefers strategic indie content",
    "constructed_prompt": "Describe strategic depth, engagement consistency, and stylistic preferences.",
    "profile": "A user who values strategic challenge and thoughtful gameplay, showing consistent interest in indie titles with depth."
  },
  "item": {
    "cue": "appeals to strategy-focused users",
    "constructed_prompt": "Describe core mechanics, strategic complexity, and target audience.",
    "profile": "This item emphasizes strategic decision-making and appeals to users who enjoy thoughtful and challenging experiences."
  }
}
    """
    user_profile = result["user"]["profile"]
    item_profile = result["item"]["profile"]

    user_cue= result["user"]["cue"]
    item_cue= result["item"]["cue"]
    user_constructed_prompt= result["user"]["constructed_prompt"]
    item_constructed_prompt= result["item"]["constructed_prompt"]


    total_reward=0
    predicted_rating = -1 # means error

            
    # 计算格式奖励
    format_reward = 0.0
    if user_profile is None or item_profile is None or user_cue is None or item_cue is None or user_constructed_prompt is None or item_constructed_prompt is None:
        print("Profile extraction failed or incomplete.....")
        format_reward = -1.0
        # 直接给负格式奖励，准确奖励为0
        total_reward = format_reward
        
    else:
        # 构造评分预测prompt
        rating_prompt = DENSE_DUET_RATING_PREDICTOR_PROMPT.format(
            user_title=current_user_title,
            item_title=current_item_title,
            user_avg_rating=current_user_avg_rating,
            item_avg_rating=current_item_avg_rating,
            user_profile=user_profile,
            item_profile=item_profile
        )
        
        rating_text= call(
            DENSE_DUET_RATING_SYSTEM_PROMPT,
            rating_prompt
        )
        # print("Dense rating text:", rating_text)


        predicted_rating, format_valid = extract_rating(rating_text)
          
        # 计算格式奖励
        if format_valid is False:
            format_reward = -1.0
            total_reward = format_reward
        else:
            # 计算准确奖励
            max_possible_reward = 1.0
            error = abs(predicted_rating - gt_rating)/4.0  # 归一化误差到[0,1]
            acc_reward = max_possible_reward - error
            
            # 总奖励 = 格式奖励 + 准确奖励
            total_reward = acc_reward + format_reward
    return {
        "score": total_reward,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "predicted_rating": predicted_rating,
        "gt_rating": gt_rating
    }


