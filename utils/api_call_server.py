import json
import bottle
from bottle import request
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

class Qwen3_8B_vLLM:
    """
    Qwen3-8B 后端推理 API（vLLM）
    """

    def __init__(self, model_path, tp=1):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        tp=torch.cuda.device_count()

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            gpu_memory_utilization=0.7,   # 防止 OOM
        )

        print(f"🔥 Qwen3-8B vLLM Engine 已加载 (TP={tp})")


    def chat(self, system_prompt, user_prompt,
             max_tokens=1024, thinking=False, seed=42,output_logits=False):

        # 1) Chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",    "content": user_prompt}
        ]
        # print("===== System Prompt =====")
        # print(system_prompt)

        # print("===== User Prompt =====")
        # print(user_prompt)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )


        if not output_logits:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                seed=seed
            )
        else:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                seed=seed,
                logprobs=5,    
            )

        # 3) 调用 vLLM 生成
        outputs = self.llm.generate(prompt, params)
        out=outputs[0].outputs[0]
        text =out.text.strip()

        res={}

        # 4) 解析 <think>
        if thinking:
            THINK_END = "</think>"
            if THINK_END in text:
                think_content, final = text.split(THINK_END, 1)
                think_content = think_content.replace("<think>", "").strip()
                res= {
                    "thinking": think_content,
                    "response": final.strip()
                }
            else:
                res= {"thinking": "", "response": text}
        else:
            res={"thinking": "", "response": text}
        if output_logits:
            res.update({
            "token_ids": out.token_ids,
            "logprobs": out.logprobs,  # dict[token_id] -> logprob
            })
        return res

    def chat_batch(
        self,
        system_prompts: list[str],
        user_prompts: list[str],
        max_tokens=1024,
        thinking=False,
        seed=42,
        output_logits=False
    ):
        assert len(system_prompts) == len(user_prompts)
        B = len(system_prompts)

        # 1) 构造 batch prompts
        prompts = []
        for sys_p, usr_p in zip(system_prompts, user_prompts):
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            prompts.append(prompt)

        # 2) Sampling params
        if not output_logits:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                seed=seed,
            )
        else:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                seed=seed,
                logprobs=5,
            )

        outputs = self.llm.generate(prompts, params)
        results = []

        for out in outputs:
            gen = out.outputs[0]
            text = gen.text.strip()

            if thinking:
                THINK_END = "</think>"
                if THINK_END in text:
                    think_content, final = text.split(THINK_END, 1)
                    think_content = think_content.replace("<think>", "").strip()
                    res = {
                        "thinking": think_content,
                        "response": final.strip()
                    }
                else:
                    res = {"thinking": "", "response": text}
            else:
                res = {"thinking": "", "response": text}

            if output_logits:
                # gen.logprobs: List[Dict[token_id, Logprob]]
                simple_logprobs = []

                for step_lp in gen.logprobs:
                    simple_step = {
                        int(tok): float(lp.logprob)
                        for tok, lp in step_lp.items()
                    }
                    simple_logprobs.append(simple_step)

                res.update({
                    "token_ids": gen.token_ids,
                    "logprobs": simple_logprobs,   # 现在是 List[Dict[int, float]]
                })

            results.append(res)

        return json.dumps(results, ensure_ascii=False)




# ------------------ 启动 API -----------------------

app = bottle.Bottle()

# 全局模型（单实例，多线程复用）
engine = None


@app.post("/chat")
def chat_api():
    """
    请求格式：
    {
        "system": "You are ChatGPT",
        "user": "Introduce yourself",
        "max_tokens": 512,
        "thinking": false
    }
    """

    data = json.loads(request.body.read().decode())
    system_prompt = data.get("system", "")
    user_prompt    = data.get("user", "")
    max_tokens     = data.get("max_tokens", 512)
    thinking       = data.get("thinking", False)
    seed = data.get("seed", 42)

    result = engine.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        thinking=thinking,
        seed=seed
    )

    return json.dumps(result, ensure_ascii=False)

@app.post("/chat_batch")
def chat_batch_api():
    """
    批量 Chat 接口
    请求格式：
    {
        "system": ["You are ChatGPT", ...],
        "user": ["Introduce yourself", ...],
        "max_tokens": 512,
        "thinking": false
    }
    """
    
    data = json.loads(request.body.read().decode())
    system_prompts = data.get("system", [])
    user_prompts    = data.get("user", [])
    max_tokens     = data.get("max_tokens", 512)
    thinking       = data.get("thinking", False)
    seed = data.get("seed", 42)
    output_logits = data.get("output_logits", False)

    results = engine.chat_batch(
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        max_tokens=max_tokens,
        thinking=thinking,
        seed=seed,
        output_logits=output_logits
    )

    return json.dumps(results, ensure_ascii=False)


def run_server(model_path, tp):
    global engine
    engine = Qwen3_8B_vLLM(model_path, tp)

    print("🚀 API 已启动：http://0.0.0.0:8008/chat")
    bottle.run(app, host="0.0.0.0", port=8008, server="tornado")

import os
import argparse
import torch
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/Qwen3-8B")

    args= parser.parse_args()
    
    tp=torch.cuda.device_count()

    run_server(args.model_path, tp=tp)
