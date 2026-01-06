from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from vllm.lora.request import LoRARequest
class Qwen3_8B:
    """
    Qwen3-8B inference using vLLM backend.
    """

    def __init__(self, model_path: str,tp=4,lora=False):
        # Load tokenizer normally
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.lora=lora
        if self.lora:
            self.lora_local_path=f"{model_path}/lora_adapter"
            print("Using LoRA-adapted model from:", self.lora_local_path)
            self.lora_request=LoRARequest("self_adapter_v1", 1, lora_local_path=self.lora_local_path)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            enable_prefix_caching=True,   # ✅ 开启 prefix caching
            gpu_memory_utilization=0.8,   # ✅ 常用推荐值
            enable_lora=self.lora,            # ✅ 启用 LoRA
            max_lora_rank=32,
        )

        print("🚀 Qwen3‑8B loaded using vLLM.")
    def generate_batch(
        self,
        system_prompts: list[str],
        user_prompts: list[str],
        max_new_tokens=2048,
        thinking=False,
    ):
        assert len(system_prompts) == len(user_prompts)

        messages_list = []
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
            messages_list.append(prompt)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.6 if thinking else 0.7,
            top_p=0.95 if thinking else 0.8,
            top_k=20,
            min_p=0,
        )

        outputs = self.llm.generate(
            prompts=messages_list,
            sampling_params=sampling_params,
            lora_request=self.lora_request if self.lora else None,
        )

        results = []
        for out in outputs:
            text = out.outputs[0].text.strip()
            results.append(text)

        return results

    # -----------------------------------------------------
    # Chat-style generation (system + user)
    # -----------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens=2048, thinking=False):

        # print("===== System Prompt =====")
        # print(system_prompt)

        # print("===== User Prompt =====")
        # print(user_prompt)

        # 🔥 构造 Chat Template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking  # 控制是否添加 <think> 标签
        )

        # ✅ 按照是否思维模式，使用不同 Sampling 参数
        if thinking:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0
            )

        # 🔮 执行 vLLM 生成

        outputs = self.llm.generate(
            prompt,
            sampling_params=sampling_params,
            lora_request=self.lora_request if self.lora else None,
        )

        full_output = outputs[0].outputs[0].text.strip()

        # 🤔 是否包含思维模式输出
        if thinking:
            THINK_END_TOKEN = "</think>"
            if THINK_END_TOKEN in full_output:
                thinking_content, final = full_output.split(THINK_END_TOKEN, 1)
                thinking_content = thinking_content.replace("<think>", "").strip()
                return thinking_content, final.strip()

            # fallback：没生成思考段
            return "", full_output
        else:
            # 非思维模式，直接返回
            return "", full_output

