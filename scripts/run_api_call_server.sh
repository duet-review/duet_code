pip install bottle flask transformers accelerate bitsandbytes gpustat tornado numpy pandas 
MODEL_PATH=  # models/Qwen3-8B  # set your model path here
cd duet
CUDA_VISIBLE_DEVICES=6,7 python utils/api_call_server.py --model_path $MODEL_PATH