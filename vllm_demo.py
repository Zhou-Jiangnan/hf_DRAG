from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is a",
    ]

    model_id = "meta-llama/Llama-3.2-1B"

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model=model_id)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()

