import openai
import os

openai.api_key = "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"

def gpt_generate_answer(prompt):
    # 使用gpt3.5模型text-davinci-003
    # 或使用gpt3模型text-davinci-002
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.7
    )
    message = completions.choices[0].text.strip()
    return message

def chatgpt_generate_answer(messages):
    # 使用chatgpt模型，支持多轮对话
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    res_msg = completion.choices[0].message
    return res_msg["content"].strip()


if __name__ == '__main__':
    messages = [{"role": "system", "content": "You are now a very useful assistant!"}]
    while True:
        prompt = input("Please input your question:")
        messages.append({"role": "user", "content": prompt})
        res_msg = chatgpt_generate_answer(messages)
        messages.append({"role": "assistant", "content": res_msg})
        print(res_msg)

# python3 directqe_robustness/scripts/use_chatgpt.py
