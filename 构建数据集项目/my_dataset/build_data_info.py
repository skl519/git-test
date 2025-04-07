import os
import json
import random

random.seed(0)

# 加载提示词列表
def load_label(label_file):
    with open(label_file, 'r', encoding='utf-8') as file:
        prompts = json.load(file)
    return prompts

def lode_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as file:
        res = file.readlines()
    return res

# 生成数据集
def generate_dataset(video_dir, label_dict, prompt_list):
    dataset = []

    # 遍历视频目录中的所有类别文件夹
    for category in os.listdir(video_dir):
        category_path = os.path.join(video_dir, category)
        if os.path.isdir(category_path):
            videos = [f for f in os.listdir(category_path) if f.endswith('.mp4')]
            if not videos:
                continue

            for video in videos:
                video_path = os.path.join(category_path, video)

                # 随机选择一个提示词
                selected_prompt = random.choice(prompt_list)
                # 加载回答

                # 构建消息对
                messages = [
                    {"content": selected_prompt.strip(), "role": "user"},
                    {"content": label_dict[video], "role": "assistant"}
                ]

                # 添加到数据集中
                dataset.append({
                    "messages": messages,
                    "videos": [video_path]
                })

    return dataset

def main():
    # 视频文件根目录
    video_directory = 'D:\Desktop\构建数据集项目\my_dataset'

    # 提示词列表文件路径
    prompt_file = 'prompt.txt'
    label_file = 'new_dataset_describe.json'

    # 加载提示词列表
    label_dict = load_label(label_file)
    prompt_list = lode_prompt(prompt_file)
    print(prompt_list)

    # 生成数据集
    dataset = generate_dataset(video_directory, label_dict, prompt_list)

    # 输出为JSON格式并保存
    output_json = json.dumps(dataset, ensure_ascii=False, indent=4)
    print(output_json)  # 打印输出以供检查

    with open('my_dataset.json', 'w', encoding='utf-8') as outfile:
        outfile.write(output_json)

    print("Dataset has been generated and saved to output_dataset.json")


if __name__ == "__main__":
    main()