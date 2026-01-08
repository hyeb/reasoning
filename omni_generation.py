import os
import json
import pandas as pd

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

import google.generativeai as genai
import PIL.Image
from tqdm import tqdm



load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


generate_model = genai.GenerativeModel('gemini-3-flash-preview')
evaluate_model = genai.GenerativeModel('gemini-2.5-pro')

data_dir = "/Users/hyeb/selectstar/innov_project/25pj136"
img_extensions = ('.jpg', '.jpeg', '.png', '.webp')
audio_extensions = ('.mp3', '.wav', '.aac', '.flac')

# sample_image_path = data_dir + "/omni_0013.png"
# sample_audio_path = data_dir + "/omni_clip_13_sec.mp3"


omni_prompt = """
# Role
너는 영상과 오디오를 동시에 분석하는 멀티모달 데이터셋 구축 전문가야. 주어진 이미지와 오디오 정보를 결합하여, VLM/Omni 모델의 성능을 측정할 수 있는 객관식(MCQ) QA 세트를 생성해야 해.

# Task 유형 (총 8가지)
다음 Task 중 이미지/오디오의 특성에 가장 적합한 것을 하나 선택하여 문제를 생성해줘.
1. Action & Activity: 현재 수행 중인 행위 파악
2. Story Description: 이야기 개요 및 전개 이해
3. Plot Inference: 사건의 원인, 결과, 미래 추론
4. Object Identification & Description: 객체 인식 및 속성 묘사
5. Contextual & Environmental: 주변 상황 및 환경 정보 파악
6. Identity & Relationship: 등장 인물 및 관계 파악
7. Text & Symbols: 화면 내 문자 및 기호 이해
8. Count & Quantity: 수량 및 개수 기반 추론

# 필수 지침 (Constraint)
1. 모달리티 간 상호 참조: 반드시 이미지(시각)와 오디오(청각) 정보를 모두 참조해야만 풀 수 있는 문제를 만들어. 
2. 난이도 조절: '보통' 혹은 '상' 난이도로 구성하며, 단순히 보이는 것을 묻기보다 인과관계나 맥락적 추론을 포함해줘.
3. 객관식 구성: 정답 1개와 매력적인 오답 3개를 포함한 4지 선다형으로 구성해. 정답과 유사하여 헷갈릴 수 있는 오답 하나를 꼭 포함해야해.

# Output Format (JSON)
반드시 다음 구조의 JSON 형식으로만 출력하라.
{
  "selected_task": "위 8개 Task 중 선택한 유형",
  "difficulty": "보통 또는 상",
  "question": "이미지와 오디오를 모두 고려한 구체적인 질문",
  "options": {
    "A": "선택지 1",
    "B": "선택지 2",
    "C": "선택지 3",
    "D": "선택지 4"
  },
  "answer": "정답",
  "reasoning": "시각적 요소와 청각적 요소가 어떻게 결합되어 정답이 도출되는지에 대한 근거"
}
""" 

class Options(BaseModel):
    A: str 
    B: str
    C: str 
    D: str 

class OqaOutput(BaseModel):
    selected_task: str
    difficulty: str
    question: str
    options: Options
    answer: Literal["A", "B", "C", "D"]
    reasoning: str


eval_prompt = """
# Role
너는 시각적 증거(Image)와 청각적 증거(Audio)를 통합하여 복합적인 문제를 해결하는 멀티모달 분석가다.
단일 모달리티만으로는 해결할 수 없는 문제를 두 정보의 상관관계를 분석하여 논리적으로 해결하라.

# Input Data
- Question: {question}
- Options: {options}
- Context: 제공된 이미지와 오디오 정보를 동시에 분석해야 함.

# Task Instructions
1. 멀티모달 통합 분석: 이미지에서 보이는 객체/상황과 오디오에서 들리는 소리/대사를 연결하여 정답을 도출하라.
2. 독립적 증거 제시: 정답의 근거를 설명할 때 반드시 '시각적 근거'와 '청각적 근거'를 각각 구분하여 명시하라.
3. 논리적 증명: 두 모달리티의 정보가 어떻게 상호작용하여 결론에 도달했는지 추론 과정을 기술하라.

# Response Format (JSON only)
{{
    "model_answer": "알파벳 하나 (A, B, C, D 중 택1)",
    "model_reason": {{
        "visual_evidence": "이미지에서 발견한 구체적인 시각적 단서",
        "audio_evidence": "오디오에서 확인한 구체적인 청각적 단서",
        "integrated_inference": "두 단서를 결합하여 정답을 도출한 최종 논리적 근거"
    }}
}}
"""
class Reasons(BaseModel):
    visual_evidence: str
    audio_evidence: str
    integrated_inference: str

class EvalOutput(BaseModel):
    model_answer: str
    model_reason: Reasons


def extract_json_str(raw_text: str) -> str:
    """
    응답에서 JSON 객체 부분만 잘라내는 함수.
    """
    text = raw_text.strip()

    # 코드블록 제거
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[len("json"):].strip()
            if p.startswith("{") and p.endswith("}"):
                return p

    # 첫 '{' ~ 마지막 '}' 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]

    return text


data_dict = {}

for root, dirs, files in os.walk(data_dir + "/data/omni_sample_data"):
    for file in files:
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        full_path = os.path.join(root, file)

        if name not in data_dict:
            data_dict[name] = {'img': None, 'audio': None}
        
        if ext in img_extensions:
            data_dict[name]['img'] = full_path
        elif ext in audio_extensions:
            data_dict[name]['audio'] = full_path

DATA_SAVE_PATH = data_dir + "/omni_results/omni_backup.jsonl"
OUTPUT_PATH = data_dir + "/omni_results"

## === Omni Data Generation ===
omni_datas = []
with open(DATA_SAVE_PATH, "w", encoding="utf-8") as f:
    for paths in tqdm(data_dict.values()):
        try:
            if paths["img"] is None or paths["audio"] is None:
                continue
            
            img_path = paths["img"]
            audio_path = paths["audio"]

            img = PIL.Image.open(img_path)
            audio = genai.upload_file(path=audio_path)

            omni_response = generate_model.generate_content([omni_prompt, img, audio])

            omni_json = extract_json_str(omni_response.text)
            omni_dict = json.loads(omni_json)
            omni_result = OqaOutput.model_validate(omni_dict)

            record = {
                "file_name": os.path.basename(img_path),
                "img_path": img_path,
                "audio_path": audio_path,
                **omni_result.model_dump()
            }

            omni_datas.append(omni_dict)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        except Exception as e:
            print(f"Error at {os.path.basename(img_path)}: {e}")

norm_data = pd.json_normalize(omni_datas, sep="_").to_dict(orient="records")[0]
pd.DataFrame(norm_data).to_csv(OUTPUT_PATH+"/omni_data_only.csv", index=False, encoding='utf-8-sig')
print("Save a omni data file")



## === Modality Check ===
omni_records = []
if os.path.exists(DATA_SAVE_PATH):
    with open(DATA_SAVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            omni_records.append(json.loads(line))
else:
    print(f"{DATA_SAVE_PATH} 파일이 없습니다.")


eval_results = []

for item in tqdm(omni_datas, desc="Step 2: Modality Evaluating")

## === Quality Check === 

final_results = []

for paths in tqdm(data_dict.values()):
    try: 
        if paths["img"] is None or paths["audio"] is None:
            continue

        img_path = paths['img']
        audio_path = paths['audio']

        img = PIL.Image.open(img_path)
        audio = genai.upload_file(path=audio_path)


        omni_response = generate_model.generate_content([omni_prompt, img, audio])

        omni_json = extract_json_str(omni_response.text)
        omni_dict = json.loads(omni_json)
        omni_result = OqaOutput.model_validate(omni_dict)
        print("success omni_result")


        curr_omni_eval_prompt = eval_prompt.format(
            question = omni_result.question,
            options = omni_result.options
        )

        omni_eval_response = evaluate_model.generate_content([curr_omni_eval_prompt, img, audio])

        eval_json = extract_json_str(omni_eval_response.text)
        eval_dict = json.loads(eval_json)
        eval_result = EvalOutput.model_validate(eval_dict)
        print("success eval_result")

        combined = {
            "file_name": os.path.basename(img_path),
            **omni_result.model_dump(), 
            **eval_result.model_dump()
        }

        comb_data = pd.json_normalize(combined, sep='_').to_dict(orient='records')[0]
        final_results.append(comb_data)

    except Exception as e:
        print(f"Error at {img_path}: {e}")
        continue

pd.DataFrame(final_results).to_csv("omni_final_dataset.csv", index=False, encoding='utf-8-sig')
print("Save a file")