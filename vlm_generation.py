import os
import json
import pandas as pd

from dotenv import load_dotenv
from pydantic import BaseModel

import google.generativeai as genai
import PIL.Image
from tqdm import tqdm



load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

generate_model = genai.GenerativeModel('gemini-3-flash-preview')
evaluate_model = genai.GenerativeModel('gemini-2.5-pro')

data_dir = "../data/vlm_sample_data"
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
# sample_image_path = data_file + "/data/drama_4.jpg"

vqa_prompt = """
# Role
너는 멀티모달 데이터셋 구축 전문가야. 주어진 이미지를 분석하여 VLM(Vision Language Model)의 성능을 평가하기 위한 고품질의 질의응답(QA) 세트를 생성해야 해.

# Context
이미지는 다음 6개 도메인 중 하나에 속함: [문서, 수학, 과학, 한국 역사, 한국 사회, 이벤트 상황]
각 도메인의 특성을 반영하여 이미지 속 정보를 정확히 파악해야만 풀 수 있는 문제를 만들어줘.

# Task: QA 생성 규칙
1. 시각적 근거(Visual Grounding): 텍스트만 보고 맞출 수 있는 상식 문제는 배제하고, 반드시 이미지 안의 특정 요소(수치, 텍스트, 위치, 관계, 상황 등)를 확인해야 답변 가능한 문제를 생성할 것.
2. 복합 추론: 단순히 "무엇이 있나요?"라는 질문보다는 "이미지의 정보를 바탕으로 분석/계산/추론했을 때 적절한 결론은?"과 같은 형태를 지향할 것.
3. 도메인 특화:
   - (문서) 표/그래프의 수치 읽기, 비교 분석, 트렌드 파악.
   - (수학/과학) 수식 풀이 과정, 실험 도구의 명칭이나 현상의 원리 설명.
   - (한국사/사회) 랜드마크 식별, 사회적 맥락 파악, 공공 표지판 이해 등.
   - (이벤트) 발생한 사건의 종류, 안전/재난 상황에 대한 판단.

# Output Format (JSON)
{
  "domain": "해당 도메인",
  "question": "이미지에 기반한 구체적인 질문",
  "answer": "정답 (명확하고 객관적인 단답형 또는 짧은 문장)",
  "reasoning": "이미지의 어떤 부분을 근거로 정답이 도출되었는지에 대한 설명"
}
""" 
class VqaOutput(BaseModel):
    domain: str
    question: str
    answer: str
    reasoning: str


vqa_eval_prompt = """
# Role
너는 시각적 단서들을 기반으로 문제를 해결하고 추론 과정을 증명하는 분석가야.
제공된 이미를 분석하여 질문에 대한 정답을 도출하고, 이미지 내에서 발견한 구체적인 시각적 근거를 바탕으로 그 이유를 설명하라.

# input data
- Question : {question}

# Response Format (JSON only)
{{
    "model_answer": "질문에 대한 모델의 답변",
    "model_reasoning": "이미지 내에서 발견한 구체적인 시각적 근거와 이를 통한 논리적 추론 과정 설명"
}}
"""

class EvalOutput(BaseModel):
    model_answer: str
    model_reasoning: str



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



img_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(valid_extensions):
            img_path = os.path.join(root, file)
            img_files.append(img_path)

print("img_files complete")


final_results = []

for img_path in tqdm(img_files, desc="Processing Images"):
    try:
        img = PIL.Image.open(img_path)


        vqa_response = generate_model.generate_content([vqa_prompt, img])
        
        vqa_json = extract_json_str(vqa_response.text)
        vqa_dict = json.loads(vqa_json)
        vqa_result = VqaOutput.model_validate(vqa_dict)


        #eval
        curr_vqa_eval_prompt = vqa_eval_prompt.format(
            question = vqa_result.question
        )

        eval_response = evaluate_model.generate_content([curr_vqa_eval_prompt, img])

        eval_json = extract_json_str(eval_response.text)
        eval_dict = json.loads(eval_json)
        eval_result = EvalOutput.model_validate(eval_dict)



        combined = {
            "file_name": os.path.basename(img_path),
            **vqa_result.model_dump(),
            **eval_result.model_dump()
        }

        comb_data = pd.json_normalize(combined, sep='_').to_dict(orient='records')[0]
        final_results.append(comb_data)

    except Exception as e:
        print(f"Error at {img_path}: {e}")
        continue


pd.DataFrame(final_results).to_csv("final_dataset.csv", index=False, encoding='utf-8-sig')
print("Save a file")


