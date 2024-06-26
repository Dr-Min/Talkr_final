from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

app = Flask(__name__)

# OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI()

# 시스템 메시지 (한국어 튜터 역할)
system_message = {
    "role": "system", 
    "content": """당신은 한국어를 알려주는 ai 한국어 튜터 민쌤입니다.
친구처럼 대화합니다.
가끔 자연스럽게 욕도 섞어서 대화합니다.
너무 크게 틀린것이 아니라면 교정하지 않습니다.
한국어 공부에 도움이 될만한 롤플레잉을 자주 유도합니다.
단어나 문법적으로 틀린말이 있을때 (한국어로 말하고 영어로도 말합니다.)
그리고 비아냥 거립니다. 
예를들어 "ㅋㅋㅋ이걸 몰라? ㅋㅋㅋ" , "븅신이고 ㅋㅋ' 이런식으로 답변합니다.

말이 끝나면 질문을 하기보다. 관련된 주제를 검색해서 찾아본 뒤 이야기를 이어갑니다.
말을 이어 나가기 위해 문맥상 맞지 않는 질문을 하지않습니다.
문맥상 대화를 종결하는게 깔끔하다고 질문을 하지 않고 판단하면 대화를 종결해도 됩니다.

당신이 어떤 주제를 꺼내면 먼저 그 주제에 대한 감상을 먼저 말합니다.
예를 들어)
더 글로리 알아? 나는 더 글로리의 이런점이 좋았고 특히 그 장면은 정말 대단했어.
이런식으로 답변합니다.

질문과 추천을 4대6 비율로 합니다.
예를 들어) 사이버펑크 해봤어?(질문) / 사이버펑크 무조건 해봐야돼. 정말 미친작품이거든;; (추천)
"""
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    try:
        messages = [system_message, {"role": "user", "content": user_message}]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        ai_message = response.choices[0].message.content

        # TTS 생성
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ai_message
        )

        # 오디오 데이터를 base64로 인코딩
        audio_base64 = base64.b64encode(speech_response.content).decode('utf-8')

        return jsonify({
            'message': ai_message, 
            'audio': audio_base64,
            'success': True
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'message': '죄송합니다. 오류가 발생했습니다.', 'success': False}), 500

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json['text']
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translator. Translate the following Korean text to English."},
                {"role": "user", "content": text}
            ]
        )
        translated_text = response.choices[0].message.content
        return jsonify({'translation': translated_text})
    except Exception as e:
        print(f"Translation Error: {str(e)}")
        return jsonify({'error': '번역 중 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)