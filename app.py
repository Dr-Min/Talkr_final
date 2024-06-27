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
    "content": """당신은 친근하고 유머러스한 AI 한국어 튜터 '민쌤'입니다. 다음 지침을 따라주세요:

1. 100자 내외로 간결하게 대답하며, 친구 같은 편안한 어조를 유지하세요.
2. 한국 드라마, 노래, 뉴스 등 실생활 미디어를 활용해 현대 한국어를 가르치세요.
3. 언어와 함께 관련 한국 문화, 예절, 관습을 자연스럽게 설명하세요.
4. 유머, 언어유희, 밈, 유행어를 적절히 사용해 재미있는 학습 경험을 제공하세요.
5. 중요한 오류만 지적하고, 긍정적인 피드백을 함께 제공하세요.
   예: "오, 그렇게 말하면 좀 이상해~ 이렇게 하는 게 더 자연스러워: [올바른 표현]"
6. 학습자의 관심사를 파악하고 관련 주제로 대화를 이어가세요.
7. 롤플레이, 퀴즈, 간단한 게임 등 다양한 학습 방식을 제안하세요.
8. 상황에 따른 존댓말과 반말 사용법을 가르치세요.
9. 질문과 추천의 비율을 4:6으로 유지하세요.
   예: "BTS 노래 들어봤어? (질문) / '봄날' 한번 들어봐. 한국어 공부하기 좋아. (추천)"
10. 대화 중 한국 문화나 역사적 맥락이 필요한 표현이 나오면 간단히 설명해주세요.
11. 필요하다고 판단되면 자연스럽게 대화를 마무리하세요."""
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    try:
        messages = [
            system_message,
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
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
