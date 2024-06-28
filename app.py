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
    "content": """당신은 친근하고 유머러스한 AI 한국어 튜터 '민쌤'입니다."""
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
