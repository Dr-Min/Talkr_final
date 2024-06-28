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
    "content": """당신은 친근하고 유머러스한 AI 한국어 튜터 '민쌤'입니다. #제시문 짧게 짧게 대화하세요. 친구처럼 대화하세요. 상대방이 말을 하면 당신이 먼저 주제를 꺼냅니다. 
    # (이하 생략, 기존 system_message 내용과 동일)"""
}

# Whisper 전사 및 GPT-4 후처리 함수
def transcribe_and_correct(audio_file):
    # Whisper로 음성을 텍스트로 변환
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    
    # GPT-4를 사용한 후처리
    system_prompt = """You are a helpful assistant for transcription correction. 
    Your task is to correct any spelling discrepancies in the transcribed Korean text. 
    Make sure to maintain the original meaning and only correct obvious errors. 
    Add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )
    
    return response.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if 'audio' in request.files:
            audio_file = request.files['audio']
            user_message = transcribe_and_correct(audio_file)
        else:
            user_message = request.json['message']
        
        messages = [
            system_message,
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
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
    # (번역 코드는 변경 없음)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
