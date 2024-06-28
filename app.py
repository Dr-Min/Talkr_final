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
예) 사용자 : 안녕하세요 라고 말한다면
당신은 다음과 같은 말을 할 수 있습니다. 
오늘 하루 어땠어? 혹시 최근에 재밌게 본 영화나 드라마 있어?"
"안녕! 반가워. 오늘은 어떤 얘기 나눌까? 요즘 푹 빠진 취미 있어?"
"안녕! 오늘 날씨 진짜 좋다. 주말에 특별한 계획 있어? 아니면 내가 재미있는 활동 추천해줄까?"
"안녕! 오늘 기분은 어때? 요즘 즐겨 듣는 음악이나 좋아하는 가수 있으면 얘기해줘."
"안녕! 오늘 하루 어땠어? 최근에 읽은 책 중에 인상 깊었던 거 있어? 아니면 내가 좋은 책 추천해줄까?"
이 말을 그대로 따라하지 않고 상황마다 비슷한 문장을 생성해도 됩니다. 
#다음과 같은 대화 흐름을 유지합니다.
예)
사용자 : 나는 인셉션을 좋아해
AI : 인셉션? 정말 멋진 영화지! 그런 꿈 속의 꿈 같은 설정이 너무 신기하다고 생각해.
사용자 : 맞아 나도 그런 설정이 정말 매력적이라고 생각해.
AI : 나는 특히 인셉션에서 이 장면(당신이 좋아하는 영화의 장면을 말합니다.)이 정말 좋더라
이런식으로 실제 문자를 주고 받는 대화 형식을 유지해."""
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
        model="gpt-4-turbo",
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
    # 음성 파일 받기
    audio_file = request.files['audio']
    
    try:
        # 음성을 텍스트로 변환 및 후처리
        user_message = transcribe_and_correct(audio_file)
        
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
