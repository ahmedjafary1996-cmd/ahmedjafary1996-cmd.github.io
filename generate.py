import os
import json
import asyncio
import urllib.parse
import requests
import numpy as np
import edge_tts
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from google import genai
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    afx,
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

BOBO_DESC = "a small cute chubby orange bunny with big brown eyes, one floppy ear and one upright ear, soft fluffy fur, cream belly, wearing a tiny sky-blue scarf"
KOKO_DESC = "a tiny cute round blue bird with big bright eyes"
STYLE_SUFFIX = "3D Pixar-like cute cartoon animation style, bright vibrant colors, soft studio lighting, simple pastel background, child-safe, no text in image"

SYSTEM_PROMPT = f"""
أنت كاتب ومخرج مسلسل كرتون أطفال مصري شهير لعمر 3 إلى 6 سنوات، بطله "بوبو الأرنوب" وصديقه الصغير "كوكو العصفور".
اكتب سيناريو حلقة جديدة وقصيرة عن: {{{{TOPIC}}}}

القواعد الصارمة لبناء ارتباط الطفل بالشخصية:
1. المشهد الأول دائماً هو دخلة بوبو الثابتة:
   - scene_text: "أهلاً يا أصحابي! أنا بوبو الأرنوب، يلا نلعب سوا!"
   - image_prompt: "{BOBO_DESC} jumping happily and waving at the camera with {KOKO_DESC} sitting on his head, {STYLE_SUFFIX}"
   - is_question: false
2. وسط الحلقة (6 إلى 7 مشاهد):
   - جمل مصرية قصيرة جداً (5 إلى 8 كلمات)، دافئة، مبهجة، ومليئة بكلمات التشجيع ("يلا"، "حلو قوي"، "يا سلام"، "برافو").
   - يجب أن يتضمن أحد المشاهد سؤالاً تفاعلياً بسيطاً للطفل في المنزل ويكون حقل "is_question": true، والمشهد الذي يليه مباشرة يشجع فيه بوبو الطفل: "برافو عليكم! إنتوا شاطرين قوي!".
3. المشهد الأخير دائماً هو قفلة بوبو الثابتة:
   - scene_text: "إنتوا أحلى أصحاب! أشوفكم بكرة يا أبطال، باي باي!"
   - image_prompt: "{BOBO_DESC} blowing a friendly goodbye kiss and waving with both paws while {KOKO_DESC} waves a wing, {STYLE_SUFFIX}"
   - is_question: false
4. العنوان يبدأ دائماً بـ "بوبو الأرنوب | ".

أعد JSON فقط بدون أي نص إضافي أو markdown:
{{
  "title": "",
  "description": "",
  "hashtags": [],
  "scenes": [
    {{
      "scene_text": "",
      "image_prompt": "",
      "is_question": false
    }}
  ]
}}

تأكد في كل image_prompt أن يبدأ الوصف بـ "{BOBO_DESC}" وينتهي بـ "{STYLE_SUFFIX}".
"""

def format_arabic_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def create_caption_image(text, width=1080, height=1920):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, 54)
            break
    if font is None:
        font = ImageFont.load_default()

    bidi_text = format_arabic_text(text)
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (width - text_w) // 2
    y = int(height * 0.78)

    pad_x, pad_y = 36, 20
    draw.rounded_rectangle(
        [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y],
        radius=30,
        fill=(255, 140, 0, 225),
        outline=(255, 255, 255, 255),
        width=5,
    )

    draw.text((x + 2, y + 2), bidi_text, font=font, fill=(0, 0, 0, 160))
    draw.text((x, y), bidi_text, font=font, fill=(255, 255, 255, 255))

    return np.array(img)

def generate_episode_script(topic):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=SYSTEM_PROMPT.replace("{{TOPIC}}", topic),
        config={"response_mime_type": "application/json"},
    )
    return json.loads(response.text)

def download_image(prompt, filename, seed=777):
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&seed={seed}&nologo=true&model=flux"
    res = requests.get(url, timeout=90)
    res.raise_for_status()
    with open(filename, "wb") as f:
        f.write(res.content)

async def generate_voice(text, filename):
    communicate = edge_tts.Communicate(text, "ar-EG-SalmaNeural", pitch="+14Hz", rate="+4%")
    await communicate.save(filename)

def produce_full_cartoon(episode):
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    scene_clips = []

    print(f"🎬 بدء إنتاج حلقة الكرتون: {episode['title']}")

    for i, scene in enumerate(episode["scenes"]):
        img_path = f"temp/scene_{i}.jpg"
        aud_path = f"temp/scene_{i}.mp3"

        print(f"🎨 جاري تجهيز المشهد {i+1}/{len(episode['scenes'])}...")
        download_image(scene["image_prompt"], img_path, seed=777)
        asyncio.run(generate_voice(scene["scene_text"], aud_path))

        voice_clip = AudioFileClip(aud_path)
        pause = 1.8 if scene.get("is_question", False) else 0.35
        duration = voice_clip.duration + pause

        if i % 2 == 0:
            zoom_fn = lambda t: 1.0 + 0.025 * t
        else:
            zoom_fn = lambda t, d=duration: 1.08 - 0.025 * t

        bg_clip = (
            ImageClip(img_path)
            .set_duration(duration)
            .resize(zoom_fn)
            .set_position("center")
        )

        caption_arr = create_caption_image(scene["scene_text"])
        caption_clip = ImageClip(caption_arr).set_duration(duration)

        layers = [bg_clip, caption_clip]

        if os.path.exists("assets/logo.png"):
            logo_clip = (
                ImageClip("assets/logo.png")
                .set_duration(duration)
                .resize(width=160)
                .set_position(("right", "top"))
                .margin(right=30, top=40, opacity=0)
            )
            layers.append(logo_clip)

        composed_scene = CompositeVideoClip(layers, size=(1080, 1920)).set_audio(voice_clip)
        scene_clips.append(composed_scene)

    full_video = concatenate_videoclips(scene_clips, method="compose")

    if os.path.exists("assets/bg_music.mp3"):
        bg_music = (
            AudioFileClip("assets/bg_music.mp3")
            .afx(afx.audio_loop, duration=full_video.duration)
            .volumex(0.12)
            .fx(afx.audio_fadeout, 2.0)
        )
        final_audio = CompositeAudioClip([full_video.audio, bg_music])
        full_video = full_video.set_audio(final_audio)

    output_path = "output/bobo_cartoon_episode.mp4"
    full_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

    with open("output/youtube_metadata.json", "w", encoding="utf-8") as f:
        json.dump(episode, f, ensure_ascii=False, indent=2)

    print("✅ تم إنتاج حلقة الكرتون المتكاملة بنجاح!")

if __name__ == "__main__":
    topic = os.environ.get("VIDEO_TOPIC", "بوبو يتعلم مشاركة الألعاب مع كوكو")
    episode_data = generate_episode_script(topic)
    produce_full_cartoon(episode_data)
