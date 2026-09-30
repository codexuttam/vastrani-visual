"""
Script to generate local 256x256 RGBA PNG emoji assets for Phase 8 AR Effect Engine.
"""

import os
from PIL import Image, ImageDraw

def create_emoji_assets(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    size = (256, 256)

    def draw_base_face(draw, color=(255, 204, 0, 255), border_color=(230, 175, 0, 255)):
        draw.ellipse([8, 8, 248, 248], fill=color, outline=border_color, width=6)

    # 1. Happy (😀)
    img_happy = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img_happy)
    draw_base_face(d)
    # Eyes
    d.ellipse([65, 80, 95, 120], fill=(40, 40, 40, 255))
    d.ellipse([161, 80, 191, 120], fill=(40, 40, 40, 255))
    # Big Smile
    d.chord([60, 110, 196, 206], start=0, end=180, fill=(40, 40, 40, 255))
    d.chord([72, 122, 184, 194], start=0, end=180, fill=(255, 255, 255, 255))
    img_happy.save(os.path.join(output_dir, "happy.png"))

    # 2. Laughing (😂)
    img_laughing = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img_laughing)
    draw_base_face(d)
    # Squinting eyes (X / arc)
    d.arc([55, 80, 105, 120], start=180, end=0, fill=(40, 40, 40, 255), width=8)
    d.arc([151, 80, 201, 120], start=180, end=0, fill=(40, 40, 40, 255), width=8)
    # Laughing mouth with tongue
    d.chord([55, 120, 201, 215], start=0, end=180, fill=(40, 40, 40, 255))
    d.chord([80, 155, 176, 215], start=0, end=180, fill=(255, 80, 100, 255))
    # Tears of joy
    d.ellipse([30, 110, 55, 150], fill=(80, 180, 255, 230))
    d.ellipse([201, 110, 226, 150], fill=(80, 180, 255, 230))
    img_laughing.save(os.path.join(output_dir, "laughing.png"))

    # 3. Cool (😎)
    img_cool = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img_cool)
    draw_base_face(d)
    # Sunglasses frame
    d.polygon([(40, 75), (120, 75), (110, 130), (50, 130)], fill=(20, 20, 20, 255))
    d.polygon([(136, 75), (216, 75), (206, 130), (146, 130)], fill=(20, 20, 20, 255))
    d.line([(115, 85), (141, 85)], fill=(20, 20, 20, 255), width=8)
    # Lens highlight
    d.line([(55, 85), (95, 120)], fill=(200, 200, 200, 180), width=4)
    d.line([(151, 85), (191, 120)], fill=(200, 200, 200, 180), width=4)
    # Cool smirk
    d.arc([75, 135, 181, 195], start=10, end=170, fill=(40, 40, 40, 255), width=8)
    img_cool.save(os.path.join(output_dir, "cool.png"))

    # 4. Angry (😡)
    img_angry = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img_angry)
    draw_base_face(d, color=(255, 70, 50, 255), border_color=(200, 40, 30, 255))
    # Slanted eyebrows
    d.line([(50, 70), (110, 100)], fill=(40, 40, 40, 255), width=10)
    d.line([(206, 70), (146, 100)], fill=(40, 40, 40, 255), width=10)
    # Angry eyes
    d.ellipse([65, 95, 95, 125], fill=(40, 40, 40, 255))
    d.ellipse([161, 95, 191, 125], fill=(40, 40, 40, 255))
    # Frowning mouth
    d.arc([75, 160, 181, 220], start=190, end=350, fill=(40, 40, 40, 255), width=9)
    img_angry.save(os.path.join(output_dir, "angry.png"))

    # 5. Love (😍)
    img_love = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img_love)
    draw_base_face(d)
    # Heart Eyes
    def draw_heart(cx, cy, s=30):
        d.ellipse([cx - s, cy - s, cx, cy], fill=(240, 30, 60, 255))
        d.ellipse([cx, cy - s, cx + s, cy], fill=(240, 30, 60, 255))
        d.polygon([(cx - s, cy - s//2), (cx + s, cy - s//2), (cx, cy + s)], fill=(240, 30, 60, 255))
    draw_heart(75, 95, 26)
    draw_heart(181, 95, 26)
    # Open loving smile
    d.chord([65, 125, 191, 210], start=0, end=180, fill=(40, 40, 40, 255))
    d.chord([85, 165, 171, 210], start=0, end=180, fill=(255, 100, 120, 255))
    img_love.save(os.path.join(output_dir, "love.png"))

    # 6. Thinking (🤔)
    img_thinking = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img_thinking)
    draw_base_face(d)
    # One raised eyebrow
    d.arc([55, 60, 105, 90], start=190, end=350, fill=(40, 40, 40, 255), width=7)
    d.line([(151, 85), (201, 85)], fill=(40, 40, 40, 255), width=7)
    # Looking up eyes
    d.ellipse([65, 80, 95, 110], fill=(40, 40, 40, 255))
    d.ellipse([75, 82, 85, 92], fill=(255, 255, 255, 255))
    d.ellipse([161, 80, 191, 110], fill=(40, 40, 40, 255))
    d.ellipse([171, 82, 181, 92], fill=(255, 255, 255, 255))
    # Pensive mouth
    d.line([(95, 165), (161, 155)], fill=(40, 40, 40, 255), width=7)
    # Thinking hand on chin
    d.ellipse([140, 180, 210, 245], fill=(245, 195, 0, 255), outline=(220, 165, 0, 255), width=4)
    d.ellipse([180, 150, 225, 210], fill=(245, 195, 0, 255), outline=(220, 165, 0, 255), width=4)
    img_thinking.save(os.path.join(output_dir, "thinking.png"))

    print(f"Generated 6 emoji RGBA assets in {output_dir}")

if __name__ == "__main__":
    create_emoji_assets("/Users/uttam7781/Documents/vastrani vsion /visioncontrol/assets/emojis")
