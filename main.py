from model import get_yolov5
import logging
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from fastapi.responses import Response
import os
from fastapi import Request, HTTPException, Header
import random
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, SourceUser, StickerMessage, ImageMessage
from dotenv import load_dotenv

load_dotenv()

channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Logo Detection API",
              description="""Upload logo image and the API will response merchant name""",
              version="0.0.1",)

model_logo = get_yolov5(0.5)

@app.post("/detectImage")
async def detect_image(file: UploadFile):
    try:
        img = Image.open(BytesIO(await file.read()))
        results = model_logo(img, size=640)
        results.render()

        img_base64 = None
        if hasattr(results, 'img'):
            img_base64 = Image.fromarray(results.img)
        elif hasattr(results, 'ims'):
            img_base64 = Image.fromarray(results.ims[0])
        elif hasattr(results, 'imgs'):
            img_base64 = Image.fromarray(results.imgs[0])

        if img_base64 is None:
            logger.error("Model output does not contain expected image attributes")
            return Response(content="Model output does not contain expected image attributes", media_type="text/plain", status_code=500)

        bytes_io = BytesIO()
        img_base64.save(bytes_io, format="jpeg")
        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return Response(content=str(e), media_type="text/plain", status_code=500)

@app.post("/getLabel")
async def detect_image_label(file: UploadFile):
    img = Image.open(BytesIO(await file.read()))
    results = model_logo(img, size=640)
    label_result = results.pandas().xyxy[0].groupby('name')[['confidence']].max().reset_index().values.tolist()
    return {"label": label_result}

@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError as e:
        raise HTTPException(status_code=400, detail="chatbot handle body error.%s" % e.message)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    text = event.message.text
    profile = line_bot_api.get_profile(event.source.user_id)
    start_word = ['Hello ', 'สวัสดีครับท่าน ', 'ยินดีต้อนรับครับ คุณ']
    response_word = random.choice(start_word) + profile.display_name + " โหลดรูปที่มีแบรนด์สินค้าดังๆมาหน่อยสิ"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_word))

@handler.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    else:
        return

    message_content = line_bot_api.get_message_content(event.message.id)
    file_path = 'inference/' + str(random.randint(1, 10000)) + '.' + ext
    with open(file_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
    image = Image.open(file_path)
    results = model_logo(image, size=640)
    response_label = str(results.pandas().xyxy[0].groupby('name')[['confidence']].max().reset_index().values.tolist())
    line_bot_api.reply_message(event.reply_token, [TextSendMessage(text='Object detection result:' + response_label)])
