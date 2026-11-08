
import asyncio
import logging
import os
import tempfile

import easyocr
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BufferedInputFile
from dotenv import load_dotenv


# Загружаем переменные из файла .env
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError(
        "Не найден BOT_TOKEN. Проверь файл .env"
    )


# Настройка логирования
logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()


# 1. Модель для печатного текста: EasyOCR

print("Загружаю EasyOCR...")

reader = easyocr.Reader(
    ["ru", "en"],
    gpu=False
)

print("EasyOCR загружен!")


# 2. Модель для рукописного текста: TrOCR

TROCR_MODEL = "kazars24/trocr-base-handwritten-ru"

print("Загружаю TrOCR для рукописного текста...")

handwriting_processor = TrOCRProcessor.from_pretrained(
    TROCR_MODEL
)

handwriting_model = VisionEncoderDecoderModel.from_pretrained(
    TROCR_MODEL
)

print("TrOCR загружен!")


# Режим каждого пользователя Telegram
# Ключ — Telegram user ID
# Значение — handwriting или printed
user_modes = {}


# 3. Функция распознавания рукописного текста

def recognize_handwriting(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    pixel_values = handwriting_processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    generated_ids = handwriting_model.generate(
        pixel_values,
        max_new_tokens=256
    )

    text = handwriting_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


# 4. Команда /start

@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Я Scribble.\n\n"
        "Выбери режим распознавания:\n"
        "/handwriting — рукописный текст\n"
        "/printed — печатный текст\n\n"
        "После выбора режима отправь фотографию."
    )


# 5. Команда /handwriting

@dp.message(Command("handwriting"))
async def handwriting_handler(message: Message):
    user_modes[message.from_user.id] = "handwriting"

    await message.answer(
        "Включён режим рукописного текста.\n"
        "Теперь отправь фотографию страницы."
    )


# 6. Команда /printed

@dp.message(Command("printed"))
async def printed_handler(message: Message):
    user_modes[message.from_user.id] = "printed"

    await message.answer(
        "Включён режим печатного текста.\n"
        "Теперь отправь фотографию документа."
    )
# 7. Обработчик фотографий

@dp.message(F.photo)
async def photo_handler(message: Message):
    user_id = message.from_user.id
    mode = user_modes.get(user_id)

    # Проверяем, выбрал ли пользователь режим
    if mode is None:
        await message.answer(
            "Сначала выбери режим:\n"
            "/handwriting — рукописный текст\n"
            "/printed — печатный текст"
        )
        return

    status = await message.answer(
        "Распознаю текст. Это может занять некоторое время..."
    )

    image_path = None

    try:
        # Создаём временный файл для фотографии
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:
            image_path = temp.name

        # Скачиваем фотографию в максимальном доступном размере
        photo = message.photo[-1]

        await bot.download(
            photo.file_id,
            destination=image_path
        )

        # Вызываем нужную модель в отдельном потоке,
        # чтобы не блокировать обработку сообщений ботом
        if mode == "handwriting":
            text = await asyncio.to_thread(
                recognize_handwriting,
                image_path
            )

        else:
            results = await asyncio.to_thread(
                reader.readtext,
                image_path,
                detail=0,
                paragraph=True
            )

            text = "\n".join(results).strip()

        # Проверяем результат
        if not text:
            await status.edit_text(
                "Не удалось распознать текст.\n"
                "Попробуй отправить более чёткое фото."
            )
            return

        # Telegram ограничивает длину обычного сообщения
        if len(text) <= 4000:
            await status.edit_text(
                f"Распознанный текст:\n\n{text}"
            )

        else:
            await status.edit_text(
                "Текст большой. Отправляю его файлом."
            )

            document = BufferedInputFile(
        ext.encode("utf-8"),
                filename="recognized_text.txt"
            )

            await message.answer_document(document)

    except Exception:
        logging.exception(
            "Ошибка обработки изображения"
        )

        await status.edit_text(
            "Не удалось обработать изображение.\n"
            "Подробности ошибки смотри в терминале."
        )

    finally:
        # Удаляем временную фотографию
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                logging.exception(
                    "Не удалось удалить временный файл"
                )


# 8. Обработка документов

@dp.message(F.document)
async def document_handler(message: Message):
    await message.answer(
        "Пока поддерживаются фотографии.\n"
        "Поддержку PDF и документов добавим позже."
    )


# 9. Обработка остальных сообщений

@dp.message()
async def other_handler(message: Message):
    await message.answer(
        "Выбери режим и отправь фотографию:\n"
        "/handwriting — рукописный текст\n"
        "/printed — печатный текст"
    )


# 10. Запуск бота

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
