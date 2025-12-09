import asyncio
import logging
import time
import re
from urllib.parse import urljoin
from io import BytesIO
from selenium.common.exceptions import TimeoutException


import numpy as np
from PIL import Image

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    from tensorflow import keras
except ImportError:
    import keras

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


# ================== НАСТРОЙКИ МОДЕЛИ ==================
MODEL_PATH = "C:\\study\\МАИ\магистратура\\1 курс\\1 сем\\Управление it проектами\\clothing_multitask_mobilenetv2.keras"  
IMG_SIZE = (224, 224)  # размер входа модели, при необходимости поменяй

TYPE_CLASSES = ['hoodie', 'jacket', 'jeans', 'pants', 'sandals',
                'shirt', 'shorts', 'sneakers', 'sweater', 'tshirt']

COLOR_CLASSES = ['black', 'blue', 'brown', 'green', 'grey',
                 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

PRINT_CLASSES = ['with_print', 'no_print']

# Человеко-читаемые подписи (для сообщения пользователю)
TYPE_RU = {
    'hoodie': 'худи',
    'jacket': 'куртка',
    'jeans': 'джинсы',
    'pants': 'брюки',
    'sandals': 'босоножки',
    'shirt': 'рубашка',
    'shorts': 'шорты',
    'sneakers': 'кроссовки',
    'sweater': 'свитер',
    'tshirt': 'футболка',
}

# Вариант цветов, который будет хорошо смотреться в описании
COLOR_RU_HUMAN = {
    'black': 'чёрные',
    'blue': 'синие',
    'brown': 'коричневые',
    'green': 'зелёные',
    'grey': 'серые',
    'orange': 'оранжевые',
    'pink': 'розовые',
    'purple': 'фиолетовые',
    'red': 'красные',
    'white': 'белые',
    'yellow': 'жёлтые',
}

PRINT_RU = {
    'with_print': 'с принтом',
    'no_print': 'без принта',
}

# Для поискового запроса — почти то же, можно оставить как есть
COLOR_RU_QUERY = COLOR_RU_HUMAN
TYPE_RU_QUERY = TYPE_RU

_model = None


def get_model():
    """Ленивая загрузка модели, чтобы не грузить её лишний раз."""
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype='float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_labels_from_bytes(image_bytes: bytes):
    """Возвращает (type_label, color_label, print_label) по байтам картинки."""
    model = get_model()
    x = preprocess_image_bytes(image_bytes)
    preds = model.predict(x)

    # Ожидаем три выхода: тип, цвет, принт
    if isinstance(preds, list) and len(preds) == 3:
        type_probs, color_probs, print_probs = preds
    else:
        raise ValueError("Модель должна выдавать три выхода: тип, цвет, принт")

    type_idx = int(np.argmax(type_probs[0]))
    color_idx = int(np.argmax(color_probs[0]))
    print_idx = int(np.argmax(print_probs[0]))

    type_label = TYPE_CLASSES[type_idx]
    color_label = COLOR_CLASSES[color_idx]
    print_label = PRINT_CLASSES[print_idx]

    return type_label, color_label, print_label


def build_description_and_query(type_label: str, color_label: str, print_label: str | None = None):
    """
    Формируем человеческое описание и строку поиска.
    РЕЗУЛЬТАТ МОДЕЛИ ПО ПРИНТУ ИГНОРИРУЕМ.
    """
    type_ru = TYPE_RU.get(type_label, type_label)
    color_ru = COLOR_RU_HUMAN.get(color_label, color_label)

    # Описание для пользователя: только цвет + тип
    description = f"{color_ru} {type_ru}"

    # Поисковый запрос для сайта: тоже только цвет + тип
    search_query = " ".join(
        p for p in [
            COLOR_RU_QUERY.get(color_label, ""),
            TYPE_RU_QUERY.get(type_label, ""),
        ]
        if p
    ).strip()

    return description, search_query



# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ====== Функции парсера ======
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\u00A0', ' ').replace('\u2009', ' ').replace('\u202F', ' ')
    return s.strip()


def normalize_price(raw: str) -> str:
    if not raw:
        return ""
    r = normalize_text(raw)
    r = re.sub(r'^[OoОоTtTт]+[^\d]*', '', r)

    m = re.search(r'(\d{1,3}(?:[ \u00A0]\d{3})*|\d+)(?:\s*₽|\s*RUB| руб)?', r, flags=re.I)
    if m:
        num = m.group(1)
        num = num.replace('\u00A0', ' ')
        num = re.sub(r'\s+', ' ', num).strip()
        digits = re.sub(r'\s', '', num)
        formatted = ''
        while len(digits) > 3:
            formatted = ' ' + digits[-3:] + formatted
            digits = digits[:-3]
        formatted = digits + formatted
        return f"{formatted} ₽"
    if '₽' in r:
        r = r.replace('₽', ' ₽')
        r = re.sub(r'\s+', ' ', r).strip()
        return r
    return r


def find_search_input(driver):
    input_selectors = [
        'input[type="search"]',
        'input[name*="search"]',
        'input[placeholder*="Поиск"]',
        'input[placeholder*="поиск"]',
        'input[aria-label*="Поиск"]',
        'input[aria-label*="search"]',
        '.header-controls_control input',
        '.search-input input',
        'input'
    ]
    for sel in input_selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            if el.is_displayed():
                return el
        except Exception:
            continue
    return None


def extract_products(driver, wait):
    base = "https://www.gloria-jeans.ru/"
    card_selectors = [
        'gj-product-mini-card',
        '.product-mini-card',
        '.listing-grid__col',
        '.product-mini-card__image-wrapper',
        '.product-card',
        '.product-item',
        '.catalog-card',
        '.catalog__item',
        '[data-testid="product-card"]',
        'article'
    ]

    cards = []
    for sel in card_selectors:
        elems = driver.find_elements(By.CSS_SELECTOR, sel)
        if elems:
            cards = elems
            break

    if not cards:
        anchors = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/product/"], a[href*="/catalog/"]')
        seen = set()
        tmp = []
        for a in anchors:
            try:
                href = a.get_attribute('href') or a.get_attribute('innerHTML') or ""
                if href in seen:
                    continue
                seen.add(href)
                parent = a.find_element(By.XPATH, "./ancestor::div[1]")
                tmp.append(parent)
            except Exception:
                continue
        cards = tmp

    results = []
    seen_links = set()

    for c in cards[:1000]:
        try:
            outer = (c.get_attribute("outerHTML") or "")[:2000]
            link = ""
            title = ""

            try:
                prod_anchor = None
                try:
                    prod_anchor = c.find_element(By.CSS_SELECTOR, '.product-mini-card__name a, a[href*="/product/"]')
                except Exception:
                    anchors = c.find_elements(By.CSS_SELECTOR, 'a[href]')
                    for a in anchors:
                        h = a.get_attribute('href') or ""
                        if '/product/' in h:
                            prod_anchor = a
                            break
                if prod_anchor:
                    href = prod_anchor.get_attribute('href') or ""
                    if href.startswith('/'):
                        href = urljoin(base, href)
                    link = href
                    title = (prod_anchor.text or "").strip()
            except Exception:
                pass

            if not link:
                try:
                    anchors = c.find_elements(By.CSS_SELECTOR, 'a[href]')
                    for a in anchors:
                        h = a.get_attribute('href') or ""
                        if '/catalog/' in h:
                            continue
                        if h:
                            if h.startswith('/'):
                                h = urljoin(base, h)
                            link = h
                            if not title:
                                title = (a.text or "").strip()
                            break
                except Exception:
                    pass

            if not title:
                try:
                    el = c.find_element(By.CSS_SELECTOR, '.product-mini-card__name, .product-mini-card__name a, .product-mini-card__name span')
                    title = (el.text or "").strip()
                except Exception:
                    pass

            if link and link in seen_links:
                continue
            if link:
                seen_links.add(link)

            image_url = ""
            for sel in ['img.product-mini-card__image, img.product-mini-card__img', 'img', 'picture img']:
                try:
                    im = c.find_element(By.CSS_SELECTOR, sel)
                    image_url = im.get_attribute('src') or im.get_attribute('data-src') or ""
                    if image_url and image_url.startswith('/'):
                        image_url = urljoin(base, image_url)
                    if image_url:
                        break
                except Exception:
                    continue

            price = ""
            price_selectors = [
                'span.button__label',
                '.button__label',
                'span.price-new',
                'span.price-old',
                'span.price',
                '.product-card__price-current',
                '.product-card__price',
                '.price-current',
                '.price',
                'gj-button-price'
            ]
            for ps in price_selectors:
                try:
                    els = c.find_elements(By.CSS_SELECTOR, ps)
                    if els:
                        for el in els:
                            txt = (el.text or "").strip()
                            if txt:
                                price = txt
                                break
                        if price:
                            break
                except Exception:
                    continue

            if not price:
                m = re.search(r'(\d{1,3}(?:[\s\u00A0]\d{3})*\s*₽)', outer)
                if m:
                    price = m.group(1).strip()

            if title.strip() or link.strip():
                results.append({
                    "title": title,
                    "price": price,
                    "link": link,
                    "image": image_url
                })
        except Exception as e:
            logger.error(f"Ошибка при обработке карточки: {e}")
            continue

    return results


def run_parser(search_query: str):
    driver = uc.Chrome(headless=False)
    wait = WebDriverWait(driver, 5)

    try:
        driver.get("https://www.gloria-jeans.ru/search")
        time.sleep(4.0)

        input_el = find_search_input(driver)
        if not input_el:
            raise RuntimeError("Не удалось найти поле поиска")

        input_el.click()
        input_el.clear()
        input_el.send_keys(search_query)
        input_el.send_keys(Keys.RETURN)
        time.sleep(1.0)

        try:
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/product/"]'))
            )
        except TimeoutException:
            # Просто нет товаров по запросу — вернём пустой список
            logger.info(f"Нет товаров по запросу: {search_query}")
            return []

        time.sleep(1.0)

        products = []
        seen_links = set()

        # Первое извлечение
        initial_items = extract_products(driver, wait)
        for item in initial_items:
            link = item.get("link")
            if link and link not in seen_links:
                seen_links.add(link)
                products.append(item)

        # -------- ПЛАВНЫЙ СКРОЛЛИНГ ДО КОНЦА СТРАНИЦЫ --------
        SCROLL_STEP = 900          # шаг прокрутки
        SCROLL_DELAY = 1.0         # пауза после каждого шага
        MAX_SCROLLS = 120          # защита от бесконечного цикла
        NO_CHANGE_LIMIT = 5        # сколько раз подряд можно не видеть изменений

        last_height = driver.execute_script("return document.body.scrollHeight")
        last_seen = len(seen_links)
        no_change_count = 0
        scroll_count = 0

        while scroll_count < MAX_SCROLLS and no_change_count < NO_CHANGE_LIMIT:
            scroll_count += 1

            driver.execute_script(f"window.scrollBy(0, {SCROLL_STEP});")
            time.sleep(SCROLL_DELAY)

            new_items = extract_products(driver, wait)
            before = len(seen_links)

            for item in new_items:
                link = item.get("link")
                if link and link not in seen_links:
                    seen_links.add(link)
                    products.append(item)

            after = len(seen_links)

            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height and after == last_seen:
                no_change_count += 1
            else:
                no_change_count = 0

            last_height = new_height
            last_seen = after

        return products

    finally:
        driver.quit()


# ====== Вспомогательная функция для отправки товарами страницами ======
PAGE_SIZE = 15


async def send_products_page(update: Update, context: ContextTypes.DEFAULT_TYPE, start_idx: int = 0):
    products = context.user_data.get("products", [])
    if not products:
        await update.effective_message.reply_text("Список товаров пуст, попробуйте поиск заново.")
        return

    end_idx = min(start_idx + PAGE_SIZE, len(products))
    chunk = products[start_idx:end_idx]

    message_lines = [f"✅ Товары {start_idx + 1}–{end_idx} из {len(products)}:\n\n"]

    for i, product in enumerate(chunk, start=start_idx + 1):
        title = normalize_text(product.get("title", ""))
        price = normalize_price(product.get("price", ""))
        link = product.get("link", "")

        if len(title) > 80:
            title = title[:77] + "..."

        product_line = f"{i}. {title}\n"
        if price:
            product_line += f"   💰 Цена: {price}\n"
        if link:
            product_line += f"   🔗 [Ссылка на товар]({link})\n"
        product_line += "\n"
        message_lines.append(product_line)

    text = "".join(message_lines)

    reply_markup = None
    if end_idx < len(products):
        keyboard = [
            [InlineKeyboardButton("Показать ещё", callback_data=f"more:{end_idx}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode='Markdown',
        disable_web_page_preview=True,
        reply_markup=reply_markup
    )


# ====== Telegram Bot Handlers ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для поиска товаров в Gloria Jeans.\n"
        "Отправь мне название товара для поиска (например: джинсы) или фото предмета гардероба."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    search_query = update.message.text
    if not search_query or not search_query.strip():
        await update.message.reply_text("Пожалуйста, введите поисковый запрос")
        return

    await update.message.reply_text(f"🔍 Ищу товары по запросу: {search_query}...")

    try:
        products = await asyncio.get_event_loop().run_in_executor(
            None, run_parser, search_query
        )

        valid_products = []
        for product in products:
            title = product.get("title", "").strip()
            link = product.get("link", "").strip()
            if title or link:
                valid_products.append(product)

        if not valid_products:
            await update.message.reply_text("❌ Товары не найдены")
            return

        context.user_data["products"] = valid_products
        await send_products_page(update, context, start_idx=0)

    except Exception as e:
        logger.error(f"Ошибка парсера: {e}")
        await update.message.reply_text("❌ Произошла ошибка при поиске товаров")


# ====== ОБРАБОТКА ФОТО + КЛАССИФИКАЦИЯ ======
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь прислал фото — распознаём и уточняем, то ли это."""
    if not update.message or not update.message.photo:
        return

    photo = update.message.photo[-1]  # самая большая по размеру
    file = await photo.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    image_bytes = bio.getvalue()

    try:
        loop = asyncio.get_event_loop()
        type_label, color_label, print_label = await loop.run_in_executor(
            None, predict_labels_from_bytes, image_bytes
        )

        description, search_query = build_description_and_query(
            type_label, color_label, print_label
        )

        # Сохраним в user_data, чтобы использовать при "Да"
        context.user_data["last_prediction"] = {
            "type": type_label,
            "color": color_label,
            "print": print_label,
            "description": description,
            "search_query": search_query,
        }

        keyboard = [
            [
                InlineKeyboardButton("Да, искать такие", callback_data="confirm:yes"),
                InlineKeyboardButton("Нет, это не то", callback_data="confirm:no"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"Я думаю, на фото: {description}.\n"
            f"Искать такие товары в магазине?",
            reply_markup=reply_markup,
        )

    except Exception as e:
        logger.exception("Ошибка при распознавании изображения: %s", e)
        await update.message.reply_text(
            "Не удалось распознать одежду на фото. "
            "Пожалуйста, отправьте текстовый запрос (например: серые джинсы)."
        )


async def show_more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопки 'Показать ещё'."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("more:"):
        return

    try:
        start_idx = int(data.split(":")[1])
    except Exception:
        start_idx = 0

    await send_products_page(update, context, start_idx=start_idx)


async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик подтверждения распознанного фото."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    choice = data.split(":", 1)[1] if ":" in data else ""

    if choice == "yes":
        pred = context.user_data.get("last_prediction")
        if not pred:
            await query.message.reply_text(
                "Не нашёл последнее распознанное фото. "
                "Пожалуйста, отправьте текстовый запрос."
            )
            return

        search_query = pred["search_query"]
        description = pred["description"]

        await query.message.reply_text(
            f"🔍 Ищу товары, похожие на: {description}\n"
            f"(по запросу: {search_query})"
        )

        try:
            products = await asyncio.get_event_loop().run_in_executor(
                None, run_parser, search_query
            )

            valid_products = []
            for product in products:
                title = product.get("title", "").strip()
                link = product.get("link", "").strip()
                if title or link:
                    valid_products.append(product)

            if not valid_products:
                await query.message.reply_text("❌ Похожие товары не найдены")
                return

            context.user_data["products"] = valid_products
            await send_products_page(update, context, start_idx=0)

        except Exception as e:
            logger.error(f"Ошибка парсера (confirm): {e}")
            await query.message.reply_text("❌ Произошла ошибка при поиске товаров")

    else:
        # Пользователь не согласился — просим текстовый запрос
        await query.message.reply_text(
            "Хорошо, тогда напишите, пожалуйста, что вы хотите найти "
            "в виде текста (например: серые джинсы)."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)


# ====== Main Bot Setup ======
def main():
    application = Application.builder().token("BOT_TOKEN").build()

    application.add_handler(CommandHandler("start", start))

    # Сначала обрабатываем фото, потом текст
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Разные callback'и по паттерну
    application.add_handler(CallbackQueryHandler(show_more, pattern=r"^more:"))
    application.add_handler(CallbackQueryHandler(handle_confirm, pattern=r"^confirm:"))

    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
