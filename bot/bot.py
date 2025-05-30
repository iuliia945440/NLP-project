# Импорт библиотек
import os
import pandas as pd
import joblib
import telebot
from telebot import types

from dotenv import load_dotenv

load_dotenv()


# Максимальная длина сообщения для Telegram
MAX_MESSAGE_LENGTH = 2000

# Загрузка датасета с вопросами и ответами
data_dir = './data'
dataset = pd.read_csv(f'{data_dir}/full_dataset.csv', index_col=False)

# Загрузка предобученного векторайзера и модели поиска ближайших соседей
vectorizer = joblib.load('./model/tfidf_vectorizer.joblib')
model = joblib.load('./model/nearest_neighbors_model.joblib')

# Векторизация всех вопросов из датасета
X = vectorizer.transform(dataset['Quest'])

# Инициализация бота с указанным токеном
TOKEN = os.environ.get('TOKEN')
bot = telebot.TeleBot(TOKEN)

# Временное хранилище результатов для пользователей (по chat_id)
user_results = {}


# Функция для разбиения длинных сообщений на части
def split_message(text, max_length=MAX_MESSAGE_LENGTH):
    if len(text) <= max_length:
        return [text]
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break
        split_pos = text[:max_length].rfind('\n')
        if split_pos == -1 or split_pos < max_length // 2:
            split_pos = text[:max_length].rfind(' ')
        if split_pos == -1:
            split_pos = max_length
        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    return parts


# Функция для поиска наиболее похожих совпадений по запросу пользователя
def find_closest_matches(query, max_results=5, distance_threshold=0.7):
    query = query.strip().lower()  # Очистка строки
    query_vec = vectorizer.transform([query])  # Векторизация запроса

    # Поиск ближайших совпадений
    distances, indices = model.kneighbors(query_vec, n_neighbors=max_results)

    # Формируем список совпадений с фильтрацией по расстоянию
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if dist <= distance_threshold:
            results.append({
                'Quest': dataset.iloc[idx]['Quest'],
                'Answer': dataset.iloc[idx]['Ans'],
                'URL': dataset.iloc[idx]['URL'],
                'Department': dataset.iloc[idx]['Department'],
                'Distance': dist
            })

    # Сортировка по возрастанию дистанции
    results = sorted(results, key=lambda x: x['Distance'])
    return results


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Задать вопрос")
    markup.add(btn1)
    bot.send_message(message.chat.id,
                     "Привет! Я бот, который поможет найти ответ. Нажми кнопку ниже.",
                     reply_markup=markup)


# Обработчик нажатия на кнопку "Задать вопрос"
@bot.message_handler(func=lambda msg: msg.text == "Задать вопрос")
def ask_question(message):
    msg = bot.send_message(message.chat.id, "Напиши свой вопрос:")
    bot.register_next_step_handler(msg, process_question)


# Обработчик полученного вопроса от пользователя
def process_question(message):
    user_query = message.text
    try:
        # Находим все совпадения по запросу
        matches = find_closest_matches(user_query, max_results=5)

        if not matches:
            # Если совпадений нет — сообщаем об этом
            bot.send_message(message.chat.id, "Извините, не нашлось подходящего ответа.")
            return

        # Берём самое близкое совпадение
        best_match = matches[0]
        user_results[message.chat.id] = matches  # Сохраняем результат по chat_id

        # Формируем клавиатуру с вариантами вывода информации
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("Ответ")
        btn2 = types.KeyboardButton("Отдел")
        btn3 = types.KeyboardButton("URL")
        btn4 = types.KeyboardButton("Новый вопрос")
        markup.add(btn1, btn2, btn3, btn4)

        # Сообщаем пользователю о найденном совпадении
        bot.send_message(message.chat.id,
                         "Нашёл подходящий вариант! Выбери, что хочешь посмотреть:",
                         reply_markup=markup)

    except Exception as e:
        bot.send_message(message.chat.id, f"Произошла ошибка: {str(e)}. Попробуй снова.")


# Обработчик нажатий на кнопки "Ответ", "Отдел", "URL", "Новый вопрос"
@bot.message_handler(func=lambda msg: msg.text in ["Ответ", "Отдел", "URL", "Новый вопрос"])
def handle_choice(message):
    chat_id = message.chat.id

    if chat_id not in user_results:
        bot.send_message(chat_id, "Сначала задай вопрос, нажав кнопку 'Задать вопрос'. (/start)")
        return

    matches = user_results[chat_id]

    if message.text == "Ответ":
        response = ""
        for i, match in enumerate(matches, start=1):
            response += f"🔸 <strong>Вопрос {i}:</strong>\n{match['Quest']}\n\n"
            response += f"<strong>Ответ:</strong>\n{match['Answer']}\n\n"
        parts = split_message(response, MAX_MESSAGE_LENGTH)
        for part in parts:
            bot.send_message(chat_id, part, parse_mode='HTML')

    elif message.text == "Отдел":
        department = matches[0]['Department']
        bot.send_message(chat_id, f"Отдел: {department}")

    elif message.text == "URL":
        url = matches[0]['URL']
        bot.send_message(chat_id, f"Ссылка: {url}")

    elif message.text == "Новый вопрос":
        ask_question(message)


# Точка входа — запуск бота
if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()
