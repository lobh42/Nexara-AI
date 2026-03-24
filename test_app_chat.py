from ai_engine import chat_query

context = {
    "equipment_health": [],
    "patterns": [],
    "schedule": [],
    "near_misses": [],
}
chat_history = []

user_input = "Hello"
chat_history.append({"role": "user", "content": user_input})
result = chat_query(user_input, context, chat_history)

print("Result:", result)
