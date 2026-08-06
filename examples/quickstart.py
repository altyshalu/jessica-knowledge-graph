#!/usr/bin/env python3
"""
JKG 7.1 Quickstart — запусти и проверь память за 30 секунд.
Требуется: pip install -e .
"""
import os
from jkg import HybridMemory

# 1. Инициализация
hm = HybridMemory()

# 2. Запомнить факты
print("🧠 Запоминаю факты...")
hm.remember("""
Алиса — CTO в Acme Corp, AI-стартапе из Сан-Франциско.
Acme Corp делает AI-агентов для enterprise. 
Главный конкурент — BetaCorp (YC W24).
Алиса использует Python, PyTorch и Kubernetes.
""")
print(f"  Статистика: {hm.stats()}")

# 3. Графовый поиск (точные связи)
print("\n🔍 Граф: Где работает Алиса?")
for r in hm.recall("Где работает Алиса")[:3]:
    print(f"  {r['entity']} {r['predicate']} {r['value']}")

# 4. Семантический поиск (синонимы!)
print("\n🔍 Семантика: Кто rivals компании Алисы?")
for r in hm.recall("rivals соперники конкуренты Алисы")[:3]:
    print(f"  {r['entity']} {r['predicate']} {r['value']} (RRF: {r.get('rrf_score', 0):.4f})")

# 5. Путь между сущностями
print("\n🗺️  Путь: Алиса → BetaCorp?")
path = hm.find_path("алиса", "betacorp")
if path.get("path"):
    print(f"  {' '.join(path['path'])} ({path['length']} hops)")

# 6. Вопрос-ответ
print("\n💬 ASK: Расскажи про компанию Алисы и её конкурентов")
result = hm.ask("Расскажи про компанию Алисы и её конкурентов")
print(f"  {result['answer']}")

print("\n✅ JKG 7.1 работает!")
