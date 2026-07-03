import requests, json

# Hero
r = requests.get('http://127.0.0.1:8000/api/hero')
print('HERO:', r.status_code, r.json().get('title','ERROR'))

# All media
r2 = requests.get('http://127.0.0.1:8000/api/media')
print('ALL MEDIA:', r2.status_code, 'count =', len(r2.json()))

# Category filter
r3 = requests.get('http://127.0.0.1:8000/api/media/Anime')
print('ANIME:', r3.status_code, 'count =', len(r3.json()))

# Recommendations
r4 = requests.get('http://127.0.0.1:8000/recommend/Inception')
d4 = r4.json()
print('RECS for Inception:', r4.status_code)
for rec in d4.get('recommendations', []):
    print('  ->', rec['title'], f"(score={rec['score']})")
