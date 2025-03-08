content = 'from web_api import app'
content = content.replace(
    'from web_api import app',
    'try:\n    from web_api import app\nexcept ImportError:\n    from service.web_api import app'
)
print(content)

with open('test_output.txt', 'w') as f:
    f.write(content)
print('Written to test_output.txt') 