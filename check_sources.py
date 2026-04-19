import os
import re

with open('CMakeLists.txt', 'r') as f:
    content = f.read()

# Find set(CECE_SRCS ...)
match = re.search(r'set\(CECE_SRCS(.*?)\)', content, re.DOTALL)
sources = []
if match:
    sources.extend(match.group(1).split())

# Find list(APPEND CECE_SRCS ...)
matches = re.findall(r'list\(APPEND CECE_SRCS(.*?)\)', content, re.DOTALL)
for m in matches:
    sources.extend(m.split())

missing = []
for src in sources:
    if src.startswith('${'): continue
    if not os.path.exists(src):
        missing.append(src)

if missing:
    print("Missing files:")
    for m in missing:
        print(m)
else:
    print("All files exist.")
