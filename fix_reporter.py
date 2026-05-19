"""Fix the invalid escape sequence in reporter_agent.py"""
import pathlib

p = pathlib.Path(r"c:\Users\aliei\OneDrive\Desktop\generative_AI\project\agents\reporter_agent.py")
content = p.read_text(encoding="utf-8")

# Show the raw bytes of line 96 for debugging
lines = content.splitlines(keepends=True)
print(f"Line 96 repr: {repr(lines[95])}")

# The fix: replace the backslash-pipe with a raw-string version
old_fragment = r".replace('|', '\|')"
new_fragment = r".replace('|', r'\|')"

if old_fragment in content:
    content = content.replace(old_fragment, new_fragment)
    p.write_text(content, encoding="utf-8")
    print("FIXED: replaced invalid escape sequence")
else:
    print("Pattern not found - checking alternative encoding...")
    # Try looking for the actual bytes
    raw = p.read_bytes()
    idx = raw.find(b"replace")
    if idx >= 0:
        print(f"Found 'replace' at byte {idx}: {raw[idx:idx+30]}")
