import sys, re

msg = sys.stdin.read().strip()

# Remove conventional commit prefixes
msg = re.sub(r'^(feat|fix|docs|chore|refactor|style|test|perf)(\([^)]*\))?\s*:\s*', '', msg)

# Fix Vietnamese messages
replacements = {
    'update kaggle notebook voi log-residual, giam lr, tang early stopping de toi uu mape': 'update kaggle notebook with log-residual, lower lr, higher early stopping',
}

# Check for Vietnamese chars and replace known ones
viet_map = {
    'ghi nh\u1eadn k\u1ebft qu\u1ea3 chung cu\u1ed9c v\u00f2ng 3 xu\u1ea5t s\u1eafc': 'record round 3 final results',
    'c\u1eadp nh\u1eadt k\u1ebft qu\u1ea3 v\u00f2ng 2': 'record round 2 results',
}

if msg in replacements:
    msg = replacements[msg]
if msg in viet_map:
    msg = viet_map[msg]

print(msg)
