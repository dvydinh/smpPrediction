import os
import subprocess

result = subprocess.run(['git', 'log', '--reverse', '--format=%h|%s'], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

commits = []
for line in lines:
    if not line: continue
    parts = line.split('|', 1)
    commits.append((parts[0], parts[1]))

os.system('git checkout --orphan temp_root')
os.system('git rm -rf .')

author_str = 'dvydinh <doanvy.dinh27@gmail.com>'

first = True
for h, msg in commits:
    if ':' in msg:
        prefix = msg.split(':')[0]
        if ' ' not in prefix and len(prefix) < 10:
            msg = msg.split(':', 1)[1].strip()
            
    print(f"Applying {h}: {msg}")
    
    if first:
        os.system(f'git read-tree {h}')
        os.system(f'git checkout {h} -- .')
        os.system('git add .')
        subprocess.run(['git', 'commit', '-m', msg, f'--author={author_str}'], check=False)
        first = False
    else:
        os.system(f'git cherry-pick {h}')
        subprocess.run(['git', 'commit', '--amend', '-m', msg, f'--author={author_str}', '--allow-empty'], check=False)

os.system('git branch -f main temp_root')
os.system('git checkout main')
os.system('git push -f origin main')
os.system('git branch -D temp_root')
