import os
import subprocess

with open(".git/config", "a") as f:
    f.write('\n[remote "origin"]\n\turl = https://github.com/kofi2552/Project_PDS.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n')

print("Config updated. Now running git branch...")
subprocess.run(["git", "branch", "-M", "main"])
print("Branch updated. Now pushing to origin...")
subprocess.run(["git", "push", "-u", "origin", "main"])
print("Push complete.")
