import os
import shutil

os.makedirs('codeForAssistant', exist_ok=True)
for root, dirs, files in os.walk('.'):
    if not files:
        continue
    for file in files: 
        if file.endswith('.py') or file.endswith(".yml"):
            os.makedirs(os.path.join("codeForAssistant",*file.split("\\")[:-1]), exist_ok=True)
            try:
                shutil.copy(os.path.join(root, file), os.path.join('codeForAssistant',*file.split("\\")[:-1], file.split("\\")[-1].split(".")[0] + '.txt'))
            except Exception as e:
                print(e)