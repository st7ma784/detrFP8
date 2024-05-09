import os
import shutil
endings=[".py",".yml","Dockerfile",".md"]
os.makedirs('codeForAssistant', exist_ok=True)
for root, dirs, files in os.walk('.'):
    if not files:
        continue
    for file in files: 
        if any([file.endswith(a) for a in endings]):
            os.makedirs(os.path.join("codeForAssistant",*file.split("\\")[:-1]), exist_ok=True)
            try:
                shutil.copy(os.path.join(root, file), os.path.join('codeForAssistant',*file.split("\\")[:-1], file.split("\\")[-1].split(".")[0] + '.txt'))
            except Exception as e:
                print(e)