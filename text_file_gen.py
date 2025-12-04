import os
current_pwd = os.getcwd()
recurrece_plots_path = os.path.join(current_pwd, "recurrence_plots")
files = os.listdir(recurrece_plots_path)
print(files)
with open(os.path.join(recurrece_plots_path, "data.txt"), "w") as f:
    for file in files:
        if file != "data.txt":
            f.write(file + "\n")

