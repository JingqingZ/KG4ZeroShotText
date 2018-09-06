import os

def clean_folders():
    basefolder = "../results/"
    for subdir in os.listdir(basefolder):
        print(subdir)
        save_epoch = list()
        for logfile in os.listdir(basefolder + subdir + "/logs/"):
            if "test" in logfile and "full" not in logfile:
                os.remove(basefolder + subdir + "/logs/" + logfile)
            elif "test" in logfile and "full" in logfile:
                save_epoch.append(logfile.split("_")[2].split(".")[0])
        for modelfile in os.listdir(basefolder + subdir + "/models/"):
            if "checkpoint" in modelfile:
                continue
            else:
                keep = False
                for epoch in save_epoch:
                    if ("-%s." % epoch) in modelfile:
                        keep = True
                        break
                if not keep:
                    # print(modelfile)
                    os.remove(basefolder + subdir + "/models/" + modelfile)


if __name__ == "__main__":
    clean_folders()
