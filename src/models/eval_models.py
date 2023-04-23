import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        lb = pd.read_csv("reports/results/leaderboard.csv")
    except:
        print("no leaderboard?? run training first")
        return

    plt.figure()
    plt.title("AUC by model (area jam classifier)")
    plt.bar(lb["model"], lb["auc"])
    plt.savefig("reports/figures/model_auc.png", bbox_inches="tight")
    plt.close()
    print("saved reports/figures/model_auc.png")

if __name__ == "__main__":
    main()