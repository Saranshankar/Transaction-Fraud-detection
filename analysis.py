import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/creditcard_reduced.csv")

counts = data["Class"].value_counts()

plt.bar(["Normal","Fraud"], counts)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")

plt.show()
