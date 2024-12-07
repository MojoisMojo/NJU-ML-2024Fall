import matplotlib.pyplot as plt


def plot_SEE_K(sees, ks):
    plt.plot(sees, ks, marker="o")
    plt.xlabel("K")
    plt.ylabel("SEE")
    plt.title("SEE-K")
    plt.grid()
    plt.savefig("./images/SEE-K.png")
    plt.close()


if __name__ == "__main__":
    pass
