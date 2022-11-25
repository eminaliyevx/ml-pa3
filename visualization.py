import matplotlib.pyplot as plt


# 2. Visualization
def plot(df):
    # Qiymet (Y) vs Yurush (X1)
    df.plot(kind="scatter", x="Yurush", y="Qiymet")
    plt.show()

    # Qiymet (Y) vs Buraxilish ili (X2)
    df.plot(kind="scatter", x="Buraxilish ili", y="Qiymet")
    plt.show()

    # 3D plot of all three values (Y, X1, X2)
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Yurush")
    ax.set_ylabel("Buraxilish ili")
    ax.set_zlabel("Qiymet")
    ax.scatter3D(df["Yurush"], df["Buraxilish ili"], df["Qiymet"])
    plt.show()
