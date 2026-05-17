import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

epochs = list(range(1, 31))

ae_train = [1.0818,0.0014,0.0004,0.0002,0.0001,0.0001,0.0000,0.0000,0.0000,0.0000,
            0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
            0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000]
ae_val   = [0.0032,0.0006,0.0002,0.0001,0.0001,0.0000,0.0000,0.0000,0.0000,0.0000,
            0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
            0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000]

cgan_g = [4.5696,4.8560,4.8483,4.8490,4.8488,4.8441,4.8434,4.8408,4.8383,4.8320,
          4.8241,4.8199,4.8161,4.8096,4.8030,4.7988,4.7916,4.7853,4.7814,4.7745,
          4.7689,4.7614,4.7525,4.7470,4.7412,4.7345,4.7277,4.7198,4.7119,4.7064]
cgan_d = [0.4264,0.3575,0.3519,0.3496,0.3484,0.3474,0.3459,0.3451,0.3456,0.3457,
          0.3468,0.3468,0.3477,0.3492,0.3500,0.3519,0.3530,0.3542,0.3560,0.3582,
          0.3593,0.3617,0.3639,0.3659,0.3669,0.3689,0.3705,0.3730,0.3750,0.3767]

lstm_train = [5.3004,5.1939,5.1862,5.1823,5.1800,5.1784,5.1770,5.1763,5.1753,5.1751,
              5.1746,5.1743,5.1743,5.1740,5.1737,5.1735,5.1730,5.1727,5.1727,5.1671,
              5.0383,4.9760,4.9527,4.9381,4.9286,4.9194,4.9109,4.9028,4.8956,4.8890]
lstm_val   = [5.2026,5.1887,5.1835,5.1794,5.1794,5.1789,5.1783,5.1762,5.1767,5.1768,
              5.1770,5.1760,5.1774,5.1769,5.1766,5.1766,5.1777,5.1769,5.1772,5.1337,
              4.9917,4.9628,4.9448,4.9384,4.9307,4.9248,4.9164,4.9123,4.9053,4.8997]

tr_train = [5.3419,5.2049,5.1940,5.1885,5.1854,5.1841,5.1825,5.1813,5.1828,5.1797,
            5.1795,5.1780,5.1781,5.1774,5.1769,5.1767,5.1767,5.1759,5.1757,5.1752,
            5.1750,5.1746,5.1751,5.1754,5.1741,5.1740,5.1742,5.1734,5.1738,5.1739]
tr_val   = [5.2177,5.1943,5.1813,5.1864,5.1831,5.1804,5.1787,5.1857,5.1811,5.1777,
            5.1774,5.1751,5.1769,5.1748,5.1752,5.1759,5.1739,5.1748,5.1745,5.1751,
            5.1721,5.1747,5.1760,5.1747,5.1762,5.1731,5.1726,5.1725,5.1735,5.1748]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training & Validation Loss — All Models", fontsize=14, fontweight="bold")

ax = axes[0][0]
ax.plot(epochs, ae_train, label="Train", color="steelblue")
ax.plot(epochs, ae_val,   label="Val",   color="orange", linestyle="--")
ax.set_yscale("log")
ax.set_title("AE (Autoencoder)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0][1]
ax.plot(epochs, cgan_g, label="Generator",     color="steelblue")
ax.plot(epochs, cgan_d, label="Discriminator", color="orange", linestyle="--")
ax.set_title("CGAN")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1][0]
ax.plot(epochs, lstm_train, label="Train", color="steelblue")
ax.plot(epochs, lstm_val,   label="Val",   color="orange", linestyle="--")
ax.set_title("LSTM")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(epochs, tr_train, label="Train", color="steelblue")
ax.plot(epochs, tr_val,   label="Val",   color="orange", linestyle="--")
ax.set_title("Transformer")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
print("saved → loss_curves.png")
plt.show()


fig2, ax2 = plt.subplots(figsize=(9, 5))
models     = ["AE", "CGAN", "LSTM", "Transformer"]
csr_all    = [77.3, 100.0, 100.0, 100.0]
csr_leap   = [77.3, 100.0, 100.0, 100.0]
csr_day    = [100.0, 100.0, 100.0, 100.0]

x = range(len(models))
width = 0.3
ax2.bar([i - width for i in x], csr_all,  width, label="Overall CSR", color="steelblue")
ax2.bar([i          for i in x], csr_day,  width, label="Day of Week", color="seagreen")
ax2.bar([i + width  for i in x], csr_leap, width, label="Leap Year",   color="tomato")

ax2.set_xticks(list(x)); ax2.set_xticklabels(models)
ax2.set_ylim(70, 105)
ax2.set_ylabel("Satisfaction Rate (%)")
ax2.set_title("Condition Satisfaction Rate per Model")
ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("csr_chart.png", dpi=150, bbox_inches="tight")
print("saved → csr_chart.png")
plt.show()
