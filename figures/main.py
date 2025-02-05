from plot import Plotter


def main():
    plotter1 = Plotter(height=3, aspect=14/9, num_cols=3, legend_spacing=0.05, subplot_title_spacing=0.28)

    plotter1.plot(
        data_path="./figures/data/aqh_vs_peer_llama_mmlu.csv",
        plot_type="bar",
        x="# Peers",
        y="Average Query Hit",
        hue="Algorithm",
        ax=plotter1.axes[0]
    )
    # plotter1.set_subplot_title(plotter1.axes[0], "Llama 3.2 3B - MMLU")
    plotter1.set_subplot_title(plotter1.axes[0], title="(a) Llama 3.2 3B - MMLU")

    plotter1.plot(
        data_path="./figures/data/aqh_vs_peer_llama_medical.csv",
        plot_type="bar",
        x="# Peers",
        y="Average Query Hit",
        hue="Algorithm",
        ax=plotter1.axes[1]
    )
    # plotter1.set_subplot_title(plotter1.axes[0], "Llama 3.2 3B - Medical")
    plotter1.set_subplot_title(plotter1.axes[1], title="(b) Llama 3.2 3B - Medical")

    plotter1.plot(
        data_path="./figures/data/aqh_vs_peer_llama_news.csv",
        plot_type="bar",
        x="# Peers",
        y="Average Query Hit",
        hue="Algorithm",
        ax=plotter1.axes[2]
    )
    # plotter1.set_subplot_title(plotter1.axes[0], "Llama 3.2 3B - News")
    plotter1.set_subplot_title(plotter1.axes[2], title="(c) Llama 3.2 3B - News")

    plotter1.add_legend(legend_cols=3)
    plotter1.save_or_show("./figures/output/aqh_vs_peer.pdf")


if __name__ == "__main__":
    main()
