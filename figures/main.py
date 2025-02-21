from plot import Plotter


def main():

    """
    F1 vs network schemes and datasets
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.08, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/f1_scheme_datasets.csv",
        plot_type="bar",
        x="Dataset",
        y="F1 (%)",
        hue="Scheme"
    )

    plotter.add_legend(legend_cols=3)
    plotter.save_or_show("./figures/output/f1_scheme_datasets.pdf")


    """
    F1 and Average Number of Messages vs Number of Peers
    """
    plotter = Plotter(height=3.4, aspect=4/3, num_rows=2, num_cols=3, legend_spacing=0.02, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/f1_peer_llama_mmlu.csv",
        plot_type="line",
        x="# Peers",
        y="F1 (%)",
        style="Algorithm",
        hue="Algorithm",
        subplot_title="(a) Llama 3.2 3B - MMLU",
        markers=True,
        show_grid=True,
    )

    plotter.plot(
        data_path="./figures/data/f1_peer_llama_medical.csv",
        plot_type="line",
        x="# Peers",
        y="F1 (%)",
        style="Algorithm",
        hue="Algorithm",
        subplot_title="(b) Llama 3.2 3B - Medical",
        markers=True,
        show_grid=True,
    )

    plotter.plot(
        data_path="./figures/data/f1_peer_llama_news.csv",
        plot_type="line",
        x="# Peers",
        y="F1 (%)",
        style="Algorithm",
        hue="Algorithm",
        subplot_title="(c) Llama 3.2 3B - News",
        markers=True,
        show_grid=True,
    )

    plotter.plot(
        data_path="./figures/data/anm_peer_llama_mmlu.csv",
        plot_type="line",
        x="# Peers",
        y="Average Number of Messages",
        style="Algorithm",
        hue="Algorithm",
        subplot_title="(d) Llama 3.2 3B - MMLU",
        markers=True,
        show_grid=True,
    )

    plotter.plot(
        data_path="./figures/data/anm_peer_llama_medical.csv",
        plot_type="line",
        x="# Peers",
        y="Average Number of Messages",
        style="Algorithm",
        hue="Algorithm",
        subplot_title="(e) Llama 3.2 3B - Medical",
        markers=True,
        show_grid=True,
    )

    plotter.plot(
        data_path="./figures/data/anm_peer_llama_news.csv",
        plot_type="line",
        x="# Peers",
        y="Average Number of Messages",
        style="Algorithm",
        hue="Algorithm",
        subplot_title="(f) Llama 3.2 3B - News",
        markers=True,
        show_grid=True,
    )

    plotter.add_legend(legend_cols=3)
    plotter.save_or_show("./figures/output/f1_anm_peer.pdf")

    """
    F1 vs Number of Peer Attachment and Number of Peer in Network
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.06, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/f1_peer_attach_llama_mmlu.csv",
        plot_type="bar",
        x="# Peer",
        y="F1 (%)",
        hue="# Peer Attach"
    )

    plotter.add_legend(legend_cols=5)
    plotter.save_or_show("./figures/output/f1_peer_attach.pdf")

    """
    ANM vs Number of Peer Attachment and Number of Peer in Network
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.06, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/anm_peer_attach_llama_mmlu.csv",
        plot_type="bar",
        x="# Peer",
        y="Average Number of Messages",
        hue="# Peer Attach"
    )

    plotter.add_legend(legend_cols=5)
    plotter.save_or_show("./figures/output/anm_peer_attach.pdf")


    """
    F1 vs LLM models and datasets
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.06, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/f1_llm_datasets.csv",
        plot_type="bar",
        x="Dataset",
        y="F1 (%)",
        hue="LLM"
    )

    plotter.add_legend(legend_cols=3)
    plotter.save_or_show("./figures/output/f1_llm_datasets.pdf")

    """
    ANM vs LLM models and datasets
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.06, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/anm_llm_datasets.csv",
        plot_type="bar",
        x="Dataset",
        y="Average Number of Messages",
        hue="LLM"
    )

    plotter.add_legend(legend_cols=3)
    plotter.save_or_show("./figures/output/anm_llm_datasets.pdf")


    """
    F1 vs number of query neighbors
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.08, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/f1_query_neighbor_llama_mmlu.csv",
        plot_type="bar",
        x="# Peers",
        y="F1 (%)",
        hue="# Query Neighbors"
    )

    plotter.add_legend(legend_cols=4)
    plotter.save_or_show("./figures/output/f1_query_neighbor.pdf")

    """
    Average number of messages vs number of query neighbors
    """

    plotter = Plotter(height=4, aspect=3/2, num_rows=1, num_cols=1, legend_spacing=0.08, subplot_title_spacing=0.36)

    plotter.plot(
        data_path="./figures/data/anm_query_neighbor_llama_mmlu.csv",
        plot_type="bar",
        x="# Peers",
        y="Average Number of Messages",
        hue="# Query Neighbors"
    )

    plotter.add_legend(legend_cols=4)
    plotter.save_or_show("./figures/output/anm_query_neighbor.pdf")



if __name__ == "__main__":
    main()
