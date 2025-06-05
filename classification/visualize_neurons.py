import argparse
import os
import torch
import config as CFG
from modules import RobertaCBL, GPT2CBL, CBL
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
device = torch.device("cpu")

parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--model", type=str, default="roberta", help="Model type: roberta or gpt2")
parser.add_argument("--w_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/W_g.pt")
parser.add_argument("--b_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/b_g.pt")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--plot", action='store_true', help="Add this to store the plot of the Sankey diagram.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    W = torch.load(args.w_path, map_location=device)
    b = torch.load(args.b_path, map_location=device)
    fc_weight = W.T  # [num_concepts, num_labels]

    concept_set = CFG.concept_set[args.dataset]
    label_names = CFG.concepts_from_labels[args.dataset]
    top_k = args.top_k

    for class_idx, label in enumerate(label_names):
        print(f"\nTop {top_k} activated neurons for class: {label}")
        top_values, top_ids = torch.topk(fc_weight[:, class_idx], k=top_k)
        for i in range(top_k):
            neuron_id = top_ids[i].item()
            activation = top_values[i].item()
            print(f"  [{activation:.4f}] {concept_set[neuron_id]}")

# Build Sankey source-target-value lists
if args.plot:
    sankey_sources = []
    sankey_targets = []
    sankey_values = []
    sankey_labels = concept_set + label_names  # Nodes = concepts + labels

    for class_idx, label in enumerate(label_names):
        top_values, top_ids = torch.topk(fc_weight[:, class_idx], k=top_k)
        for value, concept_id in zip(top_values, top_ids):
            sankey_sources.append(concept_id.item())  # concept index
            sankey_targets.append(len(concept_set) + class_idx)  # label index
            sankey_values.append(value.item())

    # Sankey Plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_labels
        ),
        link=dict(
            source=sankey_sources,
            target=sankey_targets,
            value=sankey_values
        )
    )])
    cbl_type = os.path.splitext(os.path.basename(args.w_path))[0]
    filename = f"sankey_top{top_k}_{args.model}_{args.dataset.replace('/', '_')}_{cbl_type}.html"
    
    fig.update_layout(
    title_text="Top Activated Neurons per Class (Sankey)",
    font_size=10,
    annotations=[
        dict(
            text=filename,  # show the filename
            x=0.5,
            y=-0.1,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=12)
        )
    ]
)
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(args.w_path)), f"{args.model}_plot")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.write_html(os.path.join(plot_dir, filename))
    print(f"Sankey plot saved to {os.path.join(plot_dir, filename)}")
