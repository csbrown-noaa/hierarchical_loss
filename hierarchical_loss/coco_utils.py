import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def coco_category_dist(coco) -> Figure: # Updated return type
    """Generates a bar plot of the COCO dataset category distribution.

    This function counts all instances of each category ID in the
    `coco.anns` attribute, maps those IDs to their names via
    `coco.cats`, and generates a `matplotlib` bar plot.

    The plot is configured with count labels on top of each bar and
    rotated x-axis labels for readability.

    Parameters
    ----------
    coco : pycocotools.coco.COCO
        A COCO API object.

    Returns
    -------
    matplotlib.figure.Figure
        The generated `matplotlib` Figure object containing the plot.
        (e.g., `fig = coco_category_dist(coco); fig.savefig('dist.png')`)
    """
    cats, cnt = np.unique(list(map(lambda x: x['category_id'], coco.anns.values())), return_counts=True)
    cat_names = list(map(lambda cat_id: coco.cats[cat_id]['name'], cats))
    
    # 1. Create a new Figure and Axes
    fig, ax = plt.subplots()

    # 2. Plot on the Axes object
    ax.bar(cat_names, cnt)
    for i, value in enumerate(cnt):
        ax.text(i, value + 0.5, str(value), ha='center', va='bottom')
    
    # 3. Configure the Axes object
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_names, rotation=45, ha='right')

    # 4. Apply tight layout to the Figure
    fig.tight_layout()
    
    # 5. Return the Figure object
    return fig
