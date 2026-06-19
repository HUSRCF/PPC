import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DPI = 300
FIG_DIR = 'figures'
COLORS = {
    'final': '#4C78A8',
    'mlc': '#F58518',
    'star': '#54A24B',
    'other': '#9D755D',
}
matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

def save_fig(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    fig.savefig(f'{FIG_DIR}/{name}.png')
    print(f'Saved: {FIG_DIR}/{name}.pdf/.png')
