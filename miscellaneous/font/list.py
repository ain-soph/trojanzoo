# %%
import matplotlib.font_manager
from IPython.core.display import HTML


def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)


code = "\n".join([make_html(font) for font in sorted(
    set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))

# %%
