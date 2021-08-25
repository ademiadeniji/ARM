from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
import numpy as  np 
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import PIL
from PIL import Image
import wandb 

name = '/home/mandi/snake4_TheRevenge.png'
buffer = BytesIO()
image = Image.open(name)
image.save(buffer, format='png')
for_encoding = buffer.getvalue()
encoded = 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

embeddings = np.ones((40,2))
df = pd.DataFrame(embeddings, columns=('x', 'y'))

title = 'TSNE visualization of tasks'
output_file(f'{title}.html')

embeddings = np.ones((40,2))
df = pd.DataFrame(embeddings, columns=('x', 'y'))

title = 'TSNE visualization of tasks'
output_file(f'{title}.html')
TOOLTIPS = """
<div>
    <div>
        <img 
            src='@image'  height="42" width="42" 
            style='float: left; margin: 0px 0px 0px 0px;'
        ></img>
    </div>

</div>
"""

plot_figure = figure(
    title=title,
    plot_width=600,
    plot_height=600,
#     tools=('pan, wheel_zoom, reset'),
    tooltips=TOOLTIPS,
)

task_names = ['button' for _ in range(40)]
# df['name'] = task_names
# df['image'] = ['one_batch.png' for _ in range(40)]

datasource = ColumnDataSource({
    'x': [1,2,3,4],
    'y': [3,1,2,9],
    'image': [encoded for _ in range(4)]

})

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    #color=dict(field='digit', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=40
)

show(plot_figure)

run = wandb.init(project='MTARM', job_type='visualize')
run.log({title: wandb.Html( open(f'{title}.html')) })
