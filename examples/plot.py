from alpsplot import Figure, ting_color
import numpy as np

color_dict = {
    'resnet': ting_color['red'],
    'darts': ting_color['green'],
    'enas': ting_color['blue']
}
mark_dict = {
    'resnet': 'H',
    'darts': '^',
    'enas': 'o',
}
# batch mode
# x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # y = [96.58, 96.23, 94.07, 93.13, 92.03, 91.86, 89.46, 85.32, 65.33, 10.33, 10]  # resnet18_comp 300 (abandon)
# y = {
#     'resnet': [93.78, 93.36, 93.12, 92.28, 90.86, 88.72, 85.16, 75.44, 52.47, 10.79, 0.59],
#     'darts': [93.76, 93.14, 92.18, 91.08, 88.74, 86.05, 80.44, 63.33, 33.90, 10.26, 4.73],
#     'enas': [94.09, 92.86, 92.65, 90.76, 89.74, 86.48, 80.48, 63.18, ],
# }

x = [0, 2.5, 5, 10, 20, 30, 40]
y = {
    'resnet': [93.78, 93.79, 93.30, 92.16, 90.58, 88.77, 86.13]
}

if __name__ == '__main__':
    fig = Figure('poison_percent_cifar10')
    fig.set_axis_label('x', 'Poison Percent (%)')
    fig.set_axis_label('y', 'Model Accuracy Drop (%)')
    fig.set_axis_lim('x', lim=[0, 40], piece=4, margin=[1.0, 1.0],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 10], piece=5, margin=[1.0, 1.0],
                     _format='%d')

    for key, value in y.items():
        value = np.array(value)
        fig.curve(x=x[:len(value)], y=value[0] - value, color=color_dict[key])
        fig.scatter(x=x[:len(value)], y=value[0] - value, color=color_dict[key], marker=mark_dict[key], label=key)
    fig.set_legend()
    fig.save(folder_path='./result')
