from trojanplot import Figure, ting_color


if __name__ == '__main__':
    fig = Figure('poison_percent_cifar10')
    fig.set_axis_label('x', 'Poison Percent (%)')
    fig.set_axis_label('y', 'Model Accuracy (%)')
    fig.set_axis_lim('x', lim=[0, 100], piece=5, margin=[5.0, 5.0],
                     _format='%d')
    fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                     _format='%d')

    x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y = [96.58, 96.23, 96.23, 96.23, 96.23, 96.23, 96.23, 96.23, 96.23, 96.23, 10]

    fig.curve(x=x, y=y, color=ting_color['blue'])
    fig.scatter(x, y, color=ting_color['blue'])

    fig.save(folder_path='./result')
