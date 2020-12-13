import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def make_error_graph(error_data, validation_data):
    plt.clf()
    plt.plot(list(error_data.keys()), list(error_data.values()), 'r--', label='Training error')
    plt.plot(list(validation_data.keys()), list(validation_data.values()), 'b-', label='Validation error')

    plt.xlabel("epochs")
    plt.ylabel('error')
    plt.legend()
    plt.pause(0.001)
    plt.show(block=False)
