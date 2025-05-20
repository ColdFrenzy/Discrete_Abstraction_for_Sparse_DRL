import matplotlib.pyplot as plt

def stepwise_spectral_efficiency():
    ranges = [
        ((0.10, 0.13), 10.),
        ((0.13, 0.16), 9.10), # 0.9
        ((0.16, 0.19), 8.20), # 0.9
        ((0.19, 0.22), 7.40), # 0.8     
        ((0.22, 0.25), 6.60), # 0.8
        ((0.25, 0.28), 5.90), # 0.7
        ((0.28, 0.31), 5.30), # 0.6   
        ((0.31, 0.34), 4.80), # 0.5
        ((0.34, 0.37), 4.30), # 0.5
        ((0.37, 0.40), 3.90), # 0.4
        ((0.40, 0.43), 3.60), # 0.3
        ((0.43, 0.46), 3.30), # 0.3
        ((0.46, 0.49), 3.00), # 0.3
        ((0.49, 0.52), 2.80), # 0.2
        ((0.52, 0.55), 2.60), # 0.2
        ((0.55, 0.58), 2.40), # 0.2
        ((0.58, 0.61), 2.40), # no change
        ((0.61, 0.64), 2.40), # no change
        ((0.64, 0.67), 2.20), # 0.2
        ((0.67, 0.70), 2.00), # 0.2
        ((0.70, 0.73), 1.90), # 0.1
        ((0.73, 0.76), 1.80), # 0.1
        ((0.76, 0.79), 1.70), # 0.1
        ((0.79, 0.82), 1.65), # 0.05
        ((0.82, 0.85), 1.61), # 0.04
        ((0.85, 0.88), 1.58), # 0.03
        ((0.88, 0.91), 1.56), # 0.02
        ((0.91, 0.94), 1.54), # 0.02
        ((0.94, 0.97), 1.52), # 0.01
        ((0.97, 1.00), 1.50), # 0.01
    ]
    x = []
    y = []
    for step_range, value in ranges:
        x.append(step_range[0])
        y.append(value)
        x.append(step_range[1])
        y.append(value)

    plt.figure(figsize=(10, 5))
    plt.step(x, y, where='post', label='Stepwise Function')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Stepwise Function')
    plt.legend()
    plt.grid()
    plt.show()


def linearized_spectral_efficiency():
    x = [0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55]
    y = [10., 9.10, 8.20, 7.40, 6.60, 5.90, 5.30, 4.80, 4.30, 3.90, 3.60, 3.30, 3., 2.80, 2.60, 2.40]

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Linearized Function')
    plt.grid()
    plt.show()

if __name__ == "__main__":

    linearized_spectral_efficiency()