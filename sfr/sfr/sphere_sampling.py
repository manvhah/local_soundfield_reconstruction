import numpy as np

def fibonacci_sphere(samples):
    """ fibonacci grid on a unit sphere surface
    """
    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - pow(y,2))
        phi = ((i + 1) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x,y,z])

    return np.array(points)

if __name__ == "__main__":
    fib_sphere = fibonacci_sphere(100)
