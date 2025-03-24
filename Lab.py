from scipy.stats import norm
from csv import writer

# powierzchnia pozioma
# distribution_x = norm(loc=0, scale=200)
# distribution_y = norm(loc=0, scale=20)
# distribution_z = norm(loc=0.2, scale=0.05)

# powierzchnia pionowa
# distribution_x = norm(loc=0, scale=200)
# distribution_y = norm(loc=0, scale=20)
# distribution_z = norm(loc=0.2, scale=0.05)

# powierzchnia cylindryczna
distribution_x = norm(loc=0, scale=20)
distribution_y = norm(loc=0, scale=20)
distribution_z = norm(loc=50, scale=200)

num_points:int=2000
x = distribution_x.rvs(size=num_points)
y = distribution_y.rvs(size=num_points)
z = distribution_z.rvs(size=num_points)

points = zip(x, y ,z)

with open('Data2.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)
