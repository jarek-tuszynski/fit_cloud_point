# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:40:37 2022

@author: SATuszynskiJ
"""
from fit_point_cloud import ellipse_pc, rectangle_pc, circle_pc, tunnel_pc
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

# =============================================================================
def triangle_point_cloud():
    n = 500
    sigma = 1
    straigh_edge = lambda p, q: p +  np.outer(nr.random((n,1)), q-p) 
    d = 10
    a = np.array([-d, 0])
    b = np.array([ d, 0])
    c = np.array([0, 2*d]) 
    p1 = straigh_edge(a, b)
    p2 = straigh_edge(b, c)
    p3 = straigh_edge(c, a)
    points = np.vstack((p1, p2, p3))  +  sigma * nr.normal(size=(3*n,2))
    return points 

# ============================================= ================================
def fit_points(points, ax, obj, color):
    obj.fit(points)
    h =  obj.plot(ax, color)
    return h
    
# ============================================= ================================
def triangle_test():
    points = triangle_point_cloud()
    h0 = plt.plot(points[:,0], points[:,1], '.', zorder=1)
    plt.axis('equal')
    plt.axis('off')
    ax = plt.gca()
    
    h1 = fit_points(points, ax, circle_pc(), 'r')
    h2 = fit_points(points, ax, ellipse_pc(), 'm')
    h3 = fit_points(points, ax, rectangle_pc(), 'g')
    h4 = fit_points(points, ax, tunnel_pc(), 'y')
    ax.legend((h0[0], h1, h2, h3, h4), ('point cloud', 'circle', 'ellipse', 'rectangle', 'tunnel'), loc='upper left')

    plt.savefig('fit_triangle.png', dpi=199)
    
# ============================================= ================================
def main():
    triangle_test()

# =============================================================================
if __name__ == "__main__":
    main()