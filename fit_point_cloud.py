# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 08:29:37 2022

@author: SATuszynskiJ
"""
import numpy as np
import numpy.random as nr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from   matplotlib.patches import Ellipse, Rectangle, Circle, Polygon
from numpy.linalg import norm

# =============================================================================
# === circle class
# =============================================================================
class circle_pc:
    # circle is (x-k)^2 + (y-m)^2 = r^2
    
    #--------------------------------------------------------------------------
    def __init__(self,  xyr=[None, None, None]):
        self.init(xyr)

    #--------------------------------------------------------------------------
    def init(self, xyr):
        self.k, self.m, self.r = xyr[0], xyr[1], xyr[2]
    
    #--------------------------------------------------------------------------
    def get(self):
        return np.array([self.k, self.m, self.r])

    #--------------------------------------------------------------------------
    def plot(self, ax, color='g'):
        ax.add_patch(Circle((self.k, self.m), self.r, edgecolor=color, lw=2, 
                      fc='None', zorder=2))
    
    #--------------------------------------------------------------------------
    def point_cloud(self, n, sigma, fraction=1):   
        theta  = fraction * 2 * np.pi * nr.random((n,1))
        circle = np.hstack([np.cos(theta), np.sin(theta)]) 
        origin = np.array([self.k, self.m])
        noise  = sigma * nr.normal(size=(n,2))
        return origin + self.r*circle + noise    
    
    #--------------------------------------------------------------------------
    def distance(self, points):
        '''
        mean of distances from point array "points" to a circle defined by array 
        xyr =[k,m,r] where circle is (x-k)^2 + (y-m)^2 = r^2
        '''
        origin = np.array([self.k, self.m])
        pts = points - origin# change the origin to (k,m)
        R = norm(pts, axis=1)
        return np.abs(R - self.r).mean()
    
    #--------------------------------------------------------------------------
    def fit(self, points):
        '''
        fit a circle to a point cloud data

        '''
        # (x-k)^2 + (y-m)^2 = r^2
        # x^2 + y^2 = (2*k) *x + (2*m) *y + (r^2 - k^2 - m^2)
        x = points[:,0]
        y = points[:,1]
        r2 = x*x + y*y
        b = np.ones_like(x)
        A = np.column_stack([x, y, b])
        t = np.linalg.lstsq(A,r2,rcond=None)[0].squeeze() # solve least square ||At-r^2||^2
        self.k = t[0]/2
        self.m = t[1]/2
        self.r = np.sqrt(t[2] + self.k**2 + self.m**2)
        return self.get()

    #--------------------------------------------------------------------------
    def test(self, display=False, tol=1.1):
        p0 = [-50, 50, 30]
        self.init(p0)

        n_point = 500  # number of points
        sigma   = 5    # randomness
        fraction = 0.8
        points  = self.point_cloud(n_point, sigma, fraction)
        d0 = self.distance(points)

    
        if display:
            plt.plot(points[:,0], points[:,1], 'b.', zorder=1)
            plt.axis('equal')
            ax = plt.gca()
            self.plot(ax, 'r')
        
        self.fit(points)
        p1 = self.get().astype(int)
        d1 = self.distance(points)
        print(p0, p1, d1/d0)
        assert d1/d0<1.05, 'circle test failed'
        if display:
            self.plot(ax, 'g')


# =============================================================================
# === ellipse class
# =============================================================================
class ellipse_pc:
    # ellipse defined as (x-k)^2/a^2 + (y-m)^2/b^2 = 1

    #--------------------------------------------------------------------------
    def __init__(self,  xyab=[None, None, None, None]):
        self.init(xyab)

    #--------------------------------------------------------------------------
    def init(self, xyab):
        self.k, self.m, self.a, self.b = xyab[0], xyab[1], xyab[2], xyab[3]
    
    #--------------------------------------------------------------------------
    def get(self):
        return np.array([self.k, self.m, self.a, self.b])

    #--------------------------------------------------------------------------
    def plot(self, ax, color='g'):
        ax.add_patch(Ellipse(xy=(self.k, self.m), width=2*self.a, height=2*self.b, 
                       edgecolor=color, lw=2, fc='None', zorder=2))

    #--------------------------------------------------------------------------
    def point_cloud(self, n, sigma, fraction=1):
        theta   = fraction * 2 * np.pi * nr.random((n,1))
        ellipse = np.hstack([self.a*np.cos(theta), self.b*np.sin(theta)]) 
        origin  = np.array([self.k, self.m])
        noise   = sigma * nr.normal(size=(n,2))
        return origin + ellipse + noise   
    
    #--------------------------------------------------------------------------
    def distance(self, points):
        '''
        mean of distances from point array "points" to an ellipse defined by array 
        xyab =[k, m, a, b] where ellipse is (x-k)^2/a^2 + (y-m)^2/b^2 = 1
        '''
        d_sum = 0
        for pt in points: 
            e = self.closest_point(pt)
            d_sum += norm(pt-e)   
        return d_sum / points.shape[0]  
    
    #--------------------------------------------------------------------------
    def fit(self, points):
        '''
        fit a ellipse to a point cloud data
        '''
        x = points[:,0]
        y = points[:,1]
        one = np.ones_like(x)
        A = np.column_stack([x*x, y*y, x, y])
        t = np.linalg.lstsq(A,one,rcond=None)[0].squeeze() # solve least square ||At-1||^2
        # from t[0] *x^2 + t[1] *y^2 + t[2] *x + t[3] *y = 1 get m, k , a, b
        success = self._convert(t)
        if not success:
            t = circle_pc().fit(points)  
            self.init(t[0], t[1], t[2], t[2])
        return self.get()
        
    #--------------------------------------------------------------------------
    def _convert(self, t):
        ''' Convert elipse from t[0] *x^2 + t[1] *y^2 + t[2] *x + t[3] *y = 1 
        to (x-k)^2/a^2 + (y-m)^2/b^2 = 1 representation
        '''
        # (x-k)^2/a + (y-m)^2/b = 1
        # (x^2-2kx+k^2)/a + (y^2-2my+m^2)/b = 1
        # (1/a) *x^2 + (1/b) *y^2 + (-2k/a) *x + (-2m/b) *y + (k^2/a + m^2/b -1) = 0
        # if d = k^2/a + m^2/b -1
        #  (-1/a*d) *x^2 + (-1/b*d) *y^2 + (2k/a*d) *x + (2m/b*d) *y = 1 
        if t[0]==0 or t[1]==0:
            return False

        # in this function a = self.a^2 and b = self.b^2
        k = -t[2]/(2*t[0]) # k = (-1/2) * (2k/a*d) / (-1/a*d)
        m = -t[3]/(2*t[1]) # m = (-1/2) * (2m/b*d) / (-1/b*d)
        c = t[1]/t[0]  # c = (-1/b*d) / (-1/a*d) = a/b -> a = bc
    
        # 1/t[0] + 1/t[1] = -(a+b)*d = -(c+1)*d*b -> d*b = -(1/t[0] + 1/t[1]) / (c+1)
        db = -(1/t[0] + 1/t[1]) / (c+1)
        # d = k^2/a + m^2/b -1 = k^2/cb + m^2/b -1 -> d*b+b = k^2/c + m^2
        b = k**2/c + m**2 - db
        a = b*c
        if a<0 or b<0:
            return False
        
        # verify results
        d =  k**2/a + m**2/b -1
        tt = np.array([ -1/(a*d), -1/(b*d), 2*k/(a*d), 2*m/(b*d)])
        dt = norm(t-tt)
        assert dt<1e-10, 'error'
        
        self.k, self.m, self.a, self.b = k, m, np.sqrt(a), np.sqrt(b)
        return True
        
    #--------------------------------------------------------------------------
    def closest_point(self, p0):
        '''
        distance from point "p0" to the ellipse
        '''
        origin = np.array([self.k, self.m])
        a, b = self.a, self.b
        c = a*a - b*b
        g = np.array([c/a, -c/b]) 
        f = np.array([a, b]) 
        p = np.abs(p0-origin) # point in top right quarter
        t = 0.7071 * np.ones((2))
        
        for i in range(3):
            d = f * t     # current best guess about location of the closest point
            e = g * t**3
            r = d-e
            q = p-e
            t = (q * norm(r) / norm(q) + e)/f
            t = np.clip(t, 0, 1)
            t /= norm(t) # make into unit vector
        
        cp = np.copysign(f * t, p0-origin) + origin
        return cp
    
    #--------------------------------------------------------------------------
    def test(self, display=False, tol=1.1):
        p0 = [50, 50, 30, 20]
        self.init(p0)

        n_point = 500  # number of points
        sigma   = 5    # randomness
        fraction = 0.8
        points  = self.point_cloud(n_point, sigma, fraction)
        d0 = self.distance(points)

    
        if display:
            plt.plot(points[:,0], points[:,1], 'b.', zorder=1)
            plt.axis('equal')
            ax = plt.gca()
            self.plot(ax, 'r')
        
        p1 = self.fit(points).astype(int)
        d1 = self.distance(points)
        print(p0, p1, d1/d0)
        assert d1/d0<1.05, 'ellipse test failed'
        if display:
            self.plot(ax, 'g')
    
# =============================================================================
# === rectangle class
# =============================================================================
class rectangle_pc:
    # rectangle is defined by left, bottom point and width/height
            
    #--------------------------------------------------------------------------
    def __init__(self,  lbwh=[None, None, None, None]):
        self.init(lbwh)

    #--------------------------------------------------------------------------
    def init(self, lbwh):
        self.l, self.b, self.w, self.h = lbwh[0], lbwh[1], lbwh[2], lbwh[3]
    
    #--------------------------------------------------------------------------
    def get(self):
        return np.array([self.l, self.b, self.w, self.h ])

    #--------------------------------------------------------------------------
    def plot(self, ax, color='g'):
        ax.add_patch(Rectangle(xy=(self.l, self.b), width=self.w, height=self.h, 
                         edgecolor=color, lw=2, fc='None', zorder=2))

    #--------------------------------------------------------------------------
    def _get_corners(self):
        a = np.array([self.l, self.b])
        b = np.array([0     , self.h]) + a
        c = np.array([self.w, self.h]) + a
        d = np.array([self.w, 0     ]) + a
        return a, b, c, d

    #--------------------------------------------------------------------------
    def _perimeter(self, n):
        m = np.abs(np.array([self.h, self.w, self.h, self.w]))
        m = np.round(m*n/m.sum()).astype(int)
        m[0] = n - m[1:].sum()
        return m

    #--------------------------------------------------------------------------
    def point_cloud(self, n, sigma, fraction):
        p = self._perimeter(n)
        a, b, c, d = self._get_corners()
        straigh_edge = lambda p, q, m:  p + (q-p) * nr.random((m,2)) +  sigma * nr.normal(size=(m,2))
        k = round(fraction*4)+1
        p1 = straigh_edge(a, b, p[0])
        p2 = straigh_edge(b, c, p[1])
        if k>=3:
            p3 = straigh_edge(c, d, p[2])
        if k>=4:       
            p4 = straigh_edge(d, a, p[3])
        points = np.vstack((p1, p2, p3, p4))
        return points  
    
    #--------------------------------------------------------------------------
    def distance(self, points):
        '''  mean of distances from point array "points" to the rectangle  ''' 
        return self._distance_balance(points)[0]
    
    #--------------------------------------------------------------------------
    def _distance_balance(self, points):
        '''
        mean of distances from point array "points" to the rectangle
        also returns panalty for imbalenced fit, when most points fit one edge
        '''             
        a, b, c, d = self._get_corners()
        d1 = pts2line(points, a, b)
        d2 = pts2line(points, b, c)
        d3 = pts2line(points, c, d)
        d4 = pts2line(points, d, a)
        dist = np.hstack((d1, d2, d3, d4))
        d_mean = dist.min(axis=1).mean()
        
        n_point = points.shape[0]
        count1  = np.bincount(dist.argmin(axis=1), minlength=4)  # actual count
        count0  = self._perimeter(n_point)  # expected count based on equal distribution
        balance = np.sum(np.abs(count1 - count0))/n_point # fraction of the points in the wrong bin
        #balance = (max(count)-min(count))/sum(count)
        return np.array([d_mean, d_mean*balance])
          
    #--------------------------------------------------------------------------
    def fit(self, points):
        '''
        use iterative optimization function to find a rectangle that best fits the data
    
        '''
        def pts2rectangle(ltwh, points):
            d = rectangle_pc(ltwh)._distance_balance(points)
            return d.sum()

        #f = ellipse_pc()
        t = ellipse_pc().fit(points)
        f = 0.9
        t0 = np.array([t[0]-f*t[2], t[1]-f*t[3], 2*f*t[2], 2*f*t[3]])    
        #res = minimize(pts2rectangle, t0, args=points, method='Nelder-Mead', options={'xatol':0.05})
        res = minimize(pts2rectangle, t0, args=points, method='Powell', options={'xtol':0.05})
        self.init(res['x'])
        return self.get()
    
    #--------------------------------------------------------------------------
    def test(self, display=False, tol=1.1):
        p0 = [-80, -50, 50, 40]
        self.init(p0)

        n_point = 500  # number of points
        sigma   = 4    # randomness
        fraction = 0.7
        points  = self.point_cloud(n_point, sigma, fraction)
        d0 = self.distance(points)

    
        if display:
            plt.plot(points[:,0], points[:,1], 'b.', zorder=1)
            plt.axis('equal')
            ax = plt.gca()
            self.plot(ax, 'r')
        
        p1 = self.fit(points).astype(int)
        d1 = self.distance(points)
        print(p0, p1, d1/d0)
        assert d1/d0<1.05, 'rectangle test failed'
        if display:
            self.plot(ax,'g')
        

# =============================================================================
# === tunnel class
# =============================================================================
class tunnel_pc:
    '''rectangle is defined by left, bottom point, width & height of the lower 
    # section and height of the dome modeled as a half of a n ellipse '''
            
    #--------------------------------------------------------------------------
    def __init__(self,  lbwhd=[None, None, None, None, None]):
        self.init(lbwhd)

    #--------------------------------------------------------------------------
    def init(self, t):
        self.l, self.b, self.w, self.h, self.d = t[0], t[1], t[2], t[3], t[4]
    
    #--------------------------------------------------------------------------
    def get(self):
        return np.array([self.l, self.b, self.w, self.h, self.d ])

    #--------------------------------------------------------------------------
    def plot(self, ax, color='g'):
        a, b, c, d = self._get_corners()      
        poly = np.vstack((a, self._get_dome(100), d, a))
        return ax.add_patch(Polygon(xy=poly, edgecolor=color, lw=2, fc='None', zorder=2))

    #--------------------------------------------------------------------------
    def _get_dome(self, n):
        ''' get coordinates of the dome part with "n" points'''
        anchor = np.array([self.l, self.b+self.h])
        x = np.linspace(-1, 1, n)[...,None]
        y = np.sqrt(1-x**2)
        dome = np.hstack((self.w/2*(1+x), self.d*y))
        return anchor + dome

    #--------------------------------------------------------------------------
    def _get_corners(self):
        ''' get coordinates of 4 corners of the rectangle part '''
        a = np.array([self.l, self.b])
        b = np.array([0     , self.h]) + a
        c = np.array([self.w, self.h]) + a
        d = np.array([self.w, 0     ]) + a
        return a, b, c, d

    #--------------------------------------------------------------------------
    def _perimeter(self, n):
        a, b= self.w/2, self.d # half-width or "a";  dome height or "b"
        p = np.pi * np.sqrt((a*a + b*b)/2) # appriximate perimeter of the dome 
        m = np.abs(np.array([self.h, p, self.h, self.w]))
        m = np.round(m*n/m.sum()).astype(int)
        m[0] = n - m[1:].sum()
        return m

    #--------------------------------------------------------------------------
    def point_cloud(self, n, sigma, fraction):
        straigh_edge = lambda p, q, m:  p + (q-p) * nr.random((m,2)) +  sigma * nr.normal(size=(m,2))
        m = self._perimeter(n)
      
        a, b, c, d = self._get_corners()
        k = round(fraction*4)+1
        p1 = straigh_edge(a, b, m[0])
        p2 = self._get_dome(m[1]) + sigma*nr.normal(size=(m[1],2))               
        if k>=3:
            p3 = straigh_edge(c, d, m[2])
        if k>=4:       
            p4 = straigh_edge(d, a, m[3])
        points = np.vstack((p1, p2, p3, p4))
        return points  
    
    #--------------------------------------------------------------------------
    def distance(self, points):
        '''  mean of distances from point array "points" to the rectangle  ''' 
        return self._distance_balance(points)[0]
    
    #--------------------------------------------------------------------------
    def _distance_balance(self, points):
        '''
        mean of distances from point array "points" to the rectangle
        also returns panalty for imbalenced fit, when most points fit one edge
        '''             
        a, b, c, d = self._get_corners()
        d1 = pts2line(points, a, b)       
        d2 = self.pts2dome(points, (b+c)/2 )      
        d3 = pts2line(points, c, d)
        d4 = pts2line(points, d, a)
        dist = np.hstack((d1, d2, d3, d4))
        d_mean = dist.min(axis=1).mean()
        
        n_point = points.shape[0]
        count1  = np.bincount(dist.argmin(axis=1), minlength=4)  # actual count
        count0  = self._perimeter(n_point)  # expected count based on equal distribution
        balance = np.sum(np.abs(count1 - count0))/n_point # fraction of the points in the wrong bin
        
        return np.array([d_mean, d_mean*balance])
    
    #--------------------------------------------------------------------------
    def pts2dome(self, points, origin):
        ''' Approximate distance between each point and upper half of an ellipse.
            Calculate:
             1) d1 - vertical distance from each point to the upper half of an ellipse 
             2) d2 - distance to a point of intersection between ellipse and 
                    (0,0)->(x,y) line
             3) d3 & d4 - distance to the left and right-most point
            '''
        a, b= self.w/2, self.d # half-width or "a";  dome height or "b"
        pts = points - origin  # pt in a coordinate system centered at ellipse origin
        # at this point distance between pt to the dome is tha same as a distance 
        # from point "p" to upper half of x^2/a^2 + y^2/b^2 = 1 ellipse
        c = 1/(a*a)
        x, y = pts[:,0], pts[:,1]
        d1 = np.abs(y - b*np.sqrt(np.abs(1 - c*x*x))) # vertical distance
        d1[np.abs(x)>a] = 1e6 # distance not good for points not overlapping with the ellipse

        r = norm(pts, axis=1) # distance from each point to the origin
        c = np.maximum(np.sqrt(b*b*x*x + a*a*y*y), 1e-6)
        e = a*b / c # e*points is an intersection between ellipse and (0,0)->(x,y)
        d2 = np.abs(r*(1-e))
        d2[y<0] = 1e6 # distance not good for points below the horizontal axis
        
        d3 = np.sqrt((x-a)**2 + y**2) # distance to the rightmost point
        d4 = np.sqrt((x+a)**2 + y**2) # distance to the  leftmost point     
        d = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))[..., None]
        return d
        
    #--------------------------------------------------------------------------
    def fit2(self, points):
        '''
        use iterative optimization function to find a rectangle that best fits the data    
        '''
        def pts2tunnel(ltwhd, points):
            d = tunnel_pc(ltwhd)._distance_balance(points)
            return d.sum() # add the mean distance and a measure adding panalty for badly balanced fit

        t = ellipse_pc().fit(points)
        f = 0.9
        t0 = np.array([t[0]-f*t[2], t[1]-f*t[3], 2*f*t[2], f*t[3], f*t[3]])    
        res = minimize(pts2tunnel, t0, args=points, method='Nelder-Mead', options={'xatol':0.05})
        self.init(res['x'])
        return self.get()
    
    #--------------------------------------------------------------------------
    def fit(self, points):
        '''
        use iterative optimization function to find a rectangle that best fits the data    
        '''
        t = rectangle_pc().fit(points)
        def pts2tunnel(hd, points):            
            ltwhd = [t[0], t[1], t[2], hd[0], hd[1]]
            d = tunnel_pc(ltwhd)._distance_balance(points)
            return d.sum() # add the mean distance and a measure adding panalty for badly balanced fit

        t0 = np.array([t[3]/2, t[3]/2])    
        res = minimize(pts2tunnel, t0, args=points, method='Nelder-Mead', options={'xatol':0.05})
        hd = res['x']
        self.init([t[0], t[1], t[2], hd[0], hd[1]])
        return self.get()
    
    #--------------------------------------------------------------------------
    def test(self, display=False, tol=1.1):
        p0 = [25, -50, 50, 30, 15]
        self.init(p0)

        n_point = 500  # number of points
        sigma   = 4    # randomness
        fraction = 1
        points  = self.point_cloud(n_point, sigma, fraction)
        d0 = self.distance(points)
    
        if display:
            h1 = plt.plot(points[:,0], points[:,1], 'b.', zorder=1)
            plt.axis('equal')
            plt.axis('off')
            plt.ylim([-70,100])
            ax = plt.gca()
            h2 = self.plot(ax,'r')
        
            # el = ellipse_pc()
            # el.fit(points)
            # el.plot(ax, 'c')
            # t = el.get()
            # t0 = np.array([t[0]-t[2], t[1]-t[3], 2*t[2], t[3], t[3]])  
            # self.init(t0)
            # self.plot(ax,'m')

        p1 = self.fit(points).astype(int)
        d1 = self.distance(points)
        print(p0, p1, d1/d0)
        assert d1/d0<1.05, 'tunnel test failed'
        if display:
            h3 = self.plot(ax, 'g')
            ax.legend((h1[0], h2, h3), ('point cloud', 'original', 'best fit'), 
                      loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07))
        

# =============================================================================
# === point to simple shape functions
# =============================================================================
def pts2line(pt, a, b):
    '''
    distance from point "p" to a line segment defined by endpoints a, b
    '''
    v = b-a
    u = pt-a
    m = norm(v) # length of v
    t = np.dot(u/m, v/m)
    t = np.clip(t, 0, 1)
    d = norm(np.outer(t,v) - u, axis=1)  # t*v is the nearest point; d is a distance from it to pt
    return d[..., None]

    
# ============================================= ================================
def self_test():
    display = True
    tol = 1.1 # 10% diviation from the original is allowed
    f = circle_pc()
    f.test(display, tol)

    f = ellipse_pc()
    f.test(display, tol)
    
    f = rectangle_pc()
    f.test(display, tol)
    
    f = tunnel_pc()
    f.test(display, tol)
    if display:
        plt.savefig('point cloud fit.png', dpi=199)
    

# =============================================================================
if __name__ == "__main__":
    self_test()
