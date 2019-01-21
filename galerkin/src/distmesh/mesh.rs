use std::fmt;

#[derive(Debug, Copy, Clone)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl fmt::Display for Point2D {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Triangle {
    // Indices referring to the points of the mesh
    pub a: i32,
    pub b: i32,
    pub c: i32,
}

#[derive(Debug)]
pub struct Mesh {
    pub points: Vec<Point2D>,
    pub triangles: Vec<Triangle>,
}

pub struct ExpandedTriangle {
    pub ax: f64,
    pub ay: f64,
    pub bx: f64,
    pub by: f64,
    pub cx: f64,
    pub cy: f64,
}

impl ExpandedTriangle {
    fn a_to_b(&self) -> (f64, f64) {
        (self.bx - self.ax, self.by - self.ay)
    }

    fn b_to_a(&self) -> (f64, f64) {
        (self.ax - self.bx, self.ay - self.by)
    }

    fn b_to_c(&self) -> (f64, f64) {
        (self.cx - self.bx, self.cy - self.by)
    }

    fn c_to_b(&self) -> (f64, f64) {
        (self.bx - self.cx, self.by - self.cy)
    }

    fn c_to_a(&self) -> (f64, f64) {
        (self.ax - self.cx, self.ay - self.cy)
    }

    fn a_to_c(&self) -> (f64, f64) {
        (self.cx - self.ax, self.cy - self.ay)
    }

    fn cross_product(v1: (f64, f64), v2: (f64, f64)) -> f64 {
        let (x1, y1) = v1;
        let (x2, y2) = v2;
        x1 * y2 - y1 * x2
    }

    pub fn contains(&self, point: Point2D, tolerance: f64) -> bool {
        assert!(tolerance >= 0. && tolerance < 1.);

        let a = (self.ax, self.ay);
        let p = (point.x, point.y);
        let det_p_ac = Self::cross_product(p, self.a_to_c());
        let det_a_ac = Self::cross_product(a, self.a_to_c());
        let det_ab_ac = Self::cross_product(self.a_to_b(), self.a_to_c());

        let phi = (det_p_ac - det_a_ac) / det_ab_ac;

        let det_p_ab = Self::cross_product(p, self.a_to_b());
        let det_a_ab = Self::cross_product(a, self.a_to_b());

        let psi = -(det_p_ab - det_a_ab) / det_ab_ac;

        phi > -tolerance && psi > -tolerance && phi + psi < (1. + tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains() {
        let triangle = ExpandedTriangle {
            ax: 0.0,
            ay: 0.0,
            bx: 1.0,
            by: 0.0,
            cx: 0.0,
            cy: 1.0,
        };
        assert!(triangle.contains(Point2D { x: 0.2, y: 0.3 }, 0.));
        assert!(!triangle.contains(Point2D { x: 0.9, y: 0.3 }, 0.));
    }

    #[test]
    fn test_contains_away_from_origin() {
        let triangle = ExpandedTriangle {
            ax: 3.0,
            ay: 5.0,
            bx: 8.0,
            by: 5.0,
            cx: 4.0,
            cy: 7.0,
        };
        assert!(triangle.contains(Point2D { x: 5.0, y: 5.5 }, 0.));
        assert!(!triangle.contains(Point2D { x: 5.8, y: 6.5 }, 0.));
    }

    #[test]
    fn test_chirality() {
        let triangle = ExpandedTriangle {
            ax: 8.0,
            ay: 5.0,
            bx: 3.0,
            by: 5.0,
            cx: 4.0,
            cy: 7.0,
        };
        assert!(triangle.contains(Point2D { x: 5.0, y: 5.5 }, 0.));
        assert!(!triangle.contains(Point2D { x: 5.8, y: 6.5 }, 0.));
    }

    #[test]
    fn test_contains_tolerance() {
        let triangle = ExpandedTriangle {
            ax: 0.0,
            ay: 0.0,
            bx: 1.0,
            by: 0.0,
            cx: 1.0,
            cy: 1.0,
        };
        assert!(!triangle.contains(Point2D { x: 0.5, y: -0.001 }, 0.0));
        assert!(triangle.contains(Point2D { x: 0.5, y: -0.001 }, 0.002));
    }
}
