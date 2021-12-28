extern crate ndarray;
extern crate ndarray_linalg;
use dmsuite::*;
use gnuplot::{AutoOption::Fix, AxesCommon, Color, Figure};
use ndarray::*;
use ndarray_linalg::*;

// Number of grid points
const SIM_N: usize = 180;

// Total time
const SIM_TIME: f64 = 10.;

/// Solve heat equation in 1D
pub struct Heat1D {
    u: Array1<f64>,
    x: Array1<f64>,
    delta_t: f64,
    time: f64,
    mat: Array2<f64>,
}

impl Heat1D {
    pub fn new(dim: usize) -> Heat1D {
        // Set solution vector
        let u = Array1::<f64>::zeros(dim);
        // Get Fourier differentiation matrix
        let (x, d2) = fourdif(dim, 2);
        // Get timestep size
        let delta_t = 0.1;
        let time = 0.;
        // Build matrix
        let mut a = Array2::<f64>::eye(dim) - delta_t * d2;
        let eye = Array2::<f64>::eye(dim);
        a.slice_mut(s![0, ..]).assign(&eye.slice(s![0, ..])); // BC LEFT
        a.slice_mut(s![dim - 1, ..])
            .assign(&eye.slice(s![dim - 1, ..])); // BC RIGHT
        let mat = a.inv().unwrap();
        Heat1D {
            u,
            x,
            delta_t,
            time,
            mat,
        }
    }

    pub fn plot(&self) {
        let x: Vec<f64> = self.x.to_vec();
        let y: Vec<f64> = self.u.to_vec();
        let mut fg = Figure::new();
        let axes = fg.axes2d();
        // Scale the x data so it lines up with the width given.
        axes.lines(&x, &y, &[Color("red")]);
        // Set the window.
        axes.set_x_range(Fix(0.), Fix(x[x.len() - 1]));
        axes.set_y_range(Fix(0.), Fix(1.));
        // "wxt" is the GUI name and "768,768" is the size of it in pixels.
        fg.set_terminal("wxt size 768,768", "");
        fg.show().unwrap();
    }

    pub fn iterate(&mut self, max_time: f64) {
        loop {
            self.update();
            println!("Time: {:.5}", self.time);
            if self.time > max_time {
                break;
            }
        }
    }

    pub fn update(&mut self) {
        let n = self.u.len();
        //(1-k*dt*D2) unew = uold
        self.u = self.mat.dot(&self.u);
        // Set edge
        self.u[0] = 0.0;
        self.u[n - 1] = 0.0;
        // Update time
        self.time += self.delta_t;
    }
}

fn main() {
    // Initialize
    let mut sim = Heat1D::new(SIM_N);
    sim.u = sim.x.mapv(|x| (0.5 * x).sin()); // Initial disturbance
    sim.plot();
    sim.iterate(SIM_TIME);
    sim.plot();
    println!("{:?}", sim.u[SIM_N / 2]);
}
