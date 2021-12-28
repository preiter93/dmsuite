extern crate realfft;
extern crate rustfft;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
extern crate ndarray;
use ndarray::*;

/// Wrapper around fourdifft_vec to use with ndarray
pub fn fourdifft(indata: &mut Array1<f64>, der: usize) -> Array1<f64> {
    // Convert ndarray to vec
    let mut invec: Vec<f64> = indata.to_vec();
    // Call fourdifft_vec
    let outvec = fourdifft_vec(&mut invec, der);
    // Convert vec to ndarray
    Array::from(outvec)
}

// Perform differntiation via real to complex FFT
// Input:
// indata:	Vector of type Vec<f64> containing function values
//          on the intervall [0,2pi[
// der: 	Derivative order of type usize
// Output:
// outdata: Vector containing the approximate der'th derivative
pub fn fourdifft_vec(indata: &mut Vec<f64>, der: usize) -> Vec<f64> {
    let length = indata.len();
    // make a planner
    let mut real_planner = RealFftPlanner::<f64>::new();
    // create a FFT
    let r2c = real_planner.plan_fft_forward(length);
    // create an iFFT
    let c2r = real_planner.plan_fft_inverse(length);
    // output vector
    let mut spectrum = r2c.make_output_vec();
    // Forward transform the input data
    r2c.process(indata, &mut spectrum).unwrap();
    // Differentiate
    for (j, i) in &mut spectrum.iter_mut().enumerate() {
        let k = Complex::<f64>::new(0.0, j as f64);
        *i *= k.powi(der as i32);
    }
    // create an output vector
    let mut outdata = c2r.make_output_vec();
    // Backward transform of the spectrum data
    c2r.process(&mut spectrum, &mut outdata).unwrap();
    // Normalize
    for i in &mut outdata {
        *i /= length as f64;
    }
    outdata
}

#[allow(dead_code)]
// #[allow(unused_variables)]
// Perform fft and ifft (for testing)
fn fft_ifft(indata: &mut Vec<f64>) -> Vec<f64> {
    let length = indata.len();
    // make a planner
    let mut real_planner = RealFftPlanner::<f64>::new();
    // create a FFT
    let r2c = real_planner.plan_fft_forward(length);
    // create an iFFT
    let c2r = real_planner.plan_fft_inverse(length);

    // output vector
    let mut spectrum = r2c.make_output_vec();
    // Forward transform the input data
    r2c.process(indata, &mut spectrum).unwrap();
    // create an output vector
    let mut outdata = c2r.make_output_vec();
    // Backward transform of the spectrum data
    c2r.process(&mut spectrum, &mut outdata).unwrap();
    // Normalize
    for i in &mut outdata {
        *i /= length as f64;
    }
    outdata
}
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use assert_approx_eq::assert_approx_eq;
//     use itertools_num::linspace;
//     extern crate time_test;
//     use std::f64::consts::PI;
//     use time_test::*;
//     const N: usize = 10000;
//
//     #[test]
//     fn test_fourdifft() {
//         time_test!();
//         // Init
//         let n = N;
//         let l = 2. * PI;
//         let dx = l / n as f64;
//         let mut f: Vec<f64> = linspace::<f64>(0., 2. * PI - dx, n).collect();
//         for i in &mut f {
//             *i = i.sin();
//         }
//         let mut d1f: Vec<f64> = linspace::<f64>(0., 2. * PI - dx, n).collect();
//         for i in &mut d1f {
//             *i = i.cos();
//         }
//         let mut d2f: Vec<f64> = linspace::<f64>(0., 2. * PI - dx, n).collect();
//         for i in &mut d2f {
//             *i = -i.sin();
//         }
//
//         // Run
//         let mut indata: Array1<f64> = Array::from(f.clone());
//         let d1f_fourdiff = fourdifft(&mut indata, 1); // First derivative
//         let mut indata: Array1<f64> = Array::from(f.clone());
//         let d2f_fourdiff = fourdifft(&mut indata, 2); // Second derivative
//         {
//             time_test!("FFT Time");
//         }
//         // Check element-wise (maybe replace by norm)
//         let tol = 1e-5f64;
//         for (af, bf) in d1f.iter().zip(d1f_fourdiff.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//         for (af, bf) in d2f.iter().zip(d2f_fourdiff.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//     }
// }
