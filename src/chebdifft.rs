extern crate ndarray;
use ndarray::*;
use realfft::RealFftPlanner;
use rustfft::{num_complex::Complex, FftPlanner};

// De-alise: Cut spectral data, keep only [0..N*ALIAS] Modes
pub const ALIAS: f64 = 2. / 3.;

// Perform chebyshev spectral differentiation via real to complex FFT
// Input:
// f:	    Vector of type Vec<f64> containing function valuesat the Chebyshev points
//          x(k) = cos((k)*pi/(N-1)), k = 0...N-1.
// der: 	Derivative order of type usize
// Output:
// outdata: Vector containing the approximate der'th derivative
//
// Note:
// This version uses the faster real to complex fft
pub fn chebdifft(f: &mut Array1<f64>, der: usize) -> Array1<f64> {
    let n = f.len();
    let n_alias: usize = (n as f64 * ALIAS) as usize;
    //// Setup real to complex FFT
    let length = 2 * n - 2;
    // Planner
    let mut real_planner = RealFftPlanner::<f64>::new();
    // create a FFT
    let r2c = real_planner.plan_fft_forward(length);
    // output vector
    let mut spectrum = r2c.make_output_vec();

    // Extend and compute fft
    let a0 = concatenate(Axis(0), &[f.view(), (f).slice(s![1..n-1;-1])]).unwrap();
    // Forward transform of real input vector
    let mut indata: Vec<f64> = a0.to_vec();
    r2c.process(&mut indata, &mut spectrum).unwrap();

    // Unpack Real part, a0 contains Chebychev coefficient of f
    let mut a0: Array1<f64> =
        Array::from(spectrum[0..n].iter().map(|x| x.re).collect::<Vec<f64>>());
    a0 /= n as f64 - 1.0;
    a0[0] *= 0.5;
    a0[n - 1] *= 0.5;

    // Dealise: Set large cheby modes to zero
    for i in n - n_alias..n {
        a0[i] *= 0.;
    }

    // Recursion formula for computing coefficients
    // See: Spectral Methods in Fluid Dynamics by Claudio Canuto
    // Eq. (2.4.26)
    let mut a: Array2<f64> = Array::zeros((n, der + 1));
    a.slice_mut(s![.., 0]).assign(&a0);
    for ell in 1..der + 1 {
        a[[n - ell - 1, ell]] = 2. * (n - ell) as f64 * a[[n - ell, ell - 1]];
        for k in (1..n - ell - 1).rev() {
            a[[k, ell]] = a[[k + 2, ell]] + 2. * (k + 1) as f64 * a[[k + 1, ell - 1]];
            a[[0, ell]] = a[[1, ell - 1]] + a[[2, ell]] / 2.0;
        }
    }

    // Transform back into physical space
    let mut indata: Vec<f64> = vec![2. * &a[[0, der]]];
    indata.append(&mut a.slice(s![1..n - 1, der]).to_vec());
    indata.push(2. * &a[[n - 1, der]]);
    indata.append(&mut a.slice(s![1..n-1;-1, der]).to_vec());

    // Forward transform of real input vector
    r2c.process(&mut indata, &mut spectrum).unwrap();
    // Unpack Real part
    Array::from(spectrum.iter().map(|x| 0.5 * x.re).collect::<Vec<f64>>())
}

/// Perform chebyshev spectral differentiation via FFT (complex to complex)
/// Input:
///```
/// f:	    Vector of type Vec<f64> containing function valuesat the Chebyshev points
///          x(k) = cos((k)*pi/(N-1)), k = 0...N-1.
/// der: 	Derivative order of type usize
/// Output:
/// outdata: Vector containing the approximate der'th derivative
///```
/// Note:
/// This version uses the c2c fft to transform to chebyshev space,
/// which does redundant operations for real valued input (better: fn chebdifft)
#[allow(dead_code)]
pub fn chebdifft_c2c(f: &mut Array1<f64>, der: usize) -> Array1<f64> {
    let n = f.len();
    let n_alias: usize = (n as f64 * ALIAS) as usize;
    // Setup FFT
    let length = 2 * n - 2;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(length);
    let mut buffer = vec![
        Complex {
            re: 0.0f64,
            im: 0.0f64
        };
        length
    ];
    // Extend and compute fft
    let a0 = concatenate(Axis(0), &[f.view(), (f).slice(s![1..n-1;-1])]).unwrap();
    // Transfer data to buffer
    for (i, val) in &mut buffer.iter_mut().enumerate() {
        val.re = a0[i];
    }
    // Perform forward fft
    fft.process(&mut buffer);
    // Unpack Real part, a0 contains Chebychev coefficient of f
    let mut a0: Array1<f64> = Array::from(buffer[0..n].iter().map(|x| x.re).collect::<Vec<f64>>());
    a0 /= n as f64 - 1.0;
    a0[0] *= 0.5;
    a0[n - 1] *= 0.5;

    // Dealise: Set large cheby modes to zero
    for i in n - n_alias..n {
        a0[i] *= 0.;
    }

    // Recursion formula for computing coefficients
    let mut a: Array2<f64> = Array::zeros((n, der + 1));
    a.slice_mut(s![.., 0]).assign(&a0);
    for ell in 1..der + 1 {
        a[[n - ell - 1, ell]] = 2. * (n - ell) as f64 * a[[n - ell, ell - 1]];
        for k in (1..n - ell - 1).rev() {
            a[[k, ell]] = a[[k + 2, ell]] + 2. * (k + 1) as f64 * a[[k + 1, ell - 1]];
            a[[0, ell]] = a[[1, ell - 1]] + a[[2, ell]] / 2.0;
        }
    }

    // Transform back into physical space
    let mut indata: Vec<f64> = vec![2. * &a[[0, der]]];
    indata.append(&mut a.slice(s![1..n - 1, der]).to_vec());
    indata.push(2. * &a[[n - 1, der]]);
    indata.append(&mut a.slice(s![1..n-1;-1, der]).to_vec());
    // Transfer data to buffer
    for (i, val) in &mut buffer.iter_mut().enumerate() {
        val.re = indata[i];
    }
    // Perform fft
    fft.process(&mut buffer);
    // Unpack Real part
    Array::from(
        buffer[0..n]
            .iter()
            .map(|x| 0.5 * x.re)
            .collect::<Vec<f64>>(),
    )
}
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use assert_approx_eq::assert_approx_eq;
//     extern crate time_test;
//     use std::f64::consts::PI;
//     use time_test::*;
//     const N: usize = 10_000;
//     const TOL: f64 = 1e-5;
//
//     #[test]
//     fn test_chebdifft_c2c() {
//         time_test!();
//         // Init
//         let n = N;
//         let k = Array::range(0., n as f64, 1.0);
//         let x = (PI * k / (n as f64 - 1.0)).mapv(f64::cos); //Chebyshev nodes
//         let f = (&x * PI).mapv(f64::sin);
//         let d1f = PI * (&x * PI).mapv(f64::cos);
//         let d2f = -1. * PI * PI * (&x * PI).mapv(f64::sin);
//         // Run
//         let mut indata: Array1<f64> = f.clone();
//         let d1f_cheb = chebdifft_c2c(&mut indata, 1); // First derivative
//         let mut indata: Array1<f64> = f.clone();
//         let d2f_cheb = chebdifft_c2c(&mut indata, 2); // First derivative
//         {
//             time_test!("CHEBDIFFT Time");
//         }
//         // Check element-wise (maybe replace by norm)
//         let tol = TOL;
//         for (af, bf) in d1f.iter().zip(d1f_cheb.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//         for (af, bf) in d2f.iter().zip(d2f_cheb.iter()) {
//             assert_approx_eq!(af, bf, tol * 1000.); // Check correctness
//         }
//     }
//
//     #[test]
//     fn test_chebdifft_r2c() {
//         time_test!();
//         // Init
//         let n = N;
//         let k = Array::range(0., n as f64, 1.0);
//         let x = (PI * k / (n as f64 - 1.0)).mapv(f64::cos); //Chebyshev nodes
//         let f = (&x * PI).mapv(f64::sin);
//         let d1f = PI * (&x * PI).mapv(f64::cos);
//         let d2f = -1. * PI * PI * (&x * PI).mapv(f64::sin);
//         // Run
//         let mut indata: Array1<f64> = f.clone();
//         let d1f_cheb = chebdifft(&mut indata, 1); // First derivative
//         let mut indata: Array1<f64> = f.clone();
//         let d2f_cheb = chebdifft(&mut indata, 2); // First derivative
//         {
//             time_test!("CHEBDIFFT Time");
//         }
//         // Check element-wise (maybe replace by norm)
//         let tol = TOL;
//         for (af, bf) in d1f.iter().zip(d1f_cheb.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//         for (af, bf) in d2f.iter().zip(d2f_cheb.iter()) {
//             assert_approx_eq!(af, bf, tol * 1000.); // Check correctness
//         }
//     }
// }
