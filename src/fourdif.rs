extern crate ndarray;
use crate::common::*;
use ndarray::*;
use std::f64::consts::PI;

/// Calculate differentiation matrices using Fourier transform.
/// Returns the differentiation matrices Dm  corresponding to the
/// der-th derivative of the function f, at the uniform distributed
/// nodes in the interval [0,2pi]. Adapted from [1]
///
/// [1] https://github.com/labrosse/dmsuite
#[allow(non_snake_case)]
pub fn fourdif(n: usize, der: usize) -> (Array1<f64>, Array2<f64>) {
    assert!(der > 0, "der must be greater than 0!");
    assert!(der < 3, "der must be less than 3!");
    assert!(n > der, "number of nodes must be greater than der!");
    let nf = n as f64;

    // Grid points
    let x = 2.0 * PI * Array::range(0., nf, 1.0) / nf;
    // grid spacing
    let dx = 2.0 * PI / nf;
    // Indices for flipping trick
    let nn1 = ((nf - 1.0) / 2.0).floor() as usize;
    let nn2 = ((nf - 1.0) / 2.0).ceil() as usize;
    let mut col = Array::zeros(n);
    let mut row = -col.clone();
    // der=1
    if der == 1 {
        // Compute first column of 1st derivative matrix
        let col1 = 0.5 * Array::range(1., nf, 1.0).map(|x| (-1_i32).pow(*x as u32) as f64);
        let topc = if n % 2 == 0 {
            1.0 / (dx / 2. * Array::range(1., nn2 as f64 + 1., 1.0)).mapv(f64::tan)
        } else {
            1.0 / (dx / 2. * Array::range(1., nn2 as f64 + 1., 1.0)).mapv(f64::sin)
        };
        // Stack
        let col1 = if n % 2 == 0 {
            // col1*stack![Axis(0),topc,(-1.*&topc).slice(s![0..nn1;-1])]
            col1 * concatenate(Axis(0), &[topc.view(), (-1. * &topc).slice(s![0..nn1;-1])]).unwrap()
        } else {
            // col1*stack![Axis(0),topc, topc.slice(s![0..nn1;-1])]
            col1 * concatenate(Axis(0), &[topc.view(), topc.slice(s![0..nn1;-1])]).unwrap()
        };
        // let col1 = stack![Axis(0),arr1(&[0.]),col1];
        let col1 = concatenate(Axis(0), &[arr1(&[0.]).view(), col1.view()]).unwrap();

        // First row
        let row1 = -col1.clone();
        // Assign to use outside this scope
        col.assign(&col1);
        row.assign(&row1);
    // der=2
    } else if der == 2 {
        // Compute first column of 1st derivative matrix
        let col1 = -0.5 * Array::range(1., nf, 1.0).map(|x| (-1_i32).pow(*x as u32) as f64);

        let topc = if n % 2 == 0 {
            1.0 / (dx / 2. * Array::range(1., nn2 as f64 + 1., 1.0))
                .mapv(f64::sin)
                .mapv(|a| a.powi(2))
        } else {
            1.0 / (dx / 2. * Array::range(1., nn2 as f64 + 1., 1.0)).mapv(f64::tan)
                / (dx / 2. * Array::range(1., nn2 as f64 + 1., 1.0)).mapv(f64::sin)
        };
        // Stack
        let (c, col1) = if n % 2 == 0 {
            (
                -PI.powi(2) / 3. / dx.powi(2) - 1. / 6.,
                //col1*stack![Axis(0),topc, topc.slice(s![0..nn1;-1])])
                col1 * concatenate(Axis(0), &[topc.view(), topc.slice(s![0..nn1;-1])]).unwrap(),
            )
        } else {
            (
                -PI.powi(2) / 3. / dx.powi(2) + 1. / 12.,
                // col1*stack![Axis(0),topc,(-1.*&topc).slice(s![0..nn1;-1])])
                col1 * concatenate(Axis(0), &[topc.view(), (-1. * &topc).slice(s![0..nn1;-1])])
                    .unwrap(),
            )
        };
        // let col1 = stack![Axis(0),arr1(&[c]),col1];
        let col1 = concatenate(Axis(0), &[arr1(&[c]).view(), col1.view()]).unwrap();

        // First row
        let row1 = col1.clone();
        // Assign to use outside this scope
        col.assign(&col1);
        row.assign(&row1);
    // der>2: Compute via fft
    } else {
        println!("Not implemented. Create via D3=D1.dot(&D2)!");
    }

    let dd = toeplitz(&col, &row);
    (x, dd)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use assert_approx_eq::assert_approx_eq;
//
//     #[test]
//     fn test_fourdif_even() {
//         // Init
//         let n = 20;
//         let tol = 1e-10f64;
//         // Run
//         let (_, d1) = fourdif(n, 1);
//         let (x, d2) = fourdif(n, 2);
//         let f = (&x).mapv(f64::sin);
//         let d1f_exp = 1. * (&x).mapv(f64::cos);
//         let d2f_exp = -1. * (&x).mapv(f64::sin);
//         let d1f = d1.dot(&f);
//         let d2f = d2.dot(&f);
//         // Check
//         for (af, bf) in d1f.iter().zip(d1f_exp.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//         for (af, bf) in d2f.iter().zip(d2f_exp.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//     }
//
//     #[test]
//     fn test_fourdif_odd() {
//         // Init
//         let n = 21;
//         let tol = 1e-10f64;
//         // Run
//         let (_, d1) = fourdif(n, 1);
//         let (x, d2) = fourdif(n, 2);
//         let f = (&x).mapv(f64::sin);
//         let d1f_exp = 1. * (&x).mapv(f64::cos);
//         let d2f_exp = -1. * (&x).mapv(f64::sin);
//         let d1f = d1.dot(&f);
//         let d2f = d2.dot(&f);
//         // Check
//         for (af, bf) in d1f.iter().zip(d1f_exp.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//         for (af, bf) in d2f.iter().zip(d2f_exp.iter()) {
//             assert_approx_eq!(af, bf, tol); // Check correctness
//         }
//     }
// }
