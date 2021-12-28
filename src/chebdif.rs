extern crate ndarray;
use crate::common::*;
use ndarray::*;
use std::f64::consts::PI;

/// Calculate differentiation matrices using Chebyshev collocation.
/// Returns the differentiation matrices Dm  corresponding to the
/// der-th derivative of the function f, at the ncheb Chebyshev nodes in the
/// interval [-1,1]. Adapted from [1]
///
/// [1] https://github.com/labrosse/dmsuite
#[allow(non_snake_case)]
pub fn chebdif(n: usize, der: usize) -> (Array1<f64>, Array2<f64>) {
    assert!(n > 0, "der must be greater than 0!");
    assert!(n > der, "number of nodes must be greater than der!");
    let nf = n as f64;
    let k = Array::range(0., nf + 1.0, 1.0);

    // Compute theta vector
    let th = &k * PI / nf;

    // Chebyshev nodes
    let x = PI * (nf - 2.0 * Array::linspace(0., nf, n + 1)) / (2. * nf);
    let x = -x.mapv(f64::sin);

    // Assemble differentiation matrices
    let T = tile(&(th / 2.0));
    // trigonometric identity
    let mut Dx = 2.0 * (&T.t() + &T).mapv(f64::sin) * (&T.t() - &T).mapv(f64::sin);
    // Flipping trick (increases numerical precision)
    let nn1 = ((nf + 1.0) / 2.0).floor() as usize;
    let nn2 = ((nf + 1.0) / 2.0).ceil() as usize;
    let slice = -flipud(&fliplr(&Dx.slice(s![0..nn2, ..])));
    Dx.slice_mut(s![nn1.., ..]).assign(&slice);
    // Set diag
    set_diag(&mut Dx, 1.0);
    let Dx = Dx.t();

    // C
    let mut C = toeplitz1(&k.map(|x| (-1_i32).pow(*x as u32) as f64));
    let slice = Slice::new(0, Some(n as isize + 1), n as isize); // 1 and -1
    for a in C.slice_axis_mut(Axis(0), slice) {
        *a *= 2.;
    }
    for a in C.slice_axis_mut(Axis(1), slice) {
        *a *= 0.5;
    }

    // Z
    let mut Z = 1.0 / &Dx;
    set_diag(&mut Z, 0.);

    // initialize differentiation matrices.
    let mut D = Array2::<f64>::eye(n + 1);

    for deriv in 0..der {
        // O: np.tile(np.diag(D), (ncheb + 1, 1)).
        let O = tile(&D.diag());
        D = (deriv as f64 + 1.0) * &Z * (&C * &O.t() - &D);
        let sum = -D.sum_axis(Axis(1));
        set_diag_from_arr(&mut D, &sum);
    }
    (x, -D)
}
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use assert_approx_eq::assert_approx_eq;
//
//     #[test]
//     fn test_chebdif() {
//         // Init
//         let n = 20;
//         let tol = 1e-4f64;
//         // Run
//         let (_, d1) = chebdif(n, 1);
//         let (x, d2) = chebdif(n, 2);
//         let f = (PI * &x).mapv(f64::sin);
//         let d1f_exp = PI * (PI * &x).mapv(f64::cos);
//         let d2f_exp = PI * PI * (PI * &x).mapv(f64::sin);
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
//     #[should_panic]
//     fn test_chebdif_fail() {
//         // Init
//         let n = 20;
//         // Run
//         let (_, _) = chebdif(n, n + 1); // must panic: der>n
//     }
// }
